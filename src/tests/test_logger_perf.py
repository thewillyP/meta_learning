"""Logger speed + durability tests for SQLiteLogger.

Two kinds of test here:

1. Deterministic correctness of the speed-related changes (always run, no data needed):
   - writes are buffered and only hit disk on flush()
   - flush() commits the whole batch in one transaction
   - writes are append-only (no index, so write cost stays flat at scale); replayed points just append
   - the compat `scalars` view dedups to the newest row per (series, iteration) and keeps the old shape

2. Full-pipeline throughput parity (slow, skipped without datasets):
   runs the real main.py flow (app.runApp) once with HDF5Logger and once with SQLiteLogger on the
   same reduced VAE_BETA_OHO config and asserts sqlite keeps pace with hdf5. NOTE: total wall-clock is
   dominated by JAX compile + the 3-level meta unroll, so this is a coarse regression guard against the
   logger becoming a bottleneck again (the old per-insert path was ~3x slower) — not a microbenchmark.
   The sharp number is the `[timing] ... log=Xs` segment app.py prints during a real run.
"""

import dataclasses
import os
import sqlite3
import tempfile
import time

import pytest

import configs as configs_module
from meta_learn_lib.config import GodConfig
from meta_learn_lib.config_converter import make_converter
from meta_learn_lib.checkpoint import NullCheckpointManager
from meta_learn_lib.logger import SQLiteLogger, HDF5Logger
from meta_learn_lib import app

PIPELINE_EPOCHS = int(os.environ.get("PIPELINE_EPOCHS", "50"))


def make_sqlite(dirpath: str, task_id: str) -> SQLiteLogger:
    return SQLiteLogger(dirpath, task_id, 1)


def log_point(lg: SQLiteLogger, series: str, iteration: int, value: float, run_label: str) -> None:
    lg.log_scalar(series, run_label, value, iteration, 1000, 0)


def test_writes_are_buffered_until_flush():
    d = tempfile.mkdtemp()
    lg = make_sqlite(d, "buf")
    for s in range(5):
        log_point(lg, f"train/loss/{s}", 1, float(s), "runA")
    reader = sqlite3.connect(f"file:{lg.db_file}?mode=ro", uri=True)
    assert reader.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 0  # nothing on disk yet
    lg.flush()
    assert reader.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 5
    reader.close()


def test_flush_commits_whole_batch_atomically():
    d = tempfile.mkdtemp()
    lg = make_sqlite(d, "batch")
    for it in range(1, 4):
        for s in range(3):
            log_point(lg, f"train/loss/{s}", it, float(it * 10 + s), "runA")
        lg.flush()
    con = sqlite3.connect(lg.db_file)
    assert con.execute("SELECT COUNT(*) FROM scalar_data").fetchone()[0] == 9
    assert con.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 9  # via compat view
    con.close()


def test_replay_view_returns_newest_value():
    """A preempt+resume replays steps from the last checkpoint. Writes are append-only (duplicates allowed
    in storage); the compat view resolves each (series, iteration) to the newest row by rowid."""
    d = tempfile.mkdtemp()
    lg = make_sqlite(d, "replay")
    for it in range(1, 6):
        log_point(lg, "train/loss", it, float(it), "runA")
        lg.flush()
    con = sqlite3.connect(lg.db_file)
    assert con.execute("SELECT COUNT(*) FROM scalar_data").fetchone()[0] == 5
    # replay iterations 3..5 with new values, as a resumed run would
    for it in range(3, 6):
        log_point(lg, "train/loss", it, float(it * 100), "runA")
        lg.flush()
    assert con.execute("SELECT COUNT(*) FROM scalar_data").fetchone()[0] == 8  # storage appends the replays
    rows = con.execute("SELECT iteration, value FROM scalars ORDER BY iteration").fetchall()
    con.close()
    assert rows == [(1, 1.0), (2, 2.0), (3, 300.0), (4, 400.0), (5, 500.0)]  # view shows the newest per point


def test_compat_view_matches_old_schema():
    d = tempfile.mkdtemp()
    lg = make_sqlite(d, "view")
    log_point(lg, "train/level0/loss", 7, 1.5, "runA")
    lg.flush()
    con = sqlite3.connect(lg.db_file)
    cols = [r[1] for r in con.execute("PRAGMA table_info(scalars)")]
    assert cols == ["series", "iteration", "value", "run_label"]
    assert con.execute("SELECT series, iteration, value, run_label FROM scalars").fetchone() == (
        "train/level0/loss",
        7,
        1.5,
        "runA",
    )
    con.close()


@pytest.mark.slow
def test_sqlite_vs_hdf5_write_scaling():
    """Head-to-head: does sqlite's write cost grow past hdf5's as the table fills? A crossover can only
    happen if sqlite degrades with size while hdf5 stays flat. Measure each backend's real per-step write
    throughput in a window on an empty table, then again after the table has grown by ~0.5M rows, and
    compare. No JAX — pure write path, runs on CPU."""
    n_series = 24
    window = 2000  # iterations measured at real per-step (per-flush) cadence
    fill = 20_000  # iterations written between the two windows to grow the table
    max_count = 2 * window + fill + 10

    def sqlite_window(lg, start: int) -> float:
        t = time.perf_counter()
        for it in range(start, start + window):
            for s in range(n_series):
                log_point(lg, f"train/loss/{s}", it, float(it + s), "run")
            lg.flush()
        return window * n_series / (time.perf_counter() - t)

    def sqlite_fill(lg, start: int) -> None:
        for it in range(start, start + fill):
            for s in range(n_series):
                log_point(lg, f"train/loss/{s}", it, float(it + s), "run")
            if it % 1000 == 0:
                lg.flush()
        lg.flush()

    def hdf5_window(lg, start: int, n: int) -> float:
        t = time.perf_counter()
        for it in range(start, start + n):
            for s in range(n_series):
                lg.log_scalar(f"train/loss/{s}", "run", float(it + s), it, max_count, 0)
        return n * n_series / (time.perf_counter() - t)

    d = tempfile.mkdtemp()
    sq = make_sqlite(d, "sqscale")
    sq_small = sqlite_window(sq, 1)
    sqlite_fill(sq, 1 + window)
    sq_large = sqlite_window(sq, 1 + window + fill)

    h = HDF5Logger(d, "hscale", 1)
    h_small = hdf5_window(h, 1, window)
    hdf5_window(h, 1 + window, fill)  # grow it; timing ignored
    h_large = hdf5_window(h, 1 + window + fill, window)
    h.file.flush()

    rows = sqlite3.connect(sq.db_file).execute("SELECT COUNT(*) FROM scalar_data").fetchone()[0]
    print(
        f"\nwrite scaling over {rows:,} rows (real per-step cadence):"
        f"\n  sqlite: empty={sq_small:,.0f} rows/s  full={sq_large:,.0f} rows/s  self-slowdown={sq_small / sq_large:.2f}x"
        f"\n  hdf5:   empty={h_small:,.0f} rows/s  full={h_large:,.0f} rows/s  self-slowdown={h_small / h_large:.2f}x"
        f"\n  sqlite/hdf5: empty={sq_small / h_small:.2f}x  full={sq_large / h_large:.2f}x"
    )
    # neither backend may degrade as its own table grows ...
    assert sq_large >= sq_small / 1.5, f"sqlite degraded {sq_small / sq_large:.2f}x as table grew"
    assert h_large >= h_small / 1.5, f"hdf5 degraded {h_small / h_large:.2f}x as table grew"
    # ... and sqlite must not lose ground to hdf5 at large size vs small (that would be the crossover)
    assert (sq_large / h_large) >= (sq_small / h_small) * 0.7, "sqlite lost ground to hdf5 as the table grew"


def pipeline_config(epochs: int) -> GodConfig:
    """main.py's config, shrunk: clearml_run on, no sample generators, no checkpointing."""
    converter = make_converter()
    c = configs_module.VAE_BETA_OHO
    c = dataclasses.replace(c, clearml_run=True, epochs=epochs, sample_generators=[], checkpoint_every_n_epochs=0)
    return converter.structure(converter.unstructure(c), GodConfig)


@pytest.mark.slow
def test_sqlite_keeps_pace_with_hdf5_full_pipeline():
    import os
    import jax

    # the VAE pipeline runs at x32 like main.py; conftest enables x64 for the math tests, so override here
    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        config = pipeline_config(epochs=PIPELINE_EPOCHS)
        if not os.path.isdir(config.data_root_dir):
            pytest.skip(f"data_root_dir not found: {config.data_root_dir}")

        d = tempfile.mkdtemp()
        # warmup: populates the JAX compilation cache so neither timed run pays compile cost (logger is not in the JIT)
        app.runApp(pipeline_config(epochs=2), [], NullCheckpointManager())

        t0 = time.perf_counter()
        app.runApp(config, [HDF5Logger(d, "hdf5", config.checkpoint_every_n_minibatches)], NullCheckpointManager())
        t_hdf5 = time.perf_counter() - t0

        t0 = time.perf_counter()
        app.runApp(config, [SQLiteLogger(d, "sqlite", config.checkpoint_every_n_minibatches)], NullCheckpointManager())
        t_sqlite = time.perf_counter() - t0
    finally:
        jax.config.update("jax_enable_x64", prev_x64)

    rows = sqlite3.connect(f"{d}/metrics_sqlite.sqlite").execute("SELECT COUNT(*) FROM scalars").fetchone()[0]
    print(
        f"\nfull pipeline: hdf5={t_hdf5:.2f}s  sqlite={t_sqlite:.2f}s  ratio={t_sqlite / t_hdf5:.2f}x  sqlite_rows={rows}"
    )
    assert rows > 0
    assert t_sqlite <= t_hdf5 * 1.5, f"sqlite {t_sqlite:.2f}s vs hdf5 {t_hdf5:.2f}s — logger regressed"
