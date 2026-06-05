"""Logger speed + durability tests for SQLiteLogger.

Two kinds of test here:

1. Deterministic correctness of the speed-related changes (always run, no data needed):
   - writes are buffered and only hit disk on flush()
   - flush() commits the whole batch in one transaction
   - a replayed (series, iteration) after a preempt overwrites instead of duplicating (idempotent, like hdf5)
   - the compat `scalars` view returns the pre-normalization shape so readers are unaffected

2. Full-pipeline throughput parity (slow, skipped without datasets):
   runs the real main.py flow (app.runApp) once with HDF5Logger and once with SQLiteLogger on the
   same reduced VAE_BETA_OHO config and asserts sqlite keeps pace with hdf5. NOTE: total wall-clock is
   dominated by JAX compile + the 3-level meta unroll, so this is a coarse regression guard against the
   logger becoming a bottleneck again (the old per-insert path was ~3x slower) — not a microbenchmark.
   The sharp number is the `[timing] ... log=Xs` segment app.py prints during a real run.
"""

import dataclasses
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


def test_replay_overwrites_not_duplicates():
    """A preempt+resume replays steps from the last model checkpoint; those rows must overwrite, not pile up."""
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
    rows = con.execute("SELECT iteration, value FROM scalars ORDER BY iteration").fetchall()
    con.close()
    assert rows == [(1, 1.0), (2, 2.0), (3, 300.0), (4, 400.0), (5, 500.0)]


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
        config = pipeline_config(epochs=12)
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
    print(f"\nfull pipeline: hdf5={t_hdf5:.2f}s  sqlite={t_sqlite:.2f}s  ratio={t_sqlite / t_hdf5:.2f}x  sqlite_rows={rows}")
    assert rows > 0
    assert t_sqlite <= t_hdf5 * 1.5, f"sqlite {t_sqlite:.2f}s vs hdf5 {t_hdf5:.2f}s — logger regressed"
