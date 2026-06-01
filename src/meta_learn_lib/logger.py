import clearml
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Iterator, Protocol
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import accumulate
from operator import mul
import sqlite3
import threading
import queue
import itertools
import math
import umap

from meta_learn_lib.lib_types import STAT, Tag


class Logger(Protocol):
    def log_scalar(
        self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ): ...

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray): ...

    def flush(self) -> None: ...


class HDF5Logger:
    def __init__(self, log_dir: str, task_id: str, checkpoint_every: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.log_file = self.log_dir / f"metrics_{task_id}.h5"
        self.image_logger = MatplotlibLogger(str(self.log_dir / f"images_{task_id}"))
        self.file = h5py.File(self.log_file, "a")
        self.file.attrs["task_id"] = task_id
        self.file.attrs.setdefault("created", datetime.now().isoformat())

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.image_logger.log_image(title, series, iteration, image)

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        f = self.file
        if title not in f:
            dataset = f.create_dataset(title, shape=(max_count,), dtype=np.float64, fillvalue=np.nan)
            dataset.attrs["series"] = series
            f.create_dataset(f"{title}_iterations", shape=(max_count,), dtype=np.int32, fillvalue=-1)

        absolute_iteration = iteration + iteration_offset
        idx = absolute_iteration // self.checkpoint_every - 1
        if 0 <= idx < len(f[title]):
            f[title][idx] = value
            f[f"{title}_iterations"][idx] = absolute_iteration

    def flush(self) -> None:
        pass


class SQLiteLogger:
    def __init__(self, log_dir: str, task_id: str, checkpoint_every: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.db_file = self.log_dir / f"metrics_{task_id}.sqlite"
        self.image_logger = MatplotlibLogger(str(self.log_dir / f"images_{task_id}"))
        self.conn = sqlite3.connect(str(self.db_file), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS scalars (
                series TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                value REAL NOT NULL,
                run_label TEXT
            )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_series_iter ON scalars(series, iteration)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('task_id', ?)", (task_id,))
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('created', ?)", (datetime.now().isoformat(),))
        self.buffer: list[tuple] = []

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.image_logger.log_image(title, series, iteration, image)

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        # Note: the Logger Protocol's `title` arg is the data-series path (e.g. "train/level0/loss/0/0/0"),
        # and `series` is the run label (e.g. "sos_beta_oho_2conv"). The SQL columns are named after their
        # actual contents, not after the protocol parameter names.
        absolute_iteration = iteration + iteration_offset
        self.buffer.append((title, absolute_iteration, float(value), series))

    def flush(self) -> None:
        if not self.buffer:
            return
        # explicit BEGIN/COMMIT because `with self.conn` is a no-op under isolation_level=None
        self.conn.execute("BEGIN")
        self.conn.executemany(
            "INSERT INTO scalars(series, iteration, value, run_label) VALUES (?, ?, ?, ?)",
            self.buffer,
        )
        self.conn.execute("COMMIT")
        self.buffer.clear()


class ClearMLLogger:
    def __init__(self, task: clearml.Task):
        self.task = task

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        self.task.get_logger().report_scalar(
            title=title, series=series, value=value, iteration=iteration + iteration_offset
        )

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.task.get_logger().report_image(title=title, series=series, iteration=iteration, image=image)

    def flush(self) -> None:
        pass


class ConsoleLogger:
    def __init__(self):
        pass

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        pass

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        print(f"[{title}] {series} @ {iteration + iteration_offset}/{max_count}: {value}")

    def flush(self) -> None:
        pass


class MatplotlibLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(lambda: {"iterations": [], "values": [], "title": None})

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        fig = plt.figure()
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        plt.title(f"{title} - {series}")
        plt.axis("off")
        safe_title = title.replace("/", "_")
        safe_series = series.replace("/", "_")
        save_path = self.save_dir / f"{safe_title}_{safe_series}_{iteration}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        self.data[series]["iterations"].append(iteration + iteration_offset)
        self.data[series]["values"].append(value)
        self.data[series]["title"] = title

    def flush(self) -> None:
        pass

    def generate_figures(self):
        for series, data in self.data.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data["iterations"], data["values"], linewidth=1)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.set_title(f"{data['title']} - {series}")
            ax.grid(True, alpha=0.3)

            save_path = self.save_dir / f"{series.replace('/', '_')}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        for logger in self.loggers:
            logger.log_image(title, series, iteration, image)

    def log_scalar(self, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int):
        for logger in self.loggers:
            logger.log_scalar(title, series, value, iteration, max_count, iteration_offset)

    def flush(self) -> None:
        for logger in self.loggers:
            logger.flush()


def compute_strides(shape: tuple[int, ...]) -> list[int]:
    strides = list(accumulate(reversed(shape), mul, initial=1))
    return list(reversed(strides[:-1]))


class ScalarLogger:
    def __init__(
        self,
        logger: Logger,
        num_levels: int,
        total_iterations: int,
        checkpoint_every: int,
        log_title: str,
        iteration_offset: int,
    ):
        self.logger = logger
        self.num_levels = num_levels
        self.total_iterations = total_iterations
        self.checkpoint_every = checkpoint_every
        self.log_title = log_title
        self.global_step = 0
        self.iteration_offset = iteration_offset

    def for_each_entry(
        self,
        stats: STAT,
        iterate_tags: tuple[Tag, ...],
    ) -> Iterator[tuple[str, int, np.ndarray]]:
        for key, ns in stats.items():
            data = np.asarray(ns.data)
            assert len(ns.axes) == data.ndim, f"{key}: axes {ns.axes} mismatch ndim {data.ndim}"

            scan_axes = [i for i, t in enumerate(ns.axes) if t == "scan"]
            batch_axes = [i for i, t in enumerate(ns.axes) if t in iterate_tags]
            has_time = any(t == "time" for t in ns.axes)
            scan_shape = tuple(data.shape[i] for i in scan_axes)
            batch_shape = tuple(data.shape[i] for i in batch_axes)
            scan_strides = compute_strides(scan_shape)

            for scan_idx in itertools.product(*(range(d) for d in scan_shape)):
                local_step = sum(i * st for i, st in zip(scan_idx, scan_strides))
                iteration = self.global_step + local_step + 1
                for batch_idx in itertools.product(*(range(d) for d in batch_shape)):
                    full_idx = [slice(None)] * data.ndim
                    for ax, val in zip(scan_axes, scan_idx):
                        full_idx[ax] = val
                    for ax, val in zip(batch_axes, batch_idx):
                        full_idx[ax] = val
                    suffix = "/".join(str(i) for i in batch_idx)
                    series = f"{key}/{suffix}" if suffix else key
                    sub = data[tuple(full_idx)]
                    if not has_time:
                        sub = sub[None, ...]
                    yield series, iteration, sub

    def log(self, stats: STAT) -> None:
        for series, iteration, sub in self.for_each_entry(stats, ("batch", "minibatch")):
            if sub.ndim != 1:
                continue
            if iteration % self.checkpoint_every != 0:
                continue
            v = float(np.nanmean(sub))
            if np.isnan(v):
                continue
            self.logger.log_scalar(
                series,
                self.log_title,
                v,
                iteration,
                self.total_iterations,
                self.iteration_offset,
            )
        max_scan_size = max(
            (math.prod(d for d, t in zip(ns.data.shape, ns.axes) if t == "scan") for ns in stats.values()),
            default=1,
        )
        self.global_step += max_scan_size

    def log_image_stats(self, stats: STAT, title: str) -> None:
        for series, iteration, sub in self.for_each_entry(stats, ("batch", "minibatch")):
            if sub.ndim != 4:
                continue
            if iteration % self.checkpoint_every != 0:
                continue
            for t in range(sub.shape[0]):
                img = np.transpose(sub[t], (1, 2, 0))
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                t_series = f"{series}/t{t}" if sub.shape[0] > 1 else series
                self.logger.log_image(title, t_series, iteration, img)

    def log_plot_stats(self, stats: STAT, title: str) -> None:
        for series, iteration, sub in self.for_each_entry(stats, ("batch", "minibatch")):
            if sub.ndim < 2:
                continue
            if iteration % self.checkpoint_every != 0:
                continue
            data = sub.reshape(sub.shape[0], -1)
            fig, ax = plt.subplots(figsize=(8, 4))
            for j in range(data.shape[1]):
                ax.plot(np.arange(data.shape[0]), data[:, j], linewidth=1, label=f"f{j}")
            ax.set_xlabel("step" if data.shape[0] == 1 else "time")
            ax.set_ylabel("value")
            ax.set_title(f"{title} - {series}")
            if data.shape[1] <= 12:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
            plt.close(fig)
            self.logger.log_image(title, series, iteration, img)

    def log_scalar_stats(self, stats: STAT, title: str) -> None:
        iteration = self.global_step
        for key, ns in stats.items():
            v = float(np.asarray(ns.data).mean())
            if np.isnan(v):
                continue
            series = f"{title}/{key}" if title else key
            self.logger.log_scalar(
                series,
                self.log_title,
                v,
                iteration,
                self.total_iterations,
                self.iteration_offset,
            )

    def log_grid_stats(
        self,
        stats: STAT,
        title: str,
        rows: int,
        cols: int,
        iterate_tags: tuple[Tag, ...],
        z_ticks: tuple[list[float], list[float]] | None,
    ) -> None:
        for series, iteration, sub in self.for_each_entry(stats, iterate_tags):
            if iteration % self.checkpoint_every != 0:
                continue
            tile = sub[0].reshape(rows, cols, *sub.shape[2:])
            c, h, w = tile.shape[2], tile.shape[3], tile.shape[4]
            grid_img = tile.transpose(0, 3, 1, 4, 2).reshape(rows * h, cols * w, c)
            if z_ticks is None:
                self.logger.log_image(title, series, iteration, grid_img)
                continue
            row_ticks, col_ticks = z_ticks
            fig, ax = plt.subplots(figsize=(max(cols * 0.6, 4), max(rows * 0.6, 4)))
            display = grid_img.squeeze(-1) if c == 1 else grid_img
            ax.imshow(display, cmap="gray" if c == 1 else None)
            ax.set_xticks([(j + 0.5) * w for j in range(cols)])
            ax.set_yticks([(i + 0.5) * h for i in range(rows)])
            ax.set_xticklabels([f"{v:.2f}" for v in col_ticks], fontsize=7)
            ax.set_yticklabels([f"{v:.2f}" for v in row_ticks], fontsize=7)
            ax.set_xlabel("z1")
            ax.set_ylabel("z0")
            ax.set_title(f"{title} - {series}")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
            plt.close(fig)
            self.logger.log_image(title, series, iteration, img)

    def log_umap_stats(self, stats: STAT, title: str) -> None:
        preds = {k.removesuffix("/prediction"): ns for k, ns in stats.items() if k.endswith("/prediction")}
        labels = {k.removesuffix("/label"): ns for k, ns in stats.items() if k.endswith("/label")}
        for prefix in preds.keys() & labels.keys():
            pred_iter = self.for_each_entry({f"{prefix}/prediction": preds[prefix]}, ("batch", "time"))
            label_iter = self.for_each_entry({f"{prefix}/label": labels[prefix]}, ("batch", "time"))
            for (series, iteration, pred_sub), (_, _, label_sub) in zip(pred_iter, label_iter):
                if iteration % self.checkpoint_every != 0:
                    continue
                pred_flat = pred_sub.reshape(pred_sub.shape[0], -1)
                # label_sub may be (n,) scalar labels or (n, *F) vector labels.
                label_per_sample = label_sub.reshape(label_sub.shape[0], -1)
                if pred_flat.shape[-1] == 2:
                    embedding = pred_flat
                else:
                    n_neighbors = min(15, pred_flat.shape[0] - 1)
                    embedding = umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(pred_flat)
                # Scalar labels → one categorical plot. Vector labels → one continuous plot per component.
                if label_per_sample.shape[1] == 1:
                    components = [("", label_per_sample[:, 0], "tab10")]
                else:
                    components = [
                        (f"/dim{i}", label_per_sample[:, i], "viridis") for i in range(label_per_sample.shape[1])
                    ]
                for suffix, color_values, cmap in components:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color_values, cmap=cmap, s=10)
                    ax.set_title(f"{title} - {series}{suffix}")
                    ax.grid(True, alpha=0.3)
                    fig.colorbar(scatter, ax=ax)
                    fig.canvas.draw()
                    img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
                    plt.close(fig)
                    self.logger.log_image(title, f"{series}{suffix}", iteration, img)

    def log_grid_deformation_stats(self, stats: STAT, title: str, n_per_axis: int) -> None:
        preds = {k.removesuffix("/prediction"): ns for k, ns in stats.items() if k.endswith("/prediction")}
        labels = {k.removesuffix("/label"): ns for k, ns in stats.items() if k.endswith("/label")}
        for prefix in preds.keys() & labels.keys():
            pred_iter = self.for_each_entry({f"{prefix}/prediction": preds[prefix]}, ("batch", "time"))
            label_iter = self.for_each_entry({f"{prefix}/label": labels[prefix]}, ("batch", "time"))
            for (series, iteration, pred_sub), (_, _, label_sub) in zip(pred_iter, label_iter):
                if iteration % self.checkpoint_every != 0:
                    continue
                pred_flat = pred_sub.reshape(pred_sub.shape[0], -1)
                label_flat = label_sub.reshape(label_sub.shape[0], -1)
                if pred_flat.shape[1] != 2 or label_flat.shape[1] != 2:
                    continue
                if pred_flat.shape[0] != n_per_axis * n_per_axis:
                    continue
                z = pred_flat.reshape(n_per_axis, n_per_axis, 2)
                cxy = label_flat.reshape(n_per_axis, n_per_axis, 2)
                cx_min, cx_max = float(cxy[..., 0].min()), float(cxy[..., 0].max())
                cy_min, cy_max = float(cxy[..., 1].min()), float(cxy[..., 1].max())
                cx_norm = (cxy[..., 0] - cx_min) / max(cx_max - cx_min, 1e-9)
                cy_norm = (cxy[..., 1] - cy_min) / max(cy_max - cy_min, 1e-9)
                rgb = np.stack([cx_norm, cy_norm, np.full_like(cx_norm, 0.5)], axis=-1)
                fig, ax = plt.subplots(figsize=(8, 8))
                for i in range(n_per_axis):
                    ax.plot(z[i, :, 0], z[i, :, 1], color="gray", linewidth=0.7, alpha=0.5)
                    ax.plot(z[:, i, 0], z[:, i, 1], color="gray", linewidth=0.7, alpha=0.5)
                ax.scatter(
                    z[..., 0].reshape(-1),
                    z[..., 1].reshape(-1),
                    c=rgb.reshape(-1, 3),
                    s=30,
                    edgecolors="black",
                    linewidths=0.4,
                )
                ax.set_xlabel("z1")
                ax.set_ylabel("z2")
                ax.set_title(f"{title} - {series}  (cx→R, cy→G)")
                ax.grid(True, alpha=0.3)
                fig.canvas.draw()
                img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
                plt.close(fig)
                self.logger.log_image(title, series, iteration, img)


class ThreadedScalarLogger:
    def __init__(
        self,
        logger: Logger,
        loggers: list[Logger],
        num_levels: int,
        total_iterations: int,
        checkpoint_every: int,
        log_title: str,
        iteration_offset: int,
        scalar_queue_size: int,
        sample_queue_size: int,
    ):
        self.scalar_logger = ScalarLogger(
            logger, num_levels, total_iterations, checkpoint_every, log_title, iteration_offset
        )
        self.loggers = loggers
        self.scalar_queue: queue.Queue = queue.Queue(maxsize=scalar_queue_size)
        self.sample_queue: queue.Queue = queue.Queue(maxsize=sample_queue_size)
        self.stop_event = threading.Event()
        self.scalar_worker = threading.Thread(target=self._drain, args=(self.scalar_queue,), daemon=True)
        self.sample_worker = threading.Thread(target=self._drain, args=(self.sample_queue,), daemon=True)
        self.scalar_worker.start()
        self.sample_worker.start()

    def _drain(self, q: queue.Queue) -> None:
        while not self.stop_event.is_set():
            try:
                item = q.get(timeout=0.1)
                if item is None:
                    q.task_done()
                    break
                kind, payload = item[0], item[1:]
                match kind:
                    case "scalar":
                        stats, _ = payload
                        self.scalar_logger.log(stats)
                    case "image":
                        stats, title = payload
                        self.scalar_logger.log_image_stats(stats, title)
                    case "plot":
                        stats, title = payload
                        self.scalar_logger.log_plot_stats(stats, title)
                    case "umap":
                        stats, title = payload
                        self.scalar_logger.log_umap_stats(stats, title)
                    case "grid":
                        stats, title, rows, cols, iterate_tags, z_ticks = payload
                        self.scalar_logger.log_grid_stats(stats, title, rows, cols, iterate_tags, z_ticks)
                    case "scalar_stats":
                        stats, title = payload
                        self.scalar_logger.log_scalar_stats(stats, title)
                    case "grid_deformation":
                        stats, title, n_per_axis = payload
                        self.scalar_logger.log_grid_deformation_stats(stats, title, n_per_axis)
                self.scalar_logger.logger.flush()
                q.task_done()
            except queue.Empty:
                continue

    def log(self, stats: STAT) -> None:
        if self.stop_event.is_set():
            return
        self.scalar_queue.put(("scalar", stats, None))

    def log_image_stats(self, stats: STAT, title: str) -> None:
        if self.stop_event.is_set():
            return
        self.sample_queue.put(("image", stats, title))

    def log_plot_stats(self, stats: STAT, title: str) -> None:
        if self.stop_event.is_set():
            return
        self.sample_queue.put(("plot", stats, title))

    def log_umap_stats(self, stats: STAT, title: str) -> None:
        if self.stop_event.is_set():
            return
        self.sample_queue.put(("umap", stats, title))

    def log_grid_stats(
        self,
        stats: STAT,
        title: str,
        rows: int,
        cols: int,
        iterate_tags: tuple[Tag, ...],
        z_ticks: tuple[list[float], list[float]] | None,
    ) -> None:
        if self.stop_event.is_set():
            return
        self.sample_queue.put(("grid", stats, title, rows, cols, iterate_tags, z_ticks))

    def log_scalar_stats(self, stats: STAT, title: str) -> None:
        if self.stop_event.is_set():
            return
        self.scalar_queue.put(("scalar_stats", stats, title))

    def log_grid_deformation_stats(self, stats: STAT, title: str, n_per_axis: int) -> None:
        if self.stop_event.is_set():
            return
        self.sample_queue.put(("grid_deformation", stats, title, n_per_axis))

    def flush(self) -> None:
        self.scalar_queue.join()
        self.sample_queue.join()
        self.scalar_logger.logger.flush()

    def shutdown(self) -> None:
        self.stop_event.set()
        for q in (self.scalar_queue, self.sample_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break
            q.put(None)
        self.scalar_worker.join(timeout=5.0)
        self.sample_worker.join(timeout=5.0)
        self.scalar_logger.logger.flush()

    def __del__(self):
        if hasattr(self, "stop_event") and not self.stop_event.is_set():
            self.shutdown()


def create_logger(
    loggers: list[Logger],
    num_levels: int,
    total_iterations: int,
    checkpoint_every: int,
    log_title: str,
    iteration_offset: int,
    scalar_queue_size: int,
    sample_queue_size: int,
) -> ThreadedScalarLogger:
    logger = MultiLogger(loggers)
    return ThreadedScalarLogger(
        logger,
        loggers,
        num_levels,
        total_iterations,
        checkpoint_every,
        log_title,
        iteration_offset,
        scalar_queue_size,
        sample_queue_size,
    )
