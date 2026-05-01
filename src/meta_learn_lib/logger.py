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
import threading
import queue
import itertools
import math
import re
import jax


class Logger(Protocol):
    def get_context(self): ...

    def close_context(self, context): ...

    def log_scalar(
        self, context, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ): ...

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray): ...


class HDF5Logger:
    def __init__(self, log_dir: str, task_id: str, checkpoint_every: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.log_file = self.log_dir / f"metrics_{task_id}.h5"

        with h5py.File(self.log_file, "a") as f:
            f.attrs["task_id"] = task_id
            f.attrs.setdefault("created", datetime.now().isoformat())

    def get_context(self) -> h5py.File:
        return h5py.File(self.log_file, "a")

    def close_context(self, f: h5py.File):
        f.close()

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        pass

    def log_scalar(
        self, f: h5py.File, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ):
        if title not in f:
            dataset = f.create_dataset(title, shape=(max_count,), dtype=np.float64, fillvalue=np.nan)
            dataset.attrs["series"] = series
            f.create_dataset(f"{title}_iterations", shape=(max_count,), dtype=np.int32, fillvalue=-1)

        absolute_iteration = iteration + iteration_offset
        idx = absolute_iteration // self.checkpoint_every - 1
        if 0 <= idx < len(f[title]):
            f[title][idx] = value
            f[f"{title}_iterations"][idx] = absolute_iteration


class ClearMLLogger:
    def __init__(self, task: clearml.Task):
        self.task = task

    def get_context(self):
        """No context needed for ClearML"""
        return None

    def close_context(self, context):
        """No context to close for ClearML"""
        pass

    def log_scalar(
        self, context, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ):
        self.task.get_logger().report_scalar(
            title=title, series=series, value=value, iteration=iteration + iteration_offset
        )

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.task.get_logger().report_image(title=title, series=series, iteration=iteration, image=image)


class ConsoleLogger:
    def __init__(self):
        pass

    def get_context(self):
        """No context needed for ConsoleLogger"""
        return None

    def close_context(self, context):
        """No context to close for ConsoleLogger"""
        pass

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        pass

    def log_scalar(
        self, context, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ):
        print(f"[{title}] {series} @ {iteration + iteration_offset}/{max_count}: {value}")


class MatplotlibLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(lambda: {"iterations": [], "values": [], "title": None})

    def get_context(self):
        return None

    def close_context(self, context):
        pass

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        fig, ax = plt.subplots()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(f"{title} - {series}")
        ax.axis("off")
        save_path = self.save_dir / f"{title}_{series}_{iteration}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_scalar(
        self, context, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ):
        self.data[series]["iterations"].append(iteration + iteration_offset)
        self.data[series]["values"].append(value)
        self.data[series]["title"] = title

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

    def get_context(self):
        return [logger.get_context() for logger in self.loggers]

    def close_context(self, contexts):
        for logger, context in zip(self.loggers, contexts):
            logger.close_context(context)

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        for logger in self.loggers:
            logger.log_image(title, series, iteration, image)

    def log_scalar(
        self, contexts, title: str, series: str, value: float, iteration: int, max_count: int, iteration_offset: int
    ):
        for logger, context in zip(self.loggers, contexts):
            logger.log_scalar(context, title, series, value, iteration, max_count, iteration_offset)


def infer_level(key: str) -> int:
    """Extract level number from stat key, e.g. 'train/level0/loss' -> 0."""
    match = re.search(r"level(\d+)", key)
    if match is None:
        raise ValueError(f"Could not find 'levelN' in key: {key}")
    return int(match.group(1))


def compute_strides(shape: tuple[int, ...]) -> list[int]:
    strides = list(accumulate(reversed(shape), mul, initial=1))
    return list(reversed(strides[:-1]))


def compute_ref_shape(stats: dict[str, np.ndarray], num_levels: int) -> tuple[int, ...]:
    """Take scan dims from the deepest-level (highest-ndim) stat as the iteration ref."""
    if not stats:
        return ()
    max_ndim = max(v.ndim for v in stats.values())
    for k, v in stats.items():
        if v.ndim == max_ndim:
            ref_level = infer_level(k)
            n_ref_pairs = num_levels - ref_level
            return tuple(v.shape[i] for i in range(0, 2 * n_ref_pairs, 2))
    return ()


class ScalarLogger:
    """Logs dict[str, jax.Array] stats.

    Each stat has shape (..., time, *feature_dims). Dims before time are a mix of
    paired (scan, vmap) blocks and trailing batch dims. Scan dims advance the shared
    iteration counter; batch dims become separate named series.

    For scalar stats use n_feature_dims=0; for images use n_feature_dims=3 (C,H,W).
    """

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
        self.counters: dict[str, int] = {}
        self.global_step = 0
        self.iteration_offset = iteration_offset

    def for_each_entry(
        self,
        stats: dict[str, np.ndarray],
        n_feature_dims: int,
    ) -> Iterator[tuple[str, int, bool, np.ndarray]]:
        """Yields (series, iteration, has_time, sub) for each scan/batch index of
        each stat. `sub` is `value[full_idx]` — shape `(time, *feature_dims)` if
        has_time, else `(*feature_dims,)`. Caller decides reduction."""
        ref_shape = compute_ref_shape(stats, self.num_levels)
        strides = compute_strides(ref_shape)

        for idx in itertools.product(*(range(d) for d in ref_shape)):
            local_step = sum(i * st for i, st in zip(idx, strides))
            iteration = self.global_step + local_step + 1

            for key, value in stats.items():
                level = infer_level(key)
                n_paired_dims = 2 * (self.num_levels - level)
                expected_with_time = n_paired_dims + level + 1
                effective_ndim = value.ndim - n_feature_dims
                has_time = effective_ndim >= expected_with_time
                n_before_time = (effective_ndim - 1) if has_time else effective_ndim

                actual_paired = min(n_paired_dims, n_before_time)
                actual_paired = actual_paired - (actual_paired % 2)

                batch_axes = list(range(1, actual_paired, 2)) + list(range(actual_paired, n_before_time))

                n_scan = actual_paired // 2
                scan_idx = idx[:n_scan]
                extra_idx = idx[n_scan:]

                if any(i != 0 for i in extra_idx):
                    continue

                batch_shape = tuple(value.shape[ax] for ax in batch_axes)
                for batch_idx in itertools.product(*(range(d) for d in batch_shape)):
                    suffix = "/".join(str(i) for i in batch_idx)
                    series = f"{key}/{suffix}" if suffix else key

                    full_idx = [None] * n_before_time
                    scan_pos = 0
                    batch_pos = 0
                    for ax in range(n_before_time):
                        if ax < actual_paired and ax % 2 == 0:
                            full_idx[ax] = scan_idx[scan_pos]
                            scan_pos += 1
                        else:
                            full_idx[ax] = batch_idx[batch_pos]
                            batch_pos += 1

                    yield series, iteration, has_time, value[tuple(full_idx)]

    def log(self, stats: dict[str, jax.Array]):
        # Bulk transfer all JAX arrays to numpy once (single device-to-host copy)
        stats = {k: np.asarray(v) for k, v in stats.items()}

        # Keep only scalar-shaped stats; image-shaped go through log_image_stats.
        scalar_stats = {
            k: v for k, v in stats.items() if v.ndim <= 2 * (self.num_levels - infer_level(k)) + infer_level(k) + 1
        }
        if not scalar_stats:
            return

        context = self.logger.get_context()

        for series, iteration, has_time, sub in self.for_each_entry(scalar_stats, 0):
            if iteration % self.checkpoint_every != 0:
                continue
            v = float(np.nanmean(sub)) if has_time else float(sub)
            if np.isnan(v):
                continue
            self.logger.log_scalar(
                context,
                series,
                self.log_title,
                v,
                iteration,
                self.total_iterations,
                self.iteration_offset,
            )

        self.global_step += math.prod(compute_ref_shape(scalar_stats, self.num_levels))
        self.logger.close_context(context)

    def log_image_stats(self, stats: dict[str, jax.Array], title: str, n_feature_dims: int) -> None:
        """Log image-shaped stats with the same scan/vmap iteration walk as scalars.
        `n_feature_dims` is the number of trailing feature dims (3 for (C,H,W)).
        Does NOT bump global_step — call after log() to land at the next iteration,
        or before log() to land at the same iteration as that step's scalars."""
        stats = {k: np.asarray(v) for k, v in stats.items()}

        # Keep only stats whose shape (after peeling features) fits scalar shape,
        # with or without a time axis.
        image_stats = {
            k: v
            for k, v in stats.items()
            if v.ndim >= n_feature_dims
            and (v.ndim - n_feature_dims) <= 2 * (self.num_levels - infer_level(k)) + infer_level(k) + 1
            and (v.ndim - n_feature_dims) >= 2 * (self.num_levels - infer_level(k)) + infer_level(k)
        }
        if not image_stats:
            return

        for series, iteration, has_time, sub in self.for_each_entry(image_stats, n_feature_dims):
            img = np.nanmean(sub, axis=0) if has_time else sub
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    img = img.squeeze(2)
            img = np.clip(img, 0.0, 1.0)
            self.logger.log_image(title, series, iteration, img)

    def log_plot_stats(self, stats: dict[str, jax.Array], title: str, n_feature_dims: int) -> None:
        """Log feature-vector stats as line plots. Renders matplotlib to a numpy RGBA
        image and dispatches via log_image. `n_feature_dims` trailing dims are
        flattened into a single 1D series for plotting."""
        stats = {k: np.asarray(v) for k, v in stats.items()}

        plot_stats = {
            k: v
            for k, v in stats.items()
            if v.ndim >= n_feature_dims
            and (v.ndim - n_feature_dims) <= 2 * (self.num_levels - infer_level(k)) + infer_level(k) + 1
            and (v.ndim - n_feature_dims) >= 2 * (self.num_levels - infer_level(k)) + infer_level(k)
        }
        if not plot_stats:
            return

        for series, iteration, has_time, sub in self.for_each_entry(plot_stats, n_feature_dims):
            # sub shape: (time, *feature_dims) if has_time else (*feature_dims,)
            if has_time:
                data = sub.reshape(sub.shape[0], -1)  # (time, num_features)
                x_label = "time"
            else:
                data = sub.flatten().reshape(1, -1)  # (1, num_features)
                x_label = "step"

            fig, ax = plt.subplots(figsize=(8, 4))
            for j in range(data.shape[1]):
                ax.plot(np.arange(data.shape[0]), data[:, j], linewidth=1, label=f"f{j}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("value")
            ax.set_title(f"{title} - {series}")
            if data.shape[1] <= 12:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            self.logger.log_image(title, series, iteration, img)


class ThreadedScalarLogger:
    """Async wrapper that processes log calls in a background thread."""

    def __init__(
        self,
        logger: Logger,
        loggers: list[Logger],
        num_levels: int,
        total_iterations: int,
        checkpoint_every: int,
        log_title: str,
        iteration_offset: int,
    ):
        self.scalar_logger = ScalarLogger(
            logger, num_levels, total_iterations, checkpoint_every, log_title, iteration_offset
        )
        self.loggers = loggers
        self.job_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._process, daemon=True)
        self.worker.start()

    def _process(self):
        while not self.stop_event.is_set():
            try:
                stats = self.job_queue.get(timeout=0.1)
                if stats is None:
                    self.job_queue.task_done()
                    break
                self.scalar_logger.log(stats)
                self.job_queue.task_done()
            except queue.Empty:
                continue

    def log_image_stats(self, stats: dict[str, jax.Array], title: str, n_feature_dims: int) -> None:
        self.scalar_logger.log_image_stats(stats, title, n_feature_dims)

    def log_plot_stats(self, stats: dict[str, jax.Array], title: str, n_feature_dims: int) -> None:
        self.scalar_logger.log_plot_stats(stats, title, n_feature_dims)

    def log(self, stats: dict[str, jax.Array]):
        if self.stop_event.is_set():
            return
        self.job_queue.put(stats)

    def flush(self):
        self.job_queue.join()

    def shutdown(self):
        self.stop_event.set()
        # Drain the queue so .join() doesn't block forever
        while not self.job_queue.empty():
            try:
                self.job_queue.get_nowait()
                self.job_queue.task_done()
            except queue.Empty:
                break
        self.job_queue.put(None)  # sentinel to unblock get()
        self.worker.join(timeout=5.0)

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
) -> ThreadedScalarLogger:
    """Construct a ThreadedScalarLogger from a list of loggers."""
    logger = MultiLogger(loggers)
    return ThreadedScalarLogger(
        logger, loggers, num_levels, total_iterations, checkpoint_every, log_title, iteration_offset
    )
