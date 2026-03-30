import clearml
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Protocol
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


class ScalarLogger:
    """Logs dict[str, jax.Array] stats.

    Each stat has shape (..., time) where dims before time are a mix of
    scan and batch dims. Scan dims advance the shared iteration counter.
    Batch dims become separate named series.
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

    def log(self, stats: dict[str, jax.Array]):
        # Bulk transfer all JAX arrays to numpy once (single device-to-host copy)
        stats = {k: np.asarray(v) for k, v in stats.items()}

        # ref_shape from deepest level's scan dims (exclude time)
        max_ndim = max(v.ndim for v in stats.values())
        for k, v in stats.items():
            if v.ndim == max_ndim:
                ref_level = infer_level(k)
                n_ref_pairs = self.num_levels - ref_level
                ref_shape = tuple(v.shape[i] for i in range(0, 2 * n_ref_pairs, 2))
                break

        strides = compute_strides(ref_shape)
        context = self.logger.get_context()

        for idx in itertools.product(*(range(d) for d in ref_shape)):
            local_step = sum(i * st for i, st in zip(idx, strides))
            iteration = self.global_step + local_step + 1

            if iteration % self.checkpoint_every != 0:
                continue

            for key, value in stats.items():
                level = infer_level(key)
                n_scan_vmap_pairs = self.num_levels - level
                n_paired_dims = 2 * n_scan_vmap_pairs
                expected_with_time = n_paired_dims + level + 1
                has_time = value.ndim >= expected_with_time

                if has_time:
                    n_before_time = value.ndim - 1
                else:
                    n_before_time = value.ndim

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

                    if has_time:
                        time_slice = value[tuple(full_idx)]
                        mean_v = float(np.nanmean(time_slice))
                        if not np.isnan(mean_v):
                            self.logger.log_scalar(
                                context,
                                series,
                                self.log_title,
                                mean_v,
                                iteration,
                                self.total_iterations,
                                self.iteration_offset,
                            )
                    else:
                        v = float(value[tuple(full_idx)])
                        if not np.isnan(v):
                            self.logger.log_scalar(
                                context,
                                series,
                                self.log_title,
                                v,
                                iteration,
                                self.total_iterations,
                                self.iteration_offset,
                            )

        self.global_step += math.prod(ref_shape)
        self.logger.close_context(context)

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.logger.log_image(title, series, iteration, image)


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

    def log_image(self, title: str, series: str, iteration: int, image: np.ndarray):
        self.scalar_logger.log_image(title, series, iteration, image)

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
