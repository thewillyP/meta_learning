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

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int): ...


class HDF5Logger:
    def __init__(self, log_dir: str, task_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task_id = task_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"metrics_{task_id}_{timestamp}.h5"
        self.metric_indices = {}

        # Initialize HDF5 file with task metadata
        with h5py.File(self.log_file, "w") as f:
            f.attrs["task_id"] = task_id
            f.attrs["created"] = datetime.now().isoformat()

    def get_context(self) -> h5py.File:
        """Get an open HDF5 file context for logging"""
        return h5py.File(self.log_file, "a")

    def close_context(self, f: h5py.File):
        """Close the HDF5 file context"""
        f.close()

    def log_scalar(self, f: h5py.File, title: str, series: str, value: float, iteration: int, max_count: int):
        """Log a scalar metric to an open HDF5 file"""
        if series not in f:
            dataset = f.create_dataset(series, shape=(max_count,), dtype=np.float64, fillvalue=np.nan)
            dataset.attrs["title"] = title
            f.create_dataset(f"{series}_iterations", shape=(max_count,), dtype=np.int32, fillvalue=-1)
            self.metric_indices[series] = 0

        idx = self.metric_indices[series]
        if idx < len(f[series]):
            f[series][idx] = value
            f[f"{series}_iterations"][idx] = iteration
            self.metric_indices[series] += 1


class ClearMLLogger:
    def __init__(self, task: clearml.Task):
        self.task = task

    def get_context(self):
        """No context needed for ClearML"""
        return None

    def close_context(self, context):
        """No context to close for ClearML"""
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        """Log a scalar metric to ClearML"""
        self.task.get_logger().report_scalar(title=title, series=series, value=value, iteration=iteration)


class PrintLogger:
    def __init__(self):
        pass

    def get_context(self):
        """No context needed for PrintLogger"""
        return None

    def close_context(self, context):
        """No context to close for PrintLogger"""
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        """Print the scalar metric to console"""
        print(f"[{title}] {series} @ {iteration}/{max_count}: {value}")


class MatplotlibLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(lambda: {"iterations": [], "values": [], "title": None})

    def get_context(self):
        return None

    def close_context(self, context):
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        self.data[series]["iterations"].append(iteration)
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

    def log_scalar(self, contexts, title: str, series: str, value: float, iteration: int, max_count: int):
        for logger, context in zip(self.loggers, contexts):
            logger.log_scalar(context, title, series, value, iteration, max_count)


def parse_level(key: str) -> int:
    """Extract level number from key like 'level0/loss' -> 0."""
    m = re.match(r"level(\d+)/", key)
    if m is None:
        raise ValueError(f"Cannot parse level from key: {key}")
    return int(m.group(1))


def reduce_batch_dims(arr: np.ndarray, level: int, num_levels: int) -> np.ndarray:
    """Average out batch dims, keeping only scan dims + time.

    Stats for level i have 2L - i dims before time:
      - First 2*(L-i) dims alternate (scan, batch, scan, batch, ...)
      - Remaining i dims are extra vmaps (all batch)
      - Last dim is time

    Scan dims are at even positions 0, 2, ..., 2*(L-i-1).
    Everything else before time is batch.
    """
    ndim = arr.ndim
    if ndim <= 1:
        return arr

    n_before_time = ndim - 1
    n_scan_vmap_pairs = num_levels - level
    n_paired_dims = 2 * n_scan_vmap_pairs

    batch_axes = list(range(1, n_paired_dims, 2)) + list(range(n_paired_dims, n_before_time))

    for ax in reversed(batch_axes):
        arr = np.nanmean(arr, axis=ax)
    return arr


def compute_strides(shape: tuple[int, ...]) -> list[int]:
    strides = list(accumulate(reversed(shape), mul, initial=1))
    return list(reversed(strides[:-1]))


class ScalarLogger:
    """Logs dict[str, np.ndarray] stats against a shared global step.

    After averaging out batch dims, each key has shape (scan, ..., scan, time).
    The scan dims (all but last) define the shared iteration space from the
    deepest level. Time is iterated per key separately — each non-nan timestep
    gets its own point via a per-key counter.
    """

    def __init__(self, logger: Logger, num_levels: int, total_iterations: int, checkpoint_every: int):
        self.logger = logger
        self.num_levels = num_levels
        self.global_step = 0
        self.total_iterations = total_iterations
        self.checkpoint_every = checkpoint_every
        self.counters: dict[str, int] = {}

    def log(self, stats: dict[str, np.ndarray]):
        reduced = {k: reduce_batch_dims(v, parse_level(k), self.num_levels) for k, v in stats.items()}

        # ref_shape from deepest level's scan dims (exclude time)
        max_ndim = max(v.ndim for v in reduced.values())
        for v in reduced.values():
            if v.ndim == max_ndim:
                ref_shape = v.shape[:-1]
                break

        strides = compute_strides(ref_shape)
        context = self.logger.get_context()

        for idx in itertools.product(*(range(d) for d in ref_shape)):
            local_step = sum(i * st for i, st in zip(idx, strides))
            iteration = self.global_step + local_step + 1

            if iteration % self.checkpoint_every != 0:
                continue

            for key, value in reduced.items():
                n_scan = value.ndim - 1
                scan_idx = idx[:n_scan]
                extra_idx = idx[n_scan:]

                if any(i != 0 for i in extra_idx):
                    continue

                time_slice = value[scan_idx]
                offset = self.counters.get(key, 0)
                logged = 0
                for v in time_slice:
                    v = float(v)
                    if not np.isnan(v):
                        self.logger.log_scalar(context, key, key, v, offset + logged, self.total_iterations)
                        logged += 1
                self.counters[key] = offset + logged

        self.global_step += math.prod(ref_shape)
        self.logger.close_context(context)


class ThreadedScalarLogger:
    """Async wrapper that processes log calls in a background thread."""

    def __init__(
        self, logger: Logger, loggers: list[Logger], num_levels: int, total_iterations: int, checkpoint_every: int
    ):
        self.scalar_logger = ScalarLogger(logger, num_levels, total_iterations, checkpoint_every)
        self.loggers = loggers
        self.job_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._process, daemon=False)
        self.worker.start()

    def _process(self):
        while not self.stop_event.is_set():
            try:
                stats = self.job_queue.get(timeout=0.001)
                if stats is None:
                    break
                self.scalar_logger.log(stats)
                self.job_queue.task_done()
            except queue.Empty:
                continue

    def log(self, stats: dict[str, jax.Array]):
        np_stats = {k: np.array(v) for k, v in stats.items()}
        self.job_queue.put(np_stats)

    def flush(self):
        self.job_queue.join()

    def shutdown(self):
        self.flush()
        self.job_queue.put(None)
        self.worker.join(timeout=5.0)
        self.stop_event.set()

    def __del__(self):
        if hasattr(self, "stop_event") and not self.stop_event.is_set():
            self.shutdown()


def create_logger(
    loggers: list[Logger], num_levels: int, total_iterations: int, checkpoint_every: int
) -> ThreadedScalarLogger:
    """Construct a ThreadedScalarLogger from a list of loggers."""
    logger = MultiLogger(loggers)
    return ThreadedScalarLogger(logger, loggers, num_levels, total_iterations, checkpoint_every)
