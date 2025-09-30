import clearml
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Protocol
import matplotlib.pyplot as plt
from collections import defaultdict


class Logger(Protocol):
    def get_context(self): ...

    def close_context(self, context): ...

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int): ...


class HDF5Logger:
    def __init__(self, log_dir: str, task_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task_id = task_id
        self.log_file = self.log_dir / f"metrics_{task_id}.h5"
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

            save_path = self.save_dir / f"{series}.png"
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
