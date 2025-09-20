import clearml
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime


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
