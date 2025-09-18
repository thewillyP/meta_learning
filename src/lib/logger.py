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
