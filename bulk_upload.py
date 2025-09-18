import h5py
import glob
import os
from clearml import Task
from concurrent.futures import ThreadPoolExecutor, as_completed


def upload_hdf5_file(hdf5_path: str, batch_size: int) -> str:
    """Upload metrics from HDF5 file to existing ClearML task"""
    try:
        with h5py.File(hdf5_path, "r") as f:
            task_id = f.attrs.get("task_id")
            if not task_id:
                return f"Error {hdf5_path}: No task_id found in file"

            task: Task = Task.get_task(task_id=task_id)
            logger = task.get_logger()

            metric_count = 0
            dataset_names = [name for name in f.keys() if not name.endswith("_iterations")]

            for series in dataset_names:
                title = f[series].attrs["title"]
                dataset = f[series]
                iter_dataset = f[f"{series}_iterations"]

                # Find actual end of data
                actual_length = 0
                for i in range(len(iter_dataset)):
                    if iter_dataset[i] == -1:
                        break
                    actual_length += 1

                for start in range(0, actual_length, batch_size):
                    end = min(start + batch_size, actual_length)
                    values = dataset[start:end]
                    iterations = iter_dataset[start:end]

                    for value, iteration in zip(values, iterations):
                        logger.report_scalar(title=title, series=series, value=float(value), iteration=int(iteration))
                        metric_count += 1

        # os.remove(hdf5_path)
        return f"Success: {hdf5_path} -> Task {task_id} ({metric_count} metrics uploaded)"

    except Exception as e:
        return f"Error {hdf5_path}: {e}"


def bulk_upload_hdf5_files(offline_log_dir: str, max_workers: int, batch_size: int):
    """Upload all HDF5 metric files in parallel to their original tasks"""

    print(f"Scanning for HDF5 metric files in {offline_log_dir}")
    print(f"Using max_workers={max_workers}, batch_size={batch_size}")

    h5_files = glob.glob(os.path.join(offline_log_dir, "metrics_*.h5"))

    if not h5_files:
        print(f"No HDF5 metric files found in {offline_log_dir}")
        return

    print(f"Found {len(h5_files)} HDF5 metric files to upload")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_hdf5_file, h5_file, batch_size): h5_file for h5_file in h5_files}

        for future in as_completed(futures):
            result = future.result()
            print(result)


if __name__ == "__main__":
    task: Task = Task.init(project_name="utilities", task_name="bulk_upload_hdf5_metrics")

    # Slurm configuration
    slurm_params = {
        "memory": "8GB",
        "time": "06:00:00",
        "cpu": 2,
        "gpu": 0,
        "log_dir": "/vast/wlp9800/logs",
        "singularity_overlay": "",
        "singularity_binds": "/scratch/wlp9800/clearml:/scratch",
        "container_source": {"sif_path": "/scratch/wlp9800/images/devenv-cpu.sif", "type": "sif_path"},
        "use_singularity": True,
        "setup_commands": "module load python/intel/3.8.6",
        "skip_python_env_install": True,
    }
    task.connect(slurm_params, name="slurm")

    # Upload configuration
    upload_config = {
        "offline_log_dir": "/scratch/offline_logs",
        "max_workers": 50,
        "batch_size": 1000,
        "clearml_run": False,
    }
    task.connect(upload_config, name="upload")

    if task.get_parameter("upload/clearml_run", cast=True):
        # Upload all offline metrics to their original tasks
        bulk_upload_hdf5_files(
            task.get_parameter("upload/offline_log_dir"),
            task.get_parameter("upload/max_workers", cast=True),
            task.get_parameter("upload/batch_size", cast=True),
        )
