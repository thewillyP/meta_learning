import h5py
import glob
import os
import time
import random
import numpy as np
from clearml import Task
from clearml.backend_api.session.client import APIClient
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_marker_file_path(hdf5_path: str, marker_dir: str) -> str:
    """Get marker file path for given HDF5 file"""
    filename = os.path.basename(hdf5_path)
    return os.path.join(marker_dir, f"{filename}.done")


def is_already_processed(hdf5_path: str, marker_dir: str) -> bool:
    """Check if HDF5 file has already been processed"""
    marker_file = get_marker_file_path(hdf5_path, marker_dir)
    return os.path.exists(marker_file)


def mark_as_processed(hdf5_path: str, marker_dir: str):
    """Create marker file to indicate processing is complete"""
    marker_file = get_marker_file_path(hdf5_path, marker_dir)
    os.makedirs(marker_dir, exist_ok=True)
    open(marker_file, "a").close()


def should_exclude_task(hdf5_path: str, excluded_task_ids: set) -> bool:
    """Check if task should be excluded based on task_id in filename"""
    if not excluded_task_ids:
        return False

    filename = os.path.basename(hdf5_path)
    return any(task_id in filename for task_id in excluded_task_ids)


def is_task_completed(task_id: str, client: APIClient) -> bool:
    """Check if task status is completed"""
    try:
        response = client.tasks.get_all(id=[task_id])
        if response and len(response) > 0:
            status = response[0].status
            return str(status) == "completed"
        return False
    except Exception as e:
        print(f"Error checking task status for {task_id}: {e}")
        return False


def upload_hdf5_file_rest_api(hdf5_path: str, batch_size: int, marker_dir: str, excluded_task_ids: set) -> str:
    """Upload metrics using ClearML REST API batch endpoint"""
    try:
        # Check if already processed
        if is_already_processed(hdf5_path, marker_dir):
            return f"Skipped {hdf5_path}: Already processed (marker file exists)"

        # Check if task should be excluded
        if should_exclude_task(hdf5_path, excluded_task_ids):
            return f"Skipped {hdf5_path}: Task excluded by configuration"

        with h5py.File(hdf5_path, "r") as f:
            task_id = f.attrs.get("task_id")
            if not task_id:
                return f"Error {hdf5_path}: No task_id found in file"

            # Get API client for authentication
            client = APIClient()

            # Check if task is completed
            if not is_task_completed(task_id, client):
                return f"Skipped {hdf5_path}: Task {task_id} not completed yet"

            dataset_names = [name for name in f.keys() if not name.endswith("_iterations")]

            # Create iterators for each series
            series_info = []
            for series in dataset_names:
                title = f[series].attrs["title"]
                dataset = f[series]
                iter_dataset = f[f"{series}_iterations"]
                actual_length = len(iter_dataset)

                series_info.append(
                    {
                        "series": series,
                        "title": title,
                        "dataset": dataset,
                        "iter_dataset": iter_dataset,
                        "length": actual_length,
                        "current_idx": 0,
                    }
                )

            total_metrics = sum(info["length"] for info in series_info)
            print(f"Total metrics to upload: {total_metrics}")

            # Stream batches across all series
            total_uploaded = 0
            batch_count = 0

            while any(info["current_idx"] < info["length"] for info in series_info):
                batch = []

                # Fill batch from all series round-robin style
                while len(batch) < batch_size and any(info["current_idx"] < info["length"] for info in series_info):
                    for info in series_info:
                        if info["current_idx"] < info["length"] and len(batch) < batch_size:
                            i = info["current_idx"]
                            value = float(info["dataset"][i])
                            # Convert NaN and inf values to -1
                            if np.isnan(value) or np.isinf(value):
                                value = -1.0
                            event = {
                                "task": task_id,
                                "type": "training_stats_scalar",
                                "metric": info["title"],
                                "variant": info["series"],
                                "value": value,
                                "iter": int(info["iter_dataset"][i]),
                                "timestamp": int(time.time() * 1000),
                            }
                            batch.append(event)
                            info["current_idx"] += 1

                if batch:  # Only upload if we have events
                    try:
                        response = client.events.add_batch(requests=batch)
                        total_uploaded += response.added
                        batch_count += 1
                        print(f"Uploaded batch {batch_count}: {response.added} metrics")

                    except Exception as e:
                        import traceback

                        return f"Error uploading batch for {hdf5_path}: {e}\nTraceback: {traceback.format_exc()}"

            # Mark as processed after successful upload
            mark_as_processed(hdf5_path, marker_dir)
            return f"Success: {hdf5_path} -> Task {task_id} ({total_uploaded} metrics uploaded)"

    except Exception as e:
        import traceback

        return f"Error {hdf5_path}: {e}\nTraceback: {traceback.format_exc()}"


def bulk_upload_hdf5_files_api(
    offline_log_dir: str,
    max_workers: int,
    batch_size: int,
    marker_dir: str,
    excluded_task_ids: str,
    check_interval: int,
):
    """Upload all HDF5 metric files using REST API batch upload in continuous loop"""
    print(f"Starting continuous upload monitoring...")
    print(f"Scanning for HDF5 metric files in {offline_log_dir}")
    print(f"Using {max_workers} workers with batch API")
    print(f"Using batch upload with batch_size={batch_size}")
    print(f"Marker files in: {marker_dir}")
    print(f"Check interval: {check_interval} seconds")

    # Parse excluded task IDs
    excluded_set = set()
    if excluded_task_ids:
        excluded_set = set(task_id.strip() for task_id in excluded_task_ids.split(",") if task_id.strip())
        print(f"Excluding {len(excluded_set)} task IDs: {excluded_set}")

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new files...")

        h5_files = glob.glob(os.path.join(offline_log_dir, "metrics_*.h5"))
        if not h5_files:
            print(f"No HDF5 metric files found in {offline_log_dir}")
        else:
            print(f"Found {len(h5_files)} HDF5 metric files")

            # Count already processed files
            already_processed = sum(1 for f in h5_files if is_already_processed(f, marker_dir))
            excluded_count = sum(1 for f in h5_files if should_exclude_task(f, excluded_set))
            files_to_process = len(h5_files) - already_processed - excluded_count

            print(f"Already processed: {already_processed}")
            print(f"Excluded by task ID: {excluded_count}")
            print(f"Files to process: {files_to_process}")

            if files_to_process > 0:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            upload_hdf5_file_rest_api, h5_file, batch_size, marker_dir, excluded_set
                        ): h5_file
                        for h5_file in h5_files
                    }

                    completed = 0
                    for future in as_completed(futures):
                        result = future.result()
                        completed += 1
                        print(f"[{completed}/{len(h5_files)}] {result}")

        print(f"Waiting {check_interval} seconds before next check...")
        time.sleep(check_interval)


if __name__ == "__main__":
    task: Task = Task.init(project_name="utilities", task_name="bulk_upload_hdf5_metrics_api")

    # Slurm configuration
    slurm_params = {
        "memory": "16GB",
        "time": "06:00:00",
        "cpu": 4,
        "gpu": 0,
        "log_dir": "/vast/wlp9800/logs",
        "singularity_overlay": "",
        "singularity_binds": "/scratch/wlp9800/clearml:/scratch,/vast/wlp9800/upload_markers:/vast/markers",
        "container_source": {"sif_path": "/scratch/wlp9800/images/devenv-cpu.sif", "type": "sif_path"},
        "use_singularity": True,
        "setup_commands": "module load python/intel/3.8.6",
        "skip_python_env_install": True,
    }
    task.connect(slurm_params, name="slurm")

    # Upload configuration for API version
    upload_config = {
        "offline_log_dir": "/scratch/offline_logs",
        "max_workers": 5,
        "batch_size": 60000,
        "clearml_run": False,
        "marker_dir": "/vast/markers",
        "excluded_task_ids": "",
        "check_interval": 300,
    }
    task.connect(upload_config, name="upload")

    if task.get_parameter("upload/clearml_run", cast=True):
        bulk_upload_hdf5_files_api(
            task.get_parameter("upload/offline_log_dir"),
            task.get_parameter("upload/max_workers", cast=True),
            task.get_parameter("upload/batch_size", cast=True),
            task.get_parameter("upload/marker_dir"),
            task.get_parameter("upload/excluded_task_ids"),
            task.get_parameter("upload/check_interval", cast=True),
        )
