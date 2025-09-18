import h5py
import glob
import os
import time
import random
import argparse
import numpy as np
from clearml.backend_api.session.client import APIClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


def upload_hdf5_file_rest_api(hdf5_path: str, batch_size: int, target_task_id: Optional[str] = None) -> str:
    """Upload metrics using ClearML REST API batch endpoint"""
    try:
        with h5py.File(hdf5_path, "r") as f:
            original_task_id = f.attrs.get("task_id")
            if not original_task_id:
                return f"Error {hdf5_path}: No task_id found in file"

            # Use target_task_id if provided, otherwise use original
            task_id = target_task_id if target_task_id else original_task_id

            # Get API client for authentication
            client = APIClient()

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

        return f"Success: {hdf5_path} -> Task {task_id} ({total_uploaded} metrics uploaded)"

    except Exception as e:
        import traceback

        return f"Error {hdf5_path}: {e}\nTraceback: {traceback.format_exc()}"


def find_hdf5_files_for_tasks(offline_log_dir: str, task_ids: List[str]) -> List[tuple]:
    """Find HDF5 files for specific task IDs and return list of (file_path, task_id) tuples"""
    matching_files = []

    for task_id in task_ids:
        h5_file = os.path.join(offline_log_dir, f"metrics_{task_id}.h5")
        if os.path.exists(h5_file):
            matching_files.append((h5_file, task_id))
            print(f"Found: {h5_file}")
        else:
            print(f"Not found: {h5_file}")

    return matching_files


def bulk_upload_hdf5_files_by_task_ids(offline_log_dir: str, task_ids: List[str], max_workers: int, batch_size: int):
    """Upload HDF5 metric files for specific task IDs using REST API batch upload"""
    print(f"Using {max_workers} workers with batch API")
    print(f"Using batch upload with batch_size={batch_size}")

    matching_files = find_hdf5_files_for_tasks(offline_log_dir, task_ids)

    if not matching_files:
        print("No matching HDF5 files found for the specified task IDs")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_hdf5_file_rest_api, h5_file, batch_size): (h5_file, task_id)
            for h5_file, task_id in matching_files
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(matching_files)}] {result}")


def main():
    parser = argparse.ArgumentParser(description="Upload HDF5 metrics for specific ClearML task IDs")
    parser.add_argument("--offline-log-dir", required=True, help="Directory containing HDF5 metric files")
    parser.add_argument("--task-ids", required=True, nargs="+", help="List of task IDs to upload metrics for")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker threads (default: 5)")
    parser.add_argument("--batch-size", type=int, default=60000, help="Batch size for API uploads (default: 60000)")

    args = parser.parse_args()

    print(f"Task IDs to process: {args.task_ids}")

    bulk_upload_hdf5_files_by_task_ids(args.offline_log_dir, args.task_ids, args.max_workers, args.batch_size)


if __name__ == "__main__":
    main()
