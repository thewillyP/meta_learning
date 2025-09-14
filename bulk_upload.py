import os
import glob
from pathlib import Path
from clearml import Task
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


def upload_and_delete_session(session_zip_path):
    try:
        Task.import_offline_session(session_folder_zip=session_zip_path)
        os.remove(session_zip_path)
        return f"Success: {session_zip_path}"
    except Exception as e:
        return f"Error {session_zip_path}: {e}"


def bulk_upload_offline_sessions(offline_cache_dir, max_workers=4):
    zip_files = glob.glob(os.path.join(offline_cache_dir, "*.zip"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_and_delete_session, zip_file): zip_file for zip_file in zip_files}

        for future in as_completed(futures):
            result = future.result()
            print(result)


if __name__ == "__main__":
    task: Task = Task.init(project_name="utilities", task_name="bulk_upload_offline_sessions")

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
    }
    task.connect(slurm_params, name="slurm")

    # Cache configuration
    cache_config = {"offline_cache_dir": "/scratch", "max_workers": 50}
    task.connect(cache_config, name="cache")

    task.execute_remotely(queue_name="willyp", clone=False, exit_process=True)

    # Get configuration from task
    cache_params = task.get_parameters("cache")
    offline_cache_dir = cache_params["offline_cache_dir"]
    max_workers = cache_params["max_workers"]

    # Upload all offline sessions in parallel
    bulk_upload_offline_sessions(offline_cache_dir, max_workers)
