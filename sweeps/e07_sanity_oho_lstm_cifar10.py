import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
# OHO_LSTM128_CIFAR10_BASE (clearml_run=false baked) — fresh base task registered locally.
# model_id (config) = 6a6b35af2229485ab78318a12f1c2b93
BASE_TASK_ID = "bbbc0bb1affa4c2aa5c7ea73c978ef06"

# Sanity check: known-good LSTM-CIFAR10 OHO setup x 5 seeds, ClearML logging.
# Purpose: verify the OHO algorithm + dataloader work on the canonical baseline,
# independent of the SOS/β-VAE collapse issues.
SEEDS = [0, 1, 2, 3, 4]


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E07_sanity_oho_lstm_cifar10_test: 1-cell smoke"
    if args.test
    else "E07_sanity_oho_lstm_cifar10: OHO_LSTM128_CIFAR10 x 5 seeds (sanity check)"
)
opt_task = Task.init(
    project_name=PROJECT,
    task_name=task_name,
    task_type=Task.TaskTypes.optimizer,
)
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

max_jobs = 1 if args.test else 100_000
test_only_dprs = (
    [dpr("config/epochs", [args.test_epochs])]
    if args.test
    else []
)

optimizer = HyperParameterOptimizer(
    base_task_id=BASE_TASK_ID,
    hyper_parameters=[
        dpr("config/clearml_run", [True]),
        dpr("config/seed/global_seed", SEEDS),
        dpr("Args/skip_jitter", [False]),
        dpr("config/logger_config/clearml/enabled", [True]),
        dpr("config/logger_config/hdf5/enabled", [False]),
        dpr("config/logger_config/sqlite/enabled", [False]),
        dpr("config/logger_config/console/enabled", [False]),
        dpr("config/logger_config/matplotlib/enabled", [False]),
        dpr("slurm/time", ["03:00:00"]),
        dpr("slurm/cpu", [4]),
        dpr("slurm/memory", ["16GB"]),
        dpr("slurm/gpu", [1]),
        dpr("slurm/log_dir", ["/scratch/wlp9800/offline_logs"]),
        dpr("slurm/skip_python_env_install", [True]),
        *test_only_dprs,
    ],
    objective_metric_title="eval_accumulated/level2/loss/0/0/0",
    objective_metric_series="eval",
    objective_metric_sign="min",
    max_number_of_concurrent_tasks=max_jobs,
    optimizer_class=GridSearch,
    execution_queue=QUEUE,
    total_max_jobs=max_jobs,
    spawn_project=PROJECT,
)

optimizer.start()
optimizer.wait()
optimizer.stop()
