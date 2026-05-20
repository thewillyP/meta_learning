import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
BASE_TASK_ID = "21030cf4c06f4111bf6a02b6761da929"  # SOS_BETA_OHO_clip_eg_shaped_no_run base task

METHOD_PREFIX = "config/levels/1/learner/optimizer_learner/method"


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


# logspace(1e-3, 1e0, 10) — wide range because clip bounds gradient, so effective step ~ lr
INNER_LR_VALUES = [
    1.0000000000e-03,
    2.1544346900e-03,
    4.6415888336e-03,
    1.0000000000e-02,
    2.1544346900e-02,
    4.6415888336e-02,
    1.0000000000e-01,
    2.1544346900e-01,
    4.6415888336e-01,
    1.0000000000e+00,
]


task_name = "E00_sos_test: 1-cell smoke" if args.test else "E00_sos: inner lr sweep on SOS (no OHO, fixed beta=1, clip on)"
opt_task = Task.init(
    project_name=PROJECT,
    task_name=task_name,
    task_type=Task.TaskTypes.optimizer,
)
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

max_jobs = 1 if args.test else 100_000
test_only_dprs = (
    [
        dpr("config/epochs", [args.test_epochs]),
        dpr("config/sample_generators/0/every_n_epochs", [1]),
        dpr("config/sample_generators/1/every_n_epochs", [1]),
        dpr("config/sample_generators/2/every_n_epochs", [1]),
        dpr("config/sample_generators/3/every_n_epochs", [1]),
        dpr("config/sample_generators/4/every_n_epochs", [1]),
    ]
    if args.test
    else []
)

optimizer = HyperParameterOptimizer(
    base_task_id=BASE_TASK_ID,
    hyper_parameters=[
        dpr("config/hyperparameters/meta1_sgd1_lr/value", INNER_LR_VALUES),
        dpr("config/hyperparameters/meta1_beta/value", [1.0]),
        dpr(f"{METHOD_PREFIX}/_type", ["IdentityLearnerConfig"]),
        dpr(f"{METHOD_PREFIX}/bptt_config/_type", ["BPTTConfig"]),
        dpr(f"{METHOD_PREFIX}/bptt_config/truncate_at", [None]),
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        dpr("config/clearml_run", [True]),
        dpr("Args/skip_jitter", [False]),
        dpr("config/logger_config/clearml/enabled", [False]),
        dpr("config/logger_config/hdf5/enabled", [True]),
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
