import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
BASE_TASK_ID = "fd5da229b21346edb1bbf733a31807c4"

OPT_PREFIX = "config/levels/1/learner/optimizer/meta2_sgd1/optimizer"
METHOD_PREFIX = "config/levels/1/learner/optimizer_learner/method"

MLR_PATH = "config/hyperparameters/meta2_sgd1_lr/value"
RTRL_BETA_PATH = f"{METHOD_PREFIX}/beta"
VAL_BETA_PATH = "config/hyperparameters/meta2_beta/value"
B1_PATH = "config/hyperparameters/meta2_sgd1_momentum/value"
TARGET_PATH = "config/levels/1/learner/optimizer/meta2_sgd1/target"
WD_INIT_PATH = "config/hyperparameters/meta1_sgd1_wd/value"
BETA_INIT_PATH = "config/hyperparameters/meta1_beta/value"
INNER_LR_PATH = "config/hyperparameters/meta1_sgd1_lr/value"

# e06 = e05 with inner lr initialized at 1e-1 instead of 1e-3.
# Goal: test if e05's seed-dependent lr→0 collapse is partly caused by the small starting lr.
# If e06 still collapses → starting point isn't the issue, the OHO dynamics + zero-boundary are.
SEEDS = [0, 1, 2, 3, 4]


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E06_sos_oho_all_lrinit1e1_test: 1-cell smoke"
    if args.test
    else "E06_sos_oho_all_lrinit1e1: β+lr+wd OHO, mlr=1e-4, lr_init=1e-1, val_beta=0 x 5 seeds"
)
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
        # OHO target = full set (β + lr + wd)
        dpr(TARGET_PATH, [["meta1_beta", "meta1_sgd1_lr", "meta1_sgd1_wd"]]),
        # seeds
        dpr("config/seed/global_seed", SEEDS),
        # Outer optimizer = additive_adam (20849b4's family)
        dpr(f"{OPT_PREFIX}/_type", ["AdamConfig"]),
        dpr(f"{OPT_PREFIX}/learning_rate", ["meta2_sgd1_lr"]),
        dpr(f"{OPT_PREFIX}/weight_decay", ["meta2_sgd1_wd"]),
        dpr(f"{OPT_PREFIX}/momentum", ["meta2_sgd1_momentum"]),
        dpr(f"{OPT_PREFIX}/second_momentum", [0.999]),
        dpr(f"{OPT_PREFIX}/eps", [1e-8]),
        dpr(f"{OPT_PREFIX}/eps_root", [0.0]),
        dpr(MLR_PATH, [1e-4]),
        dpr(B1_PATH, [0.9]),
        dpr(RTRL_BETA_PATH, [0.1]),
        dpr(VAL_BETA_PATH, [0.0]),
        # Inner inits — lr bumped from 1e-3 to 1e-1
        dpr(BETA_INIT_PATH, [1.0]),
        dpr(WD_INIT_PATH, [1e-6]),
        dpr(INNER_LR_PATH, [1e-1]),
        # clip on
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        # OHO mechanism: RTRL with same damping/finite_hvp as e01_sos
        dpr(f"{METHOD_PREFIX}/_type", ["RTRLConfig"]),
        dpr(f"{METHOD_PREFIX}/damping", [1e-5]),
        dpr(f"{METHOD_PREFIX}/start_at_step", [0]),
        dpr(f"{METHOD_PREFIX}/use_finite_hvp", [1e-3]),
        # logging + scheduling (identical to e05)
        dpr("config/clearml_run", [True]),
        dpr("Args/skip_jitter", [False]),
        dpr("config/logger_config/clearml/enabled", [False]),
        dpr("config/logger_config/hdf5/enabled", [False]),
        dpr("config/logger_config/sqlite/enabled", [True]),
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
