import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
# SOS_BETA_OHO base (clearml_run=False) — fresh upload model_id=7f90dfe3397d4c798cd21980ca9e46be
BASE_TASK_ID = "ca5c29b41eef4b5c85f2c98f4159768c"

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

# Existential ablation: is β-OHO doing real work?
# Test: keep β fixed at 1.0; let OHO adapt lr + wd. If this matches 20849b4 (β+lr+wd OHO),
# then β-OHO is decoration. If it materially underperforms, β-OHO is essential.

# Recipe = 20849b4's recipe (additive_adam, β_init=1.0, mlr=1e-5, rtrl_β=0.1, b1=0.9, meta2_β=0)
# only thing changed: target excludes meta1_beta.
SEEDS = [0, 1, 2, 3, 4]


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E03_sos_oho_lr_wd_only_test: 1-cell smoke"
    if args.test
    else "E03_sos_oho_lr_wd_only: β fixed at 1.0, OHO on (lr, wd) only x 5 seeds"
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
        # OHO target = (lr, wd) only — β is excluded from the target subset → β stays at init
        dpr(TARGET_PATH, [["meta1_sgd1_lr", "meta1_sgd1_wd"]]),
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
        dpr(MLR_PATH, [1e-5]),
        dpr(B1_PATH, [0.9]),
        dpr(RTRL_BETA_PATH, [0.1]),
        dpr(VAL_BETA_PATH, [0.0]),
        # Inner-init HPs (β fixed at 1.0 throughout because it's not in the OHO target)
        dpr(BETA_INIT_PATH, [1.0]),
        dpr(WD_INIT_PATH, [1e-6]),
        dpr(INNER_LR_PATH, [1e-3]),  # start small; OHO will adjust upward if it should
        # clip on
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        # OHO mechanism: RTRL with same damping/finite_hvp as e01_sos
        dpr(f"{METHOD_PREFIX}/_type", ["RTRLConfig"]),
        dpr(f"{METHOD_PREFIX}/damping", [1e-5]),
        dpr(f"{METHOD_PREFIX}/start_at_step", [0]),
        dpr(f"{METHOD_PREFIX}/use_finite_hvp", [1e-3]),
        # logging + scheduling (identical to e01_sos / e02)
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
