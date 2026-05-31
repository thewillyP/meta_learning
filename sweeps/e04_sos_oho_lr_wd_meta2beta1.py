import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
# SOS_BETA_OHO base with HardClip(1.0) pre-baked at level-0 model_learner so DPRs don't conflict
# with the add_clip=None parent path (model_id=0b6dbef9dde2435fb8094941aa7b78ee)
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

# e04 = e03 with meta2_beta=1.0 (val objective = full β-ELBO with KL at weight 1.0).
# Hypothesis (INFERENCE, unverified): e03's lr→0 collapse happens because val objective
# is pure reconstruction (meta2_beta=0) — OHO can shrink lr to freeze the model with no
# penalty as long as recon is decent. Adding val-KL pressure (meta2_beta=1) should keep
# OHO pressing on the model to maintain informative latents, preventing lr collapse.
SEEDS = [0, 1, 2, 3, 4]


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E04_sos_oho_lr_wd_meta2beta1_test: 1-cell smoke"
    if args.test
    else "E04_sos_oho_lr_wd_meta2beta1: β fixed at 1.0, OHO on (lr, wd), val_beta=1.0 x 5 seeds"
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
        # OHO target = (lr, wd) only — β stays at init throughout (not in target)
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
        dpr(VAL_BETA_PATH, [1.0]),  # CHANGED FROM e03: 0.0 -> 1.0 (full β-ELBO at val)
        # Inner-init HPs (β fixed at 1.0 throughout because it's not in the OHO target)
        dpr(BETA_INIT_PATH, [1.0]),
        dpr(WD_INIT_PATH, [1e-6]),
        dpr(INNER_LR_PATH, [1e-3]),
        # clip on
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        # OHO mechanism: RTRL with same damping/finite_hvp as e01_sos
        dpr(f"{METHOD_PREFIX}/_type", ["RTRLConfig"]),
        dpr(f"{METHOD_PREFIX}/damping", [1e-5]),
        dpr(f"{METHOD_PREFIX}/start_at_step", [0]),
        dpr(f"{METHOD_PREFIX}/use_finite_hvp", [1e-3]),
        # logging + scheduling (identical to e03)
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
