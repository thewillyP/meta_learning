import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer, ParameterSet


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
BASE_TASK_ID = "bee61bc5c97a41f69383623db146117c"  # SOS_BETA_OHO_clip_eg_shaped_no_run_v2 base task (with SOSGridSampleInput/GridDeformationReporter sample-gen)


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

RTRL_BETAS = [0.1, 1.0]
VAL_BETAS = [0.0, 1.0]
E00_SOS_WINNER_LR = 1e-1  # E00_sos winner (HPO 3c5b6798, common-iter=50000 loss=10.6)


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


# --- Outer optimizer variants (4) — each is a coherent bundle of structural fields ---
additive_sgd = {
    f"{OPT_PREFIX}/_type": "SGDConfig",
    f"{OPT_PREFIX}/learning_rate": "meta2_sgd1_lr",
    f"{OPT_PREFIX}/weight_decay": "meta2_sgd1_wd",
    f"{OPT_PREFIX}/momentum": "meta2_sgd1_momentum",
    B1_PATH: 0.0,
    MLR_PATH: dpr(MLR_PATH, [1e-5, 1e-4, 1e-3]),
    RTRL_BETA_PATH: dpr(RTRL_BETA_PATH, RTRL_BETAS),
    VAL_BETA_PATH: dpr(VAL_BETA_PATH, VAL_BETAS),
}

additive_adam = {
    f"{OPT_PREFIX}/_type": "AdamConfig",
    f"{OPT_PREFIX}/learning_rate": "meta2_sgd1_lr",
    f"{OPT_PREFIX}/weight_decay": "meta2_sgd1_wd",
    f"{OPT_PREFIX}/momentum": "meta2_sgd1_momentum",
    f"{OPT_PREFIX}/second_momentum": 0.999,
    f"{OPT_PREFIX}/eps": 1e-8,
    f"{OPT_PREFIX}/eps_root": 0.0,
    MLR_PATH: dpr(MLR_PATH, [1e-5, 1e-4, 1e-3]),
    RTRL_BETA_PATH: dpr(RTRL_BETA_PATH, RTRL_BETAS),
    VAL_BETA_PATH: dpr(VAL_BETA_PATH, VAL_BETAS),
    B1_PATH: dpr(B1_PATH, [0.9, 0.99]),
}

eg_sgd = {
    f"{OPT_PREFIX}/_type": "ExponentiatedGradientConfig",
    f"{OPT_PREFIX}/base/_type": "SGDConfig",
    f"{OPT_PREFIX}/base/learning_rate": "meta2_sgd1_lr",
    f"{OPT_PREFIX}/base/weight_decay": "meta2_sgd1_wd",
    f"{OPT_PREFIX}/base/momentum": "meta2_sgd1_momentum",
    B1_PATH: 0.0,
    MLR_PATH: dpr(MLR_PATH, [1e-6, 1e-5, 1e-4]),
    RTRL_BETA_PATH: dpr(RTRL_BETA_PATH, RTRL_BETAS),
    VAL_BETA_PATH: dpr(VAL_BETA_PATH, VAL_BETAS),
}

eg_adam = {
    f"{OPT_PREFIX}/_type": "ExponentiatedGradientConfig",
    f"{OPT_PREFIX}/base/_type": "AdamConfig",
    f"{OPT_PREFIX}/base/learning_rate": "meta2_sgd1_lr",
    f"{OPT_PREFIX}/base/weight_decay": "meta2_sgd1_wd",
    f"{OPT_PREFIX}/base/momentum": "meta2_sgd1_momentum",
    f"{OPT_PREFIX}/base/second_momentum": 0.999,
    f"{OPT_PREFIX}/base/eps": 1e-8,
    f"{OPT_PREFIX}/base/eps_root": 0.0,
    B1_PATH: 0.9,
    MLR_PATH: dpr(MLR_PATH, [1e-4, 1e-3, 1e-2]),
    RTRL_BETA_PATH: dpr(RTRL_BETA_PATH, RTRL_BETAS),
    VAL_BETA_PATH: dpr(VAL_BETA_PATH, VAL_BETAS),
}


# OHO target as independent DPR (uniform wd_init=1e-6 across all variants means no coupling)
TARGETS = [
    ["meta1_beta"],
    ["meta1_beta", "meta1_sgd1_lr"],
    ["meta1_beta", "meta1_sgd1_lr", "meta1_sgd1_wd"],
]


task_name = (
    "E01_sos_test: 1-cell smoke" if args.test else "E01_sos: outer-opt x mlr x RTRL x val_beta x target (clip on, SOS)"
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
        ParameterSet(parameter_combinations=[additive_sgd, additive_adam, eg_sgd, eg_adam]),
        dpr(TARGET_PATH, TARGETS),
        dpr(WD_INIT_PATH, [1e-6]),
        dpr(BETA_INIT_PATH, [1e-5, 1.0]),
        dpr(INNER_LR_PATH, [E00_SOS_WINNER_LR]),
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        dpr("config/clearml_run", [True]),
        dpr("Args/skip_jitter", [False]),
        dpr(f"{METHOD_PREFIX}/_type", ["RTRLConfig"]),
        dpr(f"{METHOD_PREFIX}/damping", [1e-4]),
        dpr(f"{METHOD_PREFIX}/start_at_step", [0]),
        dpr(f"{METHOD_PREFIX}/use_finite_hvp", [1e-3]),
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
