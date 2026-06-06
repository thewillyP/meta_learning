import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
# SOS_BETA_OHO_WIDE base task (wide arch: encoder [64,128,256], decoder [128,64])
BASE_TASK_ID = "3901784cc5804358962f565b3f383337"

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

SEEDS = [0, 1, 2, 3, 4]


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E09_sos_smallhole_wide_test: 1-cell smoke"
    if args.test
    else "E09_sos_smallhole_wide: wide arch + 10x10 hole + disjoint val/test, OHO β+lr+wd mlr=1e-4 lr_init=1e-1 x 5 seeds"
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
        dpr(TARGET_PATH, [["meta1_beta", "meta1_sgd1_lr", "meta1_sgd1_wd"]]),
        dpr("config/seed/global_seed", SEEDS),
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
        dpr(BETA_INIT_PATH, [1.0]),
        dpr(WD_INIT_PATH, [1e-6]),
        dpr(INNER_LR_PATH, [1e-1]),
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        dpr(f"{METHOD_PREFIX}/_type", ["RTRLConfig"]),
        dpr(f"{METHOD_PREFIX}/damping", [1e-5]),
        dpr(f"{METHOD_PREFIX}/start_at_step", [0]),
        dpr(f"{METHOD_PREFIX}/use_finite_hvp", [1e-3]),
        # 10x10 hole + disjoint val/test (same geometry as e08)
        dpr("config/levels/0/dataset_source/region", [[9.0, 19.0, 9.0, 19.0]]),
        dpr("config/levels/0/dataset_source/region_mode", ["exclude_region"]),
        dpr("config/levels/1/dataset_source/region", [[9.0, 14.0, 9.0, 19.0]]),
        dpr("config/levels/1/dataset_source/region_mode", ["only_region"]),
        dpr("config/levels/2/dataset_source/region", [[14.0, 19.0, 9.0, 19.0]]),
        dpr("config/levels/2/dataset_source/region_mode", ["only_region"]),
        dpr("config/levels/1/dataset/num_examples_total", [50000]),
        dpr("config/levels/2/dataset/num_examples_total", [50000]),
        # logging + scheduling (identical to e08)
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
