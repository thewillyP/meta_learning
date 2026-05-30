import argparse

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="1-cell smoke test with few epochs")
parser.add_argument("--test-epochs", type=int, default=5)
args = parser.parse_args()

PROJECT = "oho"
QUEUE = "willyp"
# Same SOS base as e01_sos so arch/data/eval are identical → directly comparable
BASE_TASK_ID = "c14b3d0807114b7db88c429920291287"

METHOD_PREFIX = "config/levels/1/learner/optimizer_learner/method"

# Existential ablation: does fixed-wd training (NO OHO) match the OHO winner (20849b4, loss=1.33)?
# wd grid spans 5 orders of magnitude, including the OHO-discovered peak (~2e-2) and the inner-init (1e-6).
WD_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SEEDS = [0, 1, 2, 3, 4]

# Fixed inner HPs match the e01_sos winner setup:
# β_init=1.0 (winning init; OHO had moved β to 0.69 by end-of-train so 1.0 is a defensible fixed value).
# inner_lr=0.1 = e00_sos winner LR.
FIXED_BETA = 1.0
FIXED_INNER_LR = 1e-1


def dpr(path: str, values: list):
    return DiscreteParameterRange(path, values=values)


task_name = (
    "E02_sos_wd_no_oho_test: 1-cell smoke"
    if args.test
    else "E02_sos_wd_no_oho: fixed-β fixed-lr wd sweep x 5 seeds (NO OHO, clip on, SOS)"
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
        # the swept axis
        dpr("config/hyperparameters/meta1_sgd1_wd/value", WD_VALUES),
        # seed replicas
        dpr("config/seed/global_seed", SEEDS),
        # fixed inner HPs
        dpr("config/hyperparameters/meta1_beta/value", [FIXED_BETA]),
        dpr("config/hyperparameters/meta1_sgd1_lr/value", [FIXED_INNER_LR]),
        # disable OHO: IdentityLearnerConfig + BPTT(truncate_at=None) per e00_sos_inner_lr pattern
        dpr(f"{METHOD_PREFIX}/_type", ["IdentityLearnerConfig"]),
        dpr(f"{METHOD_PREFIX}/bptt_config/_type", ["BPTTConfig"]),
        dpr(f"{METHOD_PREFIX}/bptt_config/truncate_at", [None]),
        # clip on (matches e01_sos winner config)
        dpr("config/levels/0/learner/model_learner/add_clip/_type", ["HardClip"]),
        dpr("config/levels/0/learner/model_learner/add_clip/threshold", [1.0]),
        # logging + scheduling (identical to e01_sos)
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
