from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, ParameterSet, GridSearch
from clearml import Task

opt_task = Task.init(
    project_name="oho",
    task_name="OHO Seed Sweep: Batch-4000,Epochs-2000,sCIFAR10,RNN-256,SGD-SGD,BPTT-RTRLFiniteHvp",
    task_type=Task.TaskTypes.optimizer,
)
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

task = Task.get_task(project_name="oho", task_name="OHO_RNN256_sCIFAR10_v3")

optimizer = HyperParameterOptimizer(
    base_task_id=task.id,
    hyper_parameters=[
        # Sweep
        DiscreteParameterRange(
            "config/seed/global_seed",
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ),
        # Inner optimizer hyperparameters
        DiscreteParameterRange("config/hyperparameters/meta1_sgd1_lr/value", values=[1.0e-3]),
        DiscreteParameterRange("config/hyperparameters/meta1_sgd1_wd/value", values=[1.0e-5]),
        # Outer optimizer hyperparameters
        DiscreteParameterRange("config/hyperparameters/meta2_sgd1_lr/value", values=[1e-3]),
        # Outer optimizer learner (level 1)
        DiscreteParameterRange(
            "config/levels/1/learner/optimizer_learner/method/_type", values=["RTRLFiniteHvpConfig"]
        ),
        DiscreteParameterRange("config/levels/1/learner/optimizer_learner/method/epsilon", values=[1e-3]),
        DiscreteParameterRange("config/levels/1/learner/optimizer_learner/method/rtrl_config/damping", values=[1e-4]),
        # Batch sizes
        DiscreteParameterRange("config/levels/0/dataset/num_examples_in_minibatch", values=[4000]),
        DiscreteParameterRange("config/levels/1/dataset/num_examples_in_minibatch", values=[4000]),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/epochs", values=[2000]),
        DiscreteParameterRange("config/data_root_dir", values=["/scratch/wlp9800/datasets"]),
        DiscreteParameterRange("config/log_dir", values=["/scratch/wlp9800/offline_logs"]),
        DiscreteParameterRange("config/log_title", values=["oho"]),
        # Logger config
        ParameterSet(
            parameter_combinations=[
                {
                    "config/logger_config/clearml/enabled": True,
                    "config/logger_config/hdf5/enabled": True,
                    "config/logger_config/console/enabled": False,
                    "config/logger_config/matplotlib/enabled": False,
                },
            ]
        ),
        # Slurm
        DiscreteParameterRange("slurm/time", values=["02:00:00"]),
        DiscreteParameterRange("slurm/cpu", values=[4]),
        DiscreteParameterRange("slurm/memory", values=["16GB"]),
        DiscreteParameterRange("slurm/gpu", values=[1]),
        DiscreteParameterRange("slurm/log_dir", values=["/scratch/wlp9800/offline_logs"]),
        DiscreteParameterRange("slurm/skip_python_env_install", values=[True]),
    ],
    objective_metric_title="eval_accumulated/level2/loss/0/0/0",
    objective_metric_series="oho",
    objective_metric_sign="min",
    max_number_of_concurrent_tasks=100_000,
    optimizer_class=GridSearch,
    execution_queue="willyp",
    total_max_jobs=100_000,
)

optimizer.start()
optimizer.wait()
optimizer.stop()
