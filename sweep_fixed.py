from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho",
    task_name="Seed+ILR Sweep: Batch-4000,Epochs-1000,CIFAR10,RNN-128-128,SGD-Adam,BPTT-ID",
    task_type=Task.TaskTypes.optimizer,
)
# task_name="Fixed Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD-Adam,BPTT-ID"
# task_name="OHO Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD/SGDN-Adam,BPTT-RTRL"
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="a38f7317fef24397a8b2f732641c0e66",  # Use the actual task ID
    hyper_parameters=[
        # Seed configurations as complete seed objects
        DiscreteParameterRange(
            "config/seed/global_seed",
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ),
        DiscreteParameterRange("config/seed/test_seed", values=[12345]),
        DiscreteParameterRange(
            "config/learners/0/optimizer/learning_rate/value",
            values=[
                # 0.01,
                # 0.019743504858348197,
                # 0.03898059840916188,
                # 0.0769613634072608,
                # 0.15194870523363546,
                # 0.29999999999999993,
                0.4,
            ],
        ),
        DiscreteParameterRange(
            "config/learners/0/optimizer/weight_decay/value",
            values=[
                9.999999999999999e-06,
                5.253055608807533e-05,
                0.00027594593229224307,
                0.0014495593273553926,
                0.007614615754863519,
                0.04000000000000001,
            ],
        ),
        DiscreteParameterRange("config/learners/1/learner/_type", values=["IdentityConfig"]),
        DiscreteParameterRange("config/learners/1/learner/epsilon", values=[None]),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[1000]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[4000]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[4000]),
        # DiscreteParameterRange("config/data/0/train_percent", values=[80.00]),
        # DiscreteParameterRange("config/data/1/train_percent", values=[20.00]),
        # DiscreteParameterRange("config/data/0/num_steps_in_timeseries", values=[28]),
        # DiscreteParameterRange("config/data/1/num_steps_in_timeseries", values=[28]),
        DiscreteParameterRange("config/learners/0/num_virtual_minibatches_per_turn", values=[1]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[10]),
        # DiscreteParameterRange("config/readout_uses_input_data", values=[False]),
        DiscreteParameterRange("config/treat_inference_state_as_online", values=[False]),
        DiscreteParameterRange("config/logger_config", values=[({"_type": "HDF5LoggerConfig"},)]),
        DiscreteParameterRange("config/data_root_dir", values=["/scratch/datasets"]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["03:30:00"]),
        DiscreteParameterRange("slurm/cpu", values=[4]),
        DiscreteParameterRange("slurm/memory", values=["16GB"]),
        DiscreteParameterRange("slurm/use_singularity", values=[True]),
        DiscreteParameterRange("slurm/skip_python_env_install", values=[True]),
        # gpu
        # DiscreteParameterRange("slurm/gpu", values=[1]),
        # DiscreteParameterRange("slurm/container_source/sif_path", values=["/scratch/wlp9800/images/devenv-gpu.sif"]),
    ],
    objective_metric_title="final_test/loss",
    objective_metric_series="final_test_loss",
    objective_metric_sign="min",
    max_number_of_concurrent_tasks=1950,
    optimizer_class=GridSearch,
    execution_queue="willyp",
    total_max_jobs=100_000,
)

# Start optimization
optimizer.start()
optimizer.wait()
optimizer.stop()
