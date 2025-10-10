from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho",
    task_name="Seed+ILR Sweep: Batch-100,Epochs-75,CIFAR10,LSTM-128,SGD-Adam,BPTT-ID",
)
# task_name="Fixed Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD-Adam,BPTT-ID"
# task_name="OHO Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD/SGDN-Adam,BPTT-RTRL"
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="930dce5babfd4313b6a0a2cdc8a111e5",  # Use the actual task ID
    hyper_parameters=[
        # Seed configurations as complete seed objects
        DiscreteParameterRange(
            "config/seed/global_seed",
            values=[760, 202, 747, 995, 972, 579, 274, 283, 201, 480, 14, 530, 842, 774, 32, 471, 102, 104, 479, 789],
        ),
        DiscreteParameterRange("config/seed/test_seed", values=[12345]),
        DiscreteParameterRange(
            "config/learners/0/optimizer/learning_rate/value",
            [
                0.001,
                0.01,
                0.1,
                0.12584989506418265,
                0.15838196087665796,
                0.19932353156386887,
                0.25084845531135197,
                0.3156925177794596,
                0.3972987023509261,
                0.5,
            ],
        ),
        DiscreteParameterRange(
            "config/learners/0/optimizer/weight_decay/value",
            [
                1.00000000e-06,
                3.24593635e-06,
                1.05361028e-05,
                3.41995189e-05,
                1.11009462e-04,
                3.60329646e-04,
                1.16960710e-03,
                3.79647018e-03,
                1.23231006e-02,
                4.00000000e-02,
            ],
        ),
        # dataset
        # DiscreteParameterRange("config/dataset/_type", values=["FashionMnistConfig"]),
        # DiscreteParameterRange("config/dataset/n_in", values=[28]),
        DiscreteParameterRange("config/learners/1/learner/_type", values=["IdentityConfig"]),
        DiscreteParameterRange("config/learners/1/learner/epsilon", values=[None]),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[75]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[100]),
        # DiscreteParameterRange("config/data/0/train_percent", values=[80.00]),
        # DiscreteParameterRange("config/data/1/train_percent", values=[20.00]),
        # DiscreteParameterRange("config/data/0/num_steps_in_timeseries", values=[28]),
        # DiscreteParameterRange("config/data/1/num_steps_in_timeseries", values=[28]),
        DiscreteParameterRange("config/learners/0/num_virtual_minibatches_per_turn", values=[1]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[400]),
        # DiscreteParameterRange("config/readout_uses_input_data", values=[False]),
        DiscreteParameterRange("config/treat_inference_state_as_online", values=[False]),
        DiscreteParameterRange("config/logger_config", values=[({"_type": "HDF5LoggerConfig"},)]),
        DiscreteParameterRange("config/data_root_dir", values=["/scratch/datasets"]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["04:00:00"]),
        DiscreteParameterRange("slurm/cpu", values=[2]),
        DiscreteParameterRange("slurm/memory", values=["14GB"]),
        DiscreteParameterRange("slurm/use_singularity", values=[True]),
        DiscreteParameterRange("slurm/skip_python_env_install", values=[True]),
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
