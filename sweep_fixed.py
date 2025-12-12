from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho",
    task_name="Fixed Seed+ILR Sweep: Batch-4000,Epochs-1000,CIFAR10,RNN-256,SGD-SGD,BPTT-ID",
    task_type=Task.TaskTypes.optimizer,
)
# task_name="Fixed Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD-Adam,BPTT-ID"
# task_name="OHO Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD/SGDN-Adam,BPTT-RTRL"
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

task = Task.get_task(project_name="oho", task_name="rnn256_base1")

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=task.id,
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
                0.0001,
                0.005,
                0.041874999999999996,
                0.07875,
                0.115625,
                0.1525,
                0.189375,
                0.22625,
                0.263125,
                0.3,
            ],
        ),
        DiscreteParameterRange(
            "config/learners/0/optimizer/weight_decay/value",
            values=[
                0.0,
                1e-06,
                3.866973986492824e-06,
                1.4953487812212205e-05,
                5.782474837716209e-05,
                0.00022360679774997895,
                0.0008646816701021308,
                0.0033437015248821097,
                0.012930006815315507,
                0.049999999999999996,
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
        DiscreteParameterRange("slurm/time", values=["04:00:00"]),
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
    max_number_of_concurrent_tasks=100_000,
    optimizer_class=GridSearch,
    execution_queue="willyp",
    total_max_jobs=100_000,
)

# Start optimization
optimizer.start()
optimizer.wait()
optimizer.stop()
