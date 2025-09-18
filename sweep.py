from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho", task_name="Fixed Seed+ILR Sweep: Epochs-20,FashionMNIST,GRU-128,SGD-Adam,BPTT-ID"
)
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="a2f4d504b01b4147bd7185ee60ddee06",  # Use the actual task ID
    hyper_parameters=[
        # Seed configurations as complete seed objects
        DiscreteParameterRange(
            "config/seed/global_seed",
            values=[760, 202, 747, 995, 972, 579, 274, 283, 201, 480, 14, 530, 842, 774, 32, 471, 102, 104, 479, 789],
        ),
        DiscreteParameterRange("config/seed/test_seed", values=[12345]),
        DiscreteParameterRange(
            "config/learners/0/optimizer/learning_rate",
            values=[
                1.00000000e-05,
                3.45417413e-05,
                1.19313189e-04,
                4.12128530e-04,
                1.42356370e-03,
                4.91723692e-03,
                1.69849925e-02,
                5.86691217e-02,
                2.02653362e-01,
                # 7.00000000e-01,
            ],
            # values=[1.0e-2, 1.0e-3],
        ),
        # dataset
        DiscreteParameterRange("config/dataset/_type", values=["FashionMnistConfig"]),
        DiscreteParameterRange("config/dataset/n_in", values=[28]),
        # OHO
        # DiscreteParameterRange("config/learners/1/learner/_type", values=["RTRLConfig"]),
        # # DiscreteParameterRange("config/learners/1/optimizer/_type", values=["SGDConfig"]),
        # # DiscreteParameterRange("config/learners/1/optimizer/momentum", values=[0.0]),
        # # DiscreteParameterRange("config/learners/1/optimizer/learning_rate", values=[1.0e-4, 1.0e-5]),
        # DiscreteParameterRange("config/learners/1/optimizer/learning_rate", values=[1.0e-3, 1.0e-4]),
        # DiscreteParameterRange("config/learners/0/optimizer/_type", values=["SGDConfig", "SGDNormalizedConfig"]),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[20]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/data/0/train_percent", values=[83.33]),
        DiscreteParameterRange("config/data/1/train_percent", values=[16.67]),
        DiscreteParameterRange("config/data/0/num_steps_in_timeseries", values=[28]),
        DiscreteParameterRange("config/data/1/num_steps_in_timeseries", values=[28]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[500]),
        DiscreteParameterRange("config/readout_uses_input_data", values=[False]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["01:30:00"]),
        DiscreteParameterRange("slurm/cpu", values=[2]),
        DiscreteParameterRange("slurm/memory", values=["12GB"]),
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
