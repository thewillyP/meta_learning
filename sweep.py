from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(project_name="oho", task_name="Seed+LR Sweep: MLP,SGD-ADAM,BPTT-ID")
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="138684a2d9864a4da05470fb61195306",  # Use the actual task ID
    hyper_parameters=[
        # Seed configurations as complete seed objects
        DiscreteParameterRange(
            "config/seed",
            values=[
                {"data_seed": 74274, "parameter_seed": 25223, "test_seed": 12345},
                {"data_seed": 41030, "parameter_seed": 17164, "test_seed": 12345},
                {"data_seed": 21471, "parameter_seed": 76771, "test_seed": 12345},
                {"data_seed": 43250, "parameter_seed": 39069, "test_seed": 12345},
                {"data_seed": 72537, "parameter_seed": 66096, "test_seed": 12345},
                {"data_seed": 53199, "parameter_seed": 27512, "test_seed": 12345},
                {"data_seed": 52890, "parameter_seed": 45433, "test_seed": 12345},
                {"data_seed": 51110, "parameter_seed": 60780, "test_seed": 12345},
                {"data_seed": 37103, "parameter_seed": 28963, "test_seed": 12345},
                {"data_seed": 65874, "parameter_seed": 20730, "test_seed": 12345},
            ],
        ),
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
                7.00000000e-01,
            ],
        ),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[10]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/test_batch_size", values=[100]),
        DiscreteParameterRange("config/data/0/train_percent", values=[83.33]),
        DiscreteParameterRange("config/data/1/train_percent", values=[16.67]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[500]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["01:00:00"]),
        DiscreteParameterRange("slurm/cpu", values=[2]),
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
