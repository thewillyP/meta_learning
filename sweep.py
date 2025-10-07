from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho",
    task_name="Seed+ILR Sweep: Batch-100,Epochs-50,FashionMNIST,RNN-128,SGD-Adam,BPTT-ID",
)
# task_name="Fixed Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD-Adam,BPTT-ID"
# task_name="OHO Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD/SGDN-Adam,BPTT-RTRL"
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="a7e169cbbcf3464391ab31e118a5e815",  # Use the actual task ID
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
                1.00000000e-04,
                4.64158883e-04,
                2.15443469e-03,
                1.00000000e-02,
                1.32046925e-02,
                1.74363903e-02,
                2.30242172e-02,
                3.04027708e-02,
                4.01459239e-02,
                5.30114579e-02,
                7.00000000e-02,
                8.46927715e-02,
                1.02469508e-01,
                1.23977523e-01,
                1.50000000e-01,
            ],
        ),
        # dataset
        # DiscreteParameterRange("config/dataset/_type", values=["FashionMnistConfig"]),
        # DiscreteParameterRange("config/dataset/n_in", values=[28]),
        # OHO
        # DiscreteParameterRange("config/learners/1/learner/_type", values=["RTRLFiniteHvpConfig"]),
        # DiscreteParameterRange("config/learners/1/learner/epsilon", values=[1.0e-3]),
        # DiscreteParameterRange(
        #     "config/learners/1/optimizer/learning_rate/value",
        #     values=[0.0001],
        # ),
        # DiscreteParameterRange(
        #     "config/learners/0/optimizer/learning_rate/value",
        #     values=[
        #         0.1,
        #         0.01,
        #         0.001,
        #         0.0001,
        #     ],
        # ),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[50]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[100]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[100]),
        # DiscreteParameterRange("config/data/0/train_percent", values=[80.00]),
        # DiscreteParameterRange("config/data/1/train_percent", values=[20.00]),
        # DiscreteParameterRange("config/data/0/num_steps_in_timeseries", values=[28]),
        # DiscreteParameterRange("config/data/1/num_steps_in_timeseries", values=[28]),
        DiscreteParameterRange("config/learners/0/num_virtual_minibatches_per_turn", values=[1]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[500]),
        # DiscreteParameterRange("config/readout_uses_input_data", values=[False]),
        DiscreteParameterRange("config/treat_inference_state_as_online", values=[False]),
        DiscreteParameterRange("config/logger_config", values=[({"_type": "HDF5LoggerConfig"},)]),
        DiscreteParameterRange("config/data_root_dir", values=["/scratch/datasets"]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["01:30:00"]),
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
