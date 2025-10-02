from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch
from clearml import Task

# Create optimizer task
opt_task = Task.init(
    project_name="oho",
    task_name="OHO Seed+ILR Sweep: Batch-1000,Epochs-150,CIFAR10,RNN-128,SGD-Adam,BPTT-RTRLFiniteHvpConfig",
)
# task_name="Fixed Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD-Adam,BPTT-ID"
# task_name="OHO Seed+ILR Sweep: Batch-2,Epochs-20,FashionMNIST,MLP,SGD/SGDN-Adam,BPTT-RTRL"
opt_task.execute_remotely(queue_name="services", clone=False, exit_process=True)

# Configure optimizer
optimizer = HyperParameterOptimizer(
    base_task_id="d7a4bcbdad274c0f80984730eaf1910d",  # Use the actual task ID
    hyper_parameters=[
        # Seed configurations as complete seed objects
        DiscreteParameterRange(
            "config/seed/global_seed",
            values=[760, 202, 747, 995, 972, 579, 274, 283, 201, 480, 14, 530, 842, 774, 32, 471, 102, 104, 479, 789],
        ),
        DiscreteParameterRange("config/seed/test_seed", values=[12345]),
        # DiscreteParameterRange(
        #     "config/learners/0/optimizer/learning_rate",
        #     values=[
        #         0.0001,
        #         0.00020286934558567696,
        #         0.00041155971378360783,
        #         0.0008349284980470904,
        #         0.0016938139800964523,
        #         0.003436229336860381,
        #         0.006971055968511703,
        #         0.01414213562373095,
        #         0.028690057991701875,
        #         0.05820333289591681,
        #         0.11807672055499935,
        #         0.2395414702789557,
        #         0.4859562131612263,
        #         0.9858561894731162,
        #         2.0,
        #     ],
        # ),
        # dataset
        DiscreteParameterRange("config/dataset/_type", values=["CIFAR10Config"]),
        DiscreteParameterRange("config/dataset/n_in", values=[96]),
        # OHO
        DiscreteParameterRange("config/learners/1/learner/_type", values=["RTRLFiniteHvpConfig"]),
        DiscreteParameterRange("config/learners/1/learner/epsilon", values=[2.0e-4]),
        DiscreteParameterRange(
            "config/learners/1/optimizer/learning_rate/value",
            values=[
                0.0001,
                0.00025119,
                0.00063096,
                0.00158489,
                0.00398107,
                0.01,
            ],
        ),
        DiscreteParameterRange(
            "config/learners/0/optimizer/learning_rate/value",
            values=[
                1.00000000e-04,
                1.35720881e-03,
                1.84201575e-02,
                2.50000000e-01,
            ],
        ),
        # Fixed parameters
        DiscreteParameterRange("config/clearml_run", values=[True]),
        DiscreteParameterRange("config/num_base_epochs", values=[150]),
        DiscreteParameterRange("config/data/0/num_examples_in_minibatch", values=[1000]),
        DiscreteParameterRange("config/data/1/num_examples_in_minibatch", values=[1000]),
        DiscreteParameterRange("config/data/0/train_percent", values=[80.00]),
        DiscreteParameterRange("config/data/1/train_percent", values=[20.00]),
        DiscreteParameterRange("config/data/0/num_steps_in_timeseries", values=[32]),
        DiscreteParameterRange("config/data/1/num_steps_in_timeseries", values=[32]),
        DiscreteParameterRange("config/learners/0/num_virtual_minibatches_per_turn", values=[1]),
        DiscreteParameterRange("config/learners/1/num_virtual_minibatches_per_turn", values=[40]),
        DiscreteParameterRange("config/readout_uses_input_data", values=[False]),
        DiscreteParameterRange("config/treat_inference_state_as_online", values=[False]),
        DiscreteParameterRange("config/logger_config/_type", values=[({"_type": "HDF5LoggerConfig"},)]),
        # Slurm configurations
        DiscreteParameterRange("slurm/time", values=["02:30:00"]),
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
