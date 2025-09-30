from typing import Union
import clearml
import random
import string
from cattrs import unstructure, Converter
from cattrs.strategies import configure_tagged_union
import random
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import ClearMLLogger, HDF5Logger, MatplotlibLogger, MultiLogger, PrintLogger
# import jax

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    _jitter_rng = random.Random()
    # time.sleep(_jitter_rng.uniform(1, 60))

    # names don't matter, can change in UI
    # clearml.Task.set_offline(True)
    task: clearml.Task = clearml.Task.init(
        project_name="temp",
        task_name="".join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        task_type=clearml.TaskTypes.training,
    )

    # Values dont matter because can change in UI
    slurm_params = SlurmParams(
        memory="8GB",
        time="01:00:00",
        cpu=2,
        gpu=0,
        log_dir="/vast/wlp9800/logs",
        singularity_overlay="",
        singularity_binds="/scratch/wlp9800/clearml:/scratch",
        container_source=SifContainerSource(sif_path="/scratch/wlp9800/images/devenv-cpu.sif"),
        use_singularity=True,
        setup_commands="module load python/intel/3.8.6",
        skip_python_env_install=True,
    )
    task.connect(unstructure(slurm_params), name="slurm")

    config = GodConfig(
        clearml_run=True,
        data_root_dir="/tmp",
        log_dir="/scratch/offline_logs",
        dataset=MnistConfig(7),
        # dataset=DelayAddOnlineConfig(3, 4, 1, 20, 20),
        num_base_epochs=100,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(global_seed=14, data_seed=1, parameter_seed=1, test_seed=12345),
        loss_fn="cross_entropy_with_integer_labels",
        # loss_fn="mse",
        transition_function={
            # 0: GRULayer(
            #     n=128,
            #     # activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=128,
            #     use_bias=True,
            # ),
            0: NNLayer(
                n=32,
                activation_fn="tanh",
                use_bias=True,
            ),
            # 1: NNLayer(
            #     n=128,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=64,
            #     use_bias=True,
            # ),
            # 1: NNLayer(
            #     n=64,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
        },
        # readout_function=FeedForwardConfig(
        #     ffw_layers={
        #         0: NNLayer(n=10, activation_fn="identity", use_bias=True),
        #     }
        # ),
        readout_function=FeedForwardConfig(
            ffw_layers={
                # 0: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                # 1: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                # 2: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                0: NNLayer(n=10, activation_fn="identity", use_bias=True),
            }
        ),
        learners={
            0: LearnConfig(  # normal feedforward backprop
                learner=BPTTConfig(),
                # optimizer=SGDNormalizedConfig(
                #     learning_rate=0.029240177382128668,
                #     momentum=0.0,
                # ),
                # optimizer=SGDConfig(
                #     learning_rate=0.1,
                #     momentum=0.0,
                # ),
                optimizer=SGDClipConfig(
                    learning_rate=0.1,
                    # learning_rate=0.029240177382128668,
                    momentum=0.0,
                    clip_threshold=1.0,
                    clip_sharpness=100.0,
                ),
                # optimizer=AdamConfig(
                #     learning_rate=0.001,
                # ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                learner=IdentityConfig(),
                # learner=RTRLFiniteHvpConfig(epsilon=1e-4),
                optimizer=AdamConfig(
                    learning_rate=0.001,
                ),
                # optimizer=SGDConfig(
                #     learning_rate=1e-3,
                #     momentum=0.0,
                # ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=100,
            ),
        },
        data={
            0: DataConfig(
                train_percent=83.333,
                num_examples_in_minibatch=500,
                num_steps_in_timeseries=112,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=16.667,
                num_examples_in_minibatch=500,
                num_steps_in_timeseries=112,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=False,
        logger_config=(ClearMLLoggerConfig(),),
        treat_inference_state_as_online=True,
    )

    converter = Converter()
    configure_tagged_union(Union[NNLayer, GRULayer, LSTMLayer], converter)
    configure_tagged_union(
        Union[
            RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig, RTRLHessianDecompConfig, RTRLFiniteHvpConfig
        ],
        converter,
    )
    configure_tagged_union(Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig], converter)
    configure_tagged_union(Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig, CIFAR10Config], converter)
    configure_tagged_union(
        Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig], converter
    )

    # Need two connects in order to change config in UI as well as make it HPO-able since HPO can't add new hyperparameter fields
    _config = task.connect_configuration(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)

    _config = task.connect(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)

    loggers = []
    for log_config in config.logger_config:
        match log_config:
            case HDF5LoggerConfig():
                logger = HDF5Logger(config.log_dir, task.task_id)
            case ClearMLLoggerConfig():
                logger = ClearMLLogger(task)
            case PrintLoggerConfig():
                logger = PrintLogger()
            case MatplotlibLoggerConfig(save_dir):
                logger = MatplotlibLogger(save_dir)
            case _:
                raise ValueError("Invalid logger configuration.")
        loggers.append(logger)

    logger = MultiLogger(loggers)

    app.runApp(config, logger)


if __name__ == "__main__":
    main()
