from typing import Union
import clearml
import random
import string
from cattrs import unstructure, Converter
import random
import time
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import ClearMLLogger, HDF5Logger, MatplotlibLogger, MultiLogger, PrintLogger
from meta_learn_lib.util import setup_flattened_union
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
        data_root_dir="/scratch/datasets",
        log_dir="/scratch/offline_logs",
        # dataset=CIFAR10Config(96),
        # dataset=FashionMnistConfig(784),
        dataset=FashionMnistConfig(28),
        num_base_epochs=64,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(global_seed=11111, data_seed=1, parameter_seed=1, test_seed=12345),
        loss_fn="cross_entropy_with_integer_labels",
        transition_function={
            # 0: IdentityLayer(activation_fn="identity"),
            # 0: GRULayer(
            #     n=128,
            #     # activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=128,
            #     use_bias=True,
            #     use_in_readout=True,
            # ),
            0: NNLayer(
                n=16,
                activation_fn="tanh",
                use_bias=True,
                use_in_readout=True,
                layer_norm=None,
            ),
            # 1: NNLayer(
            #     n=16,
            #     activation_fn="tanh",
            #     use_bias=True,
            #     use_in_readout=False,
            #     layer_norm=None,
            # ),
            # 2: NNLayer(
            #     n=16,
            #     activation_fn="tanh",
            #     use_bias=True,
            #     use_in_readout=True,
            #     layer_norm=None,
            # ),
            # 1: NNLayer(
            #     n=128,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
        },
        readout_function=FeedForwardConfig(
            ffw_layers={
                # 0: NNLayer(
                #     n=16,
                #     activation_fn="tanh",
                #     use_bias=True,
                #     use_in_readout=False,
                #     layer_norm=None,
                # ),
                # 1: NNLayer(
                #     n=16,
                #     activation_fn="tanh",
                #     use_bias=True,
                #     use_in_readout=False,
                #     layer_norm=None,
                # ),
                0: NNLayer(
                    n=10,
                    activation_fn="identity",
                    use_bias=True,
                    use_in_readout=False,
                    layer_norm=None,
                ),
            }
        ),
        learners={
            0: LearnConfig(  # normal feedforward backprop
                learner=BPTTConfig(),
                # optimizer=AdamConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=1e-3,
                #         learnable=True,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=1e-5,
                #         learnable=True,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                # ),
                optimizer=SGDConfig(
                    learning_rate=HyperparameterConfig(
                        value=0.01,
                        learnable=True,
                        hyperparameter_parametrization=HyperparameterConfig.squared(1),
                    ),
                    weight_decay=HyperparameterConfig(
                        value=1e-5,
                        learnable=True,
                        hyperparameter_parametrization=HyperparameterConfig.squared(1),
                    ),
                    momentum=0.0,
                ),
                # optimizer=SGDClipConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=0.2,
                #         learnable=True,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=0.0,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     momentum=0.0,
                #     clip_threshold=1.0,
                #     clip_sharpness=100.0,
                # ),
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                # learner=IdentityConfig(),
                # learner=RFLOConfig(0.4),
                learner=RTRLHessianDecompConfig(1e-5),
                # learner=RTRLConfig(),
                optimizer=AdamConfig(
                    learning_rate=HyperparameterConfig(
                        value=1e-4,
                        learnable=False,
                        hyperparameter_parametrization=HyperparameterConfig.identity(),
                    ),
                    weight_decay=HyperparameterConfig(
                        value=0.0,
                        learnable=False,
                        hyperparameter_parametrization=HyperparameterConfig.identity(),
                    ),
                    # momentum=0.0,
                ),
                # optimizer=ExponentiatedGradientConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=1e-3,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=0.0,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     momentum=0.9,
                # ),
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=375,
            ),
        },
        data={
            0: DataConfig(
                train_percent=80.0,
                num_examples_in_minibatch=64,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=20.0,
                num_examples_in_minibatch=64,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=True,
        logger_config=(ClearMLLoggerConfig(),),
        treat_inference_state_as_online=False,
    )

    converter = Converter()
    setup_flattened_union(
        converter,
        Union[
            HyperparameterConfig.identity,
            HyperparameterConfig.softplus,
            HyperparameterConfig.relu,
            HyperparameterConfig.softrelu,
            HyperparameterConfig.silu_positive,
            HyperparameterConfig.squared,
        ],
    )
    setup_flattened_union(converter, Union[NNLayer, GRULayer, LSTMLayer, IdentityLayer])
    setup_flattened_union(
        converter,
        Union[
            RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig, RTRLHessianDecompConfig, RTRLFiniteHvpConfig
        ],
    )
    setup_flattened_union(
        converter, Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig, ExponentiatedGradientConfig]
    )
    setup_flattened_union(converter, Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig, CIFAR10Config])
    setup_flattened_union(
        converter,
        Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig],
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

    for logger in loggers:
        match logger:
            case MatplotlibLogger():
                logger.generate_figures()


if __name__ == "__main__":
    main()
