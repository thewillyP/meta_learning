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
    time.sleep(_jitter_rng.uniform(1, 60))

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
        clearml_run=False,
        data_root_dir="/scratch/datasets",
        log_dir="/scratch/offline_logs",
        # dataset=CIFAR10Config(96),
        # dataset=FashionMnistConfig(784),
        # dataset=FashionMnistConfig(28),
        # dataset=DelayAddOnlineConfig(15, 17, 1, 100_000, 5000),
        dataset=MnistConfig(28, False),
        num_base_epochs=600,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(global_seed=274, data_seed=1, parameter_seed=1, test_seed=12345),
        loss_fn="cross_entropy_with_integer_labels",
        # loss_fn="cross_entropy",
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
            #     use_random_init=False,
            # ),
            0: NNLayer(
                n=32,
                activation_fn="tanh",
                use_bias=True,
                use_in_readout=True,
                layer_norm=None,
                use_random_init=False,
            ),
            # 1: NNLayer(
            #     n=128,
            #     activation_fn="identity",
            #     use_bias=True,
            #     use_in_readout=True,
            #     layer_norm=None,
            #     use_random_init=False,
            # ),
            # 1: NNLayer(
            #     n=64,
            #     activation_fn="tanh",
            #     use_bias=True,
            #     use_in_readout=False,
            #     layer_norm=LayerNorm(1e-4, False, False),
            #     use_random_init=True,
            # ),
            # 2: NNLayer(
            #     n=64,
            #     activation_fn="tanh",
            #     use_bias=True,
            #     use_in_readout=False,
            #     layer_norm=LayerNorm(1e-4, False, False),
            #     use_random_init=True,
            # ),
            # 3: NNLayer(
            #     n=64,
            #     activation_fn="tanh",
            #     use_bias=True,
            #     use_in_readout=True,
            #     layer_norm=LayerNorm(1e-4, False, False),
            #     use_random_init=True,
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
                #     n=128,
                #     activation_fn="tanh",
                #     use_bias=True,
                #     use_in_readout=False,
                #     layer_norm=None,
                #     use_random_init=False,
                # ),
                # 1: NNLayer(
                #     n=128,
                #     activation_fn="tanh",
                #     use_bias=True,
                #     use_in_readout=False,
                #     layer_norm=None,
                #     use_random_init=False,
                # ),
                # 2: NNLayer(
                #     n=128,
                #     activation_fn="tanh",
                #     use_bias=True,
                #     use_in_readout=False,
                #     layer_norm=None,
                #     use_random_init=False,
                # ),
                0: NNLayer(
                    n=10,
                    activation_fn="identity",
                    use_bias=True,
                    use_in_readout=False,
                    layer_norm=None,
                    use_random_init=False,
                ),
            }
        ),
        learners={
            0: LearnConfig(
                learner=BPTTConfig(),
                # optimizer=RecurrenceConfig(
                #     recurrent_optimizer=SGDConfig(
                #         learning_rate=HyperparameterConfig(
                #             value=0.01,
                #             learnable=False,
                #             hyperparameter_parametrization=HyperparameterConfig.squared(1),
                #         ),
                #         weight_decay=HyperparameterConfig(
                #             value=0.0,
                #             learnable=False,
                #             hyperparameter_parametrization=HyperparameterConfig.squared(1),
                #         ),
                #         momentum=0.85,
                #         add_clip=Clip(
                #             threshold=1.0,
                #             sharpness=100.0,
                #         ),
                #     ),
                #     # recurrent_optimizer=AdamConfig(
                #     #     learning_rate=HyperparameterConfig(
                #     #         value=0.001,
                #     #         learnable=False,
                #     #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     #     ),
                #     #     weight_decay=HyperparameterConfig(
                #     #         value=0.0,
                #     #         learnable=False,
                #     #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     #     ),
                #     #     add_clip=None,
                #     # ),
                #     readout_optimizer=SGDConfig(
                #         learning_rate=HyperparameterConfig(
                #             value=0.001,
                #             learnable=True,
                #             hyperparameter_parametrization=HyperparameterConfig.squared(1),
                #         ),
                #         weight_decay=HyperparameterConfig(
                #             value=0.0,
                #             learnable=False,
                #             hyperparameter_parametrization=HyperparameterConfig.squared(1),
                #         ),
                #         momentum=0.0,
                #         add_clip=None,
                #     ),
                # ),
                # optimizer=AdamConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=1e-3,
                #         learnable=True,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=0.0,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     add_clip=None,
                # ),
                optimizer=SGDConfig(
                    learning_rate=HyperparameterConfig(
                        value=1e-3,
                        learnable=True,
                        hyperparameter_parametrization=HyperparameterConfig.squared(1),
                    ),
                    weight_decay=HyperparameterConfig(
                        value=1e-5,
                        learnable=True,
                        hyperparameter_parametrization=HyperparameterConfig.squared(1),
                    ),
                    momentum=0.0,
                    add_clip=Clip(
                        threshold=1.0,
                        sharpness=1000.0,
                    ),
                ),
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                # learner=IdentityConfig(),
                # learner=RFLOConfig(0.4),
                # learner=RTRLFiniteHvpConfig(1e-3, start_at_step=0),
                learner=RTRLConfig(start_at_step=0, momentum1=0.9, momentum2=0.9),
                # learner=UOROConfig(1.0),
                # optimizer=SGDConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=0.01,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=0.0,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     momentum=0.0,
                #     add_clip=None,
                # ),
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
                    add_clip=None,
                ),
                # optimizer=ExponentiatedGradientConfig(
                #     learning_rate=HyperparameterConfig(
                #         value=1e-2,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     weight_decay=HyperparameterConfig(
                #         value=0.0,
                #         learnable=False,
                #         hyperparameter_parametrization=HyperparameterConfig.identity(),
                #     ),
                #     add_clip=None,
                # ),
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
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=16.667,
                num_examples_in_minibatch=500,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=False,
        logger_config=(ClearMLLoggerConfig(),),
        # logger_config=(PrintLoggerConfig(), MatplotlibLoggerConfig("../")),
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
        converter,
        Union[SGDConfig, SGDNormalizedConfig, AdamConfig, ExponentiatedGradientConfig, ExponentiatedGradientAdamConfig],
    )
    setup_flattened_union(
        converter,
        Union[
            SGDConfig,
            SGDNormalizedConfig,
            AdamConfig,
            ExponentiatedGradientConfig,
            RecurrenceConfig,
            ExponentiatedGradientAdamConfig,
        ],
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
