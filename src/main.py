import random
import string
import random
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import HDF5Logger, MatplotlibLogger, MultiLogger, PrintLogger
# import jax

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    config = GodConfig(
        clearml_run=True,
        data_root_dir="/tmp",
        log_dir="/scratch/offline_logs",
        dataset=CIFAR10Config(96),
        # dataset=DelayAddOnlineConfig(t1=10, t2=12, tau_task=1, n=100_000, nTest=1_000),
        num_base_epochs=200,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(global_seed=4352, data_seed=1, parameter_seed=1, test_seed=12345),
        loss_fn="cross_entropy_with_integer_labels",
        # loss_fn="cross_entropy",
        transition_function={
            # 0: GRULayer(
            #     n=256,
            #     # activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=256,
            #     use_bias=True,
            # ),
            0: NNLayer(
                n=256,
                activation_fn="tanh",
                use_bias=True,
            ),
            # 1: NNLayer(
            #     n=128,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 2: NNLayer(
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
                #     learning_rate=HyperparameterConfig(value=0.029240177382128668, learnable=True),
                #     weight_decay=HyperparameterConfig(value=0.0, learnable=False),
                #     momentum=0.0,
                # ),
                # optimizer=SGDConfig(
                #     learning_rate=HyperparameterConfig(value=0.1, learnable=True),
                #     weight_decay=HyperparameterConfig(value=0.0, learnable=False),
                #     momentum=0.0,
                # ),
                optimizer=SGDClipConfig(
                    # learning_rate=HyperparameterConfig(value=0.1, learnable=True),
                    learning_rate=HyperparameterConfig(
                        value=0.029240177382128668, learnable=True, hyperparameter_parametrization="softplus"
                    ),
                    weight_decay=HyperparameterConfig(
                        value=0.001, learnable=True, hyperparameter_parametrization="softplus"
                    ),
                    momentum=0.0,
                    clip_threshold=5.0,
                    clip_sharpness=100.0,
                ),
                # optimizer=AdamConfig(
                #     learning_rate=HyperparameterConfig(value=0.001, learnable=True),
                #     weight_decay=HyperparameterConfig(value=0.0, learnable=False),
                # ),
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                # learner=IdentityConfig(),
                learner=RTRLFiniteHvpConfig(epsilon=1e-3),
                optimizer=AdamConfig(
                    learning_rate=HyperparameterConfig(
                        value=0.02, learnable=False, hyperparameter_parametrization="identity"
                    ),
                    weight_decay=HyperparameterConfig(
                        value=0.0, learnable=False, hyperparameter_parametrization="identity"
                    ),
                ),
                # optimizer=SGDConfig(
                #     learning_rate=HyperparameterConfig(value=0.001, learnable=True),
                #     weight_decay=HyperparameterConfig(value=0.0, learnable=False),
                #     momentum=0.0,
                # ),
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=40,
            ),
        },
        data={
            0: DataConfig(
                train_percent=80.000,
                num_examples_in_minibatch=1000,
                num_steps_in_timeseries=32,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=20.000,
                num_examples_in_minibatch=1000,
                num_steps_in_timeseries=32,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=False,
        logger_config=(MatplotlibLoggerConfig("./figures"), PrintLoggerConfig()),
        treat_inference_state_as_online=False,
    )

    loggers = []
    for log_config in config.logger_config:
        match log_config:
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
