import random
import string
import random
from lib import app
from lib.config import *
from lib.logger import HDF5Logger, MatplotlibLogger, PrintLogger
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
        logger_config=MatplotlibLoggerConfig("./figures"),
        treat_inference_state_as_online=True,
    )

    match config.logger_config:
        case PrintLoggerConfig():
            logger = PrintLogger()
        case MatplotlibLoggerConfig(save_dir):
            logger = MatplotlibLogger(save_dir)
        case _:
            raise ValueError("Invalid logger configuration.")

    app.runApp(config, logger)

    match logger:
        case MatplotlibLogger():
            logger.generate_figures()


if __name__ == "__main__":
    main()
