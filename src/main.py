import jax.numpy as jnp
import copy
from configs import *
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import *
# import jax

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    config = OHO_RNN256
    config = copy.replace(config, logger_config=[MatplotlibLoggerConfig(save_dir="/scratch/wlp9800/offline_logs")])

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

    app.runApp(config, loggers)


if __name__ == "__main__":
    main()
