import jax.numpy as jnp
import copy
from configs import *
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import *
from meta_learn_lib.checkpoint import NullCheckpointManager
# import jax

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    config = VAE_BETA_OHO
    config = copy.replace(
        config,
        logger_config=copy.replace(
            config.logger_config,
            matplotlib=MatplotlibLoggerConfig(save_dir=config.log_dir, enabled=True),
        ),
    )

    loggers = []
    lc = config.logger_config
    if lc.console.enabled:
        loggers.append(ConsoleLogger())
    if lc.matplotlib.enabled:
        loggers.append(MatplotlibLogger(lc.matplotlib.save_dir))

    app.runApp(config, loggers, NullCheckpointManager())


if __name__ == "__main__":
    main()
