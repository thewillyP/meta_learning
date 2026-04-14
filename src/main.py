import os
import argparse
import copy
import jax
import jax.numpy as jnp
from configs import *
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.config_converter import make_converter
from meta_learn_lib.logger import *
from meta_learn_lib.checkpoint import NullCheckpointManager

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main(cache_dir: str):
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    config = VAE_BETA_OHO
    config = copy.replace(
        config,
        logger_config=copy.replace(
            config.logger_config,
            matplotlib=MatplotlibLoggerConfig(save_dir=config.log_dir, enabled=True),
        ),
    )

    # Round-trip through cattrs so that fields annotated as jax.Array get
    # promoted from Python literals to runtime jnp.asarray placeholders.
    converter = make_converter()
    config = converter.structure(converter.unstructure(config), GodConfig)

    loggers = []
    lc = config.logger_config
    if lc.console.enabled:
        loggers.append(ConsoleLogger())
    if lc.matplotlib.enabled:
        loggers.append(MatplotlibLogger(lc.matplotlib.save_dir))

    app.runApp(config, loggers, NullCheckpointManager())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, help="JAX persistent compilation cache directory")
    args = parser.parse_args()
    main(cache_dir=args.cache_dir)
