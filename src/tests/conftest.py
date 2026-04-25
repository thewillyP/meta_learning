import os
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)