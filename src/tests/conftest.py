import os
import jax

jax.config.update("jax_enable_x64", True)

cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax"))
if cache_dir.lower() in ("", "none", "off", "disable"):
    jax.config.update("jax_compilation_cache_dir", None)
else:
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
