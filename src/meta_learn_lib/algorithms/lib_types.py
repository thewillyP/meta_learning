from meta_learn_lib.lib_types import PRNG
import jax

type UORO_AUX = tuple[jax.Array, jax.Array, PRNG]
