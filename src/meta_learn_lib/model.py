from meta_learn_lib.category.lib_types import *
from meta_learn_lib.category.paralens import *

type Params = tuple[Params, Params] | jax.Array | Unit
type Layer = ParaLens[jax.Array, jax.Array, jax.Array, jax.Array, Params, Params]


def leaf(f: Callable[[jax.Array, jax.Array], jax.Array]) -> Layer:
    def g(p: Params, x: jax.Array) -> jax.Array:
        match p:
            case jax.Array():
                return f(p, x)
            case Unit():
                return f(jnp.zeros(0), x)
            case (l, r):
                return g(r, g(l, x))

    return para_autodiff(g)
