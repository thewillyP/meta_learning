from meta_learn_lib.category.lib_types import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy, to_mealy

type Params = tuple[Params, Params] | jax.Array | Unit
type Layer = Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, Params, Params]


def leaf(f: Callable[[jax.Array, jax.Array], jax.Array]) -> Layer:
    def h(rp: tuple[Unit, Params], x: jax.Array) -> jax.Array:
        _, p = rp
        return g(p, x)

    def g(p: Params, x: jax.Array) -> jax.Array:
        match p:
            case jax.Array():
                return f(p, x)
            case Unit():
                return f(jnp.zeros(0), x)
            case (l, r):
                return g(r, g(l, x))

    return to_mealy(para_autodiff(h))
