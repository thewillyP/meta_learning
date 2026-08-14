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


def supervised[S, HP, P](
    model: Mealy[S, S, jax.Array, jax.Array, jax.Array, jax.Array, HP, HP, P, P],
    f: Callable[[jax.Array, jax.Array], jax.Array],
) -> Mealy[
    tuple[tuple[S, Unit], Unit],
    tuple[tuple[S, Unit], Unit],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    jax.Array,
    jax.Array,
    tuple[tuple[HP, Unit], Unit],
    tuple[tuple[HP, Unit], Unit],
    tuple[tuple[P, Unit], Unit],
    tuple[tuple[P, Unit], Unit],
]:

    def h(hp_p: tuple[Unit, Unit], yt: tuple[jax.Array, jax.Array]) -> jax.Array:
        y, t = yt
        return f(y, t)

    loss = to_mealy(para_autodiff(h))
    id_t = to_mealy(to_paralens(identity(Proxy[tuple[jax.Array, jax.Array]]())))
    return (model @ id_t) >> loss
