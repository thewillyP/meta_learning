from meta_learn_lib.category.lens import identity
from meta_learn_lib.category.lib_types import Proxy, Unit
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.category.paralens import para_autodiff, to_paralens
from meta_learn_lib.construct.leaves import activation, objective
from meta_learn_lib.construct.term import Activation, Arch, Bias, Linear, Loss, Seq, Sup

from typing import overload
import equinox as eqx
import jax
from plum import dispatch


@overload
def build(
    t: Linear,
) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, eqx.nn.Linear, eqx.nn.Linear]:
    def h(hp_p: tuple[Unit, eqx.nn.Linear], x: jax.Array) -> jax.Array:
        _, layer = hp_p
        return layer(x)

    return to_mealy(para_autodiff(h))


@overload
def build(
    t: Bias,
) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, jax.Array, jax.Array]:
    def h(hp_p: tuple[Unit, jax.Array], x: jax.Array) -> jax.Array:
        _, b = hp_p
        return x + b

    return to_mealy(para_autodiff(h))


@overload
def build(t: Activation) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, Unit, Unit]:
    f = activation(t)

    def h(hp_p: tuple[Unit, Unit], x: jax.Array) -> jax.Array:
        return f(x)

    return to_mealy(para_autodiff(h))


@overload
def build(
    t: Loss,
) -> Mealy[
    Unit,
    Unit,
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    jax.Array,
    jax.Array,
    Unit,
    Unit,
    Unit,
    Unit,
]:
    f = objective(t)

    def h(hp_p: tuple[Unit, Unit], yt: tuple[jax.Array, jax.Array]) -> jax.Array:
        y, target = yt
        return f(y, target)

    return to_mealy(para_autodiff(h))


@overload
def build[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2],
) -> Mealy[
    tuple[S1, S2],
    tuple[S1, S2],
    X,
    X,
    Z,
    Z,
    tuple[HP1, HP2],
    tuple[HP1, HP2],
    tuple[P1, P2],
    tuple[P1, P2],
]:
    return build(t.first) >> build(t.second)


@overload
def build[S, X, HP, P](
    t: Sup[S, X, HP, P],
) -> Mealy[
    tuple[tuple[S, Unit], Unit],
    tuple[tuple[S, Unit], Unit],
    tuple[X, jax.Array],
    tuple[X, jax.Array],
    jax.Array,
    jax.Array,
    tuple[tuple[HP, Unit], Unit],
    tuple[tuple[HP, Unit], Unit],
    tuple[tuple[P, Unit], Unit],
    tuple[tuple[P, Unit], Unit],
]:
    id_t = to_mealy(to_paralens(identity(Proxy[tuple[jax.Array, jax.Array]]())))
    return (build(t.arch) @ id_t) >> build(t.loss)


@overload
def build[S, X, Y, HP, P](t: Arch[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError


@dispatch
def build[S, X, Y, HP, P](t: Arch[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError
