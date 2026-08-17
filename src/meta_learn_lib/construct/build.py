from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.category.paralens import para_autodiff
from meta_learn_lib.construct.leaves import activation
from meta_learn_lib.construct.term import Activation, Arch, Bias, Linear, Seq

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
def build[S1, S2, HP1, HP2, P1, P2](
    t: Seq[S1, S2, HP1, HP2, P1, P2],
) -> Mealy[
    tuple[S1, S2],
    tuple[S1, S2],
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple[HP1, HP2],
    tuple[HP1, HP2],
    tuple[P1, P2],
    tuple[P1, P2],
]:
    return build(t.first) >> build(t.second)


@overload
def build[S, HP, P](t: Arch[S, HP, P]) -> Mealy[S, S, jax.Array, jax.Array, jax.Array, jax.Array, HP, HP, P, P]:
    raise NotImplementedError


@dispatch
def build[S, HP, P](t: Arch[S, HP, P]) -> Mealy[S, S, jax.Array, jax.Array, jax.Array, jax.Array, HP, HP, P, P]:
    raise NotImplementedError
