from meta_learn_lib.algorithms.combinators import batch_data, batch_pop, learner, scan
from meta_learn_lib.algorithms.level import level, validation
from meta_learn_lib.algorithms.model import leaf
from meta_learn_lib.algorithms.optimizers import sgd
from meta_learn_lib.category.lens import identity
from meta_learn_lib.category.lib_types import Proxy, Unit
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.category.paralens import para_autodiff, to_paralens
from meta_learn_lib.lib_types import ArrayTree
from meta_learn_lib.construct.leaves import activation, alpha_of, objective
from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchPop,
    Bias,
    Linear,
    Loss,
    Meta,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Sgd,
    Sup,
    Term,
    Validator,
)

from typing import overload
import equinox as eqx
import jax
import jax.numpy as jnp
from plum import dispatch


@overload
def validator[S, X, Y, HP, P](
    v: SameModel[S, X, Y, HP, P], m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]
) -> Mealy[S, S, tuple[tuple[Y, tuple[HP, P]], X], tuple[tuple[Y, tuple[HP, P]], X], Y, Y, Unit, Unit, Unit, Unit]:
    return validation(m)


@overload
def validator[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]
) -> Mealy[SV, SV, tuple[tuple[Y, tuple[HP, P]], XV], tuple[tuple[Y, tuple[HP, P]], XV], Y, Y, HPV, HPV, PV, PV]:
    raise NotImplementedError


@dispatch
def validator[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]
) -> Mealy[SV, SV, tuple[tuple[Y, tuple[HP, P]], XV], tuple[tuple[Y, tuple[HP, P]], XV], Y, Y, HPV, HPV, PV, PV]:
    raise NotImplementedError


@overload
def build(
    t: Linear,
) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, eqx.nn.Linear, eqx.nn.Linear]:
    return leaf(lambda p, s, x: (s, p(x)))


@overload
def build(t: Bias) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, jax.Array, jax.Array]:
    return leaf(lambda p, s, x: (s, x + p))


@overload
def build(t: Activation) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, Unit, Unit]:
    f = activation(t)
    return leaf(lambda p, s, x: (s, f(x)))


@overload
def build[HPA, PA](
    t: Rnn[HPA, PA],
) -> Mealy[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    HPA,
    HPA,
    tuple[PA, eqx.nn.Linear],
    tuple[PA, eqx.nn.Linear],
]:
    read = alpha_of(t.alpha)
    f = activation(t.act)

    def h(hp_p: tuple[HPA, tuple[PA, eqx.nn.Linear]], sx: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        hp, (pa, layer) = hp_p
        s, x = sx
        a = read(hp, pa)
        s1 = (1 - a) * s + a * f(layer(jnp.concatenate([s, x])))
        return (s1, s1)

    return Mealy(para_autodiff(h))


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
    return leaf(lambda p, s, y__t: (s, f(y__t[0], y__t[1])))


@overload
def build[P](t: Sgd[P]) -> Mealy[ArrayTree, ArrayTree, P, P, P, P, Unit, Unit, jax.Array, jax.Array]:
    return sgd()


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
def build[S, X, HP, P, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, H, HPO, HPV, SV, XV, PV],
) -> Mealy[
    tuple[tuple[tuple[S, tuple[ArrayTree, P]], Unit], SV],
    tuple[tuple[tuple[S, tuple[ArrayTree, P]], Unit], SV],
    tuple[X, XV],
    tuple[X, XV],
    jax.Array,
    jax.Array,
    tuple[tuple[HPO, Unit], HPV],
    tuple[tuple[HPO, Unit], HPV],
    tuple[tuple[tuple[HP, H], Unit], PV],
    tuple[tuple[tuple[HP, H], Unit], PV],
]:
    below = build(t.below)
    return level(learner(below, build(t.opt), jnp.ones_like), validator(t.val, below))


@overload
def build[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return scan(build(t.below))


@overload
def build[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return batch_data(build(t.below))


@overload
def build[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return batch_pop(build(t.below))


@overload
def build[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError


@dispatch
def build[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError
