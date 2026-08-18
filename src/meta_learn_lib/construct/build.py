from meta_learn_lib.algorithms.combinators import batch_data, batch_pop, learner, reparametrize, scan
from meta_learn_lib.algorithms.learning import rflo, rtrl, uoro
from meta_learn_lib.algorithms.level import level, validation
from meta_learn_lib.algorithms.model import leaf
from meta_learn_lib.algorithms.optimizers import adam, additive, frozen, optimizer, sgd, sgd_normalized
from meta_learn_lib.category.lens import autodiff, identity
from meta_learn_lib.category.lib_types import Proxy, Unit
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.category.paralens import para_autodiff, to_paralens
from meta_learn_lib.lib_types import ArrayTree, JACOBIAN, PRNG
from meta_learn_lib.construct.reparam import cook, reparametrizer
from meta_learn_lib.construct.leaves import activation, objective, reader, sampler
from meta_learn_lib.construct.term import (
    Activation,
    Adam,
    BatchData,
    BatchPop,
    Bias,
    Frozen,
    Linear,
    Loss,
    Meta,
    RFLO,
    RTRL,
    Reparametrized,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Sgd,
    SgdNormalized,
    Split,
    Sup,
    Term,
    UORO,
    Validator,
)

from typing import overload
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
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
    read = reader(t.alpha)
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
def build[HL, HW, HM, P](
    t: Sgd[HL, HW, HM, P],
) -> Mealy[
    ArrayTree,
    ArrayTree,
    P,
    P,
    P,
    P,
    Unit,
    Unit,
    tuple[tuple[HL, HW], HM],
    tuple[tuple[HL, HW], HM],
]:
    r_lr, r_wd, r_m = reader(t.lr), reader(t.wd), reader(t.momentum)

    def make(h: tuple[tuple[HL, HW], HM]) -> optax.GradientTransformation:
        (lr, wd), m = h
        return sgd(r_lr(lr, Unit()), r_wd(wd, Unit()), r_m(m, Unit()))

    return optimizer(make, additive)


@overload
def build[HL, HW, HM, P](
    t: SgdNormalized[HL, HW, HM, P],
) -> Mealy[
    ArrayTree,
    ArrayTree,
    P,
    P,
    P,
    P,
    Unit,
    Unit,
    tuple[tuple[HL, HW], HM],
    tuple[tuple[HL, HW], HM],
]:
    r_lr, r_wd, r_m = reader(t.lr), reader(t.wd), reader(t.momentum)

    def make(h: tuple[tuple[HL, HW], HM]) -> optax.GradientTransformation:
        (lr, wd), m = h
        return sgd_normalized(r_lr(lr, Unit()), r_wd(wd, Unit()), r_m(m, Unit()))

    return optimizer(make, additive)


@overload
def build[HL, HW, HM, P](
    t: Adam[HL, HW, HM, P],
) -> Mealy[
    ArrayTree,
    ArrayTree,
    P,
    P,
    P,
    P,
    Unit,
    Unit,
    tuple[tuple[HL, HW], HM],
    tuple[tuple[HL, HW], HM],
]:
    r_lr, r_wd, r_m = reader(t.lr), reader(t.wd), reader(t.momentum)

    def make(h: tuple[tuple[HL, HW], HM]) -> optax.GradientTransformation:
        (lr, wd), m = h
        return adam(r_lr(lr, Unit()), r_wd(wd, Unit()), r_m(m, Unit()), t.b2, t.eps, t.eps_root)

    return optimizer(make, additive)


@overload
def build[P](t: Frozen[P]) -> Mealy[ArrayTree, ArrayTree, P, P, P, P, Unit, Unit, Unit, Unit]:
    return optimizer(lambda h: frozen(), additive)


@overload
def build[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](
    t: Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2],
) -> Mealy[
    tuple[SO1, SO2],
    tuple[SO1, SO2],
    tuple[P1, P2],
    tuple[P1, P2],
    tuple[P1, P2],
    tuple[P1, P2],
    tuple[HPO1, HPO2],
    tuple[HPO1, HPO2],
    tuple[H1, H2],
    tuple[H1, H2],
]:
    return build(t.first) @ build(t.second)


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
def build[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
) -> Mealy[
    tuple[tuple[tuple[S, tuple[SO, P]], Unit], SV],
    tuple[tuple[tuple[S, tuple[SO, P]], Unit], SV],
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
def build[S, X, Y, HP, P](
    t: RTRL[S, X, Y, HP, P],
) -> Mealy[tuple[S, JACOBIAN], tuple[S, JACOBIAN], X, X, Y, Y, tuple[HP, Unit], tuple[HP, Unit], P, P]:
    return rtrl(build(t.below))


@overload
def build[S, X, Y, HP, P, HD](
    t: RFLO[S, X, Y, HP, P, HD],
) -> Mealy[tuple[S, JACOBIAN], tuple[S, JACOBIAN], X, X, Y, Y, tuple[HP, HD], tuple[HP, HD], P, P]:
    read = reader(t.decay)
    return rflo(build(t.below), lambda hd: read(hd, Unit()))


@overload
def build[S, X, Y, HP, P](
    t: UORO[S, X, Y, HP, P],
) -> Mealy[
    tuple[S, tuple[jax.Array, jax.Array, PRNG]],
    tuple[S, tuple[jax.Array, jax.Array, PRNG]],
    X,
    X,
    Y,
    Y,
    tuple[HP, Unit],
    tuple[HP, Unit],
    P,
    P,
]:
    return uoro(build(t.below), sampler(t.noise))


@overload
def build[S, X, Y, HP, HP2, P, P2](
    t: Reparametrized[S, X, Y, HP, HP2, P, P2],
) -> Mealy[S, S, X, X, Y, Y, HP2, HP2, P2, P2]:
    expand = reparametrizer(t.r)
    below = cook(t.below)
    return reparametrize(build(t.below), autodiff(lambda hp_p: below(expand(hp_p))))


@overload
def build[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError


@dispatch
def build[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    raise NotImplementedError
