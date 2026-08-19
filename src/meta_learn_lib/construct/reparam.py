from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.leaves import reparametrization
from meta_learn_lib.construct.term import (
    Activation,
    Adam,
    BatchData,
    BatchPop,
    Bias,
    Const,
    Frozen,
    Hyper,
    Knob,
    Linear,
    Loss,
    LowRank,
    Meta,
    RFLO,
    RMap,
    RSplit,
    RTRL,
    Reparametrization,
    Reparametrized,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Sgd,
    SgdNormalized,
    Shared,
    Split,
    Sup,
    Term,
    Trained,
    UORO,
    Unconstrained,
    Validator,
)

from typing import Callable, overload
import equinox as eqx
import jax
import jax.numpy as jnp
from plum import dispatch


@overload
def reparametrizer[P](r: RMap[P]) -> Callable[[P], P]:
    forward, _ = reparametrization(r.by)
    return lambda x: jax.tree.map(forward, x)


@overload
def reparametrizer[A2, A, B2, B](r: RSplit[A2, A, B2, B]) -> Callable[[tuple[A2, B2]], tuple[A, B]]:
    first = reparametrizer(r.first)
    second = reparametrizer(r.second)

    def apply(ab: tuple[A2, B2]) -> tuple[A, B]:
        a, b = ab
        return (first(a), second(b))

    return apply


@overload
def reparametrizer(r: LowRank) -> Callable[[tuple[jax.Array, jax.Array]], eqx.nn.Linear]:
    def expand(ab: tuple[jax.Array, jax.Array]) -> eqx.nn.Linear:
        a, b = ab
        n_out, _ = a.shape
        _, n_in = b.shape
        shell = eqx.nn.Linear(n_in, n_out, use_bias=False, key=jax.random.key(0))
        return eqx.tree_at(lambda l: l.weight, shell, a @ b)

    return expand


@overload
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
    raise NotImplementedError


@dispatch
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
    raise NotImplementedError


@overload
def seed[P](r: RMap[P], p: P) -> P:
    _, invert = reparametrization(r.by)
    return jax.tree.map(invert, p)


@overload
def seed[A2, A, B2, B](r: RSplit[A2, A, B2, B], p: tuple[A, B]) -> tuple[A2, B2]:
    a, b = p
    return (seed(r.first, a), seed(r.second, b))


@overload
def seed(r: LowRank, p: eqx.nn.Linear) -> tuple[jax.Array, jax.Array]:
    u, sv, vt = jnp.linalg.svd(p.weight, full_matrices=False)
    root = jnp.sqrt(sv[: r.rank])
    return (u[:, : r.rank] * root, vt[: r.rank, :] * root[:, None])


@overload
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError


@dispatch
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError


@overload
def knob_reparam(k: Hyper) -> tuple[RMap[jax.Array], RMap[Unit]]:
    return (RMap(k.reparam), RMap(Unconstrained()))


@overload
def knob_reparam(k: Trained) -> tuple[RMap[Unit], RMap[jax.Array]]:
    return (RMap(Unconstrained()), RMap(k.reparam))


@overload
def knob_reparam(k: Const) -> tuple[RMap[Unit], RMap[Unit]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def knob_reparam[HP, P](k: Knob[HP, P]) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    raise NotImplementedError


@dispatch
def knob_reparam[HP, P](k: Knob[HP, P]) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    raise NotImplementedError


@overload
def val_reparam[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P]) -> tuple[RMap[Unit], RMap[Unit]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def val_reparam[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV],
) -> tuple[Reparametrization[HPV, HPV], Reparametrization[PV, PV]]:
    raise NotImplementedError


@dispatch
def val_reparam[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV],
) -> tuple[Reparametrization[HPV, HPV], Reparametrization[PV, PV]]:
    raise NotImplementedError


@overload
def reparam_tree(t: Linear) -> tuple[RMap[Unit], RMap[eqx.nn.Linear]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree(t: Bias) -> tuple[RMap[Unit], RMap[jax.Array]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree(t: Activation) -> tuple[RMap[Unit], RMap[Unit]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree(t: Loss) -> tuple[RMap[Unit], RMap[Unit]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree[HPA, PA](
    t: Rnn[HPA, PA],
) -> tuple[Reparametrization[HPA, HPA], RSplit[PA, PA, eqx.nn.Linear, eqx.nn.Linear]]:
    r_hp, r_p = knob_reparam(t.alpha)
    return (r_hp, RSplit(r_p, RMap(Unconstrained())))


@overload
def reparam_tree[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2],
) -> tuple[RSplit[HP1, HP1, HP2, HP2], RSplit[P1, P1, P2, P2]]:
    hp1, p1 = reparam_tree(t.first)
    hp2, p2 = reparam_tree(t.second)
    return (RSplit(hp1, hp2), RSplit(p1, p2))


@overload
def reparam_tree[S, X, HP, P](
    t: Sup[S, X, HP, P],
) -> tuple[
    RSplit[tuple[HP, Unit], tuple[HP, Unit], Unit, Unit],
    RSplit[tuple[P, Unit], tuple[P, Unit], Unit, Unit],
]:
    hp_a, p_a = reparam_tree(t.arch)
    return (
        RSplit(RSplit(hp_a, RMap(Unconstrained())), RMap(Unconstrained())),
        RSplit(RSplit(p_a, RMap(Unconstrained())), RMap(Unconstrained())),
    )


@overload
def reparam_tree[HL, HW, HM, P](
    t: Sgd[HL, HW, HM, P],
) -> tuple[RMap[Unit], RSplit[tuple[HL, HW], tuple[HL, HW], HM, HM]]:
    r_lr, _ = knob_reparam(t.lr)
    r_wd, _ = knob_reparam(t.wd)
    r_m, _ = knob_reparam(t.momentum)
    return (RMap(Unconstrained()), RSplit(RSplit(r_lr, r_wd), r_m))


@overload
def reparam_tree[HL, HW, HM, P](
    t: SgdNormalized[HL, HW, HM, P],
) -> tuple[RMap[Unit], RSplit[tuple[HL, HW], tuple[HL, HW], HM, HM]]:
    r_lr, _ = knob_reparam(t.lr)
    r_wd, _ = knob_reparam(t.wd)
    r_m, _ = knob_reparam(t.momentum)
    return (RMap(Unconstrained()), RSplit(RSplit(r_lr, r_wd), r_m))


@overload
def reparam_tree[HL, HW, HM, P](
    t: Adam[HL, HW, HM, P],
) -> tuple[RMap[Unit], RSplit[tuple[HL, HW], tuple[HL, HW], HM, HM]]:
    r_lr, _ = knob_reparam(t.lr)
    r_wd, _ = knob_reparam(t.wd)
    r_m, _ = knob_reparam(t.momentum)
    return (RMap(Unconstrained()), RSplit(RSplit(r_lr, r_wd), r_m))


@overload
def reparam_tree[P](t: Frozen[P]) -> tuple[RMap[Unit], RMap[Unit]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](
    t: Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2],
) -> tuple[RSplit[HPO1, HPO1, HPO2, HPO2], RSplit[H1, H1, H2, H2]]:
    hp1, h1 = reparam_tree(t.first)
    hp2, h2 = reparam_tree(t.second)
    return (RSplit(hp1, hp2), RSplit(h1, h2))


@overload
def reparam_tree[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    return reparam_tree(t.below)


@overload
def reparam_tree[S, X, Y, HP, P](
    t: BatchData[S, X, Y, HP, P],
) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    return reparam_tree(t.below)


@overload
def reparam_tree[S, X, Y, HP, P](
    t: BatchPop[S, X, Y, HP, P],
) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    return reparam_tree(t.below)


@overload
def reparam_tree[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P]) -> tuple[RSplit[HP, HP, Unit, Unit], Reparametrization[P, P]]:
    hp, p = reparam_tree(t.below)
    return (RSplit(hp, RMap(Unconstrained())), p)


@overload
def reparam_tree[S, X, Y, HP, P, HD](
    t: RFLO[S, X, Y, HP, P, HD],
) -> tuple[RSplit[HP, HP, HD, HD], Reparametrization[P, P]]:
    hp, p = reparam_tree(t.below)
    r_d, _ = knob_reparam(t.decay)
    return (RSplit(hp, r_d), p)


@overload
def reparam_tree[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P]) -> tuple[RSplit[HP, HP, Unit, Unit], Reparametrization[P, P]]:
    hp, p = reparam_tree(t.below)
    return (RSplit(hp, RMap(Unconstrained())), p)


@overload
def reparam_tree[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
) -> tuple[
    RSplit[tuple[HPO, Unit], tuple[HPO, Unit], HPV, HPV],
    RSplit[tuple[tuple[HP, H], Unit], tuple[tuple[HP, H], Unit], PV, PV],
]:
    hp_b, _ = reparam_tree(t.below)
    hp_o, h = reparam_tree(t.opt)
    hpv, pv = val_reparam(t.val)
    return (
        RSplit(RSplit(hp_o, RMap(Unconstrained())), hpv),
        RSplit(RSplit(RSplit(hp_b, h), RMap(Unconstrained())), pv),
    )


@overload
def reparam_tree[S, X, Y, HP, HP2, P, P2](t: Reparametrized[S, X, Y, HP, HP2, P, P2]) -> tuple[RMap[HP2], RMap[P2]]:
    return (RMap(Unconstrained()), RMap(Unconstrained()))


@overload
def reparam_tree[S, X, Y, HP, P](
    t: Shared[S, X, Y, HP, P],
) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    return reparam_tree(t.below)


@overload
def reparam_tree[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    raise NotImplementedError


@dispatch
def reparam_tree[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> tuple[Reparametrization[HP, HP], Reparametrization[P, P]]:
    raise NotImplementedError


def cook[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Callable[[tuple[HP, P]], tuple[HP, P]]:
    r_hp, r_p = reparam_tree(t)
    forward_hp = reparametrizer(r_hp)
    forward_p = reparametrizer(r_p)

    def go(hp_p: tuple[HP, P]) -> tuple[HP, P]:
        hp, p = hp_p
        return (forward_hp(hp), forward_p(p))

    return go


def store[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    r_hp, r_p = reparam_tree(t)
    hp, p = hp_p
    return (seed(r_hp, hp), seed(r_p, p))
