from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.term import (
    Validate,
    Activation,
    Adam,
    BatchData,
    BatchPop,
    Bias,
    Const,
    Declares,
    Frozen,
    HyperStorage,
    Label,
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
    Shared,
    Split,
    Sup,
    Term,
    TrainedStorage,
    UORO,
    Uses,
    Validator,
)

from typing import overload
import equinox as eqx
from jaxtyping import PyTree
import jax
import jax.numpy as jnp
from plum import dispatch


@overload
def val_source[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], hp: Unit) -> dict[str, PyTree]:
    return {}


@overload
def val_source[S, X, Y, HP, P, SV, XV, HQ, Q](
    v: Validate[S, X, Y, HP, P, SV, XV, HQ, Q], hp: Unit
) -> dict[str, PyTree]:
    return {}


@overload
def val_source[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], hp: HPV
) -> dict[str, PyTree]:
    raise NotImplementedError


@dispatch
def val_source[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], hp: HPV
) -> dict[str, PyTree]:
    raise NotImplementedError


@overload
def val_target[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], hp: Unit, shared: dict[str, PyTree]) -> Unit:
    return hp


@overload
def val_target[S, X, Y, HP, P, SV, XV, HQ, Q](
    v: Validate[S, X, Y, HP, P, SV, XV, HQ, Q], hp: Unit, shared: dict[str, PyTree]
) -> Unit:
    return hp


@overload
def val_target[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], hp: HPV, shared: dict[str, PyTree]
) -> HPV:
    raise NotImplementedError


@dispatch
def val_target[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], hp: HPV, shared: dict[str, PyTree]
) -> HPV:
    raise NotImplementedError


def label_source(l: Label, v: PyTree) -> dict[str, PyTree]:
    match l:
        case Declares(name):
            return {name: v}
        case _:
            return {}


def label_target(l: Label, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    match l:
        case Uses(name):
            if name not in shared:
                raise KeyError(f"Uses({name!r}) has no source; declared names are {sorted(shared)}")
            return shared[name]
        case _:
            return v


@overload
def sources(t: HyperStorage, hp_p: tuple[jax.Array, Unit]) -> dict[str, PyTree]:
    hp, _ = hp_p
    return label_source(t.label, hp)


@overload
def sources(t: TrainedStorage, hp_p: tuple[Unit, jax.Array]) -> dict[str, PyTree]:
    _, p = hp_p
    return label_source(t.label, p)


@overload
def sources(t: Const, hp_p: tuple[Unit, Unit]) -> dict[str, PyTree]:
    return {}


@overload
def sources(t: Linear, hp_p: tuple[Unit, eqx.nn.Linear]) -> dict[str, PyTree]:
    _, p = hp_p
    return label_source(t.label, p)


@overload
def sources(t: Bias, hp_p: tuple[Unit, jax.Array]) -> dict[str, PyTree]:
    _, p = hp_p
    return label_source(t.label, p)


@overload
def sources(t: Activation, hp_p: tuple[Unit, Unit]) -> dict[str, PyTree]:
    return {}


@overload
def sources(t: Loss, hp_p: tuple[Unit, Unit]) -> dict[str, PyTree]:
    return {}


@overload
def sources[P](t: Frozen[P], hp_p: tuple[Unit, Unit]) -> dict[str, PyTree]:
    return {}


@overload
def sources[HPA, PA](t: Rnn[HPA, PA], hp_p: tuple[HPA, tuple[PA, eqx.nn.Linear]]) -> dict[str, PyTree]:
    hp, (pa, lin) = hp_p
    return sources(t.alpha, (hp, pa)) | label_source(t.label, lin)


@overload
def sources[HL, PL, HW, PW, HM, PM, P](
    t: Sgd[HL, PL, HW, PW, HM, PM, P], hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]
) -> dict[str, PyTree]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    return sources(t.lr, (hl, pl)) | sources(t.wd, (hw, pw)) | sources(t.momentum, (hm, pm))


@overload
def sources[HL, PL, HW, PW, HM, PM, P](
    t: SgdNormalized[HL, PL, HW, PW, HM, PM, P], hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]
) -> dict[str, PyTree]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    return sources(t.lr, (hl, pl)) | sources(t.wd, (hw, pw)) | sources(t.momentum, (hm, pm))


@overload
def sources[HL, PL, HW, PW, HM, PM, P](
    t: Adam[HL, PL, HW, PW, HM, PM, P], hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]
) -> dict[str, PyTree]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    return sources(t.lr, (hl, pl)) | sources(t.wd, (hw, pw)) | sources(t.momentum, (hm, pm))


@overload
def sources[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](
    t: Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2], hp_p: tuple[tuple[HPO1, HPO2], tuple[H1, H2]]
) -> dict[str, PyTree]:
    (hp1, hp2), (h1, h2) = hp_p
    return sources(t.first, (hp1, h1)) | sources(t.second, (hp2, h2))


@overload
def sources[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2], hp_p: tuple[tuple[HP1, HP2], tuple[P1, P2]]
) -> dict[str, PyTree]:
    (hp1, hp2), (p1, p2) = hp_p
    return sources(t.first, (hp1, p1)) | sources(t.second, (hp2, p2))


@overload
def sources[S, X, HP, P](
    t: Sup[S, X, HP, P], hp_p: tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]]
) -> dict[str, PyTree]:
    ((hp_a, _), _), ((p_a, _), _) = hp_p
    return sources(t.arch, (hp_a, p_a))


@overload
def sources[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    return sources(t.below, hp_p)


@overload
def sources[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    return sources(t.below, hp_p)


@overload
def sources[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    return sources(t.below, hp_p)


@overload
def sources[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P]) -> dict[str, PyTree]:
    (hp_b, _), p = hp_p
    return sources(t.below, (hp_b, p))


@overload
def sources[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P]) -> dict[str, PyTree]:
    (hp_b, _), p = hp_p
    return sources(t.below, (hp_b, p))


@overload
def sources[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD], hp_p: tuple[tuple[HP, HD], P]) -> dict[str, PyTree]:
    (hp_b, d), p = hp_p
    return sources(t.below, (hp_b, p)) | sources(t.decay, (d, Unit()))


@overload
def sources[S, X, Y, HP, HP2, P, P2](
    t: Reparametrized[S, X, Y, HP, HP2, P, P2], hp_p: tuple[HP2, P2]
) -> dict[str, PyTree]:
    return label_source(t.label, hp_p)


@overload
def sources[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
    hp_p: tuple[tuple[tuple[HPO, Unit], HPV], tuple[tuple[tuple[HP, H], Unit], PV]],
) -> dict[str, PyTree]:
    ((hp_o, _), hpv), (((_, h), _), _) = hp_p
    return sources(t.opt, (hp_o, h)) | val_source(t.val, hpv)


@overload
def sources[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    return {}


@overload
def sources[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    raise NotImplementedError


@dispatch
def sources[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> dict[str, PyTree]:
    raise NotImplementedError


@overload
def targets(t: HyperStorage, hp_p: tuple[jax.Array, Unit], shared: dict[str, PyTree]) -> tuple[jax.Array, Unit]:
    hp, u = hp_p
    return (label_target(t.label, hp, shared), u)


@overload
def targets(t: TrainedStorage, hp_p: tuple[Unit, jax.Array], shared: dict[str, PyTree]) -> tuple[Unit, jax.Array]:
    u, p = hp_p
    return (u, label_target(t.label, p, shared))


@overload
def targets(t: Const, hp_p: tuple[Unit, Unit], shared: dict[str, PyTree]) -> tuple[Unit, Unit]:
    return hp_p


@overload
def targets(t: Linear, hp_p: tuple[Unit, eqx.nn.Linear], shared: dict[str, PyTree]) -> tuple[Unit, eqx.nn.Linear]:
    u, p = hp_p
    return (u, label_target(t.label, p, shared))


@overload
def targets(t: Bias, hp_p: tuple[Unit, jax.Array], shared: dict[str, PyTree]) -> tuple[Unit, jax.Array]:
    u, p = hp_p
    return (u, label_target(t.label, p, shared))


@overload
def targets(t: Activation, hp_p: tuple[Unit, Unit], shared: dict[str, PyTree]) -> tuple[Unit, Unit]:
    return hp_p


@overload
def targets(t: Loss, hp_p: tuple[Unit, Unit], shared: dict[str, PyTree]) -> tuple[Unit, Unit]:
    return hp_p


@overload
def targets[P](t: Frozen[P], hp_p: tuple[Unit, Unit], shared: dict[str, PyTree]) -> tuple[Unit, Unit]:
    return hp_p


@overload
def targets[HPA, PA](
    t: Rnn[HPA, PA], hp_p: tuple[HPA, tuple[PA, eqx.nn.Linear]], shared: dict[str, PyTree]
) -> tuple[HPA, tuple[PA, eqx.nn.Linear]]:
    hp, (pa, lin) = hp_p
    a, b = targets(t.alpha, (hp, pa), shared)
    return (a, (b, label_target(t.label, lin, shared)))


@overload
def targets[HL, PL, HW, PW, HM, PM, P](
    t: Sgd[HL, PL, HW, PW, HM, PM, P],
    hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]],
    shared: dict[str, PyTree],
) -> tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    a1, b1 = targets(t.lr, (hl, pl), shared)
    a2, b2 = targets(t.wd, (hw, pw), shared)
    a3, b3 = targets(t.momentum, (hm, pm), shared)
    return (((a1, a2), a3), ((b1, b2), b3))


@overload
def targets[HL, PL, HW, PW, HM, PM, P](
    t: SgdNormalized[HL, PL, HW, PW, HM, PM, P],
    hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]],
    shared: dict[str, PyTree],
) -> tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    a1, b1 = targets(t.lr, (hl, pl), shared)
    a2, b2 = targets(t.wd, (hw, pw), shared)
    a3, b3 = targets(t.momentum, (hm, pm), shared)
    return (((a1, a2), a3), ((b1, b2), b3))


@overload
def targets[HL, PL, HW, PW, HM, PM, P](
    t: Adam[HL, PL, HW, PW, HM, PM, P],
    hp_p: tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]],
    shared: dict[str, PyTree],
) -> tuple[tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM]]:
    ((hl, hw), hm), ((pl, pw), pm) = hp_p
    a1, b1 = targets(t.lr, (hl, pl), shared)
    a2, b2 = targets(t.wd, (hw, pw), shared)
    a3, b3 = targets(t.momentum, (hm, pm), shared)
    return (((a1, a2), a3), ((b1, b2), b3))


@overload
def targets[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](
    t: Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2],
    hp_p: tuple[tuple[HPO1, HPO2], tuple[H1, H2]],
    shared: dict[str, PyTree],
) -> tuple[tuple[HPO1, HPO2], tuple[H1, H2]]:
    (hp1, hp2), (h1, h2) = hp_p
    a1, b1 = targets(t.first, (hp1, h1), shared)
    a2, b2 = targets(t.second, (hp2, h2), shared)
    return ((a1, a2), (b1, b2))


@overload
def targets[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2],
    hp_p: tuple[tuple[HP1, HP2], tuple[P1, P2]],
    shared: dict[str, PyTree],
) -> tuple[tuple[HP1, HP2], tuple[P1, P2]]:
    (hp1, hp2), (p1, p2) = hp_p
    a1, b1 = targets(t.first, (hp1, p1), shared)
    a2, b2 = targets(t.second, (hp2, p2), shared)
    return ((a1, a2), (b1, b2))


@overload
def targets[S, X, HP, P](
    t: Sup[S, X, HP, P],
    hp_p: tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]],
    shared: dict[str, PyTree],
) -> tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]]:
    ((hp_a, u1), u2), ((p_a, u3), u4) = hp_p
    a, b = targets(t.arch, (hp_a, p_a), shared)
    return (((a, u1), u2), ((b, u3), u4))


@overload
def targets[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]) -> tuple[HP, P]:
    return targets(t.below, hp_p, shared)


@overload
def targets[S, X, Y, HP, P](
    t: BatchData[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]
) -> tuple[HP, P]:
    return targets(t.below, hp_p, shared)


@overload
def targets[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]) -> tuple[HP, P]:
    return targets(t.below, hp_p, shared)


@overload
def targets[S, X, Y, HP, P](
    t: RTRL[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P], shared: dict[str, PyTree]
) -> tuple[tuple[HP, Unit], P]:
    (hp_b, u), p = hp_p
    a, b = targets(t.below, (hp_b, p), shared)
    return ((a, u), b)


@overload
def targets[S, X, Y, HP, P](
    t: UORO[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P], shared: dict[str, PyTree]
) -> tuple[tuple[HP, Unit], P]:
    (hp_b, u), p = hp_p
    a, b = targets(t.below, (hp_b, p), shared)
    return ((a, u), b)


@overload
def targets[S, X, Y, HP, P, HD](
    t: RFLO[S, X, Y, HP, P, HD], hp_p: tuple[tuple[HP, HD], P], shared: dict[str, PyTree]
) -> tuple[tuple[HP, HD], P]:
    (hp_b, d), p = hp_p
    a, b = targets(t.below, (hp_b, p), shared)
    dd, _ = targets(t.decay, (d, Unit()), shared)
    return ((a, dd), b)


@overload
def targets[S, X, Y, HP, HP2, P, P2](
    t: Reparametrized[S, X, Y, HP, HP2, P, P2], hp_p: tuple[HP2, P2], shared: dict[str, PyTree]
) -> tuple[HP2, P2]:
    return label_target(t.label, hp_p, shared)


@overload
def targets[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
    hp_p: tuple[tuple[tuple[HPO, Unit], HPV], tuple[tuple[tuple[HP, H], Unit], PV]],
    shared: dict[str, PyTree],
) -> tuple[tuple[tuple[HPO, Unit], HPV], tuple[tuple[tuple[HP, H], Unit], PV]]:
    ((hp_o, u1), hpv), (((hp, h), u2), pv) = hp_p
    a, b = targets(t.opt, (hp_o, h), shared)
    return (((a, u1), val_target(t.val, hpv, shared)), (((hp, b), u2), pv))


@overload
def targets[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]) -> tuple[HP, P]:
    return hp_p


@overload
def targets[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]) -> tuple[HP, P]:
    raise NotImplementedError


@dispatch
def targets[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], shared: dict[str, PyTree]) -> tuple[HP, P]:
    raise NotImplementedError


def route[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    return targets(t, hp_p, sources(t, hp_p))


def unroute[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    def shrink(v: PyTree) -> PyTree:
        return jax.tree.map(lambda a: jnp.zeros((0,), dtype=a.dtype), v)

    return targets(t, hp_p, {name: shrink(v) for name, v in sources(t, hp_p).items()})
