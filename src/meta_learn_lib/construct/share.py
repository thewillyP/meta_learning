from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchPop,
    Bias,
    Const,
    Declares,
    Descent,
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
    Shared,
    Split,
    Sup,
    Term,
    TrainedStorage,
    UORO,
    Unlabeled,
    Uses,
    Validator,
)

from dataclasses import dataclass
from typing import overload
from jaxtyping import PyTree
import jax
import jax.numpy as jnp
from plum import dispatch


@dataclass(frozen=True)
class Slot:
    label: Label


@dataclass(frozen=True)
class Pair:
    first: "AnnTree"
    second: "AnnTree"


type AnnTree = Slot | Pair


@overload
def label_shrink(l: Anon, v: PyTree) -> PyTree:
    return v


@overload
def label_shrink(l: Declares, v: PyTree) -> PyTree:
    return v


@overload
def label_shrink(l: Uses, v: PyTree) -> PyTree:
    return jax.tree.map(lambda a: jnp.zeros((0,), dtype=a.dtype), v)


@overload
def label_shrink(l: Label, v: PyTree) -> PyTree:
    raise NotImplementedError


@dispatch
def label_shrink(l: Label, v: PyTree) -> PyTree:
    raise NotImplementedError


@overload
def label_names(l: Anon) -> frozenset[str]:
    return frozenset()


@overload
def label_names(l: Declares) -> frozenset[str]:
    return frozenset({l.name})


@overload
def label_names(l: Uses) -> frozenset[str]:
    return frozenset()


@overload
def label_names(l: Label) -> frozenset[str]:
    raise NotImplementedError


@dispatch
def label_names(l: Label) -> frozenset[str]:
    raise NotImplementedError


@overload
def label_take(l: Anon, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    return v


@overload
def label_take(l: Declares, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    return v


@overload
def label_take(l: Uses, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    return shared.get(l.name, v)


@overload
def label_take(l: Label, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    raise NotImplementedError


@dispatch
def label_take(l: Label, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    raise NotImplementedError


@overload
def label_trim(l: Anon, v: PyTree, names: frozenset[str]) -> PyTree:
    return v


@overload
def label_trim(l: Declares, v: PyTree, names: frozenset[str]) -> PyTree:
    return v


@overload
def label_trim(l: Uses, v: PyTree, names: frozenset[str]) -> PyTree:
    if l.name in names:
        return jax.tree.map(lambda a: jnp.zeros((0,), dtype=a.dtype), v)
    return v


@overload
def label_trim(l: Label, v: PyTree, names: frozenset[str]) -> PyTree:
    raise NotImplementedError


@dispatch
def label_trim(l: Label, v: PyTree, names: frozenset[str]) -> PyTree:
    raise NotImplementedError


@overload
def val_annotations[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P]) -> Slot | Pair:
    return Slot(Unlabeled())


@overload
def val_annotations[S, X, Y, HP, P, SV, XV, HPV, PV](v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV]) -> Slot | Pair:
    raise NotImplementedError


@dispatch
def val_annotations[S, X, Y, HP, P, SV, XV, HPV, PV](v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV]) -> Slot | Pair:
    raise NotImplementedError


def unpack_tree(a: AnnTree) -> tuple[AnnTree, AnnTree]:
    match a:
        case Slot():
            return (Slot(Unlabeled()), Slot(Unlabeled()))
        case Pair(first, second):
            return (first, second)


@overload
def annotations(t: HyperStorage) -> Slot | Pair:
    return Pair(Slot(t.label), Slot(Unlabeled()))


@overload
def annotations(t: TrainedStorage) -> Slot | Pair:
    return Pair(Slot(Unlabeled()), Slot(t.label))


@overload
def annotations(t: Const) -> Slot | Pair:
    return Slot(Unlabeled())


@overload
def annotations(t: Linear) -> Slot | Pair:
    return Pair(Slot(Unlabeled()), Slot(t.label))


@overload
def annotations(t: Bias) -> Slot | Pair:
    return Pair(Slot(Unlabeled()), Slot(t.label))


@overload
def annotations(t: Activation) -> Slot | Pair:
    return Slot(Unlabeled())


@overload
def annotations(t: Loss) -> Slot | Pair:
    return Slot(Unlabeled())


@overload
def annotations[HPA, PA](t: Rnn[HPA, PA]) -> Slot | Pair:
    a_hp, a_p = unpack_tree(annotations(t.alpha))
    return Pair(a_hp, Pair(a_p, Slot(t.label)))


@overload
def annotations[HL, HW, HM, P](t: Descent[HL, HW, HM, P]) -> Slot | Pair:
    _, a_lr = unpack_tree(annotations(t.lr))
    _, a_wd = unpack_tree(annotations(t.wd))
    _, a_m = unpack_tree(annotations(t.momentum))
    return Pair(Slot(Unlabeled()), Pair(Pair(a_lr, a_wd), a_m))


@overload
def annotations[P](t: Frozen[P]) -> Slot | Pair:
    return Slot(Unlabeled())


@overload
def annotations[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](t: Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2]) -> Slot | Pair:
    hp1, h1 = unpack_tree(annotations(t.first))
    hp2, h2 = unpack_tree(annotations(t.second))
    return Pair(Pair(hp1, hp2), Pair(h1, h2))


@overload
def annotations[S1, S2, X, Y, Z, HP1, HP2, P1, P2](t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2]) -> Slot | Pair:
    hp1, p1 = unpack_tree(annotations(t.first))
    hp2, p2 = unpack_tree(annotations(t.second))
    return Pair(Pair(hp1, hp2), Pair(p1, p2))


@overload
def annotations[S, X, HP, P](t: Sup[S, X, HP, P]) -> Slot | Pair:
    hp_a, p_a = unpack_tree(annotations(t.arch))
    return Pair(
        Pair(Pair(hp_a, Slot(Unlabeled())), Slot(Unlabeled())), Pair(Pair(p_a, Slot(Unlabeled())), Slot(Unlabeled()))
    )


@overload
def annotations[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
) -> Slot | Pair:
    hp_b, _ = unpack_tree(annotations(t.below))
    hp_o, h = unpack_tree(annotations(t.opt))
    hpv, pv = unpack_tree(val_annotations(t.val))
    return Pair(Pair(Pair(hp_o, Slot(Unlabeled())), hpv), Pair(Pair(Pair(hp_b, h), Slot(Unlabeled())), pv))


@overload
def annotations[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> Slot | Pair:
    return annotations(t.below)


@overload
def annotations[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P]) -> Slot | Pair:
    return annotations(t.below)


@overload
def annotations[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Slot | Pair:
    return annotations(t.below)


@overload
def annotations[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P]) -> Slot | Pair:
    hp, p = unpack_tree(annotations(t.below))
    return Pair(Pair(hp, Slot(Unlabeled())), p)


@overload
def annotations[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD]) -> Slot | Pair:
    hp, p = unpack_tree(annotations(t.below))
    d, _ = unpack_tree(annotations(t.decay))
    return Pair(Pair(hp, d), p)


@overload
def annotations[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P]) -> Slot | Pair:
    hp, p = unpack_tree(annotations(t.below))
    return Pair(Pair(hp, Slot(Unlabeled())), p)


@overload
def annotations[S, X, Y, HP, HP2, P, P2](t: Reparametrized[S, X, Y, HP, HP2, P, P2]) -> Slot | Pair:
    match t.label:
        case Unlabeled():
            return annotations(t.below)
        case _:
            return Pair(Slot(Unlabeled()), Slot(t.label))


@overload
def annotations[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P]) -> Slot | Pair:
    return annotations(t.below)


@overload
def annotations[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Slot | Pair:
    raise NotImplementedError


@dispatch
def annotations[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Slot | Pair:
    raise NotImplementedError


def collect(a: AnnTree, v: PyTree) -> dict[str, PyTree]:
    match a:
        case Slot(Declares(name)):
            return {name: v}
        case Pair(first, second):
            left, right = v
            return collect(first, left) | collect(second, right)
        case _:
            return {}


def substitute(a: AnnTree, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    match a:
        case Slot(Uses(name)):
            if name not in shared:
                raise KeyError(f"Uses({name!r}) has no source; declared names are {sorted(shared)}")
            return shared[name]
        case Pair(first, second):
            left, right = v
            return (substitute(first, left, shared), substitute(second, right, shared))
        case _:
            return v


def shrink(a: AnnTree, v: PyTree) -> PyTree:
    match a:
        case Slot(label):
            return label_shrink(label, v)
        case Pair(first, second):
            left, right = v
            return (shrink(first, left), shrink(second, right))


def declared(a: AnnTree) -> frozenset[str]:
    match a:
        case Slot(label):
            return label_names(label)
        case Pair(first, second):
            return declared(first) | declared(second)


def fill(a: AnnTree, v: PyTree, shared: dict[str, PyTree]) -> PyTree:
    match a:
        case Slot(label):
            return label_take(label, v, shared)
        case Pair(first, second):
            left, right = v
            return (fill(first, left, shared), fill(second, right, shared))


def trim(a: AnnTree, v: PyTree, names: frozenset[str]) -> PyTree:
    match a:
        case Slot(label):
            return label_trim(label, v, names)
        case Pair(first, second):
            left, right = v
            return (trim(first, left, names), trim(second, right, names))


def route[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    a = annotations(t)
    return substitute(a, hp_p, collect(a, hp_p))


def unroute[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    return shrink(annotations(t), hp_p)


def route_local[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    a = annotations(t)
    return fill(a, hp_p, collect(a, hp_p))


def unroute_local[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P]) -> tuple[HP, P]:
    a = annotations(t)
    return trim(a, hp_p, declared(a))
