from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchParams,
    BatchPop,
    Bias,
    Linear,
    Loss,
    Meta,
    Over,
    RFLO,
    RTRL,
    Reparametrized,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Shared,
    Sup,
    Term,
    UORO,
    Validate,
    Validator,
)

from dataclasses import dataclass
from typing import overload
from plum import dispatch


@dataclass(frozen=True)
class Window:
    n: int


@dataclass(frozen=True)
class Batch:
    n: int
    over: Over


type Axis = Window | Batch


@dataclass(frozen=True)
class Leaf:
    axes: tuple[Axis, ...]


type Plan = Leaf | Pair


@dataclass(frozen=True)
class Pair:
    axes: tuple[Axis, ...]
    left: Plan
    right: Plan


def prefix(axis: Axis, plan: Plan) -> Plan:
    match plan:
        case Leaf(axes):
            return Leaf((axis, *axes))
        case Pair(axes, left, right):
            return Pair((axis, *axes), left, right)


@overload
def val_data[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], below: Term[S, X, Y, HP, P]) -> Leaf | Pair:
    return data(below)


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q](
    v: Validate[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q], below: Term[S, X, Y, HP, P]
) -> Leaf | Pair:
    return data(v.term)


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Leaf | Pair:
    raise NotImplementedError


@dispatch
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Leaf | Pair:
    raise NotImplementedError


@overload
def data(t: Linear) -> Leaf | Pair:
    return Leaf(())


@overload
def data(t: Bias) -> Leaf | Pair:
    return Leaf(())


@overload
def data(t: Activation) -> Leaf | Pair:
    return Leaf(())


@overload
def data(t: Loss) -> Leaf | Pair:
    return Leaf(())


@overload
def data[HPA, PA](t: Rnn[HPA, PA]) -> Leaf | Pair:
    return Leaf(())


@overload
def data[S, X, HP, P](t: Sup[S, X, HP, P]) -> Leaf | Pair:
    return Leaf(())


@overload
def data[S1, S2, X, Y, Z, HP1, HP2, P1, P2](t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2]) -> Leaf | Pair:
    return data(t.first)


@overload
def data[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV]) -> Leaf | Pair:
    return Pair((), data(t.below), val_data(t.val, t.below))


@overload
def data[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Window(t.n), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Batch(t.n, t.over), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchParams[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Batch(t.n, t.over), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Batch(t.n, t.over), data(t.below))


@overload
def data[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P]) -> Leaf | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD]) -> Leaf | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P]) -> Leaf | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, HP2, P, P2](t: Reparametrized[S, X, Y, HP, HP2, P, P2]) -> Leaf | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P]) -> Leaf | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Leaf | Pair:
    raise NotImplementedError


@dispatch
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Leaf | Pair:
    raise NotImplementedError
