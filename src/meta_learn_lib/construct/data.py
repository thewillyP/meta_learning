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

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload
from plum import dispatch


@dataclass(frozen=True)
class Steps: ...


@dataclass(frozen=True)
class Window:
    n: int
    below: "Leaf"


@dataclass(frozen=True)
class Batch:
    n: int
    over: Over
    below: "Leaf"


type Leaf = Steps | Window | Batch


@dataclass(frozen=True)
class Pair:
    left: "Plan"
    right: "Plan"


type Plan = Leaf | Pair


def under(plan: Plan, wrap: Callable[[Leaf], Leaf]) -> Plan:
    match plan:
        case Pair(left, right):
            return Pair(under(left, wrap), under(right, wrap))
        case leaf:
            return wrap(leaf)


@overload
def val_data[S, X, Y, HP, P](
    v: SameModel[S, X, Y, HP, P], below: Term[S, X, Y, HP, P]
) -> Steps | Window | Batch | Pair:
    return data(below)


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q](
    v: Validate[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q], below: Term[S, X, Y, HP, P]
) -> Steps | Window | Batch | Pair:
    return data(v.term)


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Steps | Window | Batch | Pair:
    raise NotImplementedError


@dispatch
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Steps | Window | Batch | Pair:
    raise NotImplementedError


@overload
def data(t: Linear) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data(t: Bias) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data(t: Activation) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data(t: Loss) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data[HPA, PA](t: Rnn[HPA, PA]) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data[S, X, HP, P](t: Sup[S, X, HP, P]) -> Steps | Window | Batch | Pair:
    return Steps()


@overload
def data[S1, S2, X, Y, Z, HP1, HP2, P1, P2](t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2]) -> Steps | Window | Batch | Pair:
    return data(t.first)


@overload
def data[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
) -> Steps | Window | Batch | Pair:
    return Pair(data(t.below), val_data(t.val, t.below))


@overload
def data[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return under(data(t.below), lambda leaf: Window(t.n, leaf))


@overload
def data[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return under(data(t.below), lambda leaf: Batch(t.n, t.over, leaf))


@overload
def data[S, X, Y, HP, P](t: BatchParams[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return under(data(t.below), lambda leaf: Batch(t.n, t.over, leaf))


@overload
def data[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return under(data(t.below), lambda leaf: Batch(t.n, t.over, leaf))


@overload
def data[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD]) -> Steps | Window | Batch | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, HP2, P, P2](t: Reparametrized[S, X, Y, HP, HP2, P, P2]) -> Steps | Window | Batch | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    raise NotImplementedError


@dispatch
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Steps | Window | Batch | Pair:
    raise NotImplementedError
