from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchParams,
    BatchPop,
    Bias,
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
    Sup,
    Term,
    UORO,
    Validate,
    Validator,
)

from dataclasses import dataclass
from typing import overload
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from plum import dispatch


@dataclass(frozen=True)
class Window:
    n: int


@dataclass(frozen=True)
class Data:
    n: int


@dataclass(frozen=True)
class Params:
    n: int


@dataclass(frozen=True)
class Pop:
    n: int


type Axis = Window | Data | Params | Pop


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


def size(axis: Axis) -> int:
    match axis:
        case Window(n) | Data(n) | Params(n) | Pop(n):
            return n


def windowed(axis: Axis) -> bool:
    match axis:
        case Window():
            return True
        case Data() | Params() | Pop():
            return False


def shared(axis: Axis) -> bool:
    match axis:
        case Data():
            return True
        case Window() | Params() | Pop():
            return False


def push(axes: tuple[Axis, ...], plan: Plan) -> Plan:
    match plan:
        case Leaf(own):
            return Leaf((*axes, *own))
        case Pair(own, left, right):
            return Pair((*axes, *own), left, right)


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
    return prefix(Data(t.n), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchParams[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Params(t.n), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Leaf | Pair:
    return prefix(Pop(t.n), data(t.below))


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
