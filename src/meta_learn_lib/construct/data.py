from meta_learn_lib.construct.term import (
    BatchData,
    BatchParams,
    BatchPop,
    Meta,
    Over,
    RFLO,
    RTRL,
    Reparametrized,
    SameModel,
    Scan,
    Shared,
    Sup,
    Term,
    UORO,
    Validate,
    Validator,
)
from meta_learn_lib.data_source.source import Draw

from dataclasses import dataclass
from typing import overload
import jax
from plum import dispatch


@dataclass(frozen=True)
class Steps:
    draw: Draw


@dataclass(frozen=True)
class Chunk:
    n: int
    below: "Leaf"


@dataclass(frozen=True)
class Batch:
    n: int
    over: Over
    below: "Leaf"


type Leaf = Steps | Chunk | Batch

type Plan = Leaf | tuple[Plan, Plan]


def draw_of(leaf: Leaf) -> Draw:
    match leaf:
        case Steps(draw):
            return draw
        case Chunk(_, below) | Batch(_, _, below):
            return draw_of(below)


def redrawn(leaf: Leaf, draw: Draw) -> Leaf:
    match leaf:
        case Steps():
            return Steps(draw)
        case Chunk(n, below):
            return Chunk(n, redrawn(below, draw))
        case Batch(n, over, below):
            return Batch(n, over, redrawn(below, draw))


@overload
def val_data[S, X, Y, HP, P](
    v: SameModel[S, X, Y, HP, P], below: Term[S, X, Y, HP, P]
) -> Steps | Chunk | Batch | tuple:
    return jax.tree.map(lambda leaf: redrawn(leaf, v.draw), data(below))


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q](
    v: Validate[S, X, Y, HP, P, SV, XV, HPV, PV, HQ, Q], below: Term[S, X, Y, HP, P]
) -> Steps | Chunk | Batch | tuple:
    return data(v.term)


@overload
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Steps | Chunk | Batch | tuple:
    raise NotImplementedError


@dispatch
def val_data[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], below: Term[S, X, Y, HP, P]
) -> Steps | Chunk | Batch | tuple:
    raise NotImplementedError


@overload
def data[S, X, HP, P](t: Sup[S, X, HP, P]) -> Steps | Chunk | Batch | tuple:
    return Steps(t.draw)


@overload
def data[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV],
) -> Steps | Chunk | Batch | tuple:
    return (data(t.below), val_data(t.val, t.below))


@overload
def data[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return jax.tree.map(lambda leaf: Chunk(t.n, leaf), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return jax.tree.map(lambda leaf: Batch(t.n, t.over, leaf), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchParams[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return jax.tree.map(lambda leaf: Batch(t.n, t.over, leaf), data(t.below))


@overload
def data[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return jax.tree.map(lambda leaf: Batch(t.n, t.over, leaf), data(t.below))


@overload
def data[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return data(t.below)


@overload
def data[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD]) -> Steps | Chunk | Batch | tuple:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return data(t.below)


@overload
def data[S, X, Y, HP, HP2, P, P2](
    t: Reparametrized[S, X, Y, HP, HP2, P, P2],
) -> Steps | Chunk | Batch | tuple:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Shared[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    return data(t.below)


@overload
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    raise NotImplementedError


@dispatch
def data[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Steps | Chunk | Batch | tuple:
    raise NotImplementedError
