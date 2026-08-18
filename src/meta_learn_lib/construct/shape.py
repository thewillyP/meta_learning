from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchPop,
    Bias,
    Linear,
    Loss,
    Meta,
    RFLO,
    RTRL,
    Reparametrized,
    Rnn,
    Scan,
    Seq,
    Sup,
    Term,
    UORO,
)

from typing import overload
from plum import dispatch


@overload
def out(t: Linear, n_in: int) -> int:
    return t.n


@overload
def out(t: Bias, n_in: int) -> int:
    return n_in


@overload
def out[HPA, PA](t: Rnn[HPA, PA], n_in: int) -> int:
    return t.n


@overload
def out(t: Activation, n_in: int) -> int:
    return n_in


@overload
def out(t: Loss, n_in: int) -> int:
    return 1


@overload
def out[S1, S2, X, Y, Z, HP1, HP2, P1, P2](t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2], n_in: int) -> int:
    return out(t.second, out(t.first, n_in))


@overload
def out[S, X, HP, P](t: Sup[S, X, HP, P], n_in: int) -> int:
    return 1


@overload
def out[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](t: Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P, HD](t: RFLO[S, X, Y, HP, P, HD], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, HP2, P, P2](t: Reparametrized[S, X, Y, HP, HP2, P, P2], n_in: int) -> int:
    return out(t.below, n_in)


@overload
def out[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], n_in: int) -> int:
    raise NotImplementedError


@dispatch
def out[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], n_in: int) -> int:
    raise NotImplementedError
