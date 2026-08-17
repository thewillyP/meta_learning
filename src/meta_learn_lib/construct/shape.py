from meta_learn_lib.construct.term import Activation, Arch, Bias, Linear, Seq

from typing import overload
from plum import dispatch


@overload
def out(t: Linear, n_in: int) -> int:
    return t.n


@overload
def out(t: Bias, n_in: int) -> int:
    return n_in


@overload
def out(t: Activation, n_in: int) -> int:
    return n_in


@overload
def out[S1, S2, HP1, HP2, P1, P2](t: Seq[S1, S2, HP1, HP2, P1, P2], n_in: int) -> int:
    return out(t.second, out(t.first, n_in))


@overload
def out[S, HP, P](t: Arch[S, HP, P], n_in: int) -> int:
    raise NotImplementedError


@dispatch
def out[S, HP, P](t: Arch[S, HP, P], n_in: int) -> int:
    raise NotImplementedError
