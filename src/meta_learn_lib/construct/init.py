from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.leaves import initializer
from meta_learn_lib.construct.shape import out
from meta_learn_lib.construct.term import Activation, Arch, Bias, Linear, Seq
from meta_learn_lib.lib_types import PRNG

from typing import overload
import equinox as eqx
import jax
from plum import dispatch


@overload
def init(t: Linear, n_in: int, key: PRNG) -> tuple[tuple[Unit, eqx.nn.Linear], Unit]:
    k_layer, k_w = jax.random.split(key)
    layer = eqx.nn.Linear(n_in, t.n, use_bias=False, key=k_layer)
    w = initializer(t.init)(k_w, (t.n, n_in))
    return ((Unit(), eqx.tree_at(lambda l: l.weight, layer, w)), Unit())


@overload
def init(t: Bias, n_in: int, key: PRNG) -> tuple[tuple[Unit, jax.Array], Unit]:
    return ((Unit(), initializer(t.init)(key, (n_in,))), Unit())


@overload
def init(t: Activation, n_in: int, key: PRNG) -> tuple[tuple[Unit, Unit], Unit]:
    return ((Unit(), Unit()), Unit())


@overload
def init[S1, S2, HP1, HP2, P1, P2](
    t: Seq[S1, S2, HP1, HP2, P1, P2], n_in: int, key: PRNG
) -> tuple[tuple[tuple[HP1, HP2], tuple[P1, P2]], tuple[S1, S2]]:
    k1, k2 = jax.random.split(key)
    (hp1, p1), s1 = init(t.first, n_in, PRNG(k1))
    (hp2, p2), s2 = init(t.second, out(t.first, n_in), PRNG(k2))
    return (((hp1, hp2), (p1, p2)), (s1, s2))


@overload
def init[S, HP, P](t: Arch[S, HP, P], n_in: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError


@dispatch
def init[S, HP, P](t: Arch[S, HP, P], n_in: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError
