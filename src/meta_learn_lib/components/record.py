from dataclasses import dataclass
from functools import reduce
from typing import Callable, Protocol

import equinox as eqx
import jax
import pyrsistent
from pyrsistent import pmap
from pyrsistent.typing import PMap

from ..category.lens import Lens, wiring
from ..category.lib_types import Monoid
from ..category.paralens import ParaLens
from ..pytree import pyrsistent as _  # noqa: F401


class Add(Protocol):
    def __call__[V](self, left: V, right: V, /) -> V: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Env:
    arrays: PMap[str, jax.Array]
    norms: PMap[str, eqx.nn.LayerNorm | eqx.nn.GroupNorm]


def mapwise[V](f: Callable[[PMap[str, V]], PMap[str, V]]) -> Callable[[Env], Env]:
    return lambda e: jax.tree.map(f, e, is_leaf=lambda x: isinstance(x, pyrsistent.PMap))


def recordwise(add: Add) -> Monoid[Env, Env]:
    def mappend(left: Env, right: Env) -> Env:
        return jax.tree.map(
            lambda l, r: l.update_with(add, r), left, right, is_leaf=lambda x: isinstance(x, pyrsistent.PMap)
        )

    return Monoid(mapwise(lambda _: pmap()), mappend)


def restrict(names: frozenset[str]) -> Callable[[Env], Env]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k in names}))


def without(names: frozenset[str]) -> Callable[[Env], Env]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k not in names}))


def split(names: frozenset[str], monoid: Monoid[Env, Env]) -> Lens[Env, Env, tuple[Env, Env], tuple[Env, Env]]:
    def backward(parts: tuple[Env, Env]) -> Env:
        chosen, rest = parts
        return monoid.mappend(chosen, rest)

    return wiring(lambda e: (restrict(names)(e), without(names)(e)), backward)


def rename[X, DX, Y, DY](
    para: ParaLens[Env, Env, X, DX, Y, DY],
    transpositions: tuple[tuple[str, str], ...],
) -> ParaLens[Env, Env, X, DX, Y, DY]:
    def transpose[V](m: PMap[str, V], ab: tuple[str, str]) -> PMap[str, V]:
        def swap_name(k: str) -> str:
            a, b = ab
            match (k == a, k == b):
                case (True, _):
                    return b
                case (_, True):
                    return a
                case _:
                    return k

        return pmap({swap_name(k): v for k, v in m.items()})

    def relabel(order: tuple[tuple[str, str], ...]) -> Callable[[Env], Env]:
        return mapwise(lambda m: reduce(transpose, order, m))

    return para.reparam(wiring(relabel(transpositions), relabel(transpositions[::-1])), para.monoid)
