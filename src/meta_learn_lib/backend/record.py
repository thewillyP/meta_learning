from functools import reduce
from typing import Callable
import jax
import pyrsistent
from pyrsistent import pmap
from pyrsistent.typing import PMap

from ..category.lens import Lens, wiring
from ..category.lib_types import Monoid
from ..category.paralens import ParaLens
from ..utility.util import add_cotangents


def mapwise[E, V](f: Callable[[PMap[str, V]], PMap[str, V]]) -> Callable[[E], E]:
    return lambda e: jax.tree.map(f, e, is_leaf=lambda x: isinstance(x, pyrsistent.PMap))


def recordwise[E]() -> Monoid[E, E]:
    def mappend(left: E, right: E) -> E:
        return jax.tree.map(
            lambda l, r: l.update_with(add_cotangents, r), left, right, is_leaf=lambda x: isinstance(x, pyrsistent.PMap)
        )

    return Monoid(mapwise(lambda _: pmap()), mappend)


def restrict[E](names: frozenset[str]) -> Callable[[E], E]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k in names}))


def without[E](names: frozenset[str]) -> Callable[[E], E]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k not in names}))


def split[E](names: frozenset[str], monoid: Monoid[E, E]) -> Lens[E, E, tuple[E, E], tuple[E, E]]:
    def backward(parts: tuple[E, E]) -> E:
        chosen, rest = parts
        return monoid.mappend(chosen, rest)

    return wiring(lambda e: (restrict(names)(e), without(names)(e)), backward)


def rename[E, X, DX, Y, DY](
    para: ParaLens[E, E, X, DX, Y, DY],
    transpositions: tuple[tuple[str, str], ...],
) -> ParaLens[E, E, X, DX, Y, DY]:
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

    def relabel(order: tuple[tuple[str, str], ...]) -> Callable[[E], E]:
        return mapwise(lambda m: reduce(transpose, order, m))

    return para.reparam(wiring(relabel(transpositions), relabel(transpositions[::-1])), para.monoid)
