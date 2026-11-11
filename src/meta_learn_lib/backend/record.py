from functools import reduce
from typing import Callable
import jax
import pyrsistent
from pyrsistent import PVector, pmap
from pyrsistent.typing import PMap

from ..category.lens import Lens, wiring
from ..category.lib_types import Monoid, Proxy
from ..category.paralens import ParaLens
from ..components.env import Env, Log, Par
from ..utility.util import add_cotangents


def mapwise[V](f: Callable[[PMap[str, V]], PMap[str, V]]) -> Callable[[Env], Env]:
    return lambda e: jax.tree.map(f, e, is_leaf=lambda x: isinstance(x, pyrsistent.PMap))


def recordwise() -> Monoid[Env, Env]:
    def mappend(left: Env, right: Env) -> Env:
        return jax.tree.map(
            lambda l, r: l.update_with(add_cotangents, r), left, right, is_leaf=lambda x: isinstance(x, pyrsistent.PMap)
        )

    return Monoid(mapwise(lambda _: pmap()), mappend)


def names(e: Env) -> frozenset[str]:
    return frozenset().union(*(m.keys() for m in jax.tree.leaves(e, is_leaf=lambda x: isinstance(x, pyrsistent.PMap))))


def restrict_keys(names: frozenset[str]) -> Callable[[Env], Env]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k in names}))


def without_keys(names: frozenset[str]) -> Callable[[Env], Env]:
    return mapwise(lambda m: pmap({k: v for k, v in m.items() if k not in names}))


def split(names: frozenset[str], monoid: Monoid[Env, Env]) -> Lens[Env, Env, Env, tuple[Env, Env], tuple[Env, Env]]:
    def backward(parts: tuple[Env, Env]) -> Env:
        chosen, rest = parts
        return monoid.mappend(chosen, rest)

    return wiring(Proxy[Env](), lambda e: (restrict_keys(names)(e), without_keys(names)(e)), backward)


def join(names: frozenset[str], monoid: Monoid[Env, Env]) -> Lens[Env, tuple[Env, Env], tuple[Env, Env], Env, Env]:
    def forward(parts: tuple[Env, Env]) -> Env:
        chosen, rest = parts
        return monoid.mappend(chosen, rest)

    return wiring(Proxy[Env](), forward, lambda e: (restrict_keys(names)(e), without_keys(names)(e)))


def rename[X, DX, Y, DY](
    para: ParaLens[Log, Par, Par, X, DX, Y, DY],
    transpositions: tuple[tuple[str, str], ...],
) -> ParaLens[Log, Par, Par, X, DX, Y, DY]:
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

    return para.reparam(wiring(Proxy[Env](), relabel(transpositions), relabel(transpositions[::-1])), para.monoid)


def fold(log: PVector[Env]) -> Env:
    def step(acc: Env, entry: Env) -> Env:
        clash = names(acc) & names(entry)
        if clash:
            raise ValueError(f"logged twice: {sorted(clash)}")
        return recordwise().mappend(acc, entry)

    return reduce(step, log, Env())
