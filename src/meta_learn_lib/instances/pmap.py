from functools import reduce
from typing import Callable
from pyrsistent import pmap
from pyrsistent.typing import PMap

from ..category.lens import wiring
from ..category.lib_types import Monoid, Proxy
from ..category.paralens import ParaLens


def keywise[P, DP](_: Proxy[P], add: Callable[[DP, DP], DP]) -> Monoid[PMap[str, P], PMap[str, DP]]:
    return Monoid(lambda _: pmap(), lambda left, right: left.update_with(add, right))


def rename[P, DP, X, DX, Y, DY](
    para: ParaLens[PMap[str, P], PMap[str, DP], X, DX, Y, DY],
    transpositions: tuple[tuple[str, str], ...],
) -> ParaLens[PMap[str, P], PMap[str, DP], X, DX, Y, DY]:
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

    def relabel[V](order: tuple[tuple[str, str], ...]) -> Callable[[PMap[str, V]], PMap[str, V]]:
        return lambda m: reduce(transpose, order, m)

    return para.reparam(wiring(relabel(transpositions), relabel(transpositions[::-1])), para.monoid)
