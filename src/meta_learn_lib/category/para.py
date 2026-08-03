from dataclasses import dataclass
from typing import Callable
import jax
from jaxtyping import PyTree
from category.lens import *
from meta_learn_lib.category import lens


@dataclass(frozen=True)
class ParaLens[X1, X2, Y1, Y2, Z1, Z2]:
    arrow: Lens[tuple[Z1, X1], tuple[Z2, X2], Y1, Y2]

    def __rshift__[P1, P2, Q1, Q2, A1, A2, B1, B2, C1, C2](
        f: "ParaLens[A1, A2, B1, B2, P1, P2]",
        g: "ParaLens[B1, B2, C1, C2, Q1, Q2]",
    ) -> "ParaLens[A1, A2, C1, C2, tuple[P1, Q1], tuple[P2, Q2]]":

        return ParaLens(
            swap(Proxy[tuple[P1, P2, Q1, Q2]]()) @ identity(Proxy[tuple[A1, A2]]())
            >> assocL(Proxy[tuple[Q1, Q2, P1, P2, A1, A2]]())
            >> (identity(Proxy[tuple[Q1, Q2]]()) @ f.arrow)
            >> g.arrow
        )

    def __matmul__[P1, P2, Q1, Q2, A1, A2, B1, B2, C1, C2, D1, D2](
        f: "ParaLens[A1, A2, B1, B2, P1, P2]",
        g: "ParaLens[C1, C2, D1, D2, Q1, Q2]",
    ) -> "ParaLens[tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2], tuple[P1, Q1], tuple[P2, Q2]]":
        return ParaLens(exchange(Proxy[tuple[P1, P2, Q1, Q2, A1, A2, C1, C2]]()) >> (f.arrow @ g.arrow))


def to_paralens[A1, A2, B1, B2](f: Lens[A1, A2, B1, B2]) -> ParaLens[A1, A2, B1, B2, Unit, Unit]:
    return ParaLens(snd_unit(Proxy[tuple[A1, A2]]()) >> f)
