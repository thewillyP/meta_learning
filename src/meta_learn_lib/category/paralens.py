from dataclasses import dataclass
from meta_learn_lib.category.lens import *
from meta_learn_lib.category.lib_types import *


@dataclass(frozen=True)
class ParaLens[P1, P2, X1, X2, Y1, Y2]:
    arrow: Lens[tuple[P1, X1], tuple[P2, X2], Y1, Y2]

    def __rshift__[PF1, PF2, PG1, PG2, A1, A2, B1, B2, C1, C2](
        f: "ParaLens[PF1, PF2, A1, A2, B1, B2]",
        g: "ParaLens[PG1, PG2, B1, B2, C1, C2]",
    ) -> "ParaLens[tuple[PF1, PG1], tuple[PF2, PG2], A1, A2, C1, C2]":

        return join(ParaLens((identity(Proxy[tuple[PG1, PG2]]()) @ f.arrow) >> g.arrow))

    def __matmul__[PF1, PF2, PG1, PG2, A1, A2, B1, B2, C1, C2, D1, D2](
        f: "ParaLens[PF1, PF2, A1, A2, B1, B2]",
        g: "ParaLens[PG1, PG2, C1, C2, D1, D2]",
    ) -> "ParaLens[tuple[PF1, PG1], tuple[PF2, PG2], tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2]]":

        return ParaLens(exchange(Proxy[tuple[PF1, PF2, PG1, PG2, A1, A2, C1, C2]]()) >> (f.arrow @ g.arrow))


def para_autodiff[P, X, Y](f: Callable[[P, X], Y]) -> ParaLens[P, P, X, X, Y, Y]:
    def uncurried(px: tuple[P, X]) -> Y:
        p, x = px
        return f(p, x)

    return ParaLens(autodiff(uncurried))


def unit[A1, A2, B1, B2](f: Lens[A1, A2, B1, B2]) -> ParaLens[Unit, Unit, A1, A2, B1, B2]:
    return ParaLens(snd(Proxy[tuple[Unit, Unit, A1, A2]]()) >> f)


def join[Q1, Q2, P1, P2, A1, A2, B1, B2](
    f: ParaLens[P1, P2, tuple[Q1, A1], tuple[Q2, A2], B1, B2],
) -> ParaLens[tuple[Q1, P1], tuple[Q2, P2], A1, A2, B1, B2]:
    return ParaLens(
        (swap(Proxy[tuple[Q1, Q2, P1, P2]]()) @ identity(Proxy[tuple[A1, A2]]()))
        >> assocL(Proxy[tuple[P1, P2, Q1, Q2, A1, A2]]())
        >> f.arrow
    )


def unjoin[Q1, Q2, P1, P2, A1, A2, B1, B2](
    f: ParaLens[tuple[Q1, P1], tuple[Q2, P2], A1, A2, B1, B2],
) -> ParaLens[P1, P2, tuple[Q1, A1], tuple[Q2, A2], B1, B2]:
    return ParaLens(
        assocR(Proxy[tuple[P1, P2, Q1, Q2, A1, A2]]())
        >> (swap(Proxy[tuple[P1, P2, Q1, Q2]]()) @ identity(Proxy[tuple[A1, A2]]()))
        >> f.arrow
    )


def reparam[Q1, Q2, P1, P2, A1, A2, B1, B2](
    r: Lens[Q1, Q2, P1, P2],
    f: ParaLens[P1, P2, A1, A2, B1, B2],
) -> ParaLens[Q1, Q2, A1, A2, B1, B2]:
    return ParaLens((r @ identity(Proxy[tuple[A1, A2]]())) >> f.arrow)
