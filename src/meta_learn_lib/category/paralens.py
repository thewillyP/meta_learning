from dataclasses import dataclass
from meta_learn_lib.category.lens import *
from meta_learn_lib.category.lib_types import *


@dataclass(frozen=True)
class ParaLens[X1, X2, Y1, Y2, HP1, HP2, P1, P2]:
    arrow: Lens[tuple[tuple[HP1, P1], X1], tuple[tuple[HP2, P2], X2], Y1, Y2]

    def __rshift__[HPF1, HPF2, PF1, PF2, HPG1, HPG2, PG1, PG2, A1, A2, B1, B2, C1, C2](
        f: "ParaLens[A1, A2, B1, B2, HPF1, HPF2, PF1, PF2]",
        g: "ParaLens[B1, B2, C1, C2, HPG1, HPG2, PG1, PG2]",
    ) -> "ParaLens[A1, A2, C1, C2, tuple[HPF1, HPG1], tuple[HPF2, HPG2], tuple[PF1, PG1], tuple[PF2, PG2]]":

        return ParaLens(
            (exchange(Proxy[tuple[HPF1, HPF2, HPG1, HPG2, PF1, PF2, PG1, PG2]]()) @ identity(Proxy[tuple[A1, A2]]()))
            >> (
                swap(Proxy[tuple[tuple[HPF1, PF1], tuple[HPF2, PF2], tuple[HPG1, PG1], tuple[HPG2, PG2]]]())
                @ identity(Proxy[tuple[A1, A2]]())
            )
            >> assocL(Proxy[tuple[tuple[HPG1, PG1], tuple[HPG2, PG2], tuple[HPF1, PF1], tuple[HPF2, PF2], A1, A2]]())
            >> (identity(Proxy[tuple[tuple[HPG1, PG1], tuple[HPG2, PG2]]]()) @ f.arrow)
            >> g.arrow
        )

    def __matmul__[HPF1, HPF2, PF1, PF2, HPG1, HPG2, PG1, PG2, A1, A2, B1, B2, C1, C2, D1, D2](
        f: "ParaLens[A1, A2, B1, B2, HPF1, HPF2, PF1, PF2]",
        g: "ParaLens[C1, C2, D1, D2, HPG1, HPG2, PG1, PG2]",
    ) -> "ParaLens[tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2], tuple[HPF1, HPG1], tuple[HPF2, HPG2], tuple[PF1, PG1], tuple[PF2, PG2]]":

        return ParaLens(
            (
                exchange(Proxy[tuple[HPF1, HPF2, HPG1, HPG2, PF1, PF2, PG1, PG2]]())
                @ identity(Proxy[tuple[tuple[A1, C1], tuple[A2, C2]]]())
            )
            >> exchange(
                Proxy[tuple[tuple[HPF1, PF1], tuple[HPF2, PF2], tuple[HPG1, PG1], tuple[HPG2, PG2], A1, A2, C1, C2]]()
            )
            >> (f.arrow @ g.arrow)
        )


def to_paralens[A1, A2, B1, B2](f: Lens[A1, A2, B1, B2]) -> ParaLens[A1, A2, B1, B2, Unit, Unit, Unit, Unit]:
    return ParaLens(snd(Proxy[tuple[tuple[Unit, Unit], tuple[Unit, Unit], A1, A2]]()) >> f)


def para_autodiff[HP, P, X, Y](f: Callable[[tuple[HP, P], X], Y]) -> ParaLens[X, X, Y, Y, HP, HP, P, P]:
    def uncurried(zx: tuple[tuple[HP, P], X]) -> Y:
        params, x = zx
        return f(params, x)

    return ParaLens(autodiff(uncurried))
