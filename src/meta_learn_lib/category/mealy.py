from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *

from dataclasses import dataclass


@dataclass(frozen=True)
class Mealy[A1, A2, B1, B2, C1, C2, HP1, HP2, P1, P2]:
    """A Mealy machine as a parametric lens: ((HP, P), (A, B)) -> (A, C).

    Composition follows Katis-Sabadini-Walters `Circ`: the state objects
    TENSOR (each machine keeps its own memory) while the values CHAIN.
    """

    arrow: ParaLens[tuple[A1, B1], tuple[A2, B2], tuple[A1, C1], tuple[A2, C2], HP1, HP2, P1, P2]

    def __rshift__[S1, S2, T1, T2, X1, X2, Y1, Y2, W1, W2, HPF1, HPF2, PF1, PF2, HPG1, HPG2, PG1, PG2](
        f: "Mealy[S1, S2, X1, X2, Y1, Y2, HPF1, HPF2, PF1, PF2]",
        g: "Mealy[T1, T2, Y1, Y2, W1, W2, HPG1, HPG2, PG1, PG2]",
    ) -> "Mealy[tuple[S1, T1], tuple[S2, T2], X1, X2, W1, W2, tuple[HPF1, HPG1], tuple[HPF2, HPG2], tuple[PF1, PG1], tuple[PF2, PG2]]":

        preL = (
            assocR(Proxy[tuple[tuple[HPF1, PF1], tuple[HPF2, PF2], tuple[S1, T1], tuple[S2, T2], X1, X2]]())
            >> (
                assocR(Proxy[tuple[tuple[HPF1, PF1], tuple[HPF2, PF2], S1, S2, T1, T2]]())
                @ identity(Proxy[tuple[X1, X2]]())
            )
            >> (
                swap(Proxy[tuple[tuple[tuple[HPF1, PF1], S1], tuple[tuple[HPF2, PF2], S2], T1, T2]]())
                @ identity(Proxy[tuple[X1, X2]]())
            )
            >> assocL(Proxy[tuple[T1, T2, tuple[tuple[HPF1, PF1], S1], tuple[tuple[HPF2, PF2], S2], X1, X2]]())
            >> (
                identity(Proxy[tuple[T1, T2]]())
                @ assocL(Proxy[tuple[tuple[HPF1, PF1], tuple[HPF2, PF2], S1, S2, X1, X2]]())
            )
        )
        # (T,(S,Y)) -> ((S,T),Y)
        postL = assocR(Proxy[tuple[T1, T2, S1, S2, Y1, Y2]]()) >> (
            swap(Proxy[tuple[T1, T2, S1, S2]]()) @ identity(Proxy[tuple[Y1, Y2]]())
        )

        widenL = ParaLens(preL >> (identity(Proxy[tuple[T1, T2]]()) @ f.arrow.arrow) >> postL)

        # (Q,((S,T),Y)) -> (S,(Q,(T,Y)))
        preR = (
            swap(Proxy[tuple[tuple[HPG1, PG1], tuple[HPG2, PG2], tuple[tuple[S1, T1], Y1], tuple[tuple[S2, T2], Y2]]]())
            >> (
                assocL(Proxy[tuple[S1, S2, T1, T2, Y1, Y2]]())
                @ identity(Proxy[tuple[tuple[HPG1, PG1], tuple[HPG2, PG2]]]())
            )
            >> assocL(Proxy[tuple[S1, S2, tuple[T1, Y1], tuple[T2, Y2], tuple[HPG1, PG1], tuple[HPG2, PG2]]]())
            >> (
                identity(Proxy[tuple[S1, S2]]())
                @ swap(Proxy[tuple[tuple[T1, Y1], tuple[T2, Y2], tuple[HPG1, PG1], tuple[HPG2, PG2]]]())
            )
        )
        # (S,(T,W)) -> ((S,T),W)
        postR = assocR(Proxy[tuple[S1, S2, T1, T2, W1, W2]]())
        widenR = ParaLens(preR >> (identity(Proxy[tuple[S1, S2]]()) @ g.arrow.arrow) >> postR)

        return Mealy(widenL >> widenR)

    def __matmul__[S1, S2, T1, T2, X1, X2, Y1, Y2, U1, U2, V1, V2, HPF1, HPF2, PF1, PF2, HPG1, HPG2, PG1, PG2](
        f: "Mealy[S1, S2, X1, X2, Y1, Y2, HPF1, HPF2, PF1, PF2]",
        g: "Mealy[T1, T2, U1, U2, V1, V2, HPG1, HPG2, PG1, PG2]",
    ) -> "Mealy[tuple[S1, T1], tuple[S2, T2], tuple[X1, U1], tuple[X2, U2], tuple[Y1, V1], tuple[Y2, V2], tuple[HPF1, HPG1], tuple[HPF2, HPG2], tuple[PF1, PG1], tuple[PF2, PG2]]":
        # ((S,T),(X,U)) -> ((S,X),(T,U))
        pre = identity(
            Proxy[tuple[tuple[tuple[HPF1, HPG1], tuple[PF1, PG1]], tuple[tuple[HPF2, HPG2], tuple[PF2, PG2]]]]()
        ) @ exchange(Proxy[tuple[S1, S2, T1, T2, X1, X2, U1, U2]]())
        # ((S,Y),(T,V)) -> ((S,T),(Y,V))
        post = exchange(Proxy[tuple[S1, S2, Y1, Y2, T1, T2, V1, V2]]())
        return Mealy(ParaLens(pre >> (f.arrow @ g.arrow).arrow >> post))


def to_mealy[X1, X2, Y1, Y2, HP1, HP2, P1, P2](
    arrow: ParaLens[X1, X2, Y1, Y2, HP1, HP2, P1, P2],
) -> Mealy[Unit, Unit, X1, X2, Y1, Y2, HP1, HP2, P1, P2]:
    pre = identity(Proxy[tuple[tuple[HP1, P1], tuple[HP2, P2]]]()) @ snd(Proxy[tuple[Unit, Unit, X1, X2]]())
    return Mealy(ParaLens(pre >> arrow.arrow >> unit_intro(Proxy[tuple[Y1, Y2]]())))
