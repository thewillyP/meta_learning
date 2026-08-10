from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *

from dataclasses import dataclass


@dataclass(frozen=True)
class Mealy[A1, A2, B1, B2, C1, C2, D1, D2]:
    """A Mealy machine as a parametric lens: (D, (A, B)) -> (A, C).

    Composition follows Katis-Sabadini-Walters `Circ`: the state objects
    TENSOR (each machine keeps its own memory) while the values CHAIN.
    """

    arrow: ParaLens[tuple[A1, B1], tuple[A2, B2], tuple[A1, C1], tuple[A2, C2], D1, D2]

    def __rshift__[S1, S2, T1, T2, X1, X2, Y1, Y2, Z1, Z2, P1, P2, Q1, Q2](
        f: "Mealy[S1, S2, X1, X2, Y1, Y2, P1, P2]",
        g: "Mealy[T1, T2, Y1, Y2, Z1, Z2, Q1, Q2]",
    ) -> "Mealy[tuple[S1, T1], tuple[S2, T2], X1, X2, Z1, Z2, tuple[P1, Q1], tuple[P2, Q2]]":

        preL = (
            assocR(Proxy[tuple[P1, P2, tuple[S1, T1], tuple[S2, T2], X1, X2]]())
            >> (assocR(Proxy[tuple[P1, P2, S1, S2, T1, T2]]()) @ identity(Proxy[tuple[X1, X2]]()))
            >> (swap(Proxy[tuple[tuple[P1, S1], tuple[P2, S2], T1, T2]]()) @ identity(Proxy[tuple[X1, X2]]()))
            >> assocL(Proxy[tuple[T1, T2, tuple[P1, S1], tuple[P2, S2], X1, X2]]())
            >> (identity(Proxy[tuple[T1, T2]]()) @ assocL(Proxy[tuple[P1, P2, S1, S2, X1, X2]]()))
        )
        # (T,(S,Y)) -> ((S,T),Y)
        postL = assocR(Proxy[tuple[T1, T2, S1, S2, Y1, Y2]]()) >> (
            swap(Proxy[tuple[T1, T2, S1, S2]]()) @ identity(Proxy[tuple[Y1, Y2]]())
        )

        widenL = ParaLens(preL >> (identity(Proxy[tuple[T1, T2]]()) @ f.arrow.arrow) >> postL)

        # (Q,((S,T),Y)) -> (S,(Q,(T,Y)))
        preR = (
            swap(Proxy[tuple[Q1, Q2, tuple[tuple[S1, T1], Y1], tuple[tuple[S2, T2], Y2]]]())
            >> (assocL(Proxy[tuple[S1, S2, T1, T2, Y1, Y2]]()) @ identity(Proxy[tuple[Q1, Q2]]()))
            >> assocL(Proxy[tuple[S1, S2, tuple[T1, Y1], tuple[T2, Y2], Q1, Q2]]())
            >> (identity(Proxy[tuple[S1, S2]]()) @ swap(Proxy[tuple[tuple[T1, Y1], tuple[T2, Y2], Q1, Q2]]()))
        )
        # (S,(T,Z)) -> ((S,T),Z)
        postR = assocR(Proxy[tuple[S1, S2, T1, T2, Z1, Z2]]())
        widenR = ParaLens(preR >> (identity(Proxy[tuple[S1, S2]]()) @ g.arrow.arrow) >> postR)

        return Mealy(widenL >> widenR)


def to_mealy[X1, X2, Y1, Y2, P1, P2](
    arrow: ParaLens[X1, X2, Y1, Y2, P1, P2],
) -> Mealy[Unit, Unit, X1, X2, Y1, Y2, P1, P2]:
    pre = identity(Proxy[tuple[P1, P2]]()) @ snd(Proxy[tuple[Unit, Unit, X1, X2]]())
    return Mealy(ParaLens(pre >> arrow.arrow >> unit_intro(Proxy[tuple[Y1, Y2]]())))
