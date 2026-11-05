from dataclasses import dataclass

from .lib_types import Monoid, Proxy
from .lens import Lens, assocL, assocR, swap
from .paralens import ParaLens, unit


@dataclass(frozen=True)
class Mealy[P, DP, S, DS, X, DX, Y, DY]:
    step: ParaLens[P, DP, tuple[S, X], tuple[DS, DX], tuple[S, Y], tuple[DS, DY]]

    def then[Z, DZ](self, g: "Mealy[P, DP, S, DS, Y, DY, Z, DZ]") -> "Mealy[P, DP, S, DS, X, DX, Z, DZ]":
        return Mealy(self.step >> g.step)

    def tensor[U, DU, V, DV](
        self,
        g: "Mealy[P, DP, S, DS, U, DU, V, DV]",
    ) -> "Mealy[P, DP, S, DS, tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return self.first(Proxy[tuple[U, DU]]()) >> g.second(Proxy[tuple[Y, DY]]())

    def __rshift__[Q, DQ, T, DT, A, DA, B, DB, C, DC](
        f: "Mealy[Q, DQ, T, DT, A, DA, B, DB]",
        g: "Mealy[Q, DQ, T, DT, B, DB, C, DC]",
    ) -> "Mealy[Q, DQ, T, DT, A, DA, C, DC]":
        return f.then(g)

    def __matmul__[Q, DQ, T, DT, A, DA, B, DB, C, DC, D, DD](
        f: "Mealy[Q, DQ, T, DT, A, DA, B, DB]",
        g: "Mealy[Q, DQ, T, DT, C, DC, D, DD]",
    ) -> "Mealy[Q, DQ, T, DT, tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def reparam[Q, DQ](self, r: Lens[Q, DQ, P, DP], monoid: Monoid[Q, DQ]) -> "Mealy[Q, DQ, S, DS, X, DX, Y, DY]":
        return Mealy(self.step.reparam(r, monoid))

    def pre[W, DW](self, f: Lens[W, DW, X, DX]) -> "Mealy[P, DP, S, DS, W, DW, Y, DY]":
        return Mealy(self.step.pre(f.second(Proxy[tuple[S, DS]]())))

    def post[Z, DZ](self, g: Lens[Y, DY, Z, DZ]) -> "Mealy[P, DP, S, DS, X, DX, Z, DZ]":
        return Mealy(self.step.post(g.second(Proxy[tuple[S, DS]]())))

    def first[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "Mealy[P, DP, S, DS, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return Mealy(
            self.step.first(Proxy[tuple[C, DC]]())
            .pre(assocR(Proxy[tuple[S, DS, X, DX, C, DC]]()))
            .post(assocL(Proxy[tuple[S, DS, Y, DY, C, DC]]()))
        )

    def second[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "Mealy[P, DP, S, DS, tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return (
            self.first(Proxy[tuple[C, DC]]())
            .pre(swap(Proxy[tuple[C, DC, X, DX]]()))
            .post(swap(Proxy[tuple[Y, DY, C, DC]]()))
        )


def stateless[P, DP, S, DS, X, DX, Y, DY](
    _: Proxy[tuple[S, DS]],
    f: ParaLens[P, DP, X, DX, Y, DY],
) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    return Mealy(f.second(Proxy[tuple[S, DS]]()))


def lift[P, DP, S, DS, X, DX, Y, DY](
    _: Proxy[tuple[S, DS]],
    f: Lens[X, DX, Y, DY],
    monoid: Monoid[P, DP],
) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    return stateless(Proxy[tuple[S, DS]](), unit(f, monoid))
