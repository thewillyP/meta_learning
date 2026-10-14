from dataclasses import dataclass

from .lib_types import Proxy, Unit
from .lens import Lens, assocL, assocR, exchange, identity, swap
from .paralens import ParaLens, unit


@dataclass(frozen=True)
class Mealy[P, DP, S, DS, X, DX, Y, DY]:
    step: ParaLens[P, DP, tuple[S, X], tuple[DS, DX], tuple[S, Y], tuple[DS, DY]]

    def then[Q, DQ, T, DT, Z, DZ](
        self,
        g: "Mealy[Q, DQ, T, DT, Y, DY, Z, DZ]",
    ) -> "Mealy[tuple[P, Q], tuple[DP, DQ], tuple[S, T], tuple[DS, DT], X, DX, Z, DZ]":
        return Mealy(
            self.step.second(Proxy[tuple[T, DT]]())
            .pre(
                (swap(Proxy[tuple[S, DS, T, DT]]()) @ identity(Proxy[tuple[X, DX]]()))
                >> assocL(Proxy[tuple[T, DT, S, DS, X, DX]]())
            )
            .post(
                assocR(Proxy[tuple[T, DT, S, DS, Y, DY]]())
                >> (swap(Proxy[tuple[T, DT, S, DS]]()) @ identity(Proxy[tuple[Y, DY]]()))
                >> assocL(Proxy[tuple[S, DS, T, DT, Y, DY]]())
            )
            .then(g.step.second(Proxy[tuple[S, DS]]()))
            .post(assocR(Proxy[tuple[S, DS, T, DT, Z, DZ]]()))
        )

    def tensor[Q, DQ, T, DT, U, DU, V, DV](
        self,
        g: "Mealy[Q, DQ, T, DT, U, DU, V, DV]",
    ) -> "Mealy[tuple[P, Q], tuple[DP, DQ], tuple[S, T], tuple[DS, DT], tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return Mealy(
            (self.step @ g.step)
            .pre(exchange(Proxy[tuple[S, DS, T, DT, X, DX, U, DU]]()))
            .post(exchange(Proxy[tuple[S, DS, Y, DY, T, DT, V, DV]]()))
        )

    def __rshift__[PF, DPF, PG, DPG, SF, DSF, SG, DSG, A, DA, B, DB, C, DC](
        f: "Mealy[PF, DPF, SF, DSF, A, DA, B, DB]",
        g: "Mealy[PG, DPG, SG, DSG, B, DB, C, DC]",
    ) -> "Mealy[tuple[PF, PG], tuple[DPF, DPG], tuple[SF, SG], tuple[DSF, DSG], A, DA, C, DC]":
        return f.then(g)

    def __matmul__[PF, DPF, PG, DPG, SF, DSF, SG, DSG, A, DA, B, DB, C, DC, D, DD](
        f: "Mealy[PF, DPF, SF, DSF, A, DA, B, DB]",
        g: "Mealy[PG, DPG, SG, DSG, C, DC, D, DD]",
    ) -> "Mealy[tuple[PF, PG], tuple[DPF, DPG], tuple[SF, SG], tuple[DSF, DSG], tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def reparam[Q, DQ](self, r: Lens[Q, DQ, P, DP]) -> "Mealy[Q, DQ, S, DS, X, DX, Y, DY]":
        return Mealy(self.step.reparam(r))

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


def stateless[P, DP, X, DX, Y, DY](f: ParaLens[P, DP, X, DX, Y, DY]) -> Mealy[P, DP, Unit, Unit, X, DX, Y, DY]:
    return Mealy(f.second(Proxy[tuple[Unit, Unit]]()))


def lift[X, DX, Y, DY](f: Lens[X, DX, Y, DY]) -> Mealy[Unit, Unit, Unit, Unit, X, DX, Y, DY]:
    return stateless(unit(f))
