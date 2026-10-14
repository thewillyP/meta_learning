from dataclasses import dataclass

from .lib_types import Proxy, Unit
from .lens import Lens, assocL, assocR, exchange, identity, swap, unit_elim


@dataclass(frozen=True)
class ParaLens[P, DP, X, DX, Y, DY]:
    arrow: Lens[tuple[P, X], tuple[DP, DX], Y, DY]

    def then[Q, DQ, Z, DZ](
        self,
        g: "ParaLens[Q, DQ, Y, DY, Z, DZ]",
    ) -> "ParaLens[tuple[P, Q], tuple[DP, DQ], X, DX, Z, DZ]":
        return ParaLens((identity(Proxy[tuple[Q, DQ]]()) @ self.arrow) >> g.arrow).join()

    def tensor[Q, DQ, U, DU, V, DV](
        self,
        g: "ParaLens[Q, DQ, U, DU, V, DV]",
    ) -> "ParaLens[tuple[P, Q], tuple[DP, DQ], tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return ParaLens(exchange(Proxy[tuple[P, DP, Q, DQ, X, DX, U, DU]]()) >> (self.arrow @ g.arrow))

    def __rshift__[PF, DPF, PG, DPG, A, DA, B, DB, C, DC](
        f: "ParaLens[PF, DPF, A, DA, B, DB]",
        g: "ParaLens[PG, DPG, B, DB, C, DC]",
    ) -> "ParaLens[tuple[PF, PG], tuple[DPF, DPG], A, DA, C, DC]":
        return f.then(g)

    def __matmul__[PF, DPF, PG, DPG, A, DA, B, DB, C, DC, D, DD](
        f: "ParaLens[PF, DPF, A, DA, B, DB]",
        g: "ParaLens[PG, DPG, C, DC, D, DD]",
    ) -> "ParaLens[tuple[PF, PG], tuple[DPF, DPG], tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def reparam[Q, DQ](self, r: Lens[Q, DQ, P, DP]) -> "ParaLens[Q, DQ, X, DX, Y, DY]":
        return ParaLens((r @ identity(Proxy[tuple[X, DX]]())) >> self.arrow)

    def pre[W, DW](self, f: Lens[W, DW, X, DX]) -> "ParaLens[P, DP, W, DW, Y, DY]":
        return ParaLens((identity(Proxy[tuple[P, DP]]()) @ f) >> self.arrow)

    def post[Z, DZ](self, g: Lens[Y, DY, Z, DZ]) -> "ParaLens[P, DP, X, DX, Z, DZ]":
        return ParaLens(self.arrow >> g)

    def first[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[P, DP, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return ParaLens(assocR(Proxy[tuple[P, DP, X, DX, C, DC]]()) >> self.arrow.first(Proxy[tuple[C, DC]]()))

    def second[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[P, DP, tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return (
            self.first(Proxy[tuple[C, DC]]())
            .pre(swap(Proxy[tuple[C, DC, X, DX]]()))
            .post(swap(Proxy[tuple[Y, DY, C, DC]]()))
        )

    def join[Q, DQ, A, DA](
        self: "ParaLens[P, DP, tuple[Q, A], tuple[DQ, DA], Y, DY]",
    ) -> "ParaLens[tuple[Q, P], tuple[DQ, DP], A, DA, Y, DY]":
        return ParaLens(
            (swap(Proxy[tuple[Q, DQ, P, DP]]()) @ identity(Proxy[tuple[A, DA]]()))
            >> assocL(Proxy[tuple[P, DP, Q, DQ, A, DA]]())
            >> self.arrow
        )

    def unjoin[Q, DQ, R, DR](
        self: "ParaLens[tuple[Q, R], tuple[DQ, DR], X, DX, Y, DY]",
    ) -> "ParaLens[R, DR, tuple[Q, X], tuple[DQ, DX], Y, DY]":
        return ParaLens(
            assocR(Proxy[tuple[R, DR, Q, DQ, X, DX]]())
            >> (swap(Proxy[tuple[R, DR, Q, DQ]]()) @ identity(Proxy[tuple[X, DX]]()))
            >> self.arrow
        )


def unit[X, DX, Y, DY](f: Lens[X, DX, Y, DY]) -> ParaLens[Unit, Unit, X, DX, Y, DY]:
    return ParaLens(swap(Proxy[tuple[Unit, Unit, X, DX]]()) >> unit_elim(Proxy[tuple[X, DX]]()) >> f)
