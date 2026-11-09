from dataclasses import dataclass

from .lib_types import Monoid, Proxy
from .lens import Lens, assocL, assocR, copy, exchange, identity, snd, swap


@dataclass(frozen=True)
class ParaLens[E, P, DP, X, DX, Y, DY]:
    arrow: Lens[E, tuple[P, X], tuple[DP, DX], Y, DY]
    monoid: Monoid[P, DP]

    def then[Z, DZ](self, g: "ParaLens[E, P, DP, Y, DY, Z, DZ]") -> "ParaLens[E, P, DP, X, DX, Z, DZ]":
        return ParaLens(
            self.share(Proxy[tuple[X, DX]]())
            >> assocL(Proxy[tuple[E, P, DP, P, DP, X, DX]]())
            >> (identity(Proxy[tuple[E, P, DP]]()) @ self.arrow)
            >> g.arrow,
            self.monoid,
        )

    def tensor[U, DU, V, DV](
        self,
        g: "ParaLens[E, P, DP, U, DU, V, DV]",
    ) -> "ParaLens[E, P, DP, tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return ParaLens(
            self.share(Proxy[tuple[tuple[X, U], tuple[DX, DU]]]())
            >> exchange(Proxy[tuple[E, P, DP, P, DP, X, DX, U, DU]]())
            >> (self.arrow @ g.arrow),
            self.monoid,
        )

    def __rshift__[F, Q, DQ, A, DA, B, DB, C, DC](
        f: "ParaLens[F, Q, DQ, A, DA, B, DB]",
        g: "ParaLens[F, Q, DQ, B, DB, C, DC]",
    ) -> "ParaLens[F, Q, DQ, A, DA, C, DC]":
        return f.then(g)

    def __matmul__[F, Q, DQ, A, DA, B, DB, C, DC, D, DD](
        f: "ParaLens[F, Q, DQ, A, DA, B, DB]",
        g: "ParaLens[F, Q, DQ, C, DC, D, DD]",
    ) -> "ParaLens[F, Q, DQ, tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def reparam[Q, DQ](self, r: Lens[E, Q, DQ, P, DP], monoid: Monoid[Q, DQ]) -> "ParaLens[E, Q, DQ, X, DX, Y, DY]":
        return ParaLens(r.first(Proxy[tuple[X, DX]]()) >> self.arrow, monoid)

    def pre[W, DW](self, f: Lens[E, W, DW, X, DX]) -> "ParaLens[E, P, DP, W, DW, Y, DY]":
        return ParaLens(f.second(Proxy[tuple[P, DP]]()) >> self.arrow, self.monoid)

    def post[Z, DZ](self, g: Lens[E, Y, DY, Z, DZ]) -> "ParaLens[E, P, DP, X, DX, Z, DZ]":
        return ParaLens(self.arrow >> g, self.monoid)

    def first[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[E, P, DP, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return ParaLens(
            assocR(Proxy[tuple[E, P, DP, X, DX, C, DC]]()) >> self.arrow.first(Proxy[tuple[C, DC]]()),
            self.monoid,
        )

    def second[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[E, P, DP, tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return (
            self.first(Proxy[tuple[C, DC]]())
            .pre(swap(Proxy[tuple[E, C, DC, X, DX]]()))
            .post(swap(Proxy[tuple[E, Y, DY, C, DC]]()))
        )

    def share[W, DW](
        self, _: Proxy[tuple[W, DW]]
    ) -> Lens[E, tuple[P, W], tuple[DP, DW], tuple[tuple[P, P], W], tuple[tuple[DP, DP], DW]]:
        return copy(Proxy[tuple[E, P, DP]](), self.monoid.mappend).first(Proxy[tuple[W, DW]]())


def unit[E, P, DP, X, DX, Y, DY](f: Lens[E, X, DX, Y, DY], monoid: Monoid[P, DP]) -> ParaLens[E, P, DP, X, DX, Y, DY]:
    return ParaLens(snd(Proxy[tuple[E, P, DP, X, DX]](), monoid.mempty) >> f, monoid)
