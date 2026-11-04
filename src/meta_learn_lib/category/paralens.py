from dataclasses import dataclass
from typing import Callable

from .lib_types import Proxy
from .lens import Lens, assocL, assocR, copy, exchange, identity, snd, swap, wiring


@dataclass(frozen=True)
class ParaLens[P, DP, X, DX, Y, DY]:
    arrow: Lens[tuple[dict[str, P], X], tuple[dict[str, DP], DX], Y, DY]
    mappend: Callable[[dict[str, DP], dict[str, DP]], dict[str, DP]]

    def then[Z, DZ](self, g: "ParaLens[P, DP, Y, DY, Z, DZ]") -> "ParaLens[P, DP, X, DX, Z, DZ]":
        return ParaLens(
            self.share(Proxy[tuple[X, DX]]())
            >> assocL(Proxy[tuple[dict[str, P], dict[str, DP], dict[str, P], dict[str, DP], X, DX]]())
            >> (identity(Proxy[tuple[dict[str, P], dict[str, DP]]]()) @ self.arrow)
            >> g.arrow,
            self.mappend,
        )

    def tensor[U, DU, V, DV](
        self,
        g: "ParaLens[P, DP, U, DU, V, DV]",
    ) -> "ParaLens[P, DP, tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return ParaLens(
            self.share(Proxy[tuple[tuple[X, U], tuple[DX, DU]]]())
            >> exchange(Proxy[tuple[dict[str, P], dict[str, DP], dict[str, P], dict[str, DP], X, DX, U, DU]]())
            >> (self.arrow @ g.arrow),
            self.mappend,
        )

    def __rshift__[B, DB, C, DC, Z, DZ](
        f: "ParaLens[P, DP, B, DB, C, DC]",
        g: "ParaLens[P, DP, C, DC, Z, DZ]",
    ) -> "ParaLens[P, DP, B, DB, Z, DZ]":
        return f.then(g)

    def __matmul__[B, DB, C, DC, D, DD, E, DE](
        f: "ParaLens[P, DP, B, DB, C, DC]",
        g: "ParaLens[P, DP, D, DD, E, DE]",
    ) -> "ParaLens[P, DP, tuple[B, D], tuple[DB, DD], tuple[C, E], tuple[DC, DE]]":
        return f.tensor(g)

    def rename(self, substitution: dict[str, str]) -> "ParaLens[P, DP, X, DX, Y, DY]":
        inverse = {new: old for old, new in substitution.items()}

        def forward(p: dict[str, P]) -> dict[str, P]:
            return {inverse.get(k, k): v for k, v in p.items()}

        def backward(dp: dict[str, DP]) -> dict[str, DP]:
            return {substitution.get(k, k): v for k, v in dp.items()}

        return ParaLens(wiring(forward, backward).first(Proxy[tuple[X, DX]]()) >> self.arrow, self.mappend)

    def pre[W, DW](self, f: Lens[W, DW, X, DX]) -> "ParaLens[P, DP, W, DW, Y, DY]":
        return ParaLens(f.second(Proxy[tuple[dict[str, P], dict[str, DP]]]()) >> self.arrow, self.mappend)

    def post[Z, DZ](self, g: Lens[Y, DY, Z, DZ]) -> "ParaLens[P, DP, X, DX, Z, DZ]":
        return ParaLens(self.arrow >> g, self.mappend)

    def first[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[P, DP, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return ParaLens(
            assocR(Proxy[tuple[dict[str, P], dict[str, DP], X, DX, C, DC]]())
            >> self.arrow.first(Proxy[tuple[C, DC]]()),
            self.mappend,
        )

    def second[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "ParaLens[P, DP, tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return (
            self.first(Proxy[tuple[C, DC]]())
            .pre(swap(Proxy[tuple[C, DC, X, DX]]()))
            .post(swap(Proxy[tuple[Y, DY, C, DC]]()))
        )

    def share[W, DW](
        self, _: Proxy[tuple[W, DW]]
    ) -> Lens[
        tuple[dict[str, P], W],
        tuple[dict[str, DP], DW],
        tuple[tuple[dict[str, P], dict[str, P]], W],
        tuple[tuple[dict[str, DP], dict[str, DP]], DW],
    ]:
        return copy(Proxy[tuple[dict[str, P], dict[str, DP]]](), self.mappend).first(Proxy[tuple[W, DW]]())


def mempty[P, DP](_: dict[str, P]) -> dict[str, DP]:
    return {}


def keywise[DP](add: Callable[[DP, DP], DP]) -> Callable[[dict[str, DP], dict[str, DP]], dict[str, DP]]:
    def mappend(left: dict[str, DP], right: dict[str, DP]) -> dict[str, DP]:
        both = {name: add(left[name], right[name]) for name in left.keys() & right.keys()}
        return left | right | both

    return mappend


def unit[P, DP, X, DX, Y, DY](
    _: Proxy[P],
    f: Lens[X, DX, Y, DY],
    mappend: Callable[[dict[str, DP], dict[str, DP]], dict[str, DP]],
) -> ParaLens[P, DP, X, DX, Y, DY]:
    return ParaLens(snd(Proxy[tuple[dict[str, P], dict[str, DP], X, DX]](), mempty) >> f, mappend)
