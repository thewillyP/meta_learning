from dataclasses import dataclass
from functools import reduce
from itertools import repeat
from typing import Callable

from pyrsistent import PList, plist

from .lib_types import Proxy, Unit


@dataclass(frozen=True)
class Lens[X, DX, Y, DY]:
    run: Callable[[X], tuple[Y, Callable[[DY], DX]]]

    @property
    def get(self) -> Callable[[X], Y]:
        def _get(x: X) -> Y:
            y, _ = self.run(x)
            return y

        return _get

    @property
    def set(self) -> Callable[[X, DY], DX]:
        def _set(x: X, dy: DY) -> DX:
            _, backward = self.run(x)
            return backward(dy)

        return _set

    def then[Z, DZ](self, g: "Lens[Y, DY, Z, DZ]") -> "Lens[X, DX, Z, DZ]":
        def run(x: X) -> tuple[Z, Callable[[DZ], DX]]:
            y, backward_f = self.run(x)
            z, backward_g = g.run(y)

            return z, lambda dz: backward_f(backward_g(dz))

        return Lens(run)

    def tensor[U, DU, V, DV](
        self,
        g: "Lens[U, DU, V, DV]",
    ) -> "Lens[tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        def run(xu: tuple[X, U]) -> tuple[tuple[Y, V], Callable[[tuple[DY, DV]], tuple[DX, DU]]]:
            x, u = xu
            y, backward_f = self.run(x)
            v, backward_g = g.run(u)

            def backward(dydv: tuple[DY, DV]) -> tuple[DX, DU]:
                dy, dv = dydv
                return backward_f(dy), backward_g(dv)

            return (y, v), backward

        return Lens(run)

    def __rshift__[A, DA, B, DB, C, DC](
        f: "Lens[A, DA, B, DB]",
        g: "Lens[B, DB, C, DC]",
    ) -> "Lens[A, DA, C, DC]":
        return f.then(g)

    def __matmul__[A, DA, B, DB, C, DC, D, DD](
        f: "Lens[A, DA, B, DB]",
        g: "Lens[C, DC, D, DD]",
    ) -> "Lens[tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def pre[W, DW](self, f: "Lens[W, DW, X, DX]") -> "Lens[W, DW, Y, DY]":
        return f >> self

    def post[Z, DZ](self, g: "Lens[Y, DY, Z, DZ]") -> "Lens[X, DX, Z, DZ]":
        return self >> g

    def first[C, DC](self, _: Proxy[tuple[C, DC]]) -> "Lens[tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return self @ identity(Proxy[tuple[C, DC]]())

    def second[C, DC](self, _: Proxy[tuple[C, DC]]) -> "Lens[tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return identity(Proxy[tuple[C, DC]]()) @ self


def wiring[X, DX, Y, DY](forward: Callable[[X], Y], backward: Callable[[DY], DX]) -> Lens[X, DX, Y, DY]:
    return Lens(lambda x: (forward(x), backward))


def identity[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[X, DX, X, DX]:
    return wiring(lambda x: x, lambda dx: dx)


def swap_fwd[X, Y](xy: tuple[X, Y]) -> tuple[Y, X]:
    x, y = xy
    return y, x


def swap[X, DX, Y, DY](_: Proxy[tuple[X, DX, Y, DY]]) -> Lens[tuple[X, Y], tuple[DX, DY], tuple[Y, X], tuple[DY, DX]]:
    return wiring(swap_fwd, swap_fwd)


def assocL_fwd[X, Y, Z](xyz: tuple[tuple[X, Y], Z]) -> tuple[X, tuple[Y, Z]]:
    xy, z = xyz
    x, y = xy
    return x, (y, z)


def assocR_fwd[X, Y, Z](xyz: tuple[X, tuple[Y, Z]]) -> tuple[tuple[X, Y], Z]:
    x, yz = xyz
    y, z = yz
    return (x, y), z


def assocL[X, DX, Y, DY, Z, DZ](
    _: Proxy[tuple[X, DX, Y, DY, Z, DZ]],
) -> Lens[
    tuple[tuple[X, Y], Z],
    tuple[tuple[DX, DY], DZ],
    tuple[X, tuple[Y, Z]],
    tuple[DX, tuple[DY, DZ]],
]:
    return wiring(assocL_fwd, assocR_fwd)


def assocR[X, DX, Y, DY, Z, DZ](
    _: Proxy[tuple[X, DX, Y, DY, Z, DZ]],
) -> Lens[
    tuple[X, tuple[Y, Z]],
    tuple[DX, tuple[DY, DZ]],
    tuple[tuple[X, Y], Z],
    tuple[tuple[DX, DY], DZ],
]:
    return wiring(assocR_fwd, assocL_fwd)


def exchange_fwd[X, Y, Z, W](xyzw: tuple[tuple[X, Y], tuple[Z, W]]) -> tuple[tuple[X, Z], tuple[Y, W]]:
    xy, zw = xyzw
    x, y = xy
    z, w = zw
    return (x, z), (y, w)


def exchange[X, DX, Y, DY, Z, DZ, W, DW](
    _: Proxy[tuple[X, DX, Y, DY, Z, DZ, W, DW]],
) -> Lens[
    tuple[tuple[X, Y], tuple[Z, W]],
    tuple[tuple[DX, DY], tuple[DZ, DW]],
    tuple[tuple[X, Z], tuple[Y, W]],
    tuple[tuple[DX, DZ], tuple[DY, DW]],
]:
    return wiring(exchange_fwd, exchange_fwd)


def fst[X, DX, Z, DZ](
    _: Proxy[tuple[X, DX, Z, DZ]],
    zero: Callable[[Z], DZ],
) -> Lens[tuple[X, Z], tuple[DX, DZ], X, DX]:
    return swap(Proxy[tuple[X, DX, Z, DZ]]()) >> snd(Proxy[tuple[Z, DZ, X, DX]](), zero)


def snd[Z, DZ, X, DX](
    _: Proxy[tuple[Z, DZ, X, DX]],
    zero: Callable[[Z], DZ],
) -> Lens[tuple[Z, X], tuple[DZ, DX], X, DX]:
    def run(zx: tuple[Z, X]) -> tuple[X, Callable[[DX], tuple[DZ, DX]]]:
        z, x = zx
        return x, lambda dx: (zero(z), dx)

    return Lens(run)


def unit_intro[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[X, DX, tuple[Unit, X], tuple[Unit, DX]]:
    def backward(unit_dx: tuple[Unit, DX]) -> DX:
        _, dx = unit_dx
        return dx

    return wiring(lambda x: ((), x), backward)


def with_unit[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[X, DX, tuple[X, Unit], tuple[DX, Unit]]:
    return unit_intro(Proxy[tuple[X, DX]]()) >> swap(Proxy[tuple[Unit, Unit, X, DX]]())


def unit_elim[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[tuple[X, Unit], tuple[DX, Unit], X, DX]:
    return swap(Proxy[tuple[X, DX, Unit, Unit]]()) >> snd(Proxy[tuple[Unit, Unit, X, DX]](), lambda _: ())


def drop[X, DX](_: Proxy[tuple[X, DX]], zero: Callable[[X], DX]) -> Lens[X, DX, Unit, Unit]:
    return (
        unit_intro(Proxy[tuple[X, DX]]())
        >> swap(Proxy[tuple[Unit, Unit, X, DX]]())
        >> snd(Proxy[tuple[X, DX, Unit, Unit]](), zero)
    )


def copy[X, DX](_: Proxy[tuple[X, DX]], add: Callable[[DX, DX], DX]) -> Lens[X, DX, tuple[X, X], tuple[DX, DX]]:
    def backward(dxs: tuple[DX, DX]) -> DX:
        dx1, dx2 = dxs
        return add(dx1, dx2)

    return wiring(lambda x: (x, x), backward)


def diagonal[X, DX](
    _: Proxy[tuple[X, DX]],
    n: int,
    zero: Callable[[X], DX],
    add: Callable[[DX, DX], DX],
) -> Lens[X, DX, PList[X], PList[DX]]:
    def run(x: X) -> tuple[PList[X], Callable[[PList[DX]], DX]]:
        return plist(repeat(x, n)), lambda dxs: reduce(add, dxs, zero(x))

    return Lens(run)


def sequence[X, DX, Y, DY](local: Lens[X, DX, Y, DY]) -> Lens[PList[X], PList[DX], PList[Y], PList[DY]]:
    def run(xs: PList[X]) -> tuple[PList[Y], Callable[[PList[DY]], PList[DX]]]:
        executed = plist(map(local.run, xs))
        ys = plist(y for y, _ in executed)
        backs = plist(back for _, back in executed)

        def backward(dys: PList[DY]) -> PList[DX]:
            return plist(back(dy) for back, dy in zip(backs, dys, strict=True))

        return ys, backward

    return Lens(run)
