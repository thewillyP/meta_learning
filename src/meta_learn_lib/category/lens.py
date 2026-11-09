from dataclasses import dataclass
from functools import reduce
from itertools import repeat
from typing import Callable, Generator
from donotation import do
from pyrsistent import PList, plist

from . import writer
from .lib_types import Proxy, Unit
from .writer import Writer, pure


@dataclass(frozen=True)
class Lens[E, X, DX, Y, DY]:
    run: Callable[[X], Writer[E, tuple[Y, Callable[[DY], Writer[E, DX]]]]]

    @property
    def get(self) -> Callable[[X], Writer[E, Y]]:
        @do()
        def _get(x: X):
            y, _ = yield from self.run(x)
            return pure(Proxy[E](), y)

        return _get

    @property
    def set(self) -> Callable[[X, DY], Writer[E, DX]]:
        def _set(x: X, dy: DY) -> Writer[E, DX]:
            _, backward = self.run(x).value
            return backward(dy)

        return _set

    def then[Z, DZ](self, g: "Lens[E, Y, DY, Z, DZ]") -> "Lens[E, X, DX, Z, DZ]":
        @do()
        def run(x: X) -> Generator[None, None, Writer[E, tuple[Z, Callable[[DZ], Writer[E, DX]]]]]:
            y, backward_f = yield from self.run(x)
            z, backward_g = yield from g.run(y)

            return pure(Proxy[E](), (z, lambda dz: backward_g(dz).flat_map(backward_f)))

        return Lens(run)

    def tensor[U, DU, V, DV](
        self,
        g: "Lens[E, U, DU, V, DV]",
    ) -> "Lens[E, tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        @do()
        def run(xu: tuple[X, U]):
            x, u = xu
            y, backward_f = yield from self.run(x)
            v, backward_g = yield from g.run(u)

            @do()
            def backward(dydv: tuple[DY, DV]):
                dy, dv = dydv
                dx = yield from backward_f(dy)
                du = yield from backward_g(dv)
                return pure(Proxy[E](), (dx, du))

            return pure(Proxy[E](), ((y, v), backward))

        return Lens(run)

    def __rshift__[F, A, DA, B, DB, C, DC](
        f: "Lens[F, A, DA, B, DB]",
        g: "Lens[F, B, DB, C, DC]",
    ) -> "Lens[F, A, DA, C, DC]":
        return f.then(g)

    def __matmul__[F, A, DA, B, DB, C, DC, D, DD](
        f: "Lens[F, A, DA, B, DB]",
        g: "Lens[F, C, DC, D, DD]",
    ) -> "Lens[F, tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def pre[W, DW](self, f: "Lens[E, W, DW, X, DX]") -> "Lens[E, W, DW, Y, DY]":
        return f >> self

    def post[Z, DZ](self, g: "Lens[E, Y, DY, Z, DZ]") -> "Lens[E, X, DX, Z, DZ]":
        return self >> g

    def first[C, DC](self, _: Proxy[tuple[C, DC]]) -> "Lens[E, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return self @ identity(Proxy[tuple[E, C, DC]]())

    def second[C, DC](
        self,
        _: Proxy[tuple[C, DC]],
    ) -> "Lens[E, tuple[C, X], tuple[DC, DX], tuple[C, Y], tuple[DC, DY]]":
        return identity(Proxy[tuple[E, C, DC]]()) @ self


def wiring[E, X, DX, Y, DY](
    _: Proxy[E],
    forward: Callable[[X], Y],
    backward: Callable[[DY], DX],
) -> Lens[E, X, DX, Y, DY]:
    return Lens(lambda x: pure(_, (forward(x), lambda dy: pure(_, backward(dy)))))


def identity[E, X, DX](_: Proxy[tuple[E, X, DX]]) -> Lens[E, X, DX, X, DX]:
    return wiring(Proxy[E](), lambda x: x, lambda dx: dx)


def swap_fwd[X, Y](xy: tuple[X, Y]) -> tuple[Y, X]:
    x, y = xy
    return y, x


def swap[E, X, DX, Y, DY](
    _: Proxy[tuple[E, X, DX, Y, DY]],
) -> Lens[E, tuple[X, Y], tuple[DX, DY], tuple[Y, X], tuple[DY, DX]]:
    return wiring(Proxy[E](), swap_fwd, swap_fwd)


def assocL_fwd[X, Y, Z](xyz: tuple[tuple[X, Y], Z]) -> tuple[X, tuple[Y, Z]]:
    xy, z = xyz
    x, y = xy
    return x, (y, z)


def assocR_fwd[X, Y, Z](xyz: tuple[X, tuple[Y, Z]]) -> tuple[tuple[X, Y], Z]:
    x, yz = xyz
    y, z = yz
    return (x, y), z


def assocL[E, X, DX, Y, DY, Z, DZ](
    _: Proxy[tuple[E, X, DX, Y, DY, Z, DZ]],
) -> Lens[
    E,
    tuple[tuple[X, Y], Z],
    tuple[tuple[DX, DY], DZ],
    tuple[X, tuple[Y, Z]],
    tuple[DX, tuple[DY, DZ]],
]:
    return wiring(Proxy[E](), assocL_fwd, assocR_fwd)


def assocR[E, X, DX, Y, DY, Z, DZ](
    _: Proxy[tuple[E, X, DX, Y, DY, Z, DZ]],
) -> Lens[
    E,
    tuple[X, tuple[Y, Z]],
    tuple[DX, tuple[DY, DZ]],
    tuple[tuple[X, Y], Z],
    tuple[tuple[DX, DY], DZ],
]:
    return wiring(Proxy[E](), assocR_fwd, assocL_fwd)


def exchange_fwd[X, Y, Z, W](xyzw: tuple[tuple[X, Y], tuple[Z, W]]) -> tuple[tuple[X, Z], tuple[Y, W]]:
    xy, zw = xyzw
    x, y = xy
    z, w = zw
    return (x, z), (y, w)


def exchange[E, X, DX, Y, DY, Z, DZ, W, DW](
    _: Proxy[tuple[E, X, DX, Y, DY, Z, DZ, W, DW]],
) -> Lens[
    E,
    tuple[tuple[X, Y], tuple[Z, W]],
    tuple[tuple[DX, DY], tuple[DZ, DW]],
    tuple[tuple[X, Z], tuple[Y, W]],
    tuple[tuple[DX, DZ], tuple[DY, DW]],
]:
    return wiring(Proxy[E](), exchange_fwd, exchange_fwd)


def fst[E, X, DX, Z, DZ](
    _: Proxy[tuple[E, X, DX, Z, DZ]],
    zero: Callable[[Z], DZ],
) -> Lens[E, tuple[X, Z], tuple[DX, DZ], X, DX]:
    return swap(Proxy[tuple[E, X, DX, Z, DZ]]()) >> snd(Proxy[tuple[E, Z, DZ, X, DX]](), zero)


def snd[E, Z, DZ, X, DX](
    _: Proxy[tuple[E, Z, DZ, X, DX]],
    zero: Callable[[Z], DZ],
) -> Lens[E, tuple[Z, X], tuple[DZ, DX], X, DX]:
    def run(zx: tuple[Z, X]) -> Writer[E, tuple[X, Callable[[DX], Writer[E, tuple[DZ, DX]]]]]:
        z, x = zx
        return pure(Proxy[E](), (x, lambda dx: pure(Proxy[E](), (zero(z), dx))))

    return Lens(run)


def unit_intro[E, X, DX](_: Proxy[tuple[E, X, DX]]) -> Lens[E, X, DX, tuple[Unit, X], tuple[Unit, DX]]:
    def backward(unit_dx: tuple[Unit, DX]) -> DX:
        _, dx = unit_dx
        return dx

    return wiring(Proxy[E](), lambda x: ((), x), backward)


def with_unit[E, X, DX](_: Proxy[tuple[E, X, DX]]) -> Lens[E, X, DX, tuple[X, Unit], tuple[DX, Unit]]:
    return unit_intro(Proxy[tuple[E, X, DX]]()) >> swap(Proxy[tuple[E, Unit, Unit, X, DX]]())


def unit_elim[E, X, DX](_: Proxy[tuple[E, X, DX]]) -> Lens[E, tuple[X, Unit], tuple[DX, Unit], X, DX]:
    return swap(Proxy[tuple[E, X, DX, Unit, Unit]]()) >> snd(Proxy[tuple[E, Unit, Unit, X, DX]](), lambda _: ())


def drop[E, X, DX](_: Proxy[tuple[E, X, DX]], zero: Callable[[X], DX]) -> Lens[E, X, DX, Unit, Unit]:
    return (
        unit_intro(Proxy[tuple[E, X, DX]]())
        >> swap(Proxy[tuple[E, Unit, Unit, X, DX]]())
        >> snd(Proxy[tuple[E, X, DX, Unit, Unit]](), zero)
    )


def copy[E, X, DX](
    _: Proxy[tuple[E, X, DX]],
    add: Callable[[DX, DX], DX],
) -> Lens[E, X, DX, tuple[X, X], tuple[DX, DX]]:
    def backward(dxs: tuple[DX, DX]) -> DX:
        dx1, dx2 = dxs
        return add(dx1, dx2)

    return wiring(Proxy[E](), lambda x: (x, x), backward)


def diagonal[E, X, DX](
    _: Proxy[tuple[E, X, DX]],
    n: int,
    zero: Callable[[X], DX],
    add: Callable[[DX, DX], DX],
) -> Lens[E, X, DX, PList[X], PList[DX]]:
    def run(x: X) -> Writer[E, tuple[PList[X], Callable[[PList[DX]], Writer[E, DX]]]]:
        return pure(Proxy[E](), (plist(repeat(x, n)), lambda dxs: pure(Proxy[E](), reduce(add, dxs, zero(x)))))

    return Lens(run)


def sequence[E, X, DX, Y, DY](local: Lens[E, X, DX, Y, DY]) -> Lens[E, PList[X], PList[DX], PList[Y], PList[DY]]:
    @do()
    def run(xs: PList[X]):
        executed = yield from writer.sequence(plist(map(local.run, xs)))
        ys = plist(y for y, _ in executed)
        backs = plist(back for _, back in executed)

        def backward(dys: PList[DY]) -> Writer[E, PList[DX]]:
            return writer.sequence(plist(back(dy) for back, dy in zip(backs, dys, strict=True)))

        return pure(Proxy[E](), (ys, backward))

    return Lens(run)
