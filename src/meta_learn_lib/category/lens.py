from dataclasses import dataclass
from typing import Callable
import jax
from jaxtyping import PyTree


@dataclass(frozen=True)
class Lens[X1, X2, Y1, Y2]:
    # get: Callable[[X1], Y1]
    # set: Callable[[X1, Y2], X2]
    # run is the closure equivalent of get + set. Similar to jax.vjp concept
    run: Callable[[X1], tuple[Y1, Callable[[Y2], X2]]]

    def __rshift__[A1, A2, B1, B2, C1, C2](
        f: "Lens[A1, A2, B1, B2]",
        g: "Lens[B1, B2, C1, C2]",
    ) -> "Lens[A1, A2, C1, C2]":
        def run(a: A1) -> tuple[C1, Callable[[C2], A2]]:
            b, fb = f.run(a)
            c, gb = g.run(b)
            return c, lambda c2: fb(gb(c2))

        return Lens(run)

    def __matmul__[A1, A2, B1, B2, C1, C2, D1, D2](
        f: "Lens[A1, A2, B1, B2]",
        g: "Lens[C1, C2, D1, D2]",
    ) -> "Lens[tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2]]":
        def run(ac: tuple[A1, C1]) -> tuple[tuple[B1, D1], Callable[[tuple[B2, D2]], tuple[A2, C2]]]:
            a, c = ac
            b, fb = f.run(a)
            d, gd = g.run(c)

            def rev(bd: tuple[B2, D2]) -> tuple[A2, C2]:
                b2, d2 = bd
                return fb(b2), gd(d2)

            return (b, d), rev

        return Lens(run)


@dataclass(frozen=True)
class Proxy[A]: ...


@dataclass(frozen=True)
class Unit: ...


def identity[A1, A2](_: Proxy[tuple[A1, A2]]) -> Lens[A1, A2, A1, A2]:
    return Lens(lambda x: (x, lambda y: y))


def assocL_fwd[A, B, C](t: tuple[tuple[A, B], C]) -> tuple[A, tuple[B, C]]:
    ab, c = t
    a, b = ab
    return a, (b, c)


def assocR_fwd[A, B, C](t: tuple[A, tuple[B, C]]) -> tuple[tuple[A, B], C]:
    a, bc = t
    b, c = bc
    return (a, b), c


def assocL[A1, A2, B1, B2, C1, C2](
    _: Proxy[tuple[A1, A2, B1, B2, C1, C2]],
) -> Lens[tuple[tuple[A1, B1], C1], tuple[tuple[A2, B2], C2], tuple[A1, tuple[B1, C1]], tuple[A2, tuple[B2, C2]]]:
    def run(
        t: tuple[tuple[A1, B1], C1],
    ) -> tuple[tuple[A1, tuple[B1, C1]], Callable[[tuple[A2, tuple[B2, C2]]], tuple[tuple[A2, B2], C2]]]:
        x = assocL_fwd(t)
        return x, lambda xx: assocR_fwd(xx)

    return Lens(run)


def assocR[A1, A2, B1, B2, C1, C2](
    _: Proxy[tuple[A1, A2, B1, B2, C1, C2]],
) -> Lens[tuple[A1, tuple[B1, C1]], tuple[A2, tuple[B2, C2]], tuple[tuple[A1, B1], C1], tuple[tuple[A2, B2], C2]]:
    def run(
        t: tuple[A1, tuple[B1, C1]],
    ) -> tuple[tuple[tuple[A1, B1], C1], Callable[[tuple[tuple[A2, B2], C2]], tuple[A2, tuple[B2, C2]]]]:
        x = assocR_fwd(t)
        return x, lambda xx: assocL_fwd(xx)

    return Lens(run)


def swap_fwd[A, B](t: tuple[A, B]) -> tuple[B, A]:
    a, b = t
    return b, a


def swap[A1, A2, B1, B2](
    _: Proxy[tuple[A1, A2, B1, B2]],
) -> Lens[tuple[A1, B1], tuple[A2, B2], tuple[B1, A1], tuple[B2, A2]]:
    return Lens(lambda t: (swap_fwd(t), lambda xx: swap_fwd(xx)))


def exchange_fwd[A, B, C, D](t: tuple[tuple[A, B], tuple[C, D]]) -> tuple[tuple[A, C], tuple[B, D]]:
    (a, b), (c, d) = t
    return (a, c), (b, d)


def exchange[A1, A2, B1, B2, C1, C2, D1, D2](
    _: Proxy[tuple[A1, A2, B1, B2, C1, C2, D1, D2]],
) -> Lens[
    tuple[tuple[A1, B1], tuple[C1, D1]],
    tuple[tuple[A2, B2], tuple[C2, D2]],
    tuple[tuple[A1, C1], tuple[B1, D1]],
    tuple[tuple[A2, C2], tuple[B2, D2]],
]:
    return Lens(lambda t: (exchange_fwd(t), lambda xx: exchange_fwd(xx)))


def autodiff[A: PyTree, B: PyTree](f: Callable[[A], B]) -> Lens[A, A, B, B]:
    def run(a: A) -> tuple[B, Callable[[B], A]]:
        b, vjp = jax.vjp(f, a)
        return b, lambda b2: vjp(b2)[0]  # vjp returns a 1-tuple per argument

    return Lens(run)


def snd_unit[A1, A2](_: Proxy[tuple[A1, A2]]) -> Lens[tuple[Unit, A1], tuple[Unit, A2], A1, A2]:
    def run(za: tuple[Unit, A1]) -> tuple[A1, Callable[[A2], tuple[Unit, A2]]]:
        _, a = za
        return a, lambda a2: (Unit(), a2)

    return Lens(run)
