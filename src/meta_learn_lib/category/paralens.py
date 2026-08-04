from dataclasses import dataclass
from typing import Literal
from meta_learn_lib.category.lens import *
from meta_learn_lib.category.lib_types import *


@dataclass(frozen=True)
class ParaLens[X1, X2, Y1, Y2, Z1, Z2]:
    arrow: Lens[tuple[Z1, X1], tuple[Z2, X2], Y1, Y2]

    def __rshift__[P1, P2, Q1, Q2, A1, A2, B1, B2, C1, C2](
        f: "ParaLens[A1, A2, B1, B2, P1, P2]",
        g: "ParaLens[B1, B2, C1, C2, Q1, Q2]",
    ) -> "ParaLens[A1, A2, C1, C2, tuple[P1, Q1], tuple[P2, Q2]]":

        return ParaLens(
            (swap(Proxy[tuple[P1, P2, Q1, Q2]]()) @ identity(Proxy[tuple[A1, A2]]()))
            >> assocL(Proxy[tuple[Q1, Q2, P1, P2, A1, A2]]())
            >> (identity(Proxy[tuple[Q1, Q2]]()) @ f.arrow)
            >> g.arrow
        )

    def __matmul__[P1, P2, Q1, Q2, A1, A2, B1, B2, C1, C2, D1, D2](
        f: "ParaLens[A1, A2, B1, B2, P1, P2]",
        g: "ParaLens[C1, C2, D1, D2, Q1, Q2]",
    ) -> "ParaLens[tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2], tuple[P1, Q1], tuple[P2, Q2]]":
        return ParaLens(exchange(Proxy[tuple[P1, P2, Q1, Q2, A1, A2, C1, C2]]()) >> (f.arrow @ g.arrow))


def to_paralens[A1, A2, B1, B2](f: Lens[A1, A2, B1, B2]) -> ParaLens[A1, A2, B1, B2, Unit, Unit]:
    return ParaLens(snd(Proxy[tuple[Unit, Unit, A1, A2]]()) >> f)


def para_autodiff[P, X, Y](f: Callable[[P, X], Y]) -> ParaLens[X, X, Y, Y, P, P]:
    def uncurried(zx: tuple[P, X]) -> Y:
        z, x = zx
        return f(z, x)

    return ParaLens(autodiff(uncurried))


def batch_with_axes[P1, P2, A1, A2, B1, B2, Q1, Q2](
    f: ParaLens[A1, A2, B1, B2, P1, P2],
    to_p: Callable[[Q1], P1],
    from_p: Callable[[Batched[P2]], Q2],
    axes: Literal[0] | None,
) -> ParaLens[Batched[A1], Batched[A2], Batched[B1], Batched[B2], Q1, Q2]:
    def run(pxs: tuple[Q1, Batched[A1]]) -> tuple[Batched[B1], Callable[[Batched[B2]], tuple[Q2, Batched[A2]]]]:
        q, xs = pxs
        p = to_p(q)
        ys = jax.vmap(f.arrow.get, in_axes=((axes, 0),))((p, xs.value))

        def rev(dys: Batched[B2]) -> tuple[Q2, Batched[A2]]:
            dp, dx = jax.vmap(f.arrow.set, in_axes=((axes, 0), 0))((p, xs.value), dys.value)
            return from_p(Batched(dp)), Batched(dx)

        return Batched(ys), rev

    return ParaLens(Lens(run))


def batch_data[P1, P2, A1, A2, B1, B2](
    f: ParaLens[A1, A2, B1, B2, P1, P2],
) -> ParaLens[Batched[A1], Batched[A2], Batched[B1], Batched[B2], P1, P2]:
    return batch_with_axes(f, lambda p: p, lambda dp: jax.tree.map(lambda t: t.sum(0), dp.value), None)


def batch_pop[P1, P2, A1, A2, B1, B2](
    f: ParaLens[A1, A2, B1, B2, P1, P2],
) -> ParaLens[Batched[A1], Batched[A2], Batched[B1], Batched[B2], Batched[P1], Batched[P2]]:
    return batch_with_axes(f, lambda ps: ps.value, lambda dps: dps, 0)
