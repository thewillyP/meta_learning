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


def batch_with_axes[P1, P2, A1, A2, B1, B2](
    f: ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], P1, P2],
    to_p: Callable[[P1], P1],
    from_p: Callable[[Batched[P2]], P2],
    axes: Literal[0] | None,
) -> ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], P1, P2]:

    def run(
        pxs: tuple[P1, Axes[A1]],
    ) -> tuple[Axes[B1], Callable[[Axes[B2]], tuple[P2, Axes[A2]]]]:
        q, xs = pxs
        match xs:
            case Batched(value=inner):
                p = to_p(q)
                ys = jax.vmap(f.arrow.get, in_axes=((axes, 0),))((p, inner))

                def rev(dys: Axes[B2]) -> tuple[P2, Axes[A2]]:
                    match dys:
                        case Batched(value=d):
                            d_ax = 0
                        case _:
                            # one cotangent for the whole batch: in_axes=None broadcasts it
                            d = dys
                            d_ax = None
                    dp, dx = jax.vmap(f.arrow.set, in_axes=((axes, 0), d_ax))((p, inner), d)
                    return from_p(Batched(dp)), Batched(dx)

                return Batched(ys), rev
            case _:
                return f.arrow.run((q, xs))

    return ParaLens(Lens(run))


def batch_data[P1, P2, A1, A2, B1, B2](
    f: ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], P1, P2],
) -> ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], P1, P2]:
    return batch_with_axes(f, lambda p: p, lambda dp: jax.tree.map(lambda t: t.sum(0), dp.value), None)


def batch_pop[P1, P2, A1, A2, B1, B2](
    f: ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], Axes[P1], Axes[P2]],
) -> ParaLens[Axes[A1], Axes[A2], Axes[B1], Axes[B2], Axes[P1], Axes[P2]]:
    def unwrap(ps: Axes[P1]) -> Axes[P1]:
        match ps:
            case Batched(value=inner):
                return inner
            case _:
                return ps

    return batch_with_axes(f, unwrap, lambda dps: dps, 0)
