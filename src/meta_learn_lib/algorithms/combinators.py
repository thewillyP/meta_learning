from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy

from typing import Literal
import equinox as eqx
import jax.numpy as jnp

from meta_learn_lib.utility.util import zero_cotangent_like


def scan[S1, S2, X1, X2, Y1, Y2, Z1, Z2](
    cell: Mealy[S1, S2, Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], Z1, Z2],
) -> Mealy[S1, S2, Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], Z1, Z2]:

    def _lift[A](x: A) -> A:
        return jax.tree.map(lambda t: t[None], x)

    def _drop[A](x: A) -> A:
        return jax.tree.map(lambda t: t[0], x)

    def wrapper[A, B](a: Axes[A], _: Proxy[B]) -> tuple[Axes[A], Callable[[Axes[B]], Axes[B]]]:
        match a:
            case Seq(value=inner):
                return inner, Seq
            case _ as bare:
                return _lift(bare), _drop

    def forward(z_s_xs: tuple[Z1, tuple[S1, Axes[X1]]]) -> tuple[tuple[S1, Axes[Y1]], Seq[S1]]:
        """Forward pass AND tape.  A bare input is scanned as a length-1 sequence."""
        z, (s, xs) = z_s_xs
        inner, rewrap = wrapper(xs, Proxy[Y1]())

        def step(st: S1, x: Axes[X1]) -> tuple[S1, tuple[S1, Axes[Y1]]]:
            st_next, y = cell.arrow.arrow.get((z, (st, x)))
            return st_next, (st, y)  # emit each step's INPUT state

        s_final, (states, ys) = jax.lax.scan(step, s, inner)
        return (s_final, rewrap(ys)), Seq(states)

    @eqx.filter_custom_vjp
    def f(z_s_xs: tuple[Z1, tuple[S1, Axes[X1]]]) -> tuple[S1, Axes[Y1]]:
        out, _ = forward(z_s_xs)
        return out

    @f.def_fwd
    def f_fwd(
        perturbed: tuple[Z1, tuple[S1, Axes[X1]]],
        z_s_xs: tuple[Z1, tuple[S1, Axes[X1]]],
    ) -> tuple[tuple[S1, Axes[Y1]], Seq[S1]]:
        return forward(z_s_xs)

    @f.def_bwd
    def f_bwd(
        tape: Seq[S1],
        ct: tuple[S2, Axes[Y2]],
        perturbed: tuple[Z1, tuple[S1, Axes[X1]]],
        z_s_xs: tuple[Z1, tuple[S1, Axes[X1]]],
    ) -> tuple[Z2, tuple[S2, Axes[X2]]]:
        z, (_, xs) = z_s_xs
        d_s_final, d_ys = ct
        xs_value, _ = wrapper(xs, Proxy[Y2]())
        d_ys_value, rewrap = wrapper(d_ys, Proxy[X2]())

        def step(carry: tuple[S2, Z2], inp: tuple[S1, Axes[X1], Axes[Y2]]) -> tuple[tuple[S2, Z2], Axes[X2]]:
            d_s, d_z_acc = carry
            st, x, d_y = inp
            d_z, (d_st, d_x) = cell.arrow.arrow.set((z, (st, x)), (d_s, d_y))
            return (d_st, jax.tree.map(jnp.add, d_z_acc, d_z)), d_x

        (d_s0, d_z), d_xs = jax.lax.scan(
            step,
            (d_s_final, zero_cotangent_like(z)),
            (tape.value, xs_value, d_ys_value),
            reverse=True,
        )
        return (d_z, (d_s0, rewrap(d_xs)))

    return Mealy(para_autodiff(lambda z, s_xs: f((z, s_xs))))


def unbatch[Z](x: Axes[Z]) -> tuple[Axes[Z], Literal[0] | None]:
    match x:
        case Batched(value=v):
            return v, 0
        case _:
            # one cotangent for the whole batch: in_axes=None broadcasts it
            return x, None


def batch_with_axes[S1, S2, X1, X2, Y1, Y2, P1, P2](
    m: Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], P1, P2],
    to_p: Callable[[P1], P1],
    from_p: Callable[[Batched[P2]], P2],
    axes: Literal[0] | None,
) -> Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], P1, P2]:
    def run(
        p_sx: tuple[P1, tuple[Axes[S1], Axes[X1]]],
    ) -> tuple[
        tuple[Axes[S1], Axes[Y1]],
        Callable[[tuple[Axes[S2], Axes[Y2]]], tuple[P2, tuple[Axes[S2], Axes[X2]]]],
    ]:
        q, (ss, xs) = p_sx
        match ss, xs:
            case Batched(value=s_in), Batched(value=x_in):
                p = to_p(q)
                s2, ys = jax.vmap(m.arrow.arrow.get, in_axes=((axes, (0, 0)),))((p, (s_in, x_in)))

                def rev(d: tuple[Axes[S2], Axes[Y2]]) -> tuple[P2, tuple[Axes[S2], Axes[X2]]]:
                    d_s, d_y = d

                    ds, ds_ax = unbatch(d_s)
                    dy, dy_ax = unbatch(d_y)
                    dp, (dss, dxs) = jax.vmap(m.arrow.arrow.set, in_axes=((axes, (0, 0)), (ds_ax, dy_ax)))(
                        (p, (s_in, x_in)), (ds, dy)
                    )
                    return from_p(Batched(dp)), (Batched(dss), Batched(dxs))

                return (Batched(s2), Batched(ys)), rev
            case _:
                return m.arrow.arrow.run(p_sx)

    return Mealy(ParaLens(Lens(run)))


def batch_data[S1, S2, X1, X2, Y1, Y2, P1, P2](
    m: Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], P1, P2],
) -> Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], P1, P2]:
    return batch_with_axes(m, lambda p: p, lambda dp: jax.tree.map(lambda t: t.sum(0), dp.value), None)


def batch_pop[S1, S2, X1, X2, Y1, Y2, P1, P2](
    m: Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], Axes[P1], Axes[P2]],
) -> Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], Axes[P1], Axes[P2]]:
    return batch_with_axes(m, lambda p: unbatch(p)[0], lambda dps: dps, 0)
