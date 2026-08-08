from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import JACOBIAN

from dataclasses import dataclass
from typing import Literal
import jax.flatten_util
import jax.numpy as jnp


@dataclass(frozen=True)
class Mealy[A1, A2, B1, B2, C1, C2, D1, D2]:
    """A Mealy machine as a parametric lens: (D, (A, B)) -> (A, C).

    Composition follows Katis-Sabadini-Walters `Circ`: the state objects
    TENSOR (each machine keeps its own memory) while the values CHAIN.
    """

    arrow: ParaLens[tuple[A1, B1], tuple[A2, B2], tuple[A1, C1], tuple[A2, C2], D1, D2]

    def __rshift__[S1, S2, T1, T2, X1, X2, Y1, Y2, Z1, Z2, P1, P2, Q1, Q2](
        f: "Mealy[S1, S2, X1, X2, Y1, Y2, P1, P2]",
        g: "Mealy[T1, T2, Y1, Y2, Z1, Z2, Q1, Q2]",
    ) -> "Mealy[tuple[S1, T1], tuple[S2, T2], X1, X2, Z1, Z2, tuple[P1, Q1], tuple[P2, Q2]]":

        preL = (
            assocR(Proxy[tuple[P1, P2, tuple[S1, T1], tuple[S2, T2], X1, X2]]())
            >> (assocR(Proxy[tuple[P1, P2, S1, S2, T1, T2]]()) @ identity(Proxy[tuple[X1, X2]]()))
            >> (swap(Proxy[tuple[tuple[P1, S1], tuple[P2, S2], T1, T2]]()) @ identity(Proxy[tuple[X1, X2]]()))
            >> assocL(Proxy[tuple[T1, T2, tuple[P1, S1], tuple[P2, S2], X1, X2]]())
            >> (identity(Proxy[tuple[T1, T2]]()) @ assocL(Proxy[tuple[P1, P2, S1, S2, X1, X2]]()))
        )
        # (T,(S,Y)) -> ((S,T),Y)
        postL = assocR(Proxy[tuple[T1, T2, S1, S2, Y1, Y2]]()) >> (
            swap(Proxy[tuple[T1, T2, S1, S2]]()) @ identity(Proxy[tuple[Y1, Y2]]())
        )

        widenL = ParaLens(preL >> (identity(Proxy[tuple[T1, T2]]()) @ f.arrow.arrow) >> postL)

        # (Q,((S,T),Y)) -> (S,(Q,(T,Y)))
        preR = (
            swap(Proxy[tuple[Q1, Q2, tuple[tuple[S1, T1], Y1], tuple[tuple[S2, T2], Y2]]]())
            >> (assocL(Proxy[tuple[S1, S2, T1, T2, Y1, Y2]]()) @ identity(Proxy[tuple[Q1, Q2]]()))
            >> assocL(Proxy[tuple[S1, S2, tuple[T1, Y1], tuple[T2, Y2], Q1, Q2]]())
            >> (identity(Proxy[tuple[S1, S2]]()) @ swap(Proxy[tuple[tuple[T1, Y1], tuple[T2, Y2], Q1, Q2]]()))
        )
        # (S,(T,Z)) -> ((S,T),Z)
        postR = assocR(Proxy[tuple[S1, S2, T1, T2, Z1, Z2]]())
        widenR = ParaLens(preR >> (identity(Proxy[tuple[S1, S2]]()) @ g.arrow.arrow) >> postR)

        return Mealy(widenL >> widenR)


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

    def f_fwd(z: Z1, s_xs: tuple[S1, Axes[X1]]) -> tuple[tuple[S1, Axes[Y1]], tuple[Z1, Seq[S1], Seq[Axes[X1]]]]:
        """Forward pass AND tape.  A bare input is scanned as a length-1 sequence."""
        s, xs = s_xs
        inner, rewrap = wrapper(xs, Proxy[Y1]())

        def step(st: S1, x: Axes[X1]) -> tuple[S1, tuple[S1, Axes[Y1]]]:
            st_next, y = cell.arrow.arrow.get((z, (st, x)))
            return st_next, (st, y)  # emit each step's INPUT state

        s_final, (states, ys) = jax.lax.scan(step, s, inner)
        return (s_final, rewrap(ys)), (z, Seq(states), Seq(inner))

    @jax.custom_vjp
    def f(z: Z1, s_xs: tuple[S1, Axes[X1]]) -> tuple[S1, Axes[Y1]]:
        return f_fwd(z, s_xs)[0]

    def f_bwd(res: tuple[Z1, Seq[S1], Seq[Axes[X1]]], ct: tuple[S2, Axes[Y2]]) -> tuple[Z2, tuple[S2, Axes[X2]]]:
        z, tape, taped_xs = res
        d_s_final, d_ys = ct
        d_inner, rewrap = wrapper(d_ys, Proxy[X2]())

        def step(carry: tuple[S2, Z2], inp: tuple[S1, Axes[X1], Axes[Y2]]) -> tuple[tuple[S2, Z2], Axes[X2]]:
            d_s, d_z_acc = carry
            st, x, d_y = inp
            d_z, (d_st, d_x) = cell.arrow.arrow.set((z, (st, x)), (d_s, d_y))
            return (d_st, jax.tree.map(jnp.add, d_z_acc, d_z)), d_x

        (d_s0, d_z), d_xs = jax.lax.scan(
            step,
            (d_s_final, jax.tree.map(jnp.zeros_like, z)),
            (tape.value, taped_xs.value, d_inner),
            reverse=True,
        )
        return d_z, (d_s0, rewrap(d_xs))

    f.defvjp(f_fwd, f_bwd)
    return Mealy(para_autodiff(f))


def rtrl[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
) -> Mealy[tuple[H1, JACOBIAN], tuple[H2, JACOBIAN], X1, X2, Y1, Y2, Z1, Z2]:

    def run(
        p_sx: tuple[Z1, tuple[tuple[H1, JACOBIAN], X1]],
    ) -> tuple[
        tuple[tuple[H1, JACOBIAN], Y1],
        Callable[[tuple[tuple[H2, JACOBIAN], Y2]], tuple[Z2, tuple[tuple[H2, JACOBIAN], X2]]],
    ]:
        z, ((h0, M0), x) = p_sx
        flat_h, unflat_h = jax.flatten_util.ravel_pytree(h0)
        flat_z, unflat_z = jax.flatten_util.ravel_pytree(z)
        (h1, y), put = machine.arrow.arrow.run((z, (h0, x)))
        ignore_y = jax.tree.map(jnp.zeros_like, y)
        ignore_x = jax.tree.map(jnp.zeros_like, x)
        ignore_z = jax.tree.map(jnp.zeros_like, z)
        jvp = jax.linear_transpose(put, (h0, y))  # v -> A v, still through `put`

        def push(d_z: Z1, d_h: jax.Array) -> jax.Array:
            ((d_h_next, _),) = jvp((d_z, (unflat_h(d_h), ignore_x)))
            d_h_next_flat, _ = jax.flatten_util.ravel_pytree(d_h_next)
            return d_h_next_flat

        def row(e: jax.Array) -> jax.Array:
            dz, _ = put((unflat_h(e), ignore_y))  # dy'=0 -> dz is a row of J_z
            dz_flat, _ = jax.flatten_util.ravel_pytree(dz)
            return dz_flat

        if flat_h.size > flat_z.size:
            M1 = jax.vmap(lambda e, col: push(unflat_z(e), col), in_axes=(0, 1), out_axes=1)(jnp.eye(flat_z.size), M0)
        else:
            J_z = jax.vmap(row)(jnp.eye(flat_h.size))
            jmp_M0 = jax.vmap(lambda col: push(ignore_z, col), in_axes=1, out_axes=1)(M0)
            M1 = jmp_M0 + J_z

        def rev(
            ct: tuple[tuple[H2, JACOBIAN], Y2],
        ) -> tuple[Z2, tuple[tuple[H2, JACOBIAN], X2]]:
            (d_h_final, _), d_y = ct
            d_z_inner, (d_h0, d_x) = put((d_h_final, d_y))
            d_h0_flat, _ = jax.flatten_util.ravel_pytree(d_h0)
            boundary = unflat_z(d_h0_flat @ M0)
            d_z = jax.tree.map(jnp.add, d_z_inner, boundary)
            zero_state = jax.tree.map(jnp.zeros_like, (d_h0, M0))
            return d_z, (zero_state, d_x)

        return ((h1, JACOBIAN(M1)), y), rev

    return Mealy(ParaLens(Lens(run)))


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
