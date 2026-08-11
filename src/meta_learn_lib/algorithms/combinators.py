from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.utility.util import zero_cotangent_like

import equinox as eqx
import jax.numpy as jnp
import optax


def scan[S, X, Y, Z](
    cell: Mealy[S, S, Axes[X], Axes[X], Axes[Y], Axes[Y], Z, Z],
) -> Mealy[S, S, Axes[X], Axes[X], Axes[Y], Axes[Y], Z, Z]:

    def _lift[A](x: A) -> A:
        return jax.tree.map(lambda t: t[None] if eqx.is_array(t) else t, x)

    def _drop[A](x: A) -> A:
        return jax.tree.map(lambda t: t[0] if eqx.is_array(t) else t, x)

    def wrapper[A, B](a: Axes[A], _: Proxy[B]) -> tuple[Axes[A], Callable[[Axes[B]], Axes[B]]]:
        match a:
            case Seq(value=inner):
                return inner, Seq
            case _ as bare:
                return _lift(bare), _drop

    def forward(z_s_xs: tuple[Z, tuple[S, Axes[X]]]) -> tuple[tuple[S, Axes[Y]], Seq[S]]:
        z, (s, xs) = z_s_xs
        inner, rewrap = wrapper(xs, Proxy[Y]())
        arr_s, static_s = eqx.partition(s, eqx.is_array)
        arr_x, static_x = eqx.partition(inner, eqx.is_array)

        def y_static(ax: X) -> Y:
            _, y = cell.arrow.arrow.get((z, (s, eqx.combine(ax, static_x))))
            _, static = eqx.partition(y, eqx.is_array)
            return static

        static_y = eqx.filter_eval_shape(y_static, _drop(arr_x))

        def step(arr_st: S, ax: Axes[X]) -> tuple[S, tuple[S, Axes[Y]]]:
            st = eqx.combine(arr_st, static_s)
            x = eqx.combine(ax, static_x)
            st_next, y = cell.arrow.arrow.get((z, (st, x)))
            arr_next, _ = eqx.partition(st_next, eqx.is_array)
            arr_y, _ = eqx.partition(y, eqx.is_array)
            return arr_next, (arr_st, arr_y)

        arr_final, (arr_tape, arr_ys) = jax.lax.scan(step, arr_s, arr_x)
        return (
            (
                eqx.combine(arr_final, static_s),
                rewrap(eqx.combine(arr_ys, static_y)),
            ),
            Seq(eqx.combine(arr_tape, static_s)),
        )

    @eqx.filter_custom_vjp
    def f(z_s_xs: tuple[Z, tuple[S, Axes[X]]]) -> tuple[S, Axes[Y]]:
        out, _ = forward(z_s_xs)
        return out

    @f.def_fwd
    def f_fwd(
        perturbed: tuple[Z, tuple[S, Axes[X]]],
        z_s_xs: tuple[Z, tuple[S, Axes[X]]],
    ) -> tuple[tuple[S, Axes[Y]], Seq[S]]:
        return forward(z_s_xs)

    @f.def_bwd
    def f_bwd(
        tape: Seq[S],
        ct: tuple[S, Axes[Y]],
        perturbed: tuple[Z, tuple[S, Axes[X]]],
        z_s_xs: tuple[Z, tuple[S, Axes[X]]],
    ) -> tuple[Z, tuple[S, Axes[X]]]:
        z, (_, xs) = z_s_xs
        d_s_final, d_ys = ct
        xs_value, _ = wrapper(xs, Proxy[Y]())
        d_ys_value, rewrap = wrapper(d_ys, Proxy[X]())
        arr_tape, static_s = eqx.partition(tape.value, eqx.is_array)
        arr_x, static_x = eqx.partition(xs_value, eqx.is_array)

        def step(carry: tuple[S, Z], inp: tuple[S, Axes[X], Axes[Y]]) -> tuple[tuple[S, Z], Axes[X]]:
            d_s, d_z_acc = carry
            arr_st, ax, d_y = inp
            st = eqx.combine(arr_st, static_s)
            x = eqx.combine(ax, static_x)
            d_z, (d_st, d_x) = cell.arrow.arrow.set((z, (st, x)), (d_s, d_y))
            return (d_st, jax.tree.map(jnp.add, d_z_acc, d_z)), d_x

        (d_s0, d_z), d_xs = jax.lax.scan(
            step,
            (d_s_final, zero_cotangent_like(z)),
            (arr_tape, arr_x, d_ys_value),
            reverse=True,
        )
        return (d_z, (d_s0, rewrap(d_xs)))

    return Mealy(para_autodiff(lambda z, s_xs: f((z, s_xs))))


def unbatch[Z](x: Axes[Z]) -> tuple[Axes[Z], Callable[[object], int | None] | None]:
    match x:
        case Batched(value=v):
            return v, eqx.if_array(0)
        case _:
            # one cotangent for the whole batch: in_axes=None broadcasts it
            return x, None


def batch_with_axes[S1, S2, X1, X2, Y1, Y2, P1, P2](
    m: Mealy[Axes[S1], Axes[S2], Axes[X1], Axes[X2], Axes[Y1], Axes[Y2], P1, P2],
    to_p: Callable[[P1], P1],
    from_p: Callable[[Batched[P2]], P2],
    axes: Callable[[object], int | None] | None,
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
                s2, ys = eqx.filter_vmap(
                    m.arrow.arrow.get,
                    in_axes=((axes, (eqx.if_array(0), eqx.if_array(0))),),
                )((p, (s_in, x_in)))

                def rev(d: tuple[Axes[S2], Axes[Y2]]) -> tuple[P2, tuple[Axes[S2], Axes[X2]]]:
                    d_s, d_y = d

                    ds, ds_ax = unbatch(d_s)
                    dy, dy_ax = unbatch(d_y)
                    dp, (dss, dxs) = eqx.filter_vmap(
                        m.arrow.arrow.set,
                        in_axes=((axes, (eqx.if_array(0), eqx.if_array(0))), (ds_ax, dy_ax)),
                    )((p, (s_in, x_in)), (ds, dy))
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
    return batch_with_axes(m, lambda p: unbatch(p)[0], lambda dps: dps, eqx.if_array(0))


def learner[R, S, X, Y, Z: optax.Params, H](
    machine: Mealy[S, S, X, X, Y, Y, tuple[R, Z], tuple[R, Z]],
    opt: Mealy[optax.OptState, optax.OptState, Z, Z, Z, Z, H, H],
    d_out: Callable[[Y], Y],
) -> Mealy[
    tuple[S, tuple[optax.OptState, Z]],
    tuple[S, tuple[optax.OptState, Z]],
    X,
    X,
    Y,
    Y,
    tuple[R, H],
    tuple[R, H],
]:
    def step(
        rh: tuple[R, H], sv: tuple[tuple[S, tuple[optax.OptState, Z]], X]
    ) -> tuple[tuple[S, tuple[optax.OptState, Z]], Y]:
        r, lam = rh
        (s_m, (opt_st, theta)), x = sv
        (s_m1, y), put = machine.arrow.arrow.run(((r, theta), (s_m, x)))
        (_, d_theta), _ = put((zero_cotangent_like(s_m1), d_out(y)))
        _, rev_o = opt.arrow.arrow.run((lam, (opt_st, theta)))
        _, (opt_st1, theta1) = rev_o((opt_st, d_theta))
        return ((s_m1, (opt_st1, theta1)), y)

    return Mealy(para_autodiff(step))
