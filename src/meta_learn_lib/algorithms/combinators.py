from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.utility.util import zero_cotangent_like

import equinox as eqx
import jax.numpy as jnp
import optax


def scan[S, X, Y, HP, P](cell: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:

    def _drop[A](x: A) -> A:
        return jax.tree.map(lambda t: t[0] if eqx.is_array(t) else t, x)

    def forward(z_s_xs: tuple[tuple[HP, P], tuple[S, X]]) -> tuple[tuple[S, Y], S]:
        z, (s, xs) = z_s_xs
        arr_s, static_s = eqx.partition(s, eqx.is_array)
        arr_x, static_x = eqx.partition(xs, eqx.is_array)

        def y_static(ax: X) -> Y:
            _, y = cell.arrow.arrow.get((z, (s, eqx.combine(ax, static_x))))
            _, static = eqx.partition(y, eqx.is_array)
            return static

        static_y = eqx.filter_eval_shape(y_static, _drop(arr_x))

        def step(arr_st: S, ax: X) -> tuple[S, tuple[S, Y]]:
            st = eqx.combine(arr_st, static_s)
            x = eqx.combine(ax, static_x)
            st_next, y = cell.arrow.arrow.get((z, (st, x)))
            arr_next, _ = eqx.partition(st_next, eqx.is_array)
            arr_y, _ = eqx.partition(y, eqx.is_array)
            return arr_next, (arr_st, arr_y)

        arr_final, (arr_tape, arr_ys) = jax.lax.scan(step, arr_s, arr_x)
        return (eqx.combine(arr_final, static_s), eqx.combine(arr_ys, static_y)), eqx.combine(arr_tape, static_s)

    @eqx.filter_custom_vjp
    def f(z_s_xs: tuple[tuple[HP, P], tuple[S, X]]) -> tuple[S, Y]:
        out, _ = forward(z_s_xs)
        return out

    @f.def_fwd
    def f_fwd(
        perturbed: tuple[tuple[HP, P], tuple[S, X]],
        z_s_xs: tuple[tuple[HP, P], tuple[S, X]],
    ) -> tuple[tuple[S, Y], S]:
        return forward(z_s_xs)

    @f.def_bwd
    def f_bwd(
        tape: S,
        ct: tuple[S, Y],
        perturbed: tuple[tuple[HP, P], tuple[S, X]],
        z_s_xs: tuple[tuple[HP, P], tuple[S, X]],
    ) -> tuple[tuple[HP, P], tuple[S, X]]:
        z, (_, xs) = z_s_xs
        d_s_final, d_ys = ct
        arr_tape, static_s = eqx.partition(tape, eqx.is_array)
        arr_x, static_x = eqx.partition(xs, eqx.is_array)

        def step(carry: tuple[S, tuple[HP, P]], inp: tuple[S, X, Y]) -> tuple[tuple[S, tuple[HP, P]], X]:
            d_s, d_z_acc = carry
            arr_st, ax, d_y = inp
            st = eqx.combine(arr_st, static_s)
            x = eqx.combine(ax, static_x)
            d_z, (d_st, d_x) = cell.arrow.arrow.set((z, (st, x)), (d_s, d_y))
            return (d_st, jax.tree.map(jnp.add, d_z_acc, d_z)), d_x

        (d_s0, d_z), d_xs = jax.lax.scan(
            step,
            (d_s_final, zero_cotangent_like(z)),
            (arr_tape, arr_x, d_ys),
            reverse=True,
        )
        return (d_z, (d_s0, d_xs))

    return Mealy(para_autodiff(lambda z, s_xs: f((z, s_xs))))


def batch_with_axes[S, X, Y, HP, P](
    m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P],
    from_p: Callable[[tuple[HP, P]], tuple[HP, P]],
    axes: Callable[[object], int | None] | None,
) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    def run(
        p_sx: tuple[tuple[HP, P], tuple[S, X]],
    ) -> tuple[tuple[S, Y], Callable[[tuple[S, Y]], tuple[tuple[HP, P], tuple[S, X]]]]:
        p, sx = p_sx
        s2, ys = eqx.filter_vmap(m.arrow.arrow.get, in_axes=((axes, (eqx.if_array(0), eqx.if_array(0))),))((p, sx))

        def rev(d: tuple[S, Y]) -> tuple[tuple[HP, P], tuple[S, X]]:
            dp, dsx = eqx.filter_vmap(
                m.arrow.arrow.set,
                in_axes=((axes, (eqx.if_array(0), eqx.if_array(0))), (eqx.if_array(0), eqx.if_array(0))),
            )((p, sx), d)
            return from_p(dp), dsx

        return (s2, ys), rev

    return Mealy(ParaLens(Lens(run)))


def batch_data[S, X, Y, HP, P](m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return batch_with_axes(m, lambda dp: jax.tree.map(lambda t: t.sum(0), dp), None)


def batch_pop[S, X, Y, HP, P](m: Mealy[S, S, X, X, Y, Y, HP, HP, P, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return batch_with_axes(m, lambda dp: dp, eqx.if_array(0))


def learner[HP, HPO, S, X, Y, P, H](
    machine: Mealy[S, S, X, X, Y, Y, HP, HP, P, P],
    opt: Mealy[optax.OptState, optax.OptState, P, P, P, P, HPO, HPO, H, H],
    d_out: Callable[[Y], Y],
) -> Mealy[
    tuple[S, tuple[optax.OptState, P]],
    tuple[S, tuple[optax.OptState, P]],
    X,
    X,
    tuple[Y, tuple[HP, P]],
    tuple[Y, tuple[HP, P]],
    HPO,
    HPO,
    tuple[HP, H],
    tuple[HP, H],
]:
    def step(
        hp_h: tuple[HPO, tuple[HP, H]], sv: tuple[tuple[S, tuple[optax.OptState, P]], X]
    ) -> tuple[tuple[S, tuple[optax.OptState, P]], tuple[Y, tuple[HP, P]]]:
        hp_o, (hp, lam) = hp_h
        (s_m, (opt_st, theta)), x = sv
        (s_m1, y), put = machine.arrow.arrow.run(((hp, theta), (s_m, x)))
        (_, d_theta), _ = put((zero_cotangent_like(s_m1), d_out(y)))
        _, rev_o = opt.arrow.arrow.run(((hp_o, lam), (opt_st, theta)))
        _, (opt_st1, theta1) = rev_o((opt_st, d_theta))
        return ((s_m1, (opt_st1, theta1)), (y, (hp, theta1)))

    return Mealy(para_autodiff(step))
