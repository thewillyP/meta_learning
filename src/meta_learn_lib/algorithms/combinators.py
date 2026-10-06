from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import LOSS
from meta_learn_lib.utility.util import zero_cotangent_like

import equinox as eqx
import jax
import jax.numpy as jnp


learning_rate: ParaLens[Unit, Unit, LOSS, LOSS, Unit, Unit] = unit(Lens(lambda l: (Unit(), lambda _: jnp.ones_like(l))))


learning_rate_log: ParaLens[Unit, Unit, LOSS, LOSS, LOSS, LOSS] = post(
    snd(Proxy[tuple[Unit, Unit, LOSS, LOSS]]()),
    pre(copy(Proxy[tuple[LOSS, LOSS]]()), first(learning_rate)),
)


def mapAccum[Z, P, S, Y](
    cell: ParaLens[tuple[Z, P], tuple[Z, P], S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[tuple[Z, P], tuple[Z, P], S, S, tuple[S, Y], tuple[S, Y]]:
    """Copies of cell in a row. z is shared by every copy; ps and y gain a leading axis; s is handed from copy to copy."""

    def _drop[A](x: A) -> A:
        return jax.tree.map(lambda t: jax.ShapeDtypeStruct(t.shape[1:], t.dtype) if eqx.is_array(t) else t, x)

    def forward(zps_s: tuple[tuple[Z, P], S]) -> tuple[tuple[S, Y], S]:
        (z, ps), s = zps_s
        arr_s, static_s = eqx.partition(s, eqx.is_array)
        arr_p, static_p = eqx.partition(ps, eqx.is_array)

        def y_static(ap: P) -> Y:
            _, y = cell.arrow.get(((z, eqx.combine(ap, static_p)), s))
            _, static = eqx.partition(y, eqx.is_array)
            return static

        static_y = eqx.filter_eval_shape(y_static, _drop(arr_p))

        def step(arr_st: S, ap: P) -> tuple[S, tuple[S, Y]]:
            st = eqx.combine(arr_st, static_s)
            p = eqx.combine(ap, static_p)
            st_next, y = cell.arrow.get(((z, p), st))
            arr_next, _ = eqx.partition(st_next, eqx.is_array)
            arr_y, _ = eqx.partition(y, eqx.is_array)
            return arr_next, (arr_st, arr_y)

        arr_final, (arr_tape, arr_ys) = jax.lax.scan(step, arr_s, arr_p)
        return (eqx.combine(arr_final, static_s), eqx.combine(arr_ys, static_y)), eqx.combine(arr_tape, static_s)

    @eqx.filter_custom_vjp
    def f(zps_s: tuple[tuple[Z, P], S]) -> tuple[S, Y]:
        out, _ = forward(zps_s)
        return out

    @f.def_fwd
    def f_fwd(
        perturbed: tuple[tuple[Z, P], S],
        zps_s: tuple[tuple[Z, P], S],
    ) -> tuple[tuple[S, Y], S]:
        return forward(zps_s)

    @f.def_bwd
    def f_bwd(
        tape: S,
        ct: tuple[S, Y],
        perturbed: tuple[tuple[Z, P], S],
        zps_s: tuple[tuple[Z, P], S],
    ) -> tuple[tuple[Z, P], S]:
        (z, ps), _ = zps_s
        d_s_final, d_ys = ct
        arr_tape, static_s = eqx.partition(tape, eqx.is_array)
        arr_p, static_p = eqx.partition(ps, eqx.is_array)

        def step(carry: tuple[S, Z], inp: tuple[S, P, Y]) -> tuple[tuple[S, Z], P]:
            d_s, d_z_acc = carry
            arr_st, ap, d_y = inp
            st = eqx.combine(arr_st, static_s)
            p = eqx.combine(ap, static_p)
            (d_z, d_p), d_st = cell.arrow.set(((z, p), st), (d_s, d_y))
            return (d_st, jax.tree.map(jnp.add, d_z_acc, d_z)), d_p

        (d_s0, d_z), d_ps = jax.lax.scan(
            step,
            (d_s_final, zero_cotangent_like(z)),
            (arr_tape, arr_p, d_ys),
            reverse=True,
        )
        return ((d_z, d_ps), d_s0)

    return ParaLens(autodiff(f))


def scan[P, S, Y](
    cell: ParaLens[P, P, S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[P, P, S, S, tuple[S, Y], tuple[S, Y]]:
    """Copies of cell in a row with nothing shared: mapAccum with an empty shared part."""
    return reparam(
        unit_intro(Proxy[tuple[P, P]]()),
        mapAccum(reparam(snd(Proxy[tuple[Unit, Unit, P, P]]()), cell)),
    )


def batch[Z, P, A, B](
    cell: ParaLens[tuple[Z, P], tuple[Z, P], A, A, B, B],
) -> ParaLens[tuple[Z, P], tuple[Z, P], A, A, B, B]:
    """Copies of cell side by side. z is shared by every copy; ps, the input and the output gain a leading axis."""

    def run(zps_a: tuple[tuple[Z, P], A]) -> tuple[B, Callable[[B], tuple[tuple[Z, P], A]]]:
        zps, a = zps_a
        b = eqx.filter_vmap(cell.arrow.get, in_axes=(((None, eqx.if_array(0)), eqx.if_array(0)),))((zps, a))

        def rev(d: B) -> tuple[tuple[Z, P], A]:
            (d_z, d_ps), d_a = eqx.filter_vmap(
                cell.arrow.set,
                in_axes=(((None, eqx.if_array(0)), eqx.if_array(0)), eqx.if_array(0)),
            )((zps, a), d)
            return (jax.tree.map(lambda t: t.sum(0), d_z), d_ps), d_a

        return b, rev

    return ParaLens(Lens(run))


def optimised[K, SO, T, D, A, B](
    opt: ParaLens[K, K, tuple[SO, T], tuple[SO, T], T, T],
    body: ParaLens[tuple[T, D], tuple[T, D], A, A, B, B],
) -> ParaLens[tuple[tuple[SO, T], tuple[K, D]], tuple[tuple[SO, T], tuple[K, D]], A, A, B, B]:
    """The body with the optimiser plugged on the theta part of its port."""
    return join(first(opt) >> unjoin(body))


def put[Sigma, D, S, Y](
    body: ParaLens[tuple[Sigma, D], tuple[Sigma, D], S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[D, D, tuple[Sigma, S], tuple[Sigma, S], tuple[tuple[Sigma, S], Y], tuple[tuple[Sigma, S], Y]]:
    """Run forwards, run backwards, hand on what came back on sigma. Sigma leaves the port and becomes a wire."""

    def step(d: D, sigma_s: tuple[Sigma, S]) -> tuple[tuple[Sigma, S], Y]:
        sigma, s = sigma_s
        (s1, y), rev = body.arrow.run(((sigma, d), s))
        (sigma1, _), _ = rev((zero_cotangent_like(s1), zero_cotangent_like(y)))
        return (sigma1, s1), y

    return para_autodiff(step)


def learn[K, SO, T, D, S, Y](
    opt: ParaLens[K, K, tuple[SO, T], tuple[SO, T], T, T],
    body: ParaLens[tuple[T, D], tuple[T, D], S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[
    tuple[K, D],
    tuple[K, D],
    tuple[tuple[SO, T], S],
    tuple[tuple[SO, T], S],
    tuple[tuple[tuple[SO, T], S], Y],
    tuple[tuple[tuple[SO, T], S], Y],
]:
    return put(optimised(opt, body))
