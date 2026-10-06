from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import LOSS

import equinox as eqx
import jax
import jax.numpy as jnp


learning_rate: ParaLens[Unit, Unit, LOSS, LOSS, Unit, Unit] = unit(Lens(lambda l: (Unit(), lambda _: jnp.ones_like(l))))


learning_rate_log: ParaLens[Unit, Unit, LOSS, LOSS, LOSS, LOSS] = post(
    snd(Proxy[tuple[Unit, Unit, LOSS, LOSS]]()),
    pre(copy(Proxy[tuple[LOSS, LOSS]]()), first(learning_rate)),
)


def scan[P, S, Y](
    cell: ParaLens[P, P, S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[P, P, S, S, tuple[S, Y], tuple[S, Y]]:
    """Copies of cell in a row. The port and y gain a leading axis; s is handed from copy to copy."""

    def _drop[A](x: A) -> A:
        return jax.tree.map(lambda t: jax.ShapeDtypeStruct(t.shape[1:], t.dtype) if eqx.is_array(t) else t, x)

    def forward(ps_s: tuple[P, S]) -> tuple[tuple[S, Y], S]:
        ps, s = ps_s
        arr_s, static_s = eqx.partition(s, eqx.is_array)
        arr_p, static_p = eqx.partition(ps, eqx.is_array)

        def y_static(ap: P) -> Y:
            _, y = cell.arrow.get((eqx.combine(ap, static_p), s))
            _, static = eqx.partition(y, eqx.is_array)
            return static

        static_y = eqx.filter_eval_shape(y_static, _drop(arr_p))

        def step(arr_st: S, ap: P) -> tuple[S, tuple[S, Y]]:
            st = eqx.combine(arr_st, static_s)
            p = eqx.combine(ap, static_p)
            st_next, y = cell.arrow.get((p, st))
            arr_next, _ = eqx.partition(st_next, eqx.is_array)
            arr_y, _ = eqx.partition(y, eqx.is_array)
            return arr_next, (arr_st, arr_y)

        arr_final, (arr_tape, arr_ys) = jax.lax.scan(step, arr_s, arr_p)
        return (eqx.combine(arr_final, static_s), eqx.combine(arr_ys, static_y)), eqx.combine(arr_tape, static_s)

    @eqx.filter_custom_vjp
    def f(ps_s: tuple[P, S]) -> tuple[S, Y]:
        out, _ = forward(ps_s)
        return out

    @f.def_fwd
    def f_fwd(
        perturbed: tuple[P, S],
        ps_s: tuple[P, S],
    ) -> tuple[tuple[S, Y], S]:
        return forward(ps_s)

    @f.def_bwd
    def f_bwd(
        tape: S,
        ct: tuple[S, Y],
        perturbed: tuple[P, S],
        ps_s: tuple[P, S],
    ) -> tuple[P, S]:
        ps, _ = ps_s
        d_s_final, d_ys = ct
        arr_tape, static_s = eqx.partition(tape, eqx.is_array)
        arr_p, static_p = eqx.partition(ps, eqx.is_array)

        def step(d_s: S, inp: tuple[S, P, Y]) -> tuple[S, P]:
            arr_st, ap, d_y = inp
            st = eqx.combine(arr_st, static_s)
            p = eqx.combine(ap, static_p)
            d_p, d_st = cell.arrow.set((p, st), (d_s, d_y))
            return d_st, d_p

        d_s0, d_ps = jax.lax.scan(step, d_s_final, (arr_tape, arr_p, d_ys), reverse=True)
        return (d_ps, d_s0)

    return ParaLens(autodiff(f))
