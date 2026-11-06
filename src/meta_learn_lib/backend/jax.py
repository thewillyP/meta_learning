from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from pyrsistent import PList

from ..category import lens as structural
from ..category.lib_types import Monoid, Proxy, Unit
from ..category.lens import Lens, assocR, swap, wiring
from ..category.mealy import Mealy
from ..category.paralens import ParaLens
from ..components.env import Env
from ..components.optimizer import Optimizer
from .record import join, restrict_keys, split, without_keys
from ..utility.util import add_cotangents, zero_cotangent_like


def filter_scan[S, X, Y](
    step: Callable[[S, X], tuple[S, Y]],
    initial: S,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
) -> tuple[S, Y]:
    arr_s, static_s = eqx.partition(initial, eqx.is_array)
    arr_xs, static_x = eqx.partition(xs, eqx.is_array)

    def metadata(s: S, x: X) -> tuple[S, Y]:
        return eqx.filter(step(s, x), eqx.is_array, inverse=True)

    static_next, static_y = eqx.filter_eval_shape(
        metadata,
        initial,
        jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype) if eqx.is_array(x) else x, xs),
    )
    if not eqx.tree_equal(static_s, static_next):
        raise ValueError("scan state metadata must stay constant; use arrays for changing state")

    def run(s: S, x: X) -> tuple[S, Y]:
        return eqx.filter(step(eqx.combine(s, static_s), eqx.combine(x, static_x)), eqx.is_array)

    final, ys = jax.lax.scan(run, arr_s, arr_xs, length, reverse)
    return eqx.combine(final, static_s), eqx.combine(ys, static_y)


def autodiff[X: PyTree, Y: PyTree](f: Callable[[X], Y]) -> Lens[X, X, Y, Y]:
    def run(x: X) -> tuple[Y, Callable[[Y], X]]:
        y, vjp = eqx.filter_vjp(f, x)

        def backward(dy: Y) -> X:
            (dx,) = vjp(dy)
            return dx

        return y, backward

    return Lens(run)


def para_autodiff[P: PyTree, X: PyTree, Y: PyTree](
    f: Callable[[P, X], Y],
    monoid: Monoid[P, P],
) -> ParaLens[P, P, X, X, Y, Y]:
    def uncurried(px: tuple[P, X]) -> Y:
        p, x = px
        return f(p, x)

    return ParaLens(autodiff(uncurried), monoid)


def mealy_autodiff[P: PyTree, S: PyTree, X: PyTree, Y: PyTree](
    f: Callable[[P, S, X], tuple[S, Y]],
    monoid: Monoid[P, P],
) -> Mealy[P, P, S, S, X, X, Y, Y]:
    def transition(p: P, sx: tuple[S, X]) -> tuple[S, Y]:
        s, x = sx
        return f(p, s, x)

    return Mealy(para_autodiff(transition, monoid))


def scan[X, DX, Y, DY](
    cell: Mealy[Env, Env, Env, Env, X, DX, Y, DY],
    to_reset: frozenset[str],
    to_scan: frozenset[str],
    length: int | None,
) -> Mealy[Env, Env, Env, Env, X, DX, Y, DY]:
    merge = cell.step.monoid.mappend

    def run(
        p_sxs: tuple[Env, tuple[Env, X]],
    ) -> tuple[tuple[Env, Y], Callable[[tuple[Env, DY]], tuple[Env, tuple[Env, DX]]]]:
        params, (state, xs) = p_sxs
        p_share, ps = without_keys(to_scan)(params), restrict_keys(to_scan)(params)

        def reset(_s: Env, _p: Env) -> Env:
            return merge(without_keys(to_reset)(_s), restrict_keys(to_reset)(_p))

        def forward(s: Env, px: tuple[Env, X]) -> tuple[Env, tuple[Env, Y]]:
            p, x = px
            p = merge(p_share, p)
            s = reset(s, p)
            s_next, y = cell.step.arrow.get((p, (s, x)))
            return s_next, (s, y)

        # first element of parameters (peel index 0) defines initial state
        first = merge(p_share, jax.tree.map(lambda a: a[0] if eqx.is_array(a) else a, ps))
        s_final, (tape, ys) = filter_scan(forward, reset(state, first), (ps, xs), length)

        def backward(ds_dys: tuple[Env, DY]) -> tuple[Env, tuple[Env, DX]]:
            ds_final, dys = ds_dys

            def reverse(
                acc: tuple[Env, Env],
                step: tuple[Env, tuple[Env, X], DY],
            ) -> tuple[tuple[Env, Env], tuple[Env, DX]]:
                ds, dp_share = acc
                s, (p, x), dy = step
                p = merge(p_share, p)
                dp, (ds_prev, dx) = cell.step.arrow.set((p, (s, x)), (ds, dy))
                dp = merge(dp, restrict_keys(to_reset)(ds_prev))
                ds_prev = merge(without_keys(to_reset)(ds_prev), zero_cotangent_like(restrict_keys(to_reset)(s)))
                return (ds_prev, merge(dp_share, without_keys(to_scan)(dp))), (restrict_keys(to_scan)(dp), dx)

            (ds_initial, dp_share), (dps, dxs) = filter_scan(
                reverse,
                (ds_final, zero_cotangent_like(p_share)),
                (tape, (ps, xs), dys),
                length,
                True,
            )
            ds_initial = merge(without_keys(to_reset)(ds_initial), zero_cotangent_like(restrict_keys(to_reset)(state)))
            return merge(dp_share, dps), (ds_initial, dxs)

        return (s_final, ys), backward

    return Mealy(ParaLens(Lens(run), cell.step.monoid))


def update_rule(
    get: Callable[[Env, Env], Env],
    update: Callable[[Env, Env, Env], Env],
    monoid: Monoid[Env, Env],
) -> ParaLens[Env, Env, Env, Env, Env, Env]:
    def run(hz: tuple[Env, Env]) -> tuple[Env, Callable[[Env], tuple[Env, Env]]]:
        h, z = hz
        return get(h, z), lambda dp: (monoid.mempty(h), update(h, z, dp))

    return ParaLens(Lens(run), monoid)


def optimised[X, DX, Y, DY](opt: Optimizer, body: ParaLens[Env, Env, X, DX, Y, DY]) -> ParaLens[Env, Env, X, DX, Y, DY]:
    monoid = body.monoid
    reparam = (
        split(opt.state, monoid)
        >> structural.copy(Proxy[tuple[Env, Env]](), monoid.mappend).second(Proxy[tuple[Env, Env]]())
        >> assocR(Proxy[tuple[Env, Env, Env, Env, Env, Env]]())
        >> swap(Proxy[tuple[Env, Env, Env, Env]]()).first(Proxy[tuple[Env, Env]]())
        >> opt.rule.arrow.first(Proxy[tuple[Env, Env]]())
        >> join(opt.targets, monoid)
    )
    return body.reparam(reparam, monoid)


def put[X, DX, Y](
    body: Mealy[Env, Env, Env, Env, X, DX, Y, Y],
    state: frozenset[str],
) -> Mealy[Env, Env, Env, Env, X, X, Y, Y]:
    monoid = body.step.monoid

    def transition(r: Env, zs: Env, x: X) -> tuple[Env, Y]:
        z, s = restrict_keys(state)(zs), without_keys(state)(zs)
        (s_next, y), backward = body.step.arrow.run((monoid.mappend(z, without_keys(state)(r)), (s, x)))
        dq, _ = backward((zero_cotangent_like(s_next), zero_cotangent_like(y)))
        return monoid.mappend(restrict_keys(state)(dq), s_next), y

    return mealy_autodiff(transition, monoid)


def learn[X, DX, Y](
    opt: Optimizer, body: Mealy[Env, Env, Env, Env, X, DX, Y, Y]
) -> Mealy[Env, Env, Env, Env, X, X, Y, Y]:
    return put(Mealy(optimised(opt, body.step)), opt.state)


def batch[X, DX, Y, DY](
    cell: Mealy[Env, Env, Env, Env, X, DX, Y, DY],
    to_map: frozenset[str],
    size: int | None,
) -> Mealy[Env, Env, Env, Env, X, DX, Y, DY]:
    merge = cell.step.monoid.mappend

    def run(
        p_sxs: tuple[Env, tuple[Env, X]],
    ) -> tuple[tuple[Env, Y], Callable[[tuple[Env, DY]], tuple[Env, tuple[Env, DX]]]]:
        params, (state, xs) = p_sxs
        p_share, ps = without_keys(to_map)(params), restrict_keys(to_map)(params)

        def forward(p: Env, s: Env, x: X) -> tuple[Env, Y]:
            return cell.step.arrow.get((merge(p_share, p), (s, x)))

        def reverse(p: Env, s: Env, x: X, ds: Env, dy: DY) -> tuple[Env, tuple[Env, DX]]:
            return cell.step.arrow.set((merge(p_share, p), (s, x)), (ds, dy))

        s_final, ys = eqx.filter_vmap(forward, axis_size=size)(ps, state, xs)

        def backward(ds_dys: tuple[Env, DY]) -> tuple[Env, tuple[Env, DX]]:
            ds_final, dys = ds_dys
            dp, ds_dxs = eqx.filter_vmap(reverse, axis_size=size)(ps, state, xs, ds_final, dys)
            dp_share = jax.tree.map(lambda a: a.sum(0), without_keys(to_map)(dp))
            return merge(dp_share, restrict_keys(to_map)(dp)), ds_dxs

        return (s_final, ys), backward

    return Mealy(ParaLens(Lens(run), cell.step.monoid))


def cut[X](_: Proxy[X]) -> Lens[X, X, X, X]:
    return wiring(lambda x: x, zero_cotangent_like)


def detach[X](_: Proxy[X]) -> Lens[X, X, X, X]:
    return wiring(lambda x: x, jax.lax.stop_gradient)


def first_order(opt: Optimizer) -> Optimizer:
    return Optimizer(ParaLens(opt.rule.arrow >> detach(Proxy[Env]()), opt.rule.monoid), opt.targets, opt.state)


def copy[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[X, DX, tuple[X, X], tuple[DX, DX]]:
    return structural.copy(Proxy[tuple[X, DX]](), add_cotangents)


def drop[X](_: Proxy[X]) -> Lens[X, X, Unit, Unit]:
    return structural.drop(Proxy[tuple[X, X]](), zero_cotangent_like)


def fst[X, DX, Z](_: Proxy[tuple[X, DX, Z]]) -> Lens[tuple[X, Z], tuple[DX, Z], X, DX]:
    return structural.fst(Proxy[tuple[X, DX, Z, Z]](), zero_cotangent_like)


def snd[Z, X, DX](_: Proxy[tuple[Z, X, DX]]) -> Lens[tuple[Z, X], tuple[Z, DX], X, DX]:
    return structural.snd(Proxy[tuple[Z, Z, X, DX]](), zero_cotangent_like)


def diagonal[X](_: Proxy[X], n: int) -> Lens[X, X, PList[X], PList[X]]:
    return structural.diagonal(Proxy[tuple[X, X]](), n, zero_cotangent_like, add_cotangents)


def learning_rate() -> Lens[jax.Array, jax.Array, Unit, Unit]:
    return Lens(lambda loss: ((), lambda _: jnp.ones_like(loss)))


def learning_rate_log() -> Lens[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        copy(Proxy[tuple[jax.Array, jax.Array]]())
        >> learning_rate().first(Proxy[tuple[jax.Array, jax.Array]]())
        >> snd(Proxy[tuple[Unit, jax.Array, jax.Array]]())
    )
