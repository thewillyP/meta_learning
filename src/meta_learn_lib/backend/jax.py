from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from donotation import do
from jaxtyping import PyTree
from pyrsistent import PList, PVector

from ..category import lens as structural
from ..category.lib_types import Monoid, Proxy, Unit
from ..category.lens import Lens, assocR, swap, wiring
from ..category.mealy import Mealy
from ..category.paralens import ParaLens
from ..category.writer import Writer, pure
from ..components.env import HP, Env, Log, Par, St
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


def autodiff[X: PyTree, Y: PyTree](f: Callable[[X], Writer[Env, Y]]) -> Lens[Env, X, X, Y, Y]:
    def run(x: X) -> Writer[Env, tuple[Y, Callable[[Y], Writer[Env, X]]]]:
        def split_log(x: X) -> tuple[Y, PVector[Env]]:
            w = f(x)
            return w.value, w.log

        y, vjp, log = eqx.filter_vjp(split_log, x, has_aux=True)

        def backward(dy: Y) -> Writer[Env, X]:
            (dx,) = vjp(dy)
            return pure(Proxy[Env](), dx)

        return Writer((y, backward), log)

    return Lens(run)


def para_autodiff[P: PyTree, X: PyTree, Y: PyTree](
    f: Callable[[P, X], Writer[Env, Y]],
    monoid: Monoid[P, P],
) -> ParaLens[Env, P, P, X, X, Y, Y]:
    def uncurried(px: tuple[P, X]) -> Writer[Env, Y]:
        p, x = px
        return f(p, x)

    return ParaLens(autodiff(uncurried), monoid)


def mealy_autodiff[P: PyTree, S: PyTree, X: PyTree, Y: PyTree](
    f: Callable[[P, S, X], Writer[Env, tuple[S, Y]]],
    monoid: Monoid[P, P],
) -> Mealy[Env, P, P, S, S, X, X, Y, Y]:
    def transition(p: P, sx: tuple[S, X]) -> Writer[Env, tuple[S, Y]]:
        s, x = sx
        return f(p, s, x)

    return Mealy(para_autodiff(transition, monoid))


def scan[X, DX, Y, DY](
    cell: Mealy[Log, Par, Par, St, St, X, DX, Y, DY],
    to_reset: frozenset[str],
    to_scan: frozenset[str],
    length: int | None,
) -> Mealy[Log, Par, Par, St, St, X, DX, Y, DY]:
    merge = cell.step.monoid.mappend

    def run(
        p_sxs: tuple[Env, tuple[Env, X]],
    ) -> Writer[Env, tuple[tuple[Env, Y], Callable[[tuple[Env, DY]], Writer[Env, tuple[Env, tuple[Env, DX]]]]]]:
        params, (state, xs) = p_sxs
        p_share, ps = without_keys(to_scan)(params), restrict_keys(to_scan)(params)

        def reset(_s: Env, _p: Env) -> Env:
            return merge(without_keys(to_reset)(_s), restrict_keys(to_reset)(_p))

        def forward(s: Env, px: tuple[Env, X]) -> tuple[Env, tuple[Env, Y, PVector[Env]]]:
            p, x = px
            p = merge(p_share, p)
            s = reset(s, p)
            w = cell.step.arrow.get((p, (s, x)))
            s_next, y = w.value
            return s_next, (s, y, w.log)

        # first element of parameters (peel index 0) defines initial state
        first = merge(p_share, jax.tree.map(lambda a: a[0] if eqx.is_array(a) else a, ps))
        s_final, (tape, ys, logs) = filter_scan(forward, reset(state, first), (ps, xs), length)

        def backward(ds_dys: tuple[Env, DY]) -> Writer[Env, tuple[Env, tuple[Env, DX]]]:
            ds_final, dys = ds_dys

            def reverse(
                acc: tuple[Env, Env],
                step: tuple[Env, tuple[Env, X], DY],
            ) -> tuple[tuple[Env, Env], tuple[Env, DX, PVector[Env]]]:
                ds, dp_share = acc
                s, (p, x), dy = step
                p = merge(p_share, p)
                w = cell.step.arrow.set((p, (s, x)), (ds, dy))
                dp, (ds_prev, dx) = w.value
                dp = merge(dp, restrict_keys(to_reset)(ds_prev))
                ds_prev = merge(without_keys(to_reset)(ds_prev), zero_cotangent_like(restrict_keys(to_reset)(s)))
                return (ds_prev, merge(dp_share, without_keys(to_scan)(dp))), (restrict_keys(to_scan)(dp), dx, w.log)

            (ds_initial, dp_share), (dps, dxs, blogs) = filter_scan(
                reverse,
                (ds_final, zero_cotangent_like(p_share)),
                (tape, (ps, xs), dys),
                length,
                True,
            )
            ds_initial = merge(without_keys(to_reset)(ds_initial), zero_cotangent_like(restrict_keys(to_reset)(state)))
            return Writer((merge(dp_share, dps), (ds_initial, dxs)), blogs)

        return Writer(((s_final, ys), backward), logs)

    return Mealy(ParaLens(Lens(run), cell.step.monoid))


def optimizer_reparam(
    get: Callable[[HP, Par], Par],
    update: Callable[[HP, Par, Par], Writer[Log, Par]],
    monoid: Monoid[HP, HP],
) -> ParaLens[Log, HP, HP, Par, Par, Par, Par]:
    def run(hz: tuple[HP, Par]) -> Writer[Log, tuple[Par, Callable[[Par], Writer[Log, tuple[HP, Par]]]]]:
        h, z = hz

        @do()
        def backward(dp: Par):
            z_next = yield from update(h, z, dp)
            return pure(Proxy[Log](), (monoid.mempty(h), z_next))

        return pure(Proxy[Log](), (get(h, z), backward))

    return ParaLens(Lens(run), monoid)


def learn[X, DX, Y](
    opt: Optimizer,
    body: Mealy[Log, Par, Par, St, St, X, DX, Y, Y],
) -> Mealy[Log, HP, HP, Par, Par, X, X, Y, Y]:
    monoid = body.step.monoid
    reparam = (
        split(opt.state, monoid)
        >> structural.copy(Proxy[tuple[Env, Env, Env]](), monoid.mappend).second(Proxy[tuple[Env, Env]]())
        >> assocR(Proxy[tuple[Env, Env, Env, Env, Env, Env, Env]]())
        >> swap(Proxy[tuple[Env, Env, Env, Env, Env]]()).first(Proxy[tuple[Env, Env]]())
        >> opt.reparam.arrow.first(Proxy[tuple[Env, Env]]())
        >> join(opt.targets, monoid)
    )
    optimised = body.step.reparam(reparam, monoid)

    @do()
    def transition(r: HP, zs: Par, x: X):
        z, s = restrict_keys(opt.state)(zs), without_keys(opt.state)(zs)
        (s_next, y), backward = yield from optimised.arrow.run((monoid.mappend(z, without_keys(opt.state)(r)), (s, x)))
        dq, _ = yield from backward((zero_cotangent_like(s_next), zero_cotangent_like(y)))
        zs_next = monoid.mappend(restrict_keys(opt.state)(dq), s_next)
        return pure(Proxy[Log](), (zs_next, y))

    return mealy_autodiff(transition, monoid)


def batch[X, DX, Y, DY](
    cell: Mealy[Log, Par, Par, St, St, X, DX, Y, DY],
    to_map: frozenset[str],
    size: int | None,
) -> Mealy[Log, Par, Par, St, St, X, DX, Y, DY]:
    merge = cell.step.monoid.mappend

    def run(
        p_sxs: tuple[Env, tuple[Env, X]],
    ) -> Writer[Env, tuple[tuple[Env, Y], Callable[[tuple[Env, DY]], Writer[Env, tuple[Env, tuple[Env, DX]]]]]]:
        params, (state, xs) = p_sxs
        p_share, ps = without_keys(to_map)(params), restrict_keys(to_map)(params)

        def forward(p: Env, s: Env, x: X) -> Writer[Env, tuple[Env, Y]]:
            return cell.step.arrow.get((merge(p_share, p), (s, x)))

        def reverse(p: Env, s: Env, x: X, ds: Env, dy: DY) -> Writer[Env, tuple[Env, tuple[Env, DX]]]:
            return cell.step.arrow.set((merge(p_share, p), (s, x)), (ds, dy))

        w = eqx.filter_vmap(forward, axis_size=size)(ps, state, xs)

        def backward(ds_dys: tuple[Env, DY]) -> Writer[Env, tuple[Env, tuple[Env, DX]]]:
            ds_final, dys = ds_dys
            bw = eqx.filter_vmap(reverse, axis_size=size)(ps, state, xs, ds_final, dys)
            dp, ds_dxs = bw.value
            dp_share = jax.tree.map(lambda a: a.sum(0), without_keys(to_map)(dp))
            return Writer((merge(dp_share, restrict_keys(to_map)(dp)), ds_dxs), bw.log)

        return Writer((w.value, backward), w.log)

    return Mealy(ParaLens(Lens(run), cell.step.monoid))


def cut[X](_: Proxy[X]) -> Lens[Env, X, X, X, X]:
    return wiring(Proxy[Env](), lambda x: x, zero_cotangent_like)


def detach[X](_: Proxy[X]) -> Lens[Env, X, X, X, X]:
    return wiring(Proxy[Env](), lambda x: x, jax.lax.stop_gradient)


def copy[X, DX](_: Proxy[tuple[X, DX]]) -> Lens[Env, X, DX, tuple[X, X], tuple[DX, DX]]:
    return structural.copy(Proxy[tuple[Env, X, DX]](), add_cotangents)


def drop[X](_: Proxy[X]) -> Lens[Env, X, X, Unit, Unit]:
    return structural.drop(Proxy[tuple[Env, X, X]](), zero_cotangent_like)


def fst[X, DX, Z](_: Proxy[tuple[X, DX, Z]]) -> Lens[Env, tuple[X, Z], tuple[DX, Z], X, DX]:
    return structural.fst(Proxy[tuple[Env, X, DX, Z, Z]](), zero_cotangent_like)


def snd[Z, X, DX](_: Proxy[tuple[Z, X, DX]]) -> Lens[Env, tuple[Z, X], tuple[Z, DX], X, DX]:
    return structural.snd(Proxy[tuple[Env, Z, Z, X, DX]](), zero_cotangent_like)


def diagonal[X](_: Proxy[X], n: int) -> Lens[Env, X, X, PList[X], PList[X]]:
    return structural.diagonal(Proxy[tuple[Env, X, X]](), n, zero_cotangent_like, add_cotangents)


def learning_rate() -> Lens[Env, jax.Array, jax.Array, Unit, Unit]:
    return Lens(lambda loss: pure(Proxy[Env](), ((), lambda _: pure(Proxy[Env](), jnp.ones_like(loss)))))


def learning_rate_log() -> Lens[Env, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        copy(Proxy[tuple[jax.Array, jax.Array]]())
        >> learning_rate().first(Proxy[tuple[jax.Array, jax.Array]]())
        >> snd(Proxy[tuple[Unit, jax.Array, jax.Array]]())
    )
