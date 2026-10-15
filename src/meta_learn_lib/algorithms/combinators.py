from typing import Callable

import equinox as eqx
import jax

from ..category.lib_types import Proxy, Unit
from ..category.lens import Lens, swap, unit_intro
from ..category.mealy import Mealy
from ..category.paralens import ParaLens
from ..jax_backend import add_cotangents, filter_scan, fst, mealy_autodiff, snd, zero_cotangent_like


def update_rule[H, Z, P, DP](
    get: Callable[[H, Z], P],
    update: Callable[[H, Z, DP], Z],
) -> ParaLens[H, Unit, Z, Z, P, DP]:
    def run(hz: tuple[H, Z]) -> tuple[P, Callable[[DP], tuple[Unit, Z]]]:
        h, z = hz
        return get(h, z), lambda dp: ((), update(h, z, dp))

    return ParaLens(Lens(run))


def optimised[H, DH, Z, P, DP, R, DR, X, DX, Y, DY](
    opt: ParaLens[H, DH, Z, Z, P, DP],
    body: ParaLens[tuple[P, R], tuple[DP, DR], X, DX, Y, DY],
) -> ParaLens[tuple[Z, tuple[H, R]], tuple[Z, tuple[DH, DR]], X, DX, Y, DY]:
    return (opt.first(Proxy[tuple[X, DX]]()) >> body.unjoin()).join()


def put[Z, R, DR, S, X, DX, Y](
    body: Mealy[tuple[Z, R], tuple[Z, DR], S, S, X, DX, Y, Y],
) -> Mealy[R, R, tuple[Z, S], tuple[Z, S], X, X, Y, Y]:
    def transition(r: R, zs: tuple[Z, S], x: X) -> tuple[tuple[Z, S], Y]:
        z, s = zs
        (s_next, y), backward = body.step.arrow.run(((z, r), (s, x)))
        (z_next, _), _ = backward((zero_cotangent_like(s_next), zero_cotangent_like(y)))
        return (z_next, s_next), y

    return mealy_autodiff(transition)


def learn[H, DH, Z, P, DP, R, DR, S, X, DX, Y](
    opt: ParaLens[H, DH, Z, Z, P, DP],
    body: Mealy[tuple[P, R], tuple[DP, DR], S, S, X, DX, Y, Y],
) -> Mealy[tuple[H, R], tuple[H, R], tuple[Z, S], tuple[Z, S], X, X, Y, Y]:
    return put(Mealy(optimised(opt, body.step)))


def mapAccum[Z, P, DP, S, DS, X, DX, Y, DY](
    cell: Mealy[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY],
    length: int | None = None,
) -> Mealy[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY]:
    def run(
        zps_sxs: tuple[tuple[Z, P], tuple[S, X]],
    ) -> tuple[tuple[S, Y], Callable[[tuple[DS, DY]], tuple[tuple[Z, DP], tuple[DS, DX]]]]:
        (z, ps), (s, xs) = zps_sxs

        def forward(s: S, px: tuple[P, X]) -> tuple[S, tuple[S, Y]]:
            p, x = px
            s_next, y = cell.step.arrow.get(((z, p), (s, x)))
            return s_next, (s, y)

        s_final, (tape, ys) = filter_scan(forward, s, (ps, xs), length)

        def backward(ds_dys: tuple[DS, DY]) -> tuple[tuple[Z, DP], tuple[DS, DX]]:
            ds_final, dys = ds_dys

            def reverse(
                carry: tuple[DS, Z],
                step: tuple[S, tuple[P, X], DY],
            ) -> tuple[tuple[DS, Z], tuple[DP, DX]]:
                ds, dz = carry
                s, (p, x), dy = step
                (dz_step, dp), (ds_previous, dx) = cell.step.arrow.set(((z, p), (s, x)), (ds, dy))
                return (ds_previous, add_cotangents(dz, dz_step)), (dp, dx)

            (ds_initial, dz), (dps, dxs) = filter_scan(
                reverse,
                (ds_final, zero_cotangent_like(z)),
                (tape, (ps, xs), dys),
                length,
                True,
            )
            return (dz, dps), (ds_initial, dxs)

        return (s_final, ys), backward

    return Mealy(ParaLens(Lens(run)))


def scan[P, DP, S, DS, X, DX, Y, DY](
    cell: Mealy[P, DP, S, DS, X, DX, Y, DY],
    length: int | None = None,
) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    return mapAccum(cell.reparam(snd(Proxy[tuple[Unit, P, DP]]())), length).reparam(unit_intro(Proxy[tuple[P, DP]]()))


def scan_shared[P, S, DS, X, DX, Y, DY](
    cell: Mealy[P, P, S, DS, X, DX, Y, DY],
    length: int | None = None,
) -> Mealy[P, P, S, DS, X, DX, Y, DY]:
    return mapAccum(cell.reparam(fst(Proxy[tuple[P, P, Unit]]())), length).reparam(
        unit_intro(Proxy[tuple[P, P]]()) >> swap(Proxy[tuple[Unit, Unit, P, P]]())
    )


def batch[Z, P, DP, S, DS, X, DX, Y, DY](
    cell: Mealy[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY],
    size: int | None = None,
) -> Mealy[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY]:
    def run(
        zps_sxs: tuple[tuple[Z, P], tuple[S, X]],
    ) -> tuple[tuple[S, Y], Callable[[tuple[DS, DY]], tuple[tuple[Z, DP], tuple[DS, DX]]]]:
        result = eqx.filter_vmap(
            cell.step.arrow.get,
            in_axes=(((None, eqx.if_array(0)), (eqx.if_array(0), eqx.if_array(0))),),
            axis_size=size,
        )(zps_sxs)

        def backward(ds_dys: tuple[DS, DY]) -> tuple[tuple[Z, DP], tuple[DS, DX]]:
            (dzs, dps), ds_dxs = eqx.filter_vmap(
                cell.step.arrow.set,
                in_axes=(
                    ((None, eqx.if_array(0)), (eqx.if_array(0), eqx.if_array(0))),
                    (eqx.if_array(0), eqx.if_array(0)),
                ),
                axis_size=size,
            )(zps_sxs, ds_dys)
            return (jax.tree.map(lambda dz: dz.sum(0), dzs), dps), ds_dxs

        return result, backward

    return Mealy(ParaLens(Lens(run)))


def batch_data[P, S, DS, X, DX, Y, DY](
    cell: Mealy[P, P, S, DS, X, DX, Y, DY],
    size: int | None = None,
) -> Mealy[P, P, S, DS, X, DX, Y, DY]:
    return batch(cell.reparam(fst(Proxy[tuple[P, P, Unit]]())), size).reparam(
        unit_intro(Proxy[tuple[P, P]]()) >> swap(Proxy[tuple[Unit, Unit, P, P]]())
    )


def batch_pop[P, DP, S, DS, X, DX, Y, DY](
    cell: Mealy[P, DP, S, DS, X, DX, Y, DY],
    size: int | None = None,
) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    return batch(cell.reparam(snd(Proxy[tuple[Unit, P, DP]]())), size).reparam(unit_intro(Proxy[tuple[P, DP]]()))
