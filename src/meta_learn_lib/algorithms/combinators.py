from typing import Callable

from ..category.lib_types import Proxy, Unit
from ..category.lens import Lens
from ..category.mealy import Mealy
from ..category.paralens import ParaLens
from ..jax_backend import mealy_autodiff, zero_cotangent_like


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
