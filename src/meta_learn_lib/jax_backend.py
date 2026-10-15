from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from pyrsistent import PList

from .category import lens as structural
from .category.lib_types import Proxy, Unit
from .category.lens import Lens
from .category.mealy import Mealy
from .category.paralens import ParaLens


def zero_cotangent_like[T](tree: T) -> T:
    return jax.tree.map(jnp.zeros_like, eqx.filter(tree, eqx.is_inexact_array))


def add_cotangents[T](a: T, b: T) -> T:
    return eqx.combine(
        jax.tree.map(
            jnp.add,
            eqx.filter(a, eqx.is_inexact_array),
            eqx.filter(b, eqx.is_inexact_array),
        ),
        eqx.filter(a, eqx.is_inexact_array, inverse=True),
    )


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
) -> ParaLens[P, P, X, X, Y, Y]:
    def uncurried(px: tuple[P, X]) -> Y:
        p, x = px
        return f(p, x)

    return ParaLens(autodiff(uncurried))


def mealy_autodiff[P: PyTree, S: PyTree, X: PyTree, Y: PyTree](
    f: Callable[[P, S, X], tuple[S, Y]],
) -> Mealy[P, P, S, S, X, X, Y, Y]:
    def transition(p: P, sx: tuple[S, X]) -> tuple[S, Y]:
        s, x = sx
        return f(p, s, x)

    return Mealy(para_autodiff(transition))


def local_state[Q, DQ, P, DP, S, X, DX, Y, DY](
    initial: Lens[Q, DQ, S, S],
    body: Mealy[P, DP, S, S, X, DX, Y, DY],
) -> ParaLens[tuple[Q, P], tuple[DQ, DP], X, DX, Y, DY]:
    return body.local_state(initial, zero_cotangent_like)


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


learning_rate: Lens[jax.Array, jax.Array, Unit, Unit] = Lens(lambda loss: ((), lambda _: jnp.ones_like(loss)))

learning_rate_log: Lens[jax.Array, jax.Array, jax.Array, jax.Array] = (
    copy(Proxy[tuple[jax.Array, jax.Array]]())
    >> learning_rate.first(Proxy[tuple[jax.Array, jax.Array]]())
    >> snd(Proxy[tuple[Unit, jax.Array, jax.Array]]())
)
