from typing import Callable
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from donotation import do

from ..backend.record import restrict_keys, without_keys
from ..category.lens import Lens
from ..category.lib_types import Proxy
from ..category.mealy import Mealy
from ..category.paralens import ParaLens
from ..category.writer import pure
from ..components.env import Log, Par, St
from ..lib_types import JACOBIAN, PRNG
from ..utility.distributions import SAMPLER
from ..utility.util import zero_cotangent_like


def immediate_influence(
    push: Callable[[jax.Array, jax.Array], jax.Array],
    row: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int],
) -> jax.Array:
    n, p = shape
    if n > p:
        return jax.vmap(lambda e: push(e, jnp.zeros(n)), in_axes=0, out_axes=1)(jnp.eye(p))
    else:
        return jax.vmap(row)(jnp.eye(n))


def rtrl_like[X, Y](
    cell: Mealy[Log, Par, Par, St, St, X, X, Y, Y],
    targets: frozenset[str],
    influence: frozenset[str],
    update_influence: Callable[
        [
            Par,
            Callable[[jax.Array, jax.Array], jax.Array],
            Callable[[jax.Array], jax.Array],
            St,
        ],
        St,
    ],
    boundary: Callable[[Par, St, Y, jax.Array, St], jax.Array],
) -> Mealy[Log, Par, Par, St, St, X, X, Y, Y]:
    merge = cell.step.monoid.mappend

    @do()
    def run(p_sx: tuple[Par, tuple[St, X]]):
        p, (s, x) = p_sx
        s0, w0 = without_keys(influence)(s), restrict_keys(influence)(s)
        _, unflat_s = jax.flatten_util.ravel_pytree(eqx.filter(s0, eqx.is_inexact_array))
        _, unflat_theta = jax.flatten_util.ravel_pytree(eqx.filter(restrict_keys(targets)(p), eqx.is_inexact_array))
        (s1, y), put = yield from cell.step.arrow.run((p, (s0, x)))
        ignore_p = zero_cotangent_like(without_keys(targets)(p))
        ignore_x = zero_cotangent_like(x)
        ignore_y = zero_cotangent_like(y)
        jvp = jax.linear_transpose(lambda ct: put(ct).value, zero_cotangent_like((s1, y)))

        def push(d_theta: jax.Array, d_s: jax.Array) -> jax.Array:
            ((d_s_next, _),) = jvp((merge(ignore_p, unflat_theta(d_theta)), (unflat_s(d_s), ignore_x)))
            d_s_next_flat, _ = jax.flatten_util.ravel_pytree(d_s_next)
            return d_s_next_flat

        def row(e: jax.Array) -> jax.Array:
            dp, _ = put((unflat_s(e), ignore_y)).value
            d_theta_flat, _ = jax.flatten_util.ravel_pytree(restrict_keys(targets)(dp))
            return d_theta_flat

        w1 = update_influence(p, push, row, w0)

        @do()
        def rev(ct: tuple[St, Y]):
            ds_final, dy = ct
            ds_final = without_keys(influence)(ds_final)
            dp, (ds0, dx) = yield from put((ds_final, dy))
            ds0_flat, _ = jax.flatten_util.ravel_pytree(ds0)
            dp = merge(dp, unflat_theta(boundary(p, ds_final, dy, ds0_flat, w0)))
            return pure(Proxy[Log](), (dp, (zero_cotangent_like(s), dx)))

        return pure(Proxy[Log](), ((merge(s1, w1), y), rev))

    return Mealy(ParaLens(Lens(run), cell.step.monoid))


def rtrl[X, Y](
    cell: Mealy[Log, Par, Par, St, St, X, X, Y, Y],
    targets: frozenset[str],
    influence: str,
) -> Mealy[Log, Par, Par, St, St, X, X, Y, Y]:
    def update_influence(
        m: Par,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        w0: St,
    ) -> St:
        M0 = w0.influences[influence]
        n, p = M0.shape
        if n > p:
            M1 = jax.vmap(lambda e, col: push(e, col), in_axes=(0, 1), out_axes=1)(jnp.eye(p), M0)
        else:
            J_p = immediate_influence(push, row, (n, p))
            jmp_M0 = jax.vmap(lambda col: push(jnp.zeros(p), col), in_axes=1, out_axes=1)(M0)
            M1 = jmp_M0 + J_p
        return w0.set(influences=w0.influences.set(influence, JACOBIAN(M1)))

    def boundary(m: Par, ds_final: St, dy: Y, ds0: jax.Array, w0: St) -> jax.Array:
        return ds0 @ w0.influences[influence]

    return rtrl_like(cell, targets, frozenset({influence}), update_influence, boundary)


def uoro[X, Y](
    cell: Mealy[Log, Par, Par, St, St, X, X, Y, Y],
    targets: frozenset[str],
    distribution: SAMPLER,
    aux: str,
    prng: str,
) -> Mealy[Log, Par, Par, St, St, X, X, Y, Y]:
    def update_influence(
        m: Par,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        w0: St,
    ) -> St:
        (A0, B0), key = w0.uoros[aux], w0.prngs[prng]
        key0, key1 = jax.random.split(key)
        nu = distribution(PRNG(key0), A0.shape)
        jmp_A0 = push(jnp.zeros_like(B0), A0)
        nu_J_p = row(nu)
        rho0 = jnp.sqrt(optax.safe_norm(B0, 1e-12) / optax.safe_norm(jmp_A0, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(nu_J_p, 1e-12) / optax.safe_norm(nu, 1e-12))
        A1: jax.Array = rho0 * jmp_A0 + rho1 * nu
        B1: jax.Array = B0 / rho0 + nu_J_p / rho1
        return w0.set(uoros=w0.uoros.set(aux, (A1, B1)), prngs=w0.prngs.set(prng, PRNG(key1)))

    def boundary(m: Par, ds_final: St, dy: Y, ds0: jax.Array, w0: St) -> jax.Array:
        A0, B0 = w0.uoros[aux]
        return (ds0 @ A0) * B0

    return rtrl_like(cell, targets, frozenset({aux, prng}), update_influence, boundary)


def rflo[X, Y](
    cell: Mealy[Log, Par, Par, St, St, X, X, Y, Y],
    targets: frozenset[str],
    influence: str,
    decay: Callable[[Par], jax.Array],
) -> Mealy[Log, Par, Par, St, St, X, X, Y, Y]:
    def update_influence(
        m: Par,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        w0: St,
    ) -> St:
        alpha = decay(m)
        M0 = w0.influences[influence]
        n, p = M0.shape
        J_p = immediate_influence(push, row, (n, p))
        return w0.set(influences=w0.influences.set(influence, JACOBIAN((1 - alpha) * M0 + J_p)))

    def boundary(m: Par, ds_final: St, dy: Y, ds0: jax.Array, w0: St) -> jax.Array:
        alpha = decay(m)
        c_state, _ = jax.flatten_util.ravel_pytree(ds_final)
        c_out, _ = jax.flatten_util.ravel_pytree(dy)
        return (1 - alpha) * ((c_state + c_out) @ w0.influences[influence])

    return rtrl_like(cell, targets, frozenset({influence}), update_influence, boundary)
