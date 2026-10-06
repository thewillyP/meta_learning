from meta_learn_lib.algorithms.lib_types import UORO_AUX
from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import JACOBIAN, PRNG
from meta_learn_lib.utility.distributions import SAMPLER
from meta_learn_lib.utility.util import zero_cotangent_like

import equinox as eqx
import jax.flatten_util
import optax
import jax.numpy as jnp


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


def rtrl_like[Theta, D, M, S, Y, W](
    model: ParaLens[tuple[Theta, D], tuple[Theta, D], S, S, tuple[S, Y], tuple[S, Y]],
    update_influence: Callable[
        [M, Callable[[jax.Array, jax.Array], jax.Array], Callable[[jax.Array], jax.Array], W], W
    ],
    boundary: Callable[[M, S, Y, jax.Array, W], jax.Array],
) -> ParaLens[
    tuple[Theta, tuple[M, D]],
    tuple[Theta, tuple[M, D]],
    tuple[S, W],
    tuple[S, W],
    tuple[tuple[S, W], Y],
    tuple[tuple[S, W], Y],
]:

    def run(
        p_sw: tuple[tuple[Theta, tuple[M, D]], tuple[S, W]],
    ) -> tuple[
        tuple[tuple[S, W], Y],
        Callable[[tuple[tuple[S, W], Y]], tuple[tuple[Theta, tuple[M, D]], tuple[S, W]]],
    ]:
        (theta, (m, d)), (s0, W0) = p_sw
        _, unflat_s = jax.flatten_util.ravel_pytree(eqx.filter(s0, eqx.is_inexact_array))
        _, unflat_theta = jax.flatten_util.ravel_pytree(eqx.filter(theta, eqx.is_inexact_array))
        (s1, y), put = model.arrow.run(((theta, d), s0))
        ignore_y = zero_cotangent_like(y)
        ignore_d = zero_cotangent_like(d)
        jvp = jax.linear_transpose(put, zero_cotangent_like((s0, y)))

        def push(d_theta: jax.Array, d_s: jax.Array) -> jax.Array:
            ((d_s_next, _),) = jvp(((unflat_theta(d_theta), ignore_d), unflat_s(d_s)))
            d_s_next_flat, _ = jax.flatten_util.ravel_pytree(d_s_next)
            return d_s_next_flat

        def row(e: jax.Array) -> jax.Array:
            (d_theta, _), _ = put((unflat_s(e), ignore_y))
            d_theta_flat, _ = jax.flatten_util.ravel_pytree(d_theta)
            return d_theta_flat

        W1 = update_influence(m, push, row, W0)

        def rev(
            ct: tuple[tuple[S, W], Y],
        ) -> tuple[tuple[Theta, tuple[M, D]], tuple[S, W]]:
            (d_s_final, _), d_y = ct
            (d_theta_inner, d_d), d_s0 = put((d_s_final, d_y))
            d_s0_flat, _ = jax.flatten_util.ravel_pytree(d_s0)
            d_theta = jax.tree.map(jnp.add, d_theta_inner, unflat_theta(boundary(m, d_s_final, d_y, d_s0_flat, W0)))
            zero_state = zero_cotangent_like((s0, W0))
            return (d_theta, (zero_cotangent_like(m), d_d)), zero_state

        return ((s1, W1), y), rev

    return ParaLens(Lens(run))


def rtrl[Theta, D, S, Y](
    model: ParaLens[tuple[Theta, D], tuple[Theta, D], S, S, tuple[S, Y], tuple[S, Y]],
) -> ParaLens[
    tuple[Theta, tuple[Unit, D]],
    tuple[Theta, tuple[Unit, D]],
    tuple[S, JACOBIAN],
    tuple[S, JACOBIAN],
    tuple[tuple[S, JACOBIAN], Y],
    tuple[tuple[S, JACOBIAN], Y],
]:

    def update_influence(
        m: Unit,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        M0: JACOBIAN,
    ) -> JACOBIAN:
        n, p = M0.shape
        if n > p:
            M1 = jax.vmap(lambda e, col: push(e, col), in_axes=(0, 1), out_axes=1)(jnp.eye(p), M0)
        else:
            J_p = immediate_influence(push, row, (n, p))
            jmp_M0 = jax.vmap(lambda col: push(jnp.zeros(p), col), in_axes=1, out_axes=1)(M0)
            M1 = jmp_M0 + J_p
        return JACOBIAN(M1)

    def boundary(m: Unit, d_s_final: S, d_y: Y, d_s0: jax.Array, M0: JACOBIAN) -> jax.Array:
        return d_s0 @ M0

    return rtrl_like(model, update_influence, boundary)


def uoro[Theta, D, S, Y](
    model: ParaLens[tuple[Theta, D], tuple[Theta, D], S, S, tuple[S, Y], tuple[S, Y]],
    distribution: SAMPLER,
) -> ParaLens[
    tuple[Theta, tuple[Unit, D]],
    tuple[Theta, tuple[Unit, D]],
    tuple[S, UORO_AUX],
    tuple[S, UORO_AUX],
    tuple[tuple[S, UORO_AUX], Y],
    tuple[tuple[S, UORO_AUX], Y],
]:

    def update_influence(
        m: Unit,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        W0: UORO_AUX,
    ) -> UORO_AUX:
        A0, B0, key = W0
        key0, key1 = jax.random.split(key)
        nu = distribution(PRNG(key0), A0.shape)
        jmp_A0 = push(jnp.zeros_like(B0), A0)
        nu_J_p = row(nu)
        rho0 = jnp.sqrt(optax.safe_norm(B0, 1e-12) / optax.safe_norm(jmp_A0, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(nu_J_p, 1e-12) / optax.safe_norm(nu, 1e-12))
        A1: jax.Array = rho0 * jmp_A0 + rho1 * nu
        B1: jax.Array = B0 / rho0 + nu_J_p / rho1
        return (A1, B1, PRNG(key1))

    def boundary(m: Unit, d_s_final: S, d_y: Y, d_s0: jax.Array, W0: UORO_AUX) -> jax.Array:
        A0, B0, _ = W0
        return (d_s0 @ A0) * B0

    return rtrl_like(model, update_influence, boundary)


def rflo[Theta, D, HD, S, Y](
    model: ParaLens[tuple[Theta, D], tuple[Theta, D], S, S, tuple[S, Y], tuple[S, Y]],
    decay: Callable[[HD], jax.Array],
) -> ParaLens[
    tuple[Theta, tuple[HD, D]],
    tuple[Theta, tuple[HD, D]],
    tuple[S, JACOBIAN],
    tuple[S, JACOBIAN],
    tuple[tuple[S, JACOBIAN], Y],
    tuple[tuple[S, JACOBIAN], Y],
]:

    def update_influence(
        hd: HD,
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        M0: JACOBIAN,
    ) -> JACOBIAN:
        alpha = decay(hd)
        n, p = M0.shape
        J_p = immediate_influence(push, row, (n, p))
        return JACOBIAN((1 - alpha) * M0 + J_p)

    def boundary(hd: HD, d_s_final: S, d_y: Y, d_s0: jax.Array, M0: JACOBIAN) -> jax.Array:
        alpha = decay(hd)
        c_state, _ = jax.flatten_util.ravel_pytree(d_s_final)
        c_out, _ = jax.flatten_util.ravel_pytree(d_y)
        return (1 - alpha) * ((c_state + c_out) @ M0)

    return rtrl_like(model, update_influence, boundary)
