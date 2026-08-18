from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import JACOBIAN, PRNG
from meta_learn_lib.category.mealy import Mealy
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


def rtrl_like[HP, HPM, H, X, Y, P, W](
    machine: Mealy[H, H, X, X, Y, Y, HP, HP, P, P],
    update_influence: Callable[
        [HPM, Callable[[jax.Array, jax.Array], jax.Array], Callable[[jax.Array], jax.Array], W], W
    ],
    boundary: Callable[[HPM, H, Y, jax.Array, W], jax.Array],
) -> Mealy[tuple[H, W], tuple[H, W], X, X, Y, Y, tuple[HP, HPM], tuple[HP, HPM], P, P]:

    def run(
        p_sx: tuple[tuple[tuple[HP, HPM], P], tuple[tuple[H, W], X]],
    ) -> tuple[
        tuple[tuple[H, W], Y],
        Callable[[tuple[tuple[H, W], Y]], tuple[tuple[tuple[HP, HPM], P], tuple[tuple[H, W], X]]],
    ]:
        ((hp, hpm), p), ((h0, W0), x) = p_sx
        hp_p = (hp, p)
        _, unflat_h = jax.flatten_util.ravel_pytree(eqx.filter(h0, eqx.is_inexact_array))
        _, unflat_p = jax.flatten_util.ravel_pytree(eqx.filter(p, eqx.is_inexact_array))
        (h1, y), put = machine.arrow.arrow.run((hp_p, (h0, x)))
        ignore_y = zero_cotangent_like(y)
        ignore_x = zero_cotangent_like(x)
        ignore_hp = zero_cotangent_like(hp)
        jvp = jax.linear_transpose(put, zero_cotangent_like((h0, y)))

        def push(d_p: jax.Array, d_h: jax.Array) -> jax.Array:
            ((d_h_next, _),) = jvp(((ignore_hp, unflat_p(d_p)), (unflat_h(d_h), ignore_x)))
            d_h_next_flat, _ = jax.flatten_util.ravel_pytree(d_h_next)
            return d_h_next_flat

        def row(e: jax.Array) -> jax.Array:
            (_, dp), _ = put((unflat_h(e), ignore_y))
            dp_flat, _ = jax.flatten_util.ravel_pytree(dp)
            return dp_flat

        W1 = update_influence(hpm, push, row, W0)

        def rev(
            ct: tuple[tuple[H, W], Y],
        ) -> tuple[tuple[tuple[HP, HPM], P], tuple[tuple[H, W], X]]:
            (d_h_final, _), d_y = ct
            (d_hp, d_p_inner), (d_h0, d_x) = put((d_h_final, d_y))
            d_h0_flat, _ = jax.flatten_util.ravel_pytree(d_h0)
            d_p = jax.tree.map(jnp.add, d_p_inner, unflat_p(boundary(hpm, d_h_final, d_y, d_h0_flat, W0)))
            zero_state = zero_cotangent_like((h0, W0))
            return ((d_hp, zero_cotangent_like(hpm)), d_p), (zero_state, d_x)

        return ((h1, W1), y), rev

    return Mealy(ParaLens(Lens(run)))


def rtrl[HP, H, X, Y, P](
    machine: Mealy[H, H, X, X, Y, Y, HP, HP, P, P],
) -> Mealy[tuple[H, JACOBIAN], tuple[H, JACOBIAN], X, X, Y, Y, tuple[HP, Unit], tuple[HP, Unit], P, P]:

    def update_influence(
        hpm: Unit,
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

    def boundary(hpm: Unit, d_h_final: H, d_y: Y, d_h0: jax.Array, M0: JACOBIAN) -> jax.Array:
        return d_h0 @ M0

    return rtrl_like(machine, update_influence, boundary)


type UORO_AUX = tuple[jax.Array, jax.Array, PRNG]


def uoro[HP, H, X, Y, P](
    machine: Mealy[H, H, X, X, Y, Y, HP, HP, P, P],
    distribution: SAMPLER,
) -> Mealy[tuple[H, UORO_AUX], tuple[H, UORO_AUX], X, X, Y, Y, tuple[HP, Unit], tuple[HP, Unit], P, P]:

    def update_influence(
        hpm: Unit,
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

    def boundary(hpm: Unit, d_h_final: H, d_y: Y, d_h0: jax.Array, W0: UORO_AUX) -> jax.Array:
        A0, B0, _ = W0
        return (d_h0 @ A0) * B0

    return rtrl_like(machine, update_influence, boundary)


def rflo[HP, HD, H, X, Y, P](
    machine: Mealy[H, H, X, X, Y, Y, HP, HP, P, P],
    decay: Callable[[HD], jax.Array],
) -> Mealy[tuple[H, JACOBIAN], tuple[H, JACOBIAN], X, X, Y, Y, tuple[HP, HD], tuple[HP, HD], P, P]:

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

    def boundary(hd: HD, d_h_final: H, d_y: Y, d_h0: jax.Array, M0: JACOBIAN) -> jax.Array:
        alpha = decay(hd)
        c_state, _ = jax.flatten_util.ravel_pytree(d_h_final)
        c_out, _ = jax.flatten_util.ravel_pytree(d_y)
        return (1 - alpha) * ((c_state + c_out) @ M0)

    return rtrl_like(machine, update_influence, boundary)
