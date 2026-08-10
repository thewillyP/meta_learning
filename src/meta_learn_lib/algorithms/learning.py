from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import JACOBIAN, PRNG
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.utility.distributions import SAMPLER

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


def rtrl_like[H1, H2, X1, X2, Y1, Y2, Z1, Z2, W](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
    update_influence: Callable[[Callable[[jax.Array, jax.Array], jax.Array], Callable[[jax.Array], jax.Array], W], W],
    boundary: Callable[[H2, Y2, jax.Array, W], jax.Array],
) -> Mealy[tuple[H1, W], tuple[H2, W], X1, X2, Y1, Y2, Z1, Z2]:

    def run(
        p_sx: tuple[Z1, tuple[tuple[H1, W], X1]],
    ) -> tuple[
        tuple[tuple[H1, W], Y1],
        Callable[[tuple[tuple[H2, W], Y2]], tuple[Z2, tuple[tuple[H2, W], X2]]],
    ]:
        z, ((h0, W0), x) = p_sx
        _, unflat_h = jax.flatten_util.ravel_pytree(h0)
        _, unflat_z = jax.flatten_util.ravel_pytree(z)
        (h1, y), put = machine.arrow.arrow.run((z, (h0, x)))
        ignore_y = jax.tree.map(jnp.zeros_like, y)
        ignore_x = jax.tree.map(jnp.zeros_like, x)
        jvp = jax.linear_transpose(put, (h0, y))

        def push(d_z: jax.Array, d_h: jax.Array) -> jax.Array:
            ((d_h_next, _),) = jvp((unflat_z(d_z), (unflat_h(d_h), ignore_x)))
            d_h_next_flat, _ = jax.flatten_util.ravel_pytree(d_h_next)
            return d_h_next_flat

        def row(e: jax.Array) -> jax.Array:
            dz, _ = put((unflat_h(e), ignore_y))
            dz_flat, _ = jax.flatten_util.ravel_pytree(dz)
            return dz_flat

        W1 = update_influence(push, row, W0)

        def rev(
            ct: tuple[tuple[H2, W], Y2],
        ) -> tuple[Z2, tuple[tuple[H2, W], X2]]:
            (d_h_final, _), d_y = ct
            d_z_inner, (d_h0, d_x) = put((d_h_final, d_y))
            d_h0_flat, _ = jax.flatten_util.ravel_pytree(d_h0)
            d_z = jax.tree.map(jnp.add, d_z_inner, unflat_z(boundary(d_h_final, d_y, d_h0_flat, W0)))
            zero_state = jax.tree.map(jnp.zeros_like, (d_h0, W0))
            return d_z, (zero_state, d_x)

        return ((h1, W1), y), rev

    return Mealy(ParaLens(Lens(run)))


def rtrl[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
) -> Mealy[tuple[H1, JACOBIAN], tuple[H2, JACOBIAN], X1, X2, Y1, Y2, Z1, Z2]:

    def update_influence(
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        M0: JACOBIAN,
    ) -> JACOBIAN:
        n, p = M0.shape
        if n > p:
            M1 = jax.vmap(lambda e, col: push(e, col), in_axes=(0, 1), out_axes=1)(jnp.eye(p), M0)
        else:
            J_z = immediate_influence(push, row, M0.shape)
            jmp_M0 = jax.vmap(lambda col: push(jnp.zeros(p), col), in_axes=1, out_axes=1)(M0)
            M1 = jmp_M0 + J_z
        return JACOBIAN(M1)

    def boundary(d_h_final: H2, d_y: Y2, d_h0: jax.Array, M0: JACOBIAN) -> jax.Array:
        return d_h0 @ M0

    return rtrl_like(machine, update_influence, boundary)


type UORO_AUX = tuple[jax.Array, jax.Array, PRNG]


def uoro[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
    distribution: SAMPLER,
) -> Mealy[tuple[H1, UORO_AUX], tuple[H2, UORO_AUX], X1, X2, Y1, Y2, Z1, Z2]:

    def update_influence(
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        W0: UORO_AUX,
    ) -> UORO_AUX:
        A0, B0, key = W0
        key0, key1 = jax.random.split(key)
        nu = distribution(key0, A0.shape)
        jmp_A0 = push(jnp.zeros_like(B0), A0)
        nu_J_z = row(nu)
        rho0 = jnp.sqrt(optax.safe_norm(B0, 1e-12) / optax.safe_norm(jmp_A0, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(nu_J_z, 1e-12) / optax.safe_norm(nu, 1e-12))
        A1: jax.Array = rho0 * jmp_A0 + rho1 * nu
        B1: jax.Array = B0 / rho0 + nu_J_z / rho1
        return (A1, B1, key1)

    def boundary(d_h_final: H2, d_y: Y2, d_h0: jax.Array, W0: UORO_AUX) -> jax.Array:
        A0, B0, _ = W0
        return (d_h0 @ A0) * B0

    return rtrl_like(machine, update_influence, boundary)


def rflo[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
    alpha: jax.Array,
) -> Mealy[tuple[H1, JACOBIAN], tuple[H2, JACOBIAN], X1, X2, Y1, Y2, Z1, Z2]:

    def update_influence(
        push: Callable[[jax.Array, jax.Array], jax.Array],
        row: Callable[[jax.Array], jax.Array],
        M0: JACOBIAN,
    ) -> JACOBIAN:
        J_z = immediate_influence(push, row, M0.shape)
        return JACOBIAN((1 - alpha) * M0 + J_z)

    def boundary(d_h_final: H2, d_y: Y2, d_h0: jax.Array, M0: JACOBIAN) -> jax.Array:
        c_state, _ = jax.flatten_util.ravel_pytree(d_h_final)
        c_out, _ = jax.flatten_util.ravel_pytree(d_y)
        return (1 - alpha) * ((c_state + c_out) @ M0)

    return rtrl_like(machine, update_influence, boundary)
