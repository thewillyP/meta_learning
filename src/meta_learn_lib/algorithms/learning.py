from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import JACOBIAN, PRNG
from meta_learn_lib.category.mealy import Mealy

import jax.flatten_util
import jax.numpy as jnp
import optax


def rtrl[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
) -> Mealy[tuple[H1, JACOBIAN], tuple[H2, JACOBIAN], X1, X2, Y1, Y2, Z1, Z2]:

    def run(
        p_sx: tuple[Z1, tuple[tuple[H1, JACOBIAN], X1]],
    ) -> tuple[
        tuple[tuple[H1, JACOBIAN], Y1],
        Callable[[tuple[tuple[H2, JACOBIAN], Y2]], tuple[Z2, tuple[tuple[H2, JACOBIAN], X2]]],
    ]:
        z, ((h0, M0), x) = p_sx
        flat_h, unflat_h = jax.flatten_util.ravel_pytree(h0)
        flat_z, unflat_z = jax.flatten_util.ravel_pytree(z)
        (h1, y), put = machine.arrow.arrow.run((z, (h0, x)))
        ignore_y = jax.tree.map(jnp.zeros_like, y)
        ignore_x = jax.tree.map(jnp.zeros_like, x)
        ignore_z = jax.tree.map(jnp.zeros_like, z)
        jvp = jax.linear_transpose(put, (h0, y))  # v -> A v, still through `put`

        def push(d_z: Z1, d_h: jax.Array) -> jax.Array:
            ((d_h_next, _),) = jvp((d_z, (unflat_h(d_h), ignore_x)))
            d_h_next_flat, _ = jax.flatten_util.ravel_pytree(d_h_next)
            return d_h_next_flat

        def row(e: jax.Array) -> jax.Array:
            dz, _ = put((unflat_h(e), ignore_y))  # dy'=0 -> dz is a row of J_z
            dz_flat, _ = jax.flatten_util.ravel_pytree(dz)
            return dz_flat

        if flat_h.size > flat_z.size:
            M1 = jax.vmap(lambda e, col: push(unflat_z(e), col), in_axes=(0, 1), out_axes=1)(jnp.eye(flat_z.size), M0)
        else:
            J_z = jax.vmap(row)(jnp.eye(flat_h.size))
            jmp_M0 = jax.vmap(lambda col: push(ignore_z, col), in_axes=1, out_axes=1)(M0)
            M1 = jmp_M0 + J_z

        def rev(
            ct: tuple[tuple[H2, JACOBIAN], Y2],
        ) -> tuple[Z2, tuple[tuple[H2, JACOBIAN], X2]]:
            (d_h_final, _), d_y = ct
            d_z_inner, (d_h0, d_x) = put((d_h_final, d_y))
            d_h0_flat, _ = jax.flatten_util.ravel_pytree(d_h0)
            boundary = unflat_z(d_h0_flat @ M0)
            d_z = jax.tree.map(jnp.add, d_z_inner, boundary)
            zero_state = jax.tree.map(jnp.zeros_like, (d_h0, M0))
            return d_z, (zero_state, d_x)

        return ((h1, JACOBIAN(M1)), y), rev

    return Mealy(ParaLens(Lens(run)))


type UORO_AUX = tuple[jax.Array, jax.Array, PRNG]


def uoro[H1, H2, X1, X2, Y1, Y2, Z1, Z2](
    machine: Mealy[H1, H2, X1, X2, Y1, Y2, Z1, Z2],
) -> Mealy[tuple[H1, UORO_AUX], tuple[H2, UORO_AUX], X1, X2, Y1, Y2, Z1, Z2]:

    def run(
        p_sx: tuple[Z1, tuple[tuple[H1, UORO_AUX], X1]],
    ) -> tuple[
        tuple[tuple[H1, UORO_AUX], Y1],
        Callable[[tuple[tuple[H2, UORO_AUX], Y2]], tuple[Z2, tuple[tuple[H2, UORO_AUX], X2]]],
    ]:
        z, ((h0, (A0, B0, key)), x) = p_sx
        flat_h, unflat_h = jax.flatten_util.ravel_pytree(h0)
        _, unflat_z = jax.flatten_util.ravel_pytree(z)
        (h1, y), put = machine.arrow.arrow.run((z, (h0, x)))
        ignore_y = jax.tree.map(jnp.zeros_like, y)
        ignore_x = jax.tree.map(jnp.zeros_like, x)
        ignore_z = jax.tree.map(jnp.zeros_like, z)
        jvp = jax.linear_transpose(put, (h0, y))

        def push(d_h: jax.Array) -> jax.Array:
            ((d_h_next, _),) = jvp((ignore_z, (unflat_h(d_h), ignore_x)))
            d_h_next_flat, _ = jax.flatten_util.ravel_pytree(d_h_next)
            return d_h_next_flat

        def row(e: jax.Array) -> jax.Array:
            dz, _ = put((unflat_h(e), ignore_y))  # dy'=0 -> dz is a row of J_z
            dz_flat, _ = jax.flatten_util.ravel_pytree(dz)
            return dz_flat

        key0, key1 = jax.random.split(key)
        nu = jax.random.rademacher(key0, flat_h.shape, dtype=flat_h.dtype)
        jmp_A0 = push(A0)
        nu_J_z = row(nu)

        rho0 = jnp.sqrt(optax.safe_norm(B0, 1e-12) / optax.safe_norm(jmp_A0, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(nu_J_z, 1e-12) / optax.safe_norm(nu, 1e-12))

        A1: jax.Array = rho0 * jmp_A0 + rho1 * nu
        B1: jax.Array = B0 / rho0 + nu_J_z / rho1

        def rev(
            ct: tuple[tuple[H2, UORO_AUX], Y2],
        ) -> tuple[Z2, tuple[tuple[H2, UORO_AUX], X2]]:
            (d_h_final, _), d_y = ct
            d_z_inner, (d_h0, d_x) = put((d_h_final, d_y))
            d_h0_flat, _ = jax.flatten_util.ravel_pytree(d_h0)
            boundary = unflat_z((d_h0_flat @ A0) * B0)
            d_z = jax.tree.map(jnp.add, d_z_inner, boundary)
            zero_state = jax.tree.map(jnp.zeros_like, (d_h0, (A0, B0, key)))
            return d_z, (zero_state, d_x)

        return ((h1, (A1, B1, key1)), y), rev

    return Mealy(ParaLens(Lens(run)))
