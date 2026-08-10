from typing import Callable

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp


class Vector[T](eqx.Module):
    vector: jax.Array
    to_param: Callable[[jax.Array], T] = eqx.field(static=True)


def zero_tangent_like(value: jax.Array) -> jax.Array:
    return (
        jnp.zeros_like(value)
        if jnp.issubdtype(value.dtype, jnp.inexact)
        else jnp.zeros_like(value, dtype=jax.dtypes.float0)
    )


def add_cotangents[T](a: T, b: T) -> T:
    return eqx.combine(
        jax.tree.map(
            jnp.add,
            eqx.filter(a, eqx.is_inexact_array),
            eqx.filter(b, eqx.is_inexact_array),
        ),
        eqx.filter(a, eqx.is_inexact_array, inverse=True),
    )


def split_static[T](tree: T, filter: Callable[[object], bool]) -> tuple[T, Callable[[T], T]]:
    """Partition at the boundary: array leaves out as the port value, everything else closed over."""
    diff, static = eqx.partition(tree, filter)
    return diff, lambda d: eqx.combine(d, static)


def to_vector[T](tree: T) -> Vector[T]:
    """Convert a pytree to a Vector, which contains a flattened array and non-parameter parts."""
    params, recombine = split_static(tree, eqx.is_inexact_array)
    vector, to_param = jax.flatten_util.ravel_pytree(params)
    if vector.size == 0:
        vector = vector.astype(jnp.result_type(float))
    return Vector(vector=vector, to_param=lambda a: recombine(to_param(a)))
