from meta_learn_lib.lib_types import PRNG

import jax
import jax.numpy as jnp
from typing import Callable

type SAMPLER = Callable[[PRNG, tuple[int, ...]], jax.Array]


def scaled(distribution: SAMPLER, std: float) -> SAMPLER:
    return lambda key, shape: std * distribution(key, shape)


def uniform_unit(key: PRNG, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.uniform(key, shape, minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))


def rademacher(key: PRNG, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.rademacher(key, shape, dtype=float)
