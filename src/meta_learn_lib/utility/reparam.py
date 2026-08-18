import jax
import jax.numpy as jnp

LAMBERTW_ITERATIONS = 20


def softminus(x: jax.Array) -> jax.Array:
    return -jax.nn.softplus(-x)


def soft_relu(x: jax.Array, sharpness: float) -> jax.Array:
    return x - softminus(sharpness * x) / sharpness


def soft_clip(x: jax.Array, a: float | None, b: float | None, sharpness: float) -> jax.Array:
    c = sharpness / ((b - a) / 2) if a is not None and b is not None else sharpness
    v = x
    if a is not None:
        v = v - softminus(c * (x - a)) / c
    if b is not None:
        v = v - jax.nn.softplus(c * (x - b)) / c
    return v


def lambertw(x: jax.Array) -> jax.Array:
    w = jnp.log1p(x)
    for _ in range(LAMBERTW_ITERATIONS):
        ew = jnp.exp(w)
        wew = w * ew
        w = w - (wew - x) / (ew * (w + 1) - (w + 2) * (wew - x) / (2 * w + 2))
    return w


SILU_BIAS = lambertw(jnp.asarray(1.0 / jnp.e))


def silu_positive(x: jax.Array, scale: float) -> jax.Array:
    return x * jax.nn.sigmoid(scale * x) + SILU_BIAS


def silu_positive_inverse(y: jax.Array, scale: float) -> jax.Array:
    z = scale * (y - SILU_BIAS)
    return (z + lambertw(z * jnp.exp(-z))) / scale


def softplus_inverse(y: jax.Array) -> jax.Array:
    return jnp.log(jnp.expm1(y))
