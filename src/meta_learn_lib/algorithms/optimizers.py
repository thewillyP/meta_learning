from typing import Callable, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..category.lib_types import Unit
from ..category.paralens import ParaLens
from .combinators import update_rule


def exponentiated[T](theta: T, updates: optax.Updates) -> T:
    return jax.tree.map(lambda t, u: t * jnp.exp(u), theta, updates)


def additive[T](theta: T, updates: optax.Updates) -> T:
    return eqx.apply_updates(theta, updates)


def scale_by_sign_of_param() -> optax.GradientTransformation:
    def f(updates: optax.Updates, params: optax.Params | None) -> optax.Updates:
        return jax.tree.map(lambda u, p: u * jnp.sign(p), updates, params)

    return optax.stateless(f)


def add_scalar_wd(wd_value: jax.Array) -> optax.GradientTransformation:
    def f(updates: optax.Updates, params: optax.Params | None) -> optax.Updates:
        return jax.tree.map(lambda u: u + wd_value, updates)

    return optax.stateless(f)


def exponentiated_gradient[H](
    base: Callable[[H], optax.GradientTransformation],
) -> Callable[[tuple[jax.Array, H]], optax.GradientTransformation]:
    def make(hp: tuple[jax.Array, H]) -> optax.GradientTransformation:
        wd, base_hp = hp
        return optax.chain(scale_by_sign_of_param(), add_scalar_wd(wd), base(base_hp))

    return make


def optimizer[T, K](
    make: Callable[[K], optax.GradientTransformation],
    apply: Callable[[T, optax.Updates], T],
) -> ParaLens[K, Unit, tuple[optax.OptState, T], tuple[optax.OptState, T], T, T]:
    def get(k: K, state: tuple[optax.OptState, T]) -> T:
        _, theta = state
        return theta

    def update(k: K, state: tuple[optax.OptState, T], dtheta: T) -> tuple[optax.OptState, T]:
        memory, theta = state
        updates, memory_next = make(k).update(cast(optax.Params, dtheta), memory, cast(optax.Params, theta))
        return memory_next, apply(theta, updates)

    return update_rule(get, update)


def init_optimizer[T, K](
    make: Callable[[K], optax.GradientTransformation],
    k: K,
    theta: T,
) -> tuple[optax.OptState, T]:
    return make(k).init(cast(optax.Params, theta)), theta


def sgd(lr: jax.Array, wd: jax.Array, momentum: jax.Array) -> optax.GradientTransformation:
    return optax.chain(optax.add_decayed_weights(wd), optax.sgd(lr, momentum=momentum))


def sgd_normalized(lr: jax.Array, wd: jax.Array, momentum: jax.Array) -> optax.GradientTransformation:
    return optax.chain(
        optax.normalize_by_update_norm(scale_factor=1.0),
        optax.add_decayed_weights(wd),
        optax.sgd(lr, momentum=momentum),
    )


def adam(
    lr: jax.Array, wd: jax.Array, momentum: jax.Array, b2: float, eps: float, eps_root: float
) -> optax.GradientTransformation:
    return optax.adamw(lr, b1=momentum, b2=b2, eps=eps, eps_root=eps_root, weight_decay=wd)


def frozen() -> optax.GradientTransformation:
    return optax.set_to_zero()
