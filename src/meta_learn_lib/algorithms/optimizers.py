from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import ParaLens
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.utility.util import zero_cotangent_like

import jax
import jax.numpy as jnp
import optax
from typing import Callable


def exponentiated[T](theta: T, updates: T) -> T:
    return jax.tree.map(lambda t, u: t * jnp.exp(u), theta, updates)


def scale_by_sign_of_param() -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        return None

    def update_fn(grad, state, params):
        return grad * jnp.sign(params), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def add_scalar_wd(wd_value: jax.Array) -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        return None

    def update_fn(grad, state, params):
        return grad + wd_value, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def exponentiated_gradient[H](
    base: Callable[[H], optax.GradientTransformation],
) -> Callable[[tuple[jax.Array, H]], optax.GradientTransformation]:
    def make(hp: tuple[jax.Array, H]) -> optax.GradientTransformation:
        wd, base_hp = hp
        return optax.chain(scale_by_sign_of_param(), add_scalar_wd(wd), base(base_hp))

    return make


def optimizer[T, H](
    make: Callable[[H], optax.GradientTransformation],
    apply: Callable[[T, T], T],
) -> Mealy[optax.OptState, optax.OptState, T, T, T, T, H, H]:
    def run(
        p_sx: tuple[H, tuple[optax.OptState, T]],
    ) -> tuple[
        tuple[optax.OptState, T],
        Callable[[tuple[optax.OptState, T]], tuple[H, tuple[optax.OptState, T]]],
    ]:
        hp, (opt_st, theta) = p_sx

        def rev(ct: tuple[optax.OptState, T]) -> tuple[H, tuple[optax.OptState, T]]:
            _, d_theta = ct
            updates, opt_st1 = make(hp).update(d_theta, opt_st, theta)
            theta1 = apply(theta, updates)
            return zero_cotangent_like(hp), (opt_st1, theta1)

        return (opt_st, theta), rev

    return Mealy(ParaLens(Lens(run)))


def sgd[T]() -> Mealy[optax.OptState, optax.OptState, T, T, T, T, jax.Array, jax.Array]:
    return optimizer(optax.sgd, optax.apply_updates)
