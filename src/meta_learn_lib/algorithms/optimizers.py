from meta_learn_lib.category.lens import *
from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.paralens import ParaLens
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.utility.util import zero_cotangent_like

import jax
import jax.numpy as jnp
import optax
from typing import Callable, cast


def exponentiated[T](theta: T, updates: optax.Updates) -> T:
    return jax.tree.map(lambda t, u: t * jnp.exp(u), theta, updates)


def additive[T](theta: T, updates: optax.Updates) -> T:
    return jax.tree.map(lambda t, u: t + u, theta, updates)


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


def optimizer[T, H](
    make: Callable[[H], optax.GradientTransformation],
    apply: Callable[[T, optax.Updates], T],
) -> Mealy[optax.OptState, optax.OptState, T, T, T, T, Unit, Unit, H, H]:
    def run(
        p_sx: tuple[tuple[Unit, H], tuple[optax.OptState, T]],
    ) -> tuple[
        tuple[optax.OptState, T],
        Callable[[tuple[optax.OptState, T]], tuple[tuple[Unit, H], tuple[optax.OptState, T]]],
    ]:
        (_, hp), (opt_st, theta) = p_sx

        def rev(ct: tuple[optax.OptState, T]) -> tuple[tuple[Unit, H], tuple[optax.OptState, T]]:
            _, d_theta = ct
            updates, opt_st1 = make(hp).update(cast(optax.Params, d_theta), opt_st, cast(optax.Params, theta))
            theta1 = apply(theta, updates)
            return (Unit(), zero_cotangent_like(hp)), (opt_st1, theta1)

        return (opt_st, theta), rev

    return Mealy(ParaLens(Lens(run)))


def sgd[T]() -> Mealy[
    optax.OptState,
    optax.OptState,
    T,
    T,
    T,
    T,
    Unit,
    Unit,
    jax.Array,
    jax.Array,
]:
    return optimizer(optax.sgd, additive)
