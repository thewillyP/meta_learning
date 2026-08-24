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


def optimizer[T, HPO, H](
    make: Callable[[tuple[HPO, H]], optax.GradientTransformation],
    apply: Callable[[T, optax.Updates], T],
) -> Mealy[optax.OptState, optax.OptState, T, T, T, T, HPO, HPO, H, H]:
    def run(
        p_sx: tuple[tuple[HPO, H], tuple[optax.OptState, T]],
    ) -> tuple[
        tuple[optax.OptState, T],
        Callable[[tuple[optax.OptState, T]], tuple[tuple[HPO, H], tuple[optax.OptState, T]]],
    ]:
        hp_h, (opt_st, theta) = p_sx

        def rev(ct: tuple[optax.OptState, T]) -> tuple[tuple[HPO, H], tuple[optax.OptState, T]]:
            _, d_theta = ct
            updates, opt_st1 = make(hp_h).update(cast(optax.Params, d_theta), opt_st, cast(optax.Params, theta))
            theta1 = apply(theta, updates)
            return zero_cotangent_like(hp_h), (opt_st1, theta1)

        return (opt_st, theta), rev

    return Mealy(ParaLens(Lens(run)))


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
