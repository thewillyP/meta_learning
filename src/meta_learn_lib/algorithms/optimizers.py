from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..backend.jax import detach, update_rule
from ..category.lib_types import Monoid, Proxy
from ..category.paralens import ParaLens
from ..category.writer import pure
from ..components.env import Env
from ..components.optimizer import Optimizer


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


def exponentiated_gradient(
    base: Callable[[Env], optax.GradientTransformation],
    wd: str,
) -> Callable[[Env], optax.GradientTransformation]:
    def make(h: Env) -> optax.GradientTransformation:
        return optax.chain(scale_by_sign_of_param(), add_scalar_wd(h.arrays[wd]), base(h))

    return make


def optimizer(
    make: Callable[[Env], optax.GradientTransformation],
    apply: Callable[[Env, optax.Updates], Env],
    targets: frozenset[str],
    name: str,
    monoid: Monoid[Env, Env],
) -> Optimizer:
    def theta(z: Env) -> Env:
        return z.set(optstates=z.optstates.discard(name))

    def get(h: Env, z: Env) -> Env:
        return theta(z)

    def update(h: Env, z: Env, dp: Env):
        params, static = eqx.partition(theta(z), eqx.is_inexact_array)
        updates, optstate = make(h).update(eqx.filter(dp, eqx.is_inexact_array), z.optstates[name], params)
        return pure(
            Proxy[Env](),
            eqx.combine(apply(params, updates), static).set(optstates=z.optstates.set(name, optstate)),
        )

    return Optimizer(update_rule(get, update, monoid), targets, targets | {name})


def init_optimizer(make: Callable[[Env], optax.GradientTransformation], h: Env, theta: Env, name: str) -> Env:
    params, _ = eqx.partition(theta, eqx.is_inexact_array)
    return theta.set(optstates=theta.optstates.set(name, make(h).init(params)))


def first_order(opt: Optimizer) -> Optimizer:
    return Optimizer(ParaLens(opt.rule.arrow >> detach(Proxy[Env]()), opt.rule.monoid), opt.targets, opt.state)


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
