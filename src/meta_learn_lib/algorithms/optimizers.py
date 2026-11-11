from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..backend.jax import detach, optimizer_reparam
from ..category.lib_types import Monoid, Proxy
from ..category.paralens import ParaLens
from ..category.writer import Writer, pure
from ..components.env import HP, Log, Par
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
    base: Callable[[HP], optax.GradientTransformation],
    wd: str,
) -> Callable[[HP], optax.GradientTransformation]:
    def make(h: HP) -> optax.GradientTransformation:
        return optax.chain(scale_by_sign_of_param(), add_scalar_wd(h.arrays[wd]), base(h))

    return make


def optimizer(
    make: Callable[[HP], optax.GradientTransformation],
    apply: Callable[[Par, optax.Updates], Par],
    targets: frozenset[str],
    optstate: str,
    monoid: Monoid[HP, HP],
) -> Optimizer:
    def theta(z: Par) -> Par:
        return z.set(optstates=z.optstates.discard(optstate))

    def update(h: HP, z: Par, dp: Par) -> Writer[Log, Par]:
        params, static = eqx.partition(theta(z), eqx.is_inexact_array)
        updates, optstate_next = make(h).update(eqx.filter(dp, eqx.is_inexact_array), z.optstates[optstate], params)
        return pure(
            Proxy[Log](),
            eqx.combine(apply(params, updates), static).set(optstates=z.optstates.set(optstate, optstate_next)),
        )

    return Optimizer(optimizer_reparam(lambda h, z: theta(z), update, monoid), targets, targets | {optstate})


def init_optimizer(make: Callable[[HP], optax.GradientTransformation], h: HP, theta: Par, optstate: str) -> Par:
    params, _ = eqx.partition(theta, eqx.is_inexact_array)
    return theta.set(optstates=theta.optstates.set(optstate, make(h).init(params)))


def first_order(opt: Optimizer) -> Optimizer:
    return Optimizer(ParaLens(opt.reparam.arrow >> detach(Proxy[Par]()), opt.reparam.monoid), opt.targets, opt.state)


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
