from meta_learn_lib.algorithms.optimizers import adam, sgd, sgd_normalized
from meta_learn_lib.construct import config
from meta_learn_lib.construct.ports import Rates
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.utility.distributions import rademacher, uniform_unit

from typing import Callable, overload
import jax
import jax.numpy as jnp
import optax
from plum import dispatch


# ── activations ───────────────────────────────────────────────────────────────


@overload
def activation(t: config.Tanh) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.tanh


@overload
def activation(t: config.Relu) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.relu


@overload
def activation(t: config.Sigmoid) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.sigmoid


@overload
def activation(t: config.Softmax) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.softmax


@overload
def activation(t: config.Identity) -> Callable[[jax.Array], jax.Array]:
    return lambda x: x


@dispatch
def activation(t: config.Activation) -> Callable[[jax.Array], jax.Array]:
    raise NotImplementedError


# ── initialisers ──────────────────────────────────────────────────────────────


@overload
def initializer(t: config.Zeros) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.zeros


@overload
def initializer(t: config.LecunNormal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", in_axis=-1, out_axis=-2)


@overload
def initializer(t: config.PytorchUniform) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=-1, out_axis=-2)


@overload
def initializer(t: config.Normal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.normal(t.std)


@overload
def initializer(t: config.Orthogonal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.orthogonal()


@dispatch
def initializer(t: config.Init) -> jax.nn.initializers.Initializer:
    raise NotImplementedError


# ── losses ────────────────────────────────────────────────────────────────────


@overload
def objective(t: config.L2) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return lambda y, target: jnp.sum(optax.losses.l2_loss(y, target))


@overload
def objective(t: config.CrossEntropy) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return lambda y, target: jnp.sum(optax.losses.softmax_cross_entropy(y, target))


@dispatch
def objective(t: config.Loss) -> Callable[[jax.Array, jax.Array], jax.Array]:
    raise NotImplementedError


# ── noise ─────────────────────────────────────────────────────────────────────


@overload
def sampler(t: config.Gaussian) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return jax.random.normal


@overload
def sampler(t: config.Rademacher) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return rademacher


@overload
def sampler(t: config.UniformUnit) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return uniform_unit


@dispatch
def sampler(t: config.Noise) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    raise NotImplementedError


# ── optimisers: handed their rates, the update rule ───────────────────────────


@overload
def transformation(t: config.Sgd) -> Callable[[Rates[jax.Array]], optax.GradientTransformation]:
    return lambda k: sgd(k.learning_rate, k.weight_decay, k.momentum)


@overload
def transformation(t: config.SgdNormalized) -> Callable[[Rates[jax.Array]], optax.GradientTransformation]:
    return lambda k: sgd_normalized(k.learning_rate, k.weight_decay, k.momentum)


@overload
def transformation(t: config.Adam) -> Callable[[Rates[jax.Array]], optax.GradientTransformation]:
    return lambda k: adam(k.learning_rate, k.weight_decay, k.momentum, t.b2, t.eps, t.eps_root)


@dispatch
def transformation(
    t: config.Sgd | config.SgdNormalized | config.Adam,
) -> Callable[[Rates[jax.Array]], optax.GradientTransformation]:
    raise NotImplementedError
