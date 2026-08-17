from meta_learn_lib.construct.term import (
    Activation,
    CrossEntropy,
    L2,
    Loss,
    Identity,
    Init,
    LecunNormal,
    PytorchUniform,
    Relu,
    Sigmoid,
    Softmax,
    Tanh,
    Zeros,
)

from typing import Callable, overload
import jax
import jax.numpy as jnp
import optax
from plum import dispatch


@overload
def activation(t: Tanh) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.tanh


@overload
def activation(t: Relu) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.relu


@overload
def activation(t: Sigmoid) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.sigmoid


@overload
def activation(t: Softmax) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.softmax


@overload
def activation(t: Identity) -> Callable[[jax.Array], jax.Array]:
    return lambda x: x


@overload
def activation(t: Activation) -> Callable[[jax.Array], jax.Array]:
    raise NotImplementedError


@dispatch
def activation(t: Activation) -> Callable[[jax.Array], jax.Array]:
    raise NotImplementedError


@overload
def initializer(t: Zeros) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.zeros


@overload
def initializer(t: LecunNormal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", in_axis=-1, out_axis=-2)


@overload
def initializer(t: PytorchUniform) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=-1, out_axis=-2)


@overload
def initializer(t: Init) -> jax.nn.initializers.Initializer:
    raise NotImplementedError


@dispatch
def initializer(t: Init) -> jax.nn.initializers.Initializer:
    raise NotImplementedError


@overload
def objective(t: L2) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return lambda y, target: jnp.sum(optax.losses.l2_loss(y, target))


@overload
def objective(t: CrossEntropy) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return lambda y, target: jnp.sum(optax.losses.softmax_cross_entropy(y, target))


@overload
def objective(t: Loss) -> Callable[[jax.Array, jax.Array], jax.Array]:
    raise NotImplementedError


@dispatch
def objective(t: Loss) -> Callable[[jax.Array, jax.Array], jax.Array]:
    raise NotImplementedError
