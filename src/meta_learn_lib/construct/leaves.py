from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.term import (
    Activation,
    Adam,
    Anon,
    CrossEntropy,
    Descent,
    Gaussian,
    HyperStorage,
    Identity,
    Init,
    L2,
    Label,
    LecunNormal,
    Loss,
    Noise,
    Normal,
    Orthogonal,
    PytorchUniform,
    RMap,
    Rademacher,
    Rectify,
    Relu,
    Reparam,
    Reparametrized,
    Sgd,
    SgdNormalized,
    Sigmoid,
    SiluPositive,
    SoftClip,
    SoftRelu,
    Softmax,
    Softplus,
    Squared,
    Tanh,
    Term,
    TrainedStorage,
    Unconstrained,
    UniformUnit,
    Zeros,
)
from meta_learn_lib.algorithms.optimizers import adam, sgd, sgd_normalized
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.utility.distributions import rademacher, uniform_unit
from meta_learn_lib.utility.reparam import (
    silu_positive,
    silu_positive_inverse,
    soft_clip,
    soft_relu,
    softplus_inverse,
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
def initializer(t: Normal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.normal(t.std)


@overload
def initializer(t: Orthogonal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.orthogonal()


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


@overload
def reparametrization(
    r: Unconstrained,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: x, lambda y: y)


@overload
def reparametrization(
    r: Softplus,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (jax.nn.softplus, softplus_inverse)


@overload
def reparametrization(
    r: Rectify,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: jnp.maximum(x, 0.0), lambda y: y)


@overload
def reparametrization(
    r: SoftRelu,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: soft_relu(x, r.sharpness), lambda y: y)


@overload
def reparametrization(
    r: SiluPositive,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: silu_positive(x, r.scale), lambda y: silu_positive_inverse(y, r.scale))


@overload
def reparametrization(
    r: Squared,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: (r.scale * x) ** 2, lambda y: jnp.sqrt(y) / r.scale)


@overload
def reparametrization(
    r: SoftClip,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    return (lambda x: soft_clip(x, r.a, r.b, r.sharpness), lambda y: y)


@overload
def reparametrization(
    r: Reparam,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    raise NotImplementedError


@dispatch
def reparametrization(
    r: Reparam,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    raise NotImplementedError


@overload
def chain(t: Sgd) -> Callable[[jax.Array, jax.Array, jax.Array], optax.GradientTransformation]:
    return sgd


@overload
def chain(t: SgdNormalized) -> Callable[[jax.Array, jax.Array, jax.Array], optax.GradientTransformation]:
    return sgd_normalized


@overload
def chain(t: Adam) -> Callable[[jax.Array, jax.Array, jax.Array], optax.GradientTransformation]:
    return lambda lr, wd, m: adam(lr, wd, m, t.b2, t.eps, t.eps_root)


@overload
def chain[HL, HW, HM, P](
    t: Descent[HL, HW, HM, P],
) -> Callable[[jax.Array, jax.Array, jax.Array], optax.GradientTransformation]:
    raise NotImplementedError


@dispatch
def chain[HL, HW, HM, P](
    t: Descent[HL, HW, HM, P],
) -> Callable[[jax.Array, jax.Array, jax.Array], optax.GradientTransformation]:
    raise NotImplementedError


@overload
def sampler(t: Gaussian) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return jax.random.normal


@overload
def sampler(t: Rademacher) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return rademacher


@overload
def sampler(t: UniformUnit) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return uniform_unit


@overload
def sampler(t: Noise) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    raise NotImplementedError


@dispatch
def sampler(t: Noise) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    raise NotImplementedError


def Hyper(value: float, reparam: Reparam, label: Label) -> Term[Unit, Unit, jax.Array, jax.Array, Unit]:
    return Reparametrized(HyperStorage(value, label), RMap(reparam), Anon())


def Trained(value: float, reparam: Reparam, label: Label) -> Term[Unit, Unit, jax.Array, Unit, jax.Array]:
    return Reparametrized(TrainedStorage(value, label), RMap(reparam), Anon())
