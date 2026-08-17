from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.category.paralens import para_autodiff
from meta_learn_lib.construct.term import (
    Activation,
    Arch,
    Bias,
    Identity,
    Init,
    LecunNormal,
    Linear,
    PytorchUniform,
    Relu,
    Seq,
    Sigmoid,
    Softmax,
    Tanh,
    Zeros,
)
from meta_learn_lib.lib_types import PRNG


from typing import Callable, overload
import equinox as eqx
import jax
import jax.numpy as jnp
from plum import dispatch


@overload
def build(
    t: Linear,
) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, eqx.nn.Linear, eqx.nn.Linear]:
    def h(hp_p: tuple[Unit, eqx.nn.Linear], x: jax.Array) -> jax.Array:
        _, layer = hp_p
        return layer(x)

    return to_mealy(para_autodiff(h))


@overload
def build(
    t: Bias,
) -> Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, jax.Array, jax.Array]:
    def h(hp_p: tuple[Unit, jax.Array], x: jax.Array) -> jax.Array:
        _, b = hp_p
        return x + b

    return to_mealy(para_autodiff(h))


type Pointwise = Mealy[Unit, Unit, jax.Array, jax.Array, jax.Array, jax.Array, Unit, Unit, Unit, Unit]


@overload
def build(t: Activation) -> Pointwise:
    f = activation(t)

    def h(hp_p: tuple[Unit, Unit], x: jax.Array) -> jax.Array:
        return f(x)

    return to_mealy(para_autodiff(h))


@overload
def build[S1, S2, HP1, HP2, P1, P2](
    t: Seq[S1, S2, HP1, HP2, P1, P2],
) -> Mealy[
    tuple[S1, S2],
    tuple[S1, S2],
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple[HP1, HP2],
    tuple[HP1, HP2],
    tuple[P1, P2],
    tuple[P1, P2],
]:
    return build(t.first) >> build(t.second)


@overload
def build[S, HP, P](t: Arch[S, HP, P]) -> Mealy[S, S, jax.Array, jax.Array, jax.Array, jax.Array, HP, HP, P, P]:
    raise NotImplementedError


@dispatch
def build[S, HP, P](t: Arch[S, HP, P]) -> Mealy[S, S, jax.Array, jax.Array, jax.Array, jax.Array, HP, HP, P, P]:
    raise NotImplementedError


@overload
def out(t: Linear, n_in: int) -> int:
    return t.n


@overload
def out(t: Bias, n_in: int) -> int:
    return n_in


@overload
def out(t: Activation, n_in: int) -> int:
    return n_in


@overload
def out[S1, S2, HP1, HP2, P1, P2](t: Seq[S1, S2, HP1, HP2, P1, P2], n_in: int) -> int:
    return out(t.second, out(t.first, n_in))


@overload
def out[S, HP, P](t: Arch[S, HP, P], n_in: int) -> int:
    raise NotImplementedError


@dispatch
def out[S, HP, P](t: Arch[S, HP, P], n_in: int) -> int:
    raise NotImplementedError


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
    return jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")


@overload
def initializer(t: PytorchUniform) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")


@overload
def initializer(t: Init) -> jax.nn.initializers.Initializer:
    raise NotImplementedError


@dispatch
def initializer(t: Init) -> jax.nn.initializers.Initializer:
    raise NotImplementedError


@overload
def init(t: Linear, n_in: int, key: PRNG) -> tuple[tuple[Unit, eqx.nn.Linear], Unit]:
    k_layer, k_w = jax.random.split(key)
    layer = eqx.nn.Linear(n_in, t.n, use_bias=False, key=k_layer)
    w = initializer(t.init)(k_w, (t.n, n_in))
    return ((Unit(), eqx.tree_at(lambda l: l.weight, layer, w)), Unit())


@overload
def init(t: Bias, n_in: int, key: PRNG) -> tuple[tuple[Unit, jax.Array], Unit]:
    return ((Unit(), jnp.asarray(initializer(t.init)(key, (n_in,)))), Unit())


@overload
def init(t: Activation, n_in: int, key: PRNG) -> tuple[tuple[Unit, Unit], Unit]:
    return ((Unit(), Unit()), Unit())


@overload
def init[S1, S2, HP1, HP2, P1, P2](
    t: Seq[S1, S2, HP1, HP2, P1, P2], n_in: int, key: PRNG
) -> tuple[tuple[tuple[HP1, HP2], tuple[P1, P2]], tuple[S1, S2]]:
    k1, k2 = jax.random.split(key)
    (hp1, p1), s1 = init(t.first, n_in, PRNG(k1))
    (hp2, p2), s2 = init(t.second, out(t.first, n_in), PRNG(k2))
    return (((hp1, hp2), (p1, p2)), (s1, s2))


@overload
def init[S, HP, P](t: Arch[S, HP, P], n_in: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError


@dispatch
def init[S, HP, P](t: Arch[S, HP, P], n_in: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError
