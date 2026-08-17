from meta_learn_lib.category.lib_types import Unit

from dataclasses import dataclass
import equinox as eqx
import jax


@dataclass(frozen=True)
class Arch[S, X, Y, HP, P]: ...


@dataclass(frozen=True)
class Init: ...


@dataclass(frozen=True)
class Zeros(Init): ...


@dataclass(frozen=True)
class LecunNormal(Init): ...


@dataclass(frozen=True)
class PytorchUniform(Init): ...


@dataclass(frozen=True)
class Linear(Arch[Unit, jax.Array, jax.Array, Unit, eqx.nn.Linear]):
    n: int
    init: Init


@dataclass(frozen=True)
class Bias(Arch[Unit, jax.Array, jax.Array, Unit, jax.Array]):
    init: Init


@dataclass(frozen=True)
class Activation(Arch[Unit, jax.Array, jax.Array, Unit, Unit]): ...


@dataclass(frozen=True)
class Tanh(Activation): ...


@dataclass(frozen=True)
class Relu(Activation): ...


@dataclass(frozen=True)
class Sigmoid(Activation): ...


@dataclass(frozen=True)
class Softmax(Activation): ...


@dataclass(frozen=True)
class Identity(Activation): ...


@dataclass(frozen=True)
class Loss(Arch[Unit, tuple[jax.Array, jax.Array], jax.Array, Unit, Unit]): ...


@dataclass(frozen=True)
class L2(Loss): ...


@dataclass(frozen=True)
class CrossEntropy(Loss): ...


@dataclass(frozen=True)
class Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2](Arch[tuple[S1, S2], X, Z, tuple[HP1, HP2], tuple[P1, P2]]):
    first: Arch[S1, X, Y, HP1, P1]
    second: Arch[S2, Y, Z, HP2, P2]


@dataclass(frozen=True)
class Sup[S, X, HP, P](
    Arch[
        tuple[tuple[S, Unit], Unit],
        tuple[X, jax.Array],
        jax.Array,
        tuple[tuple[HP, Unit], Unit],
        tuple[tuple[P, Unit], Unit],
    ]
):
    arch: Arch[S, X, jax.Array, HP, P]
    loss: Loss
