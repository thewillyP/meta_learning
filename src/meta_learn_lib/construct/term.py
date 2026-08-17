from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.lib_types import ArrayTree

from dataclasses import dataclass
import equinox as eqx
import jax


@dataclass(frozen=True)
class Term[S, X, Y, HP, P]: ...


@dataclass(frozen=True)
class Init: ...


@dataclass(frozen=True)
class Zeros(Init): ...


@dataclass(frozen=True)
class LecunNormal(Init): ...


@dataclass(frozen=True)
class PytorchUniform(Init): ...


@dataclass(frozen=True)
class Linear(Term[Unit, jax.Array, jax.Array, Unit, eqx.nn.Linear]):
    n: int
    init: Init


@dataclass(frozen=True)
class Bias(Term[Unit, jax.Array, jax.Array, Unit, jax.Array]):
    init: Init


@dataclass(frozen=True)
class Activation(Term[Unit, jax.Array, jax.Array, Unit, Unit]): ...


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
class Loss(Term[Unit, tuple[jax.Array, jax.Array], jax.Array, Unit, Unit]): ...


@dataclass(frozen=True)
class L2(Loss): ...


@dataclass(frozen=True)
class CrossEntropy(Loss): ...


@dataclass(frozen=True)
class Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2](Term[tuple[S1, S2], X, Z, tuple[HP1, HP2], tuple[P1, P2]]):
    first: Term[S1, X, Y, HP1, P1]
    second: Term[S2, Y, Z, HP2, P2]


@dataclass(frozen=True)
class Sup[S, X, HP, P](
    Term[
        tuple[tuple[S, Unit], Unit],
        tuple[X, jax.Array],
        jax.Array,
        tuple[tuple[HP, Unit], Unit],
        tuple[tuple[P, Unit], Unit],
    ]
):
    arch: Term[S, X, jax.Array, HP, P]
    loss: Loss


@dataclass(frozen=True)
class Opt[P](Term[ArrayTree, P, P, Unit, jax.Array]): ...


@dataclass(frozen=True)
class Sgd[P](Opt[P]):
    lr: float


@dataclass(frozen=True)
class Validator[S, X, Y, HP, P, SV, XV, PV]: ...


@dataclass(frozen=True)
class SameModel[S, X, Y, HP, P](Validator[S, X, Y, HP, P, S, X, Unit]): ...


@dataclass(frozen=True)
class Meta[S, X, HP, P, SV, XV, PV](
    Term[
        tuple[tuple[tuple[S, tuple[ArrayTree, P]], Unit], SV],
        tuple[X, XV],
        jax.Array,
        tuple[tuple[HP, Unit], HP],
        tuple[tuple[jax.Array, Unit], PV],
    ]
):
    below: Term[S, X, jax.Array, HP, P]
    opt: Opt[P]
    val: Validator[S, X, jax.Array, HP, P, SV, XV, PV]
