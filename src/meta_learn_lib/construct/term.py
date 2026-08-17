from meta_learn_lib.category.lib_types import Unit

from dataclasses import dataclass
import equinox as eqx
import jax


@dataclass(frozen=True)
class Arch[S, HP, P]: ...


@dataclass(frozen=True)
class Init: ...


@dataclass(frozen=True)
class Zeros(Init): ...


@dataclass(frozen=True)
class LecunNormal(Init): ...


@dataclass(frozen=True)
class PytorchUniform(Init): ...


@dataclass(frozen=True)
class Linear(Arch[Unit, Unit, eqx.nn.Linear]):
    n: int
    init: Init


@dataclass(frozen=True)
class Bias(Arch[Unit, Unit, jax.Array]):
    init: Init


@dataclass(frozen=True)
class Activation(Arch[Unit, Unit, Unit]): ...


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
class Seq[S1, S2, HP1, HP2, P1, P2](Arch[tuple[S1, S2], tuple[HP1, HP2], tuple[P1, P2]]):
    first: Arch[S1, HP1, P1]
    second: Arch[S2, HP2, P2]
