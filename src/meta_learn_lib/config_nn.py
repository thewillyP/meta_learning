from dataclasses import dataclass
from typing import Literal, Union


type Activation = Literal["tanh", "relu", "sigmoid", "identity", "softmax"]


@dataclass(frozen=True)
class LayerNorm:
    epsilon: float
    use_weight: bool
    use_bias: bool


@dataclass(frozen=True)
class NNLayer:
    n: int
    activation_fn: Activation


@dataclass(frozen=True)
class RNNLayer:
    n: int
    activation_fn: Activation
    use_bias: bool
    use_random_init: bool


@dataclass(frozen=True)
class GRULayer:
    n: int
    use_bias: bool
    use_random_init: bool


@dataclass(frozen=True)
class LSTMLayer:
    n: int
    use_bias: bool
    use_random_init: bool


@dataclass(frozen=True)
class IdentityLayer:
    # no learnable parameters, just applies the activation function
    activation_fn: Activation


type LayerConfig = Union[NNLayer, RNNLayer, GRULayer, LSTMLayer, IdentityLayer]


class Node:
    def __init__(self, op: LayerConfig, inputs: list[str]) -> None:
        self.op: LayerConfig = op
        self.inputs: list[str] = inputs


type GraphConfig = dict[str, Node]
