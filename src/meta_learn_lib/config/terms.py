from dataclasses import dataclass

from .bindings import Ref, Refs
from .initialization import Init, Initialization


@dataclass(frozen=True)
class Term: ...


@dataclass(frozen=True)
class Leaf(Term): ...


@dataclass(frozen=True)
class Unary(Term):
    body: Term


@dataclass(frozen=True)
class Call(Leaf):
    name: str


@dataclass(frozen=True)
class Read(Leaf):
    ref: Ref


@dataclass(frozen=True)
class Scan(Unary):
    steps: int


@dataclass(frozen=True)
class Batch(Unary):
    size: int


@dataclass(frozen=True)
class Step:
    body: Term
    input: Ref | None
    name: str | None


@dataclass(frozen=True)
class Sequence(Term):
    steps: tuple[Step, ...]

    @classmethod
    def of(cls, *bodies: Term) -> "Sequence":
        return cls(tuple(Step(body, None, None) for body in bodies))


@dataclass(frozen=True)
class Loss: ...


@dataclass(frozen=True)
class L2(Loss): ...


@dataclass(frozen=True)
class CrossEntropy(Loss): ...


@dataclass(frozen=True)
class Reduction: ...


@dataclass(frozen=True)
class Mean(Reduction): ...


@dataclass(frozen=True)
class Sum(Reduction): ...


@dataclass(frozen=True)
class Supervised(Unary):
    loss: Loss
    reduction: Reduction


@dataclass(frozen=True)
class Reduce(Unary):
    operation: Reduction


@dataclass(frozen=True)
class Scope(Unary):
    initialize: Refs


@dataclass(frozen=True)
class Learn(Unary):
    optimizer: str


@dataclass(frozen=True)
class Computed:
    body: Term


type Recipe = Init | Initialization | Computed
type Slot = Recipe | Ref
type Control = float | Slot


@dataclass(frozen=True)
class Primitive(Leaf): ...


@dataclass(frozen=True)
class Value(Primitive):
    """A callable value declaration, initialized by a scope referencing its name."""

    initial: Slot


@dataclass(frozen=True)
class Linear(Primitive):
    n: int
    parameters: Slot


@dataclass(frozen=True)
class Bias(Primitive):
    parameters: Slot


@dataclass(frozen=True)
class Activation(Leaf): ...


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
class Rnn(Primitive):
    n: int
    act: Activation
    recurrent: Slot
    incoming: Slot
    time_constant: Control
    state: Slot


@dataclass(frozen=True)
class Rule: ...


@dataclass(frozen=True)
class Sgd(Rule):
    learning_rate: Control
    weight_decay: Control
    momentum: Control


@dataclass(frozen=True)
class SgdNormalized(Sgd): ...


@dataclass(frozen=True)
class Adam(Rule):
    learning_rate: Control
    weight_decay: Control
    momentum: Control
    b2: float
    eps: float
    eps_root: float


@dataclass(frozen=True)
class Frozen(Rule): ...


@dataclass(frozen=True)
class Group:
    name: str
    targets: Refs
    rule: Rule


@dataclass(frozen=True)
class Config:
    computations: dict[str, Term]
    optimizers: dict[str, tuple[Group, ...]]
    run: Term
