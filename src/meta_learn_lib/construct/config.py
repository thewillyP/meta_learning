from dataclasses import dataclass


# ── initialisers ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Zeros: ...


@dataclass(frozen=True)
class LecunNormal: ...


@dataclass(frozen=True)
class Normal:
    std: float


@dataclass(frozen=True)
class Orthogonal: ...


@dataclass(frozen=True)
class PytorchUniform: ...


type Init = Zeros | LecunNormal | Normal | Orthogonal | PytorchUniform


# ── noise ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gaussian: ...


@dataclass(frozen=True)
class Rademacher: ...


@dataclass(frozen=True)
class UniformUnit: ...


type Noise = Gaussian | Rademacher | UniformUnit


# ── activations ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tanh: ...


@dataclass(frozen=True)
class Relu: ...


@dataclass(frozen=True)
class Sigmoid: ...


@dataclass(frozen=True)
class Softmax: ...


@dataclass(frozen=True)
class Identity: ...


type Activation = Tanh | Relu | Sigmoid | Softmax | Identity


# ── pieces ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Linear:
    n: int
    init: Init


@dataclass(frozen=True)
class Bias:
    init: Init


@dataclass(frozen=True)
class Rnn:
    n: int
    act: Activation
    time_constant: float
    init: Init
    h0: Init


@dataclass(frozen=True)
class L2: ...


@dataclass(frozen=True)
class CrossEntropy: ...


type Loss = L2 | CrossEntropy


@dataclass(frozen=True)
class Sgd:
    learning_rate: float
    weight_decay: float
    momentum: float


@dataclass(frozen=True)
class SgdNormalized:
    learning_rate: float
    weight_decay: float
    momentum: float


@dataclass(frozen=True)
class Adam:
    learning_rate: float
    weight_decay: float
    momentum: float
    b2: float
    eps: float
    eps_root: float


@dataclass(frozen=True)
class Frozen: ...


type Opt = Sgd | SgdNormalized | Adam | Frozen

type Piece = Linear | Bias | Activation | Rnn | Loss | Opt

# ── names for what is on a wire ───────────────────────────────────────────────

type Shape = str | tuple[()] | tuple[Shape, Shape]


# ── nodes ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Leaf:
    piece: Piece
    names: tuple[str, ...]


@dataclass(frozen=True)
class LearningRateLog: ...


@dataclass(frozen=True)
class Then:
    first: "Config"
    second: "Config"


@dataclass(frozen=True)
class Beside:
    first: "Config"
    second: "Config"


@dataclass(frozen=True)
class Join:
    name: str
    below: "Config"


@dataclass(frozen=True)
class MapAccum:
    over: tuple[str, ...]
    below: "Config"


@dataclass(frozen=True)
class Batch:
    n: int
    over: tuple[str, ...]
    below: "Config"


@dataclass(frozen=True)
class Learn:
    opt: "Config"
    names: tuple[str, ...]
    below: "Config"


@dataclass(frozen=True)
class Rtrl:
    names: tuple[str, ...]
    below: "Config"


@dataclass(frozen=True)
class Rflo:
    decay: str
    decay_start: float
    names: tuple[str, ...]
    below: "Config"


@dataclass(frozen=True)
class Uoro:
    names: tuple[str, ...]
    noise: Noise
    below: "Config"


# ── shorter spellings ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class First:
    below: "Config"


@dataclass(frozen=True)
class Second:
    below: "Config"


@dataclass(frozen=True)
class Pre:
    before: Shape
    after: Shape
    below: "Config"


@dataclass(frozen=True)
class Post:
    before: Shape
    after: Shape
    below: "Config"


@dataclass(frozen=True)
class Input:
    name: str
    below: "Config"


@dataclass(frozen=True)
class Scored:
    model: "Config"
    loss: "Config"


@dataclass(frozen=True)
class Fresh:
    name: str
    below: "Config"


# ── the tree ──────────────────────────────────────────────────────────────────

type Config = (
    Leaf
    | LearningRateLog
    | Then
    | Beside
    | Join
    | MapAccum
    | Batch
    | Learn
    | Rtrl
    | Rflo
    | Uoro
    | First
    | Second
    | Pre
    | Post
    | Input
    | Scored
    | Fresh
)
