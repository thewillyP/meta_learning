from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.lib_types import ArrayTree, JACOBIAN, PRNG

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
class Normal(Init):
    std: float


@dataclass(frozen=True)
class Orthogonal(Init): ...


@dataclass(frozen=True)
class PytorchUniform(Init): ...


@dataclass(frozen=True)
class Anon: ...


@dataclass(frozen=True)
class Declares:
    name: str


@dataclass(frozen=True)
class Uses:
    name: str


type Label = Anon | Declares | Uses


@dataclass(frozen=True)
class Reparametrization[P2, P]: ...


@dataclass(frozen=True)
class Unconstrained[P](Reparametrization[P, P]): ...


@dataclass(frozen=True)
class Softplus(Reparametrization[jax.Array, jax.Array]): ...


@dataclass(frozen=True)
class Rectify(Reparametrization[jax.Array, jax.Array]): ...


@dataclass(frozen=True)
class SoftRelu(Reparametrization[jax.Array, jax.Array]):
    sharpness: float


@dataclass(frozen=True)
class SiluPositive(Reparametrization[jax.Array, jax.Array]):
    scale: float


@dataclass(frozen=True)
class Squared(Reparametrization[jax.Array, jax.Array]):
    scale: float


@dataclass(frozen=True)
class SoftClip(Reparametrization[jax.Array, jax.Array]):
    a: float | None
    b: float | None
    sharpness: float


@dataclass(frozen=True)
class RMap[P](Reparametrization[P, P]):
    by: Reparametrization[jax.Array, jax.Array]


@dataclass(frozen=True)
class PortOf[HP, P, S](Reparametrization[tuple[tuple[tuple[HP, P], S], tuple[Unit, Unit]], tuple[HP, P]]): ...


@dataclass(frozen=True)
class Demoted[HP1, HP0, H, PV, S0, SO, P0, SV](
    Reparametrization[
        tuple[
            tuple[tuple[HP1, tuple[tuple[tuple[HP0, H], Unit], PV]], tuple[tuple[tuple[S0, tuple[SO, P0]], Unit], SV]],
            tuple[Unit, Unit],
        ],
        tuple[HP0, P0],
    ]
): ...


@dataclass(frozen=True)
class RCompose[A, B, C](Reparametrization[A, C]):
    first: Reparametrization[A, B]
    second: Reparametrization[B, C]


@dataclass(frozen=True)
class Below[HP2, HP1, H2, PV2, S1, SO2, P1, SV2](
    Reparametrization[
        tuple[
            tuple[
                tuple[HP2, tuple[tuple[tuple[HP1, H2], Unit], PV2]], tuple[tuple[tuple[S1, tuple[SO2, P1]], Unit], SV2]
            ],
            tuple[Unit, Unit],
        ],
        tuple[tuple[tuple[HP1, P1], S1], tuple[Unit, Unit]],
    ]
): ...


@dataclass(frozen=True)
class RSplit[A2, A, B2, B](Reparametrization[tuple[A2, B2], tuple[A, B]]):
    first: Reparametrization[A2, A]
    second: Reparametrization[B2, B]


@dataclass(frozen=True)
class LowRank(Reparametrization[tuple[jax.Array, jax.Array], eqx.nn.Linear]):
    rank: int


@dataclass(frozen=True)
class HyperStorage(Term[Unit, Unit, jax.Array, jax.Array, Unit]):
    value: float
    label: Label


@dataclass(frozen=True)
class TrainedStorage(Term[Unit, Unit, jax.Array, Unit, jax.Array]):
    value: float
    label: Label


@dataclass(frozen=True)
class Const(Term[Unit, Unit, jax.Array, Unit, Unit]):
    value: float


@dataclass(frozen=True)
class Linear(Term[Unit, jax.Array, jax.Array, Unit, eqx.nn.Linear]):
    n: int
    init: Init
    label: Label


@dataclass(frozen=True)
class Bias(Term[Unit, jax.Array, jax.Array, Unit, jax.Array]):
    init: Init
    label: Label


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
class Rnn[HPA, PA](Term[jax.Array, jax.Array, jax.Array, HPA, tuple[PA, eqx.nn.Linear]]):
    n: int
    act: Activation
    alpha: Term[Unit, Unit, jax.Array, HPA, PA]
    init: Init
    h0: Init
    label: Label


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
class Opt[SO, HPO, H, P](Term[SO, P, P, HPO, H]): ...


@dataclass(frozen=True)
class Sgd[HL, PL, HW, PW, HM, PM, P](Opt[ArrayTree, tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM], P]):
    lr: Term[Unit, Unit, jax.Array, HL, PL]
    wd: Term[Unit, Unit, jax.Array, HW, PW]
    momentum: Term[Unit, Unit, jax.Array, HM, PM]


@dataclass(frozen=True)
class SgdNormalized[HL, PL, HW, PW, HM, PM, P](Opt[ArrayTree, tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM], P]):
    lr: Term[Unit, Unit, jax.Array, HL, PL]
    wd: Term[Unit, Unit, jax.Array, HW, PW]
    momentum: Term[Unit, Unit, jax.Array, HM, PM]


@dataclass(frozen=True)
class Adam[HL, PL, HW, PW, HM, PM, P](Opt[ArrayTree, tuple[tuple[HL, HW], HM], tuple[tuple[PL, PW], PM], P]):
    lr: Term[Unit, Unit, jax.Array, HL, PL]
    wd: Term[Unit, Unit, jax.Array, HW, PW]
    momentum: Term[Unit, Unit, jax.Array, HM, PM]
    b2: float
    eps: float
    eps_root: float


@dataclass(frozen=True)
class Frozen[P](Opt[ArrayTree, Unit, Unit, P]): ...


@dataclass(frozen=True)
class Split[SO1, HPO1, H1, P1, SO2, HPO2, H2, P2](
    Opt[tuple[SO1, SO2], tuple[HPO1, HPO2], tuple[H1, H2], tuple[P1, P2]]
):
    first: Opt[SO1, HPO1, H1, P1]
    second: Opt[SO2, HPO2, H2, P2]


@dataclass(frozen=True)
class Scan[S, X, Y, HP, P](Term[S, X, Y, HP, P]):
    below: Term[S, X, Y, HP, P]


@dataclass(frozen=True)
class BatchData[S, X, Y, HP, P](Term[S, X, Y, HP, P]):
    below: Term[S, X, Y, HP, P]
    n: int


@dataclass(frozen=True)
class BatchParams[S, X, Y, HP, P](Term[S, X, Y, HP, P]):
    below: Term[S, X, Y, HP, P]
    n: int


@dataclass(frozen=True)
class BatchPop[S, X, Y, HP, P](Term[S, X, Y, HP, P]):
    below: Term[S, X, Y, HP, P]
    n: int


@dataclass(frozen=True)
class Noise: ...


@dataclass(frozen=True)
class Gaussian(Noise): ...


@dataclass(frozen=True)
class Rademacher(Noise): ...


@dataclass(frozen=True)
class UniformUnit(Noise): ...


@dataclass(frozen=True)
class RTRL[S, X, Y, HP, P](Term[tuple[S, JACOBIAN], X, Y, tuple[HP, Unit], P]):
    below: Term[S, X, Y, HP, P]


@dataclass(frozen=True)
class RFLO[S, X, Y, HP, P, HD](Term[tuple[S, JACOBIAN], X, Y, tuple[HP, HD], P]):
    below: Term[S, X, Y, HP, P]
    decay: Term[Unit, Unit, jax.Array, HD, Unit]


@dataclass(frozen=True)
class UORO[S, X, Y, HP, P](Term[tuple[S, tuple[jax.Array, jax.Array, PRNG]], X, Y, tuple[HP, Unit], P]):
    below: Term[S, X, Y, HP, P]
    noise: Noise


@dataclass(frozen=True)
class Validator[S, X, Y, HP, P, SV, XV, HPV, PV]: ...


@dataclass(frozen=True)
class Validate[S, X, Y, HP, P, SV, XV, HQ, Q](Validator[S, X, Y, HP, P, SV, XV, Unit, Unit]):
    term: Term[SV, XV, Y, HQ, Q]
    r: Reparametrization[tuple[tuple[tuple[HP, P], S], tuple[Unit, Unit]], tuple[HQ, Q]]


@dataclass(frozen=True)
class SameModel[S, X, Y, HP, P](Validator[S, X, Y, HP, P, S, X, Unit, Unit]): ...


@dataclass(frozen=True)
class Meta[S, X, HP, P, SO, H, HPO, HPV, SV, XV, PV](
    Term[
        tuple[tuple[tuple[S, tuple[SO, P]], Unit], SV],
        tuple[X, XV],
        jax.Array,
        tuple[tuple[HPO, Unit], HPV],
        tuple[tuple[tuple[HP, H], Unit], PV],
    ]
):
    below: Term[S, X, jax.Array, HP, P]
    opt: Opt[SO, HPO, H, P]
    val: Validator[S, X, jax.Array, HP, P, SV, XV, HPV, PV]


@dataclass(frozen=True)
class Reparametrized[S, X, Y, HP, HP2, P, P2](Term[S, X, Y, HP2, P2]):
    below: Term[S, X, Y, HP, P]
    r: Reparametrization[tuple[HP2, P2], tuple[HP, P]]
    label: Label


@dataclass(frozen=True)
class Shared[S, X, Y, HP, P](Term[S, X, Y, HP, P]):
    below: Term[S, X, Y, HP, P]
