from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Union

from meta_learn_lib.lib_types import ACTIVATION_FN


@dataclass(frozen=True)
class SlurmParams:
    memory: str
    time: str
    cpu: int
    gpu: int
    log_dir: str
    skip_python_env_install: bool


@dataclass(frozen=True)
class SeedConfig:
    global_seed: int
    data_seed: int
    parameter_seed: int
    test_seed: int


@dataclass(frozen=True)
class DatasetConfig:
    num_examples_in_minibatch: int  # for online its num parallel in a batch, for offline its num ex, per validationn
    num_steps_in_timeseries: int  # for online its 1, for offline its n (could be whole if not TBPTT). Every dataset will be treated as a time series. Even trivial ones.
    num_examples_total: int  # total number of examples in dataset split
    is_test: bool  # whether this split is test or train. if test then it will source from standardized test set


@dataclass(frozen=True)
class MNISTTaskFamily:
    n_in: int
    add_spurious_pixel_to_train: bool
    domain: frozenset[Literal["mnist", "fashion_mnist"]]
    sampling: Literal["stochastic", "deterministic"]


@dataclass(frozen=True)
class CIFAR10TaskFamily:
    n_in: int


@dataclass(frozen=True)
class CIFAR100TaskFamily:
    n_in: int


@dataclass(frozen=True)
class DelayAddTaskFamily:
    t1_lb: int
    t1_ub: int
    t2_lb: int
    t2_ub: int
    tau_task_lb: int
    tau_task_ub: int
    t_train: int  # length of entire online sequence for training
    n_train: int  # number of examples. n_train=1 for online.
    t_test: int  # length of entire online sequence for testing
    n_test: int  # number of examples. n_test=1 for online.


type Task = Union[MNISTTaskFamily, CIFAR10TaskFamily, CIFAR100TaskFamily, DelayAddTaskFamily]


@dataclass(frozen=True)
class HyperparameterConfig:
    @dataclass(frozen=True)
    class identity: ...

    @dataclass(frozen=True)
    class softplus: ...

    @dataclass(frozen=True)
    class relu: ...

    @dataclass(frozen=True)
    class softrelu:
        clip: float

    @dataclass(frozen=True)
    class silu_positive:
        scale: float

    @dataclass(frozen=True)
    class squared:
        scale: float

    @dataclass(frozen=True)
    class softclip:
        a: float | None
        b: float | None
        clip: float

    type Parametrization = Union[identity, softplus, relu, softrelu, silu_positive, squared, softclip]

    value: float
    learnable: bool
    hyperparameter_parametrization: Parametrization
    min_value: float  # used for mandatory gradient projection
    max_value: float  # used for mandatory gradient projection
    id: str  # for tracking how to optimize


type HP = HyperparameterConfig | str  # str for parameter sharing


@dataclass(frozen=True)
class SoftClip:
    threshold: float
    sharpness: float


@dataclass(frozen=True)
class HardClip:
    threshold: float


@dataclass(frozen=True)
class SGDConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP
    add_clip: HardClip | SoftClip | None


@dataclass(frozen=True)
class SGDNormalizedConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: HP
    weight_decay: HP
    add_clip: HardClip | SoftClip | None


@dataclass(frozen=True)
class ExponentiatedGradientAdamConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP
    add_clip: HardClip | SoftClip | None


@dataclass(frozen=True)
class ExponentiatedGradientConfig:
    learning_rate: HP
    weight_decay: HP
    add_clip: HardClip | SoftClip | None


type Optimizer = Union[
    SGDConfig,
    SGDNormalizedConfig,
    AdamConfig,
    ExponentiatedGradientConfig,
    ExponentiatedGradientAdamConfig,
]


@dataclass(frozen=True)
class OptimizerAssignment:
    target: frozenset[str]
    optimizer: Optimizer
    per_parameter: bool  # separate hyperparameter instance per parameter vs shared


@dataclass(frozen=True)
class RTRLConfig:
    start_at_step: int
    hessian_damping: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class RTRLHessianDecompConfig:
    epsilon: float
    start_at_step: int
    hessian_damping: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class RTRLFiniteHvpConfig:
    epsilon: float
    start_at_step: int
    hessian_damping: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class BPTTConfig: ...


@dataclass(frozen=True)
class IdentityConfig: ...


@dataclass(frozen=True)
class RFLOConfig:
    time_constant: HP
    use_reverse_mode: bool


@dataclass(frozen=True)
class UOROConfig:
    std: HP


type GradientMethod = Union[
    RTRLConfig,
    RTRLHessianDecompConfig,
    RTRLFiniteHvpConfig,
    BPTTConfig,
    IdentityConfig,
    RFLOConfig,
    UOROConfig,
]


@dataclass(frozen=True)
class LearnConfig:
    model_learner: GradientMethod
    optimizer_learner: GradientMethod
    optimizer: list[OptimizerAssignment]


@dataclass(frozen=True)
class LayerNorm:
    epsilon: float
    use_weight: bool
    use_bias: bool


@dataclass(frozen=True)
class NNLayer:
    n: int
    activation_fn: ACTIVATION_FN
    use_bias: bool
    layer_norm: Optional[LayerNorm]


@dataclass(frozen=True)
class VanillaRNNLayer:
    nn_layer: NNLayer
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
class Scan:  # Wraps around a layer and repeats it in an unfold manner
    n: int | None  # if None then scan over input instead of stacking input n times
    graph: dict[str, list[str]]
    autoregressive_mask: Literal["teacher_forcing", "identity", "none"]
    input_source: str  # which node in the graph to source input I will be scanning over. This is time series
    pred_source: str  # which node in the graph to source teacher predictions.


type Node = Union[NNLayer, VanillaRNNLayer, GRULayer, LSTMLayer, Scan]


@dataclass(frozen=True)
class HDF5LoggerConfig: ...


@dataclass(frozen=True)
class ClearMLLoggerConfig: ...


@dataclass(frozen=True)
class PrintLoggerConfig: ...


@dataclass(frozen=True)
class MatplotlibLoggerConfig:
    save_dir: str


type LoggerConfig = Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig]


@dataclass(frozen=True)
class VaeObjective:
    beta: HP


@dataclass(frozen=True)
class RegressionObjective: ...


@dataclass(frozen=True)
class CrossEntropyObjective: ...


type ObjectiveFn = Union[VaeObjective, RegressionObjective, CrossEntropyObjective]


@dataclass(frozen=True)
class GodConfig:
    seed: SeedConfig
    clearml_run: bool
    data_root_dir: str
    epochs: int
    checkpoint_every_n_minibatches: int
    log_dir: str
    logger_config: list[LoggerConfig]

    # Inference
    transition_graph: dict[str, list[str]]
    transition_nodes: dict[str, Node]
    readout_graph: dict[str, list[str]]
    readout_nodes: dict[str, Node]
    objective_fn: list[ObjectiveFn]  # objective function used for validation inference at each meta level

    # Dataloading
    dataset_meta_levels: list[DatasetConfig]  # the number of meta learning levels
    dataset_meta_sources: list[Task]  # what dataset validation inference uses at each meta level

    # Learning
    learners: list[LearnConfig]
