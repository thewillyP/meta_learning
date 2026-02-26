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
    task_seed: int


@dataclass(frozen=True)
class MetaOptimizationConfig:
    batch: int  # num optimizers that run in parallel
    num_steps: int
    reset_t: int | None  # if not None, then reset the environment every reset_t steps. if None, then never reset
    track_influence_in: frozenset[int]  # which levels to track influence for


@dataclass(frozen=True)
class ValidationConfig:
    num_examples_in_minibatch: int  # for online its num parallel in a batch, for offline its num ex, per validation
    num_steps_in_timeseries: int  # for online its 1, for offline its n (could be whole if not TBPTT). Every dataset will be treated as a time series. Even trivial ones.
    num_examples_total: int  # total number of examples in dataset split
    is_test: bool  # whether this split is test or train. if test then it will source from standardized test set
    task_batch_size: int
    reset_t: int | None
    track_influence_in: frozenset[int]  # which levels to track influence for


@dataclass(frozen=True)
class MNISTTaskFamily:
    type Domain = Literal["mnist", "fashion_mnist"]
    patch_h: int
    patch_w: int
    label_last_only: bool
    add_spurious_pixel_to_train: bool
    domain: frozenset[Domain]
    normalize: bool


@dataclass(frozen=True)
class CIFAR10TaskFamily:
    patch_h: int
    patch_w: int
    label_last_only: bool


@dataclass(frozen=True)
class CIFAR100TaskFamily:
    patch_h: int
    patch_w: int
    label_last_only: bool


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
    type Kind = Literal["learning_rate", "weight_decay", "momentum", "time_constant", "kl_regularizer_beta"]

    value: float
    kind: Kind
    count: int  # how many unique parameters of this kind
    hyperparameter_parametrization: Parametrization
    min_value: float  # used for mandatory gradient projection
    max_value: float  # used for mandatory gradient projection
    level: int  # which meta level this hyperparameter belongs to. used for tracking how to optimize
    parametrizes_transition: bool  # whether this hp influences transitional dynamics vs readout


type HP = str


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


@dataclass(frozen=True)
class SGDNormalizedConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP


@dataclass(frozen=True)
class ExponentiatedGradientConfig:
    learning_rate: HP
    weight_decay: HP
    momentum: HP
    use_adam: bool


type Optimizer = Union[
    SGDConfig,
    SGDNormalizedConfig,
    AdamConfig,
    ExponentiatedGradientConfig,
]


@dataclass(frozen=True)
class OptimizerAssignment:
    target: frozenset[str]
    optimizer: Optimizer


@dataclass(frozen=True)
class RTRLConfig:
    start_at_step: int
    use_reverse_mode: bool
    damping: float


@dataclass(frozen=True)
class RTRLFiniteHvpConfig:
    epsilon: float
    rtrl_config: RTRLConfig


@dataclass(frozen=True)
class BPTTConfig:
    truncate_at: int | None  # if None, then perform full bptt


@dataclass(frozen=True)
class IdentityLearnerConfig: ...


@dataclass(frozen=True)
class RFLOConfig:
    time_constant: HP
    use_reverse_mode: bool


@dataclass(frozen=True)
class UOROConfig:
    std: float


type GradientMethod = Union[
    RTRLConfig,
    RTRLFiniteHvpConfig,
    BPTTConfig,
    IdentityLearnerConfig,
    RFLOConfig,
    UOROConfig,
]


@dataclass(frozen=True)
class GradientConfig:
    method: GradientMethod
    add_clip: HardClip | SoftClip | None  # these are placed here because vl_gr could also want to be clipped
    scale: float


@dataclass(frozen=True)
class LearnConfig:
    model_learner: GradientConfig
    optimizer_learner: GradientConfig
    optimizer: dict[str, OptimizerAssignment]


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
    time_constant: HP


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
    graph: dict[str, list[str]]
    autoregressive_mask: Literal["teacher_forcing", "identity", "erase"]
    pred_source: str  # which node in the graph to source teacher predictions.
    start_token: Literal["zeros"]


@dataclass(frozen=True)
class Repeat:
    n: int


@dataclass(frozen=True)
class Concat: ...


@dataclass(frozen=True)
class ToEmpty: ...


@dataclass(frozen=True)
class UnlabeledSource: ...


@dataclass(frozen=True)
class LabeledSource: ...


type Node = Union[
    NNLayer,
    VanillaRNNLayer,
    GRULayer,
    LSTMLayer,
    Scan,
    UnlabeledSource,
    LabeledSource,
    Repeat,
    Concat,
    ToEmpty,
]


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
class RegressionObjective: ...


@dataclass(frozen=True)
class CrossEntropyObjective:
    mode: Literal["cross_entropy_with_integer_labels", "cross_entropy"]


@dataclass(frozen=True)
class ELBOObjective:
    beta: HP
    likelihood: CrossEntropyObjective | RegressionObjective


type ObjectiveFn = Union[ELBOObjective, RegressionObjective, CrossEntropyObjective]


@dataclass(frozen=True)
class TrackLogs:
    gradient: bool
    hessian_contains_nans: bool
    largest_eigenvalue: bool
    influence_tensor: bool
    immediate_influence_tensor: bool
    largest_jac_eigenvalue: bool
    jacobian: bool


@dataclass(frozen=True)
class MetaConfig:
    objective_fn: ObjectiveFn  # objective function used for validation inference at each meta level
    dataset_validation: ValidationConfig
    dataset_source: Task  # what dataset validation inference uses at each meta level
    meta_opt: MetaOptimizationConfig  # number of simultaneous tasks at same level
    learner: LearnConfig
    test_seed: int
    track_logs: TrackLogs


@dataclass(frozen=True)
class GodConfig:
    seed: SeedConfig
    clearml_run: bool
    data_root_dir: str
    log_dir: str
    logger_config: list[LoggerConfig]
    epochs: int
    checkpoint_every_n_minibatches: int

    transition_graph: dict[str, list[str]]
    readout_graph: dict[str, list[str]]
    nodes: dict[str, Node]
    hyperparameters: dict[HP, HyperparameterConfig]

    levels: list[MetaConfig]

    label_mask_value: float
    unlabeled_mask_value: float
    num_tasks: int
