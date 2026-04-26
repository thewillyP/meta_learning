from dataclasses import dataclass
from typing import Literal, Optional, Union

import equinox as eqx
import jax

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
    sample_seed: int


@dataclass(frozen=True)
class MNISTTaskFamily:
    type Domain = Literal["mnist", "fashion_mnist"]
    type PixelTransform = Literal["normalize", "binarize", "raw"]
    patch_h: int
    patch_w: int
    label_last_only: bool
    add_spurious_pixel_to_train: bool
    domain: frozenset[Domain]
    pixel_transform: PixelTransform


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


class HyperparameterConfig(eqx.Module):
    class identity(eqx.Module): ...

    class softplus(eqx.Module): ...

    class relu(eqx.Module): ...

    class softrelu(eqx.Module):
        clip: jax.Array

    class silu_positive(eqx.Module):
        scale: jax.Array

    class squared(eqx.Module):
        scale: jax.Array

    class softclip(eqx.Module):
        a: Optional[jax.Array]
        b: Optional[jax.Array]
        clip: jax.Array

    type Parametrization = Union[identity, softplus, relu, softrelu, silu_positive, squared, softclip]
    type Kind = Literal["learning_rate", "weight_decay", "momentum", "time_constant", "kl_regularizer_beta"]

    value: float
    kind: Kind
    count: int  # how many unique parameters of this kind
    hyperparameter_parametrization: Parametrization
    min_value: jax.Array  # used for mandatory gradient projection
    max_value: jax.Array  # used for mandatory gradient projection
    level: int  # which meta level this hyperparameter belongs to. used for tracking how to optimize
    parametrizes_transition: bool  # whether this hp influences transitional dynamics vs readout


type HP = str


class SoftClip(eqx.Module):
    threshold: jax.Array
    sharpness: jax.Array


class HardClip(eqx.Module):
    threshold: jax.Array


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


class AdamConfig(eqx.Module):
    learning_rate: HP
    weight_decay: HP
    momentum: HP
    eps: jax.Array
    eps_root: jax.Array


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


class OptimizerAssignment(eqx.Module):
    target: frozenset[str]
    optimizer: Optimizer


class RTRLConfig(eqx.Module):
    start_at_step: int
    damping: jax.Array
    beta: jax.Array
    use_finite_hvp: jax.Array | None


class TikhonovRTRLConfig(eqx.Module):
    rtrl_config: RTRLConfig


class PadeRTRLConfig(eqx.Module):
    rtrl_config: RTRLConfig


class MidpointRTRLConfig(eqx.Module):
    rtrl_config: RTRLConfig


class HeunRTRLConfig(eqx.Module):
    rtrl_config: RTRLConfig


class ImplicitEulerRTRLConfig(eqx.Module):
    rtrl_config: RTRLConfig
    num_arnoldi_iters: int


@dataclass(frozen=True)
class BPTTConfig:
    truncate_at: Optional[int]  # if None, then perform full bptt


@dataclass(frozen=True)
class IdentityLearnerConfig:
    bptt_config: BPTTConfig


class RFLOConfig(eqx.Module):
    time_constant: HP
    rtrl_config: RTRLConfig


class UOROConfig(eqx.Module):
    type Distribution = Literal["uniform", "normal"]
    std: jax.Array
    distribution: Distribution
    rtrl_config: RTRLConfig


@dataclass(frozen=True)
class ImmediateLearnerConfig: ...


type GradientMethod = Union[
    RTRLConfig,
    TikhonovRTRLConfig,
    PadeRTRLConfig,
    MidpointRTRLConfig,
    HeunRTRLConfig,
    ImplicitEulerRTRLConfig,
    BPTTConfig,
    IdentityLearnerConfig,
    RFLOConfig,
    UOROConfig,
    ImmediateLearnerConfig,
]


class GradientConfig(eqx.Module):
    method: GradientMethod
    add_clip: Optional[Union[HardClip, SoftClip]]  # these are placed here because vl_gr could also want to be clipped
    scale: jax.Array


class LearnConfig(eqx.Module):
    model_learner: GradientConfig
    optimizer_learner: GradientConfig
    optimizer: dict[str, OptimizerAssignment]


class LayerNorm(eqx.Module):
    epsilon: jax.Array
    use_weight: bool
    use_bias: bool


class NNLayer(eqx.Module):
    n: int
    activation_fn: ACTIVATION_FN
    use_bias: bool
    layer_norm: Optional[LayerNorm]


class VanillaRNNLayer(eqx.Module):
    nn_layer: NNLayer
    use_random_init: bool
    time_constant: HP


@dataclass(frozen=True)
class GRULayer:
    n: int
    use_bias: bool
    use_random_init: bool
    time_constant: HP


@dataclass(frozen=True)
class LSTMLayer:
    n: int
    use_bias: bool
    use_random_init: bool
    time_constant: HP


@dataclass(frozen=True)
class Scan:  # Wraps around a layer and repeats it in an unfold manner
    graph: dict[str, frozenset[str]]
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


@dataclass(frozen=True)
class ReparameterizeLayer: ...


@dataclass(frozen=True)
class MergeOutputs: ...


@dataclass(frozen=True)
class ExtractZ:
    n: int


@dataclass(frozen=True)
class Reshape:
    shape: tuple[int, ...]


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
    ReparameterizeLayer,
    MergeOutputs,
    ExtractZ,
    Reshape,
]


@dataclass(frozen=True)
class HDF5LoggerConfig:
    enabled: bool


@dataclass(frozen=True)
class ClearMLLoggerConfig:
    enabled: bool


@dataclass(frozen=True)
class ConsoleLoggerConfig:
    enabled: bool


@dataclass(frozen=True)
class MatplotlibLoggerConfig:
    save_dir: str
    enabled: bool


@dataclass(frozen=True)
class LoggersConfig:
    clearml: ClearMLLoggerConfig
    hdf5: HDF5LoggerConfig
    console: ConsoleLoggerConfig
    matplotlib: MatplotlibLoggerConfig


type Reduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class RegressionObjective:
    reduction: Reduction


@dataclass(frozen=True)
class CrossEntropyObjective:
    mode: Literal["cross_entropy_with_integer_labels", "cross_entropy"]


@dataclass(frozen=True)
class BernoulliObjective:
    reduction: Reduction


class ELBOObjective(eqx.Module):
    @dataclass(frozen=True)
    class GaussianPosterior: ...

    class GaussianPrior(eqx.Module):
        mu: jax.Array
        log_var: jax.Array

    type Posterior = GaussianPosterior
    type Prior = GaussianPrior

    beta: HP
    likelihood: RegressionObjective | BernoulliObjective
    posterior: Posterior
    prior: Prior


type ObjectiveFn = Union[ELBOObjective, RegressionObjective, CrossEntropyObjective, BernoulliObjective]


@dataclass(frozen=True)
class GaussianSampleInput: ...


@dataclass(frozen=True)
class DataSampleInput: ...


type SampleInput = Union[GaussianSampleInput, DataSampleInput]


@dataclass(frozen=True)
class ImageReporter:
    title: str


type SampleReporter = Union[ImageReporter]


@dataclass(frozen=True)
class SampleGeneratorConfig:
    transition_graph: dict[str, frozenset[str]]
    readout_graph: dict[str, frozenset[str]]
    source_nodes: dict[str, Node]
    input_shape: tuple[int, ...]
    num_samples: int
    every_n_epochs: int
    input: SampleInput
    reporter: SampleReporter


@dataclass(frozen=True)
class TrackLogs:
    gradient: bool
    hessian_contains_nans: bool
    largest_eigenvalue: bool
    influence_tensor_norm: bool
    immediate_influence_tensor: bool
    largest_jac_eigenvalue: bool
    jacobian: bool


@dataclass(frozen=True)
class StepConfig:
    num_steps: int
    batch: int
    reset_t: Optional[int]
    track_influence_in: frozenset[int]


@dataclass(frozen=True)
class DatasetConfig:
    num_examples_in_minibatch: int
    num_examples_total: int
    is_test: bool
    augment: bool


class MetaConfig(eqx.Module):
    objective_fn: ObjectiveFn
    dataset_source: Task
    dataset: DatasetConfig
    validation: StepConfig
    nested: StepConfig
    learner: LearnConfig
    track_logs: TrackLogs
    test_seed: int


class GodConfig(eqx.Module):
    seed: SeedConfig
    clearml_run: bool
    data_root_dir: str
    log_dir: str
    log_title: str
    logger_config: LoggersConfig
    epochs: int
    checkpoint_every_n_minibatches: int
    checkpoint_every_n_epochs: int

    transition_graph: dict[str, frozenset[str]]
    readout_graph: dict[str, frozenset[str]]
    nodes: dict[str, Node]
    hyperparameters: dict[HP, HyperparameterConfig]

    levels: list[MetaConfig]

    sample_generators: list[SampleGeneratorConfig]

    label_mask_value: jax.Array
    unlabeled_mask_value: jax.Array
    num_tasks: int
    prefetch_buffer_size: int
