from dataclasses import dataclass
from typing import Literal, Optional, Union

import equinox as eqx
import jax

from meta_learn_lib.lib_types import ACTIVATION_FN, Canon, CarryTransform, PixelTransform, Uncanon


@dataclass(frozen=True)
class SlurmParams:
    memory: str
    time: str
    cpu: int
    gpu: int
    log_dir: str
    skip_python_env_install: bool
    comment: str


@dataclass(frozen=True)
class SeedConfig:
    global_seed: int
    data_seed: int
    parameter_seed: int
    task_seed: int
    sample_seed: int


@dataclass(frozen=True)
class MNISTTaskFamily:
    patch_h: int
    patch_w: int
    label_last_only: bool
    add_spurious_pixel_to_train: bool
    pixel_transform: PixelTransform


@dataclass(frozen=True)
class FashionMNISTTaskFamily:
    patch_h: int
    patch_w: int
    label_last_only: bool
    add_spurious_pixel_to_train: bool
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


@dataclass(frozen=True)
class GaussianNoiseTaskFamily:
    shape: tuple[int, ...]
    n: int


@dataclass(frozen=True)
class GridTaskFamily:
    dim: int
    min_value: float
    max_value: float
    n_per_axis: int
    tag: int
    mode: Literal["uniform", "quantile"]


@dataclass(frozen=True)
class MNISTSequenceTaskFamily:
    time_series_length: int
    pixel_transform: PixelTransform


@dataclass(frozen=True)
class SOSTaskFamily:
    grid_size: int
    sigma_x: float
    sigma_y: float
    n: int
    patch_h: int
    patch_w: int
    region: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max); ignored when region_mode="full"
    region_mode: Literal["full", "exclude_region", "only_region", "grid"]
    tag: int


@dataclass(frozen=True)
class NTMCopyTaskFamily:
    min_seq_len: int
    max_seq_len: int
    bits_per_vector: int
    n_train: int
    n_test: int


type Task = Union[
    MNISTTaskFamily,
    FashionMNISTTaskFamily,
    CIFAR10TaskFamily,
    CIFAR100TaskFamily,
    DelayAddTaskFamily,
    GaussianNoiseTaskFamily,
    GridTaskFamily,
    MNISTSequenceTaskFamily,
    SOSTaskFamily,
    NTMCopyTaskFamily,
]


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
    min_value: float  # used for mandatory gradient projection (static, baked into HLO)
    max_value: float  # used for mandatory gradient projection (static, baked into HLO)
    level: int  # which meta level this hyperparameter belongs to. used for tracking how to optimize
    parametrizes_transition: bool  # whether this hp influences transitional dynamics vs readout


type HP = str


class SoftClip(eqx.Module):
    threshold: jax.Array
    sharpness: jax.Array


class HardClip(eqx.Module):
    threshold: jax.Array


class HardClipElementwise(eqx.Module):
    threshold: jax.Array


class SoftNormClip(eqx.Module):
    bound: jax.Array
    ema_decay: jax.Array
    headroom: jax.Array
    init_ema: jax.Array
    eps_root: jax.Array


type Clip = Union[HardClip, HardClipElementwise, SoftClip, SoftNormClip]


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
    second_momentum: jax.Array
    eps: jax.Array
    eps_root: jax.Array


class ExponentiatedGradientConfig(eqx.Module):
    base: Union[SGDConfig, AdamConfig]


type Optimizer = Union[
    SGDConfig,
    SGDNormalizedConfig,
    AdamConfig,
    ExponentiatedGradientConfig,
]


class OptimizerAssignment(eqx.Module):
    target: frozenset[Canon]
    optimizer: Optimizer
    add_clip: Optional[Clip]


class InfluenceColumnClip(eqx.Module):
    threshold: jax.Array
    eps_root: jax.Array
    stop_gradient: bool


class UnitCircleClip(eqx.Module):
    margin: jax.Array
    measure: Literal["eigenvalue", "growth"]


class RTRLConfig(eqx.Module):
    start_at_step: int
    damping: jax.Array
    beta: jax.Array
    use_finite_hvp: jax.Array | None
    influence_clip: InfluenceColumnClip | None
    propagation_clip: jax.Array | None
    lr_edge_margin: jax.Array | None
    unit_circle_clip: UnitCircleClip | None


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
    add_clip: Optional[Clip]
    scale: jax.Array


class LearnConfig(eqx.Module):
    model_learner: GradientConfig
    optimizer_learner: GradientConfig
    optimizer: dict[str, OptimizerAssignment]


class LayerNorm(eqx.Module):
    epsilon: float
    use_weight: bool
    use_bias: bool


class GroupNorm(eqx.Module):
    groups: int
    epsilon: float
    channelwise_affine: bool


class NNLayer(eqx.Module):
    n: int
    activation_fn: ACTIVATION_FN
    use_bias: bool
    init: Literal["lecun_normal", "pytorch_default"]


class VanillaRNNLayer(eqx.Module):
    nn_layer: NNLayer
    layer_norm: Optional[LayerNorm | GroupNorm]
    use_random_init: bool
    time_constant: HP


@dataclass(frozen=True)
class Conv2dLayer:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    use_bias: bool


@dataclass(frozen=True)
class ConvTranspose2dLayer:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    use_bias: bool


@dataclass(frozen=True)
class MaxPool2dLayer:
    kernel_size: int
    stride: int


@dataclass(frozen=True)
class AvgPool2dLayer:
    kernel_size: int
    stride: int


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


class Scan(eqx.Module):  # Wraps around a layer and repeats it in an unfold manner
    graph: dict[Uncanon, frozenset[Uncanon]]
    autoregressive_mask: Literal["teacher_forcing", "identity", "erase"]
    carry_transform: CarryTransform
    pred_source: Uncanon  # which node in the graph to source teacher predictions.
    start_token: Literal["zeros", "input_zero"]


class MemoryScan(eqx.Module):
    graph: dict[Uncanon, frozenset[Uncanon]]
    K: int
    cell_shape: tuple[int, ...]
    reset_inner_state: bool


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
class ExtractMu:
    n: int


@dataclass(frozen=True)
class Reshape:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Activation:
    activation_fn: ACTIVATION_FN


@dataclass(frozen=True)
class Take:
    start: int
    length: int


@dataclass(frozen=True)
class Interpolate:
    n_steps: int
    start: Uncanon
    end: Uncanon


type Node = Union[
    NNLayer,
    VanillaRNNLayer,
    GRULayer,
    LSTMLayer,
    Conv2dLayer,
    ConvTranspose2dLayer,
    MaxPool2dLayer,
    AvgPool2dLayer,
    Scan,
    MemoryScan,
    UnlabeledSource,
    LabeledSource,
    Repeat,
    Concat,
    ToEmpty,
    ReparameterizeLayer,
    MergeOutputs,
    ExtractZ,
    ExtractMu,
    Reshape,
    Activation,
    LayerNorm,
    GroupNorm,
    Take,
    Interpolate,
]


@dataclass(frozen=True)
class HDF5LoggerConfig:
    enabled: bool


@dataclass(frozen=True)
class SQLiteLoggerConfig:
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
    sqlite: SQLiteLoggerConfig
    console: ConsoleLoggerConfig
    matplotlib: MatplotlibLoggerConfig
    scalar_queue_size: int
    sample_queue_size: int


type Reduction = Literal["sum", "mean", "total_sum"]


@dataclass(frozen=True)
class RegressionObjective:
    reduction: Reduction


@dataclass(frozen=True)
class CrossEntropyObjective:
    mode: Literal["cross_entropy_with_integer_labels", "cross_entropy"]


@dataclass(frozen=True)
class BernoulliObjective:
    reduction: Reduction


@dataclass(frozen=True)
class NoopObjective:
    pass


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


type ObjectiveFn = Union[ELBOObjective, RegressionObjective, CrossEntropyObjective, BernoulliObjective, NoopObjective]


@dataclass(frozen=True)
class GaussianSampleInput: ...


@dataclass(frozen=True)
class DataSampleInput: ...


@dataclass(frozen=True)
class GridSampleInput:
    min_value: float
    max_value: float
    n_per_axis: int
    mode: Literal["uniform", "quantile"]


@dataclass(frozen=True)
class InterpolationSampleInput:
    pixel_transform: PixelTransform


@dataclass(frozen=True)
class SOSGridSampleInput:
    dataset: SOSTaskFamily
    n_per_axis: int


type SampleInput = Union[
    GaussianSampleInput,
    DataSampleInput,
    GridSampleInput,
    InterpolationSampleInput,
    SOSGridSampleInput,
]


@dataclass(frozen=True)
class ImageReporter:
    title: str


@dataclass(frozen=True)
class PlotReporter:
    title: str


@dataclass(frozen=True)
class UMAPReporter:
    title: str


@dataclass(frozen=True)
class GridReporter:
    title: str
    rows: int
    cols: int
    show_z_labels: bool


@dataclass(frozen=True)
class PerSampleGridReporter:
    grid: GridReporter


@dataclass(frozen=True)
class MIGMetric:
    n_bins: int


@dataclass(frozen=True)
class ModularityMetric:
    n_bins: int


@dataclass(frozen=True)
class TCMetric:
    pass


type DisentanglementMetric = Union[MIGMetric, ModularityMetric, TCMetric]


@dataclass(frozen=True)
class DisentanglementReporter:
    title: str
    metrics: tuple[DisentanglementMetric, ...]


@dataclass(frozen=True)
class GridDeformationReporter:
    title: str
    n_per_axis: int


type SampleReporter = Union[
    ImageReporter,
    PlotReporter,
    UMAPReporter,
    GridReporter,
    PerSampleGridReporter,
    DisentanglementReporter,
    GridDeformationReporter,
]


class SampleGeneratorConfig(eqx.Module):
    transition_graph: dict[Uncanon, frozenset[Uncanon]]
    readout_graph: dict[Uncanon, frozenset[Uncanon]]
    source_nodes: dict[Canon, Node]
    aliases: dict[Uncanon, Canon]
    input_shape: tuple[int, ...]
    num_samples: int
    every_n_epochs: int
    seed: int | None
    shuffle: bool
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
    shuffle: bool


class MetaConfig(eqx.Module):
    objective_fn: ObjectiveFn
    dataset_source: Task
    dataset: DatasetConfig
    validation: StepConfig
    nested: StepConfig
    learner: LearnConfig
    track_logs: TrackLogs
    test_seed: int
    collect_predictions: bool


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

    transition_graph: dict[Uncanon, frozenset[Uncanon]]
    readout_graph: dict[Uncanon, frozenset[Uncanon]]
    nodes: dict[Canon, Node]
    aliases: dict[Uncanon, Canon]
    hyperparameters: dict[HP, HyperparameterConfig]

    levels: list[MetaConfig]

    sample_generators: list[SampleGeneratorConfig]

    label_mask_value: jax.Array
    unlabeled_mask_value: jax.Array
    num_tasks: int
    prefetch_buffer_size: int
    dataloader_chunk_size: int | None
