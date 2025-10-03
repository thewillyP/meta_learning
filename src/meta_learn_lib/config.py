from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class DockerContainerSource:
    docker_url: str
    type: str = "docker_url"


@dataclass(frozen=True)
class SifContainerSource:
    sif_path: str
    type: str = "sif_path"


@dataclass(frozen=True)
class ArtifactContainerSource:
    project: str
    dataset_name: str
    type: str = "artifact_task"


@dataclass(frozen=True)
class SlurmParams:
    memory: str
    time: str
    cpu: int
    gpu: int
    log_dir: str
    singularity_overlay: str
    singularity_binds: str
    container_source: Union[DockerContainerSource, SifContainerSource, ArtifactContainerSource]
    use_singularity: bool
    setup_commands: str
    skip_python_env_install: bool


@dataclass(frozen=True)
class SeedConfig:
    global_seed: int
    data_seed: int
    parameter_seed: int
    test_seed: int


@dataclass(frozen=True)
class MnistConfig:
    n_in: int


@dataclass(frozen=True)
class FashionMnistConfig:
    n_in: int


@dataclass(frozen=True)
class CIFAR10Config:
    n_in: int


@dataclass(frozen=True)
class DelayAddOnlineConfig:
    t1: int
    t2: int
    tau_task: bool
    n: int  # length of entire online sequence
    nTest: int


@dataclass(frozen=True)
class RTRLConfig: ...


@dataclass(frozen=True)
class RTRLHessianDecompConfig:
    epsilon: float


@dataclass(frozen=True)
class RTRLFiniteHvpConfig:
    epsilon: float


@dataclass(frozen=True)
class BPTTConfig: ...


@dataclass(frozen=True)
class IdentityConfig: ...


@dataclass(frozen=True)
class RFLOConfig:
    time_constant: float


@dataclass(frozen=True)
class UOROConfig:
    std: float


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

    value: float
    learnable: bool
    hyperparameter_parametrization: Union[identity, softplus, relu, softrelu]


@dataclass(frozen=True)
class SGDConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float


@dataclass(frozen=True)
class SGDNormalizedConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float


@dataclass(frozen=True)
class SGDClipConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float
    clip_threshold: float
    clip_sharpness: float


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig


@dataclass(frozen=True)
class DataConfig:
    train_percent: float  # % data devote to learn, meta 1 validation, meta 2 validation, etc
    num_examples_in_minibatch: int  # for online its num parallel in a batch, for offline its num ex, per validationn
    num_steps_in_timeseries: int  # for online its 1, for offline its n (could be whole if not TBPTT)
    num_times_to_avg_in_timeseries: int  # for BPTT offline if you want to consume the whole sequence, this better be num_steps_to_avg_in_timeseries = data_length / num_steps_in_timeseries. Otherwise it will partially update. For online this can be whatever, however much you want to update


@dataclass(frozen=True)
class LearnConfig:
    learner: Union[
        RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig, RTRLHessianDecompConfig, RTRLFiniteHvpConfig
    ]
    optimizer: Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig]
    lanczos_iterations: int
    track_logs: bool
    track_special_logs: bool
    num_virtual_minibatches_per_turn: int  # for data loading purposes. iterate over minibatches. orthogonal scale to # example in minibatch for memory. need for hierarchy


@dataclass(frozen=True)
class NNLayer:
    n: int
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"]
    use_bias: bool


@dataclass(frozen=True)
class GRULayer:
    n: int
    use_bias: bool


@dataclass(frozen=True)
class LSTMLayer:
    n: int
    use_bias: bool


@dataclass(frozen=True)
class FeedForwardConfig:
    ffw_layers: dict[int, NNLayer]


@dataclass(frozen=True)
class HDF5LoggerConfig: ...


@dataclass(frozen=True)
class ClearMLLoggerConfig: ...


@dataclass(frozen=True)
class PrintLoggerConfig: ...


@dataclass(frozen=True)
class MatplotlibLoggerConfig:
    save_dir: str


@dataclass(frozen=True)
class GodConfig:
    clearml_run: bool
    data_root_dir: str
    log_dir: str
    dataset: Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig, CIFAR10Config]
    num_base_epochs: int
    checkpoint_every_n_minibatches: int
    seed: SeedConfig
    loss_fn: Literal["cross_entropy", "cross_entropy_with_integer_labels", "mse"]
    transition_function: dict[int, Union[NNLayer, GRULayer, LSTMLayer]]  # if len()>1 creates stacked recurrence.
    readout_function: Union[FeedForwardConfig]
    learners: dict[int, LearnConfig]
    data: dict[int, DataConfig]
    ignore_validation_inference_recurrence: bool  # will make sparser influence tensors that ignore validation inference
    readout_uses_input_data: bool
    logger_config: tuple[Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig], ...]
    treat_inference_state_as_online: bool  # if true, influence tensors will be computed for inference state
