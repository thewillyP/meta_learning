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


@dataclass(frozen=True)
class SeedConfig:
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
class DelayAddOnlineConfig:
    t1: int
    t2: int
    tau_task: bool
    n: int  # length of entire online sequence
    nTest: int


@dataclass(frozen=True)
class RTRLConfig: ...


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
class SGDConfig:
    learning_rate: float


@dataclass(frozen=True)
class SGDNormalizedConfig(SGDConfig): ...


@dataclass(frozen=True)
class SGDClipConfig(SGDConfig):
    clip_threshold: float
    clip_sharpness: float


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: float


@dataclass(frozen=True)
class DataConfig:
    train_percent: float  # % data devote to learn, meta 1 validation, meta 2 validation, etc
    num_examples_in_minibatch: int  # for online its num parallel in a batch, for offline its num ex, per validationn
    num_steps_in_timeseries: int  # for online its 1, for offline its n (could be whole if not TBPTT)
    num_times_to_avg_in_timeseries: int  # for BPTT offline if you want to consume the whole sequence, this better be num_steps_to_avg_in_timeseries = data_length / num_steps_in_timeseries. Otherwise it will partially update. For online this can be whatever, however much you want to update


@dataclass(frozen=True)
class LearnConfig:
    learner: Union[RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig]
    optimizer: Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig]
    hyperparameter_parametrization: Literal["identity", "softplus", "relu"]
    lanczos_iterations: int
    track_logs: bool
    track_special_logs: bool


@dataclass(frozen=True)
class NNLayer:
    n: int
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"]
    use_bias: bool


@dataclass(frozen=True)
class FeedForwardConfig:
    ffw_layers: dict[int, NNLayer]


@dataclass(frozen=True)
class GodConfig:
    clearml_run: bool
    data_root_dir: str
    dataset: Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig]
    num_base_epochs: int
    checkpoint_every_n_minibatches: int
    seed: SeedConfig
    loss_fn: Literal["cross_entropy", "cross_entropy_with_integer_labels", "mse"]
    transition_function: dict[int, Union[NNLayer]]  # if len()>1 creates stacked recurrence. LSTM/GRU TBD
    readout_function: Union[FeedForwardConfig]
    learners: dict[int, LearnConfig]
    data: dict[int, DataConfig]
    num_virtual_minibatches_per_turn: int  # for data loading purposes. iterate over minibatches. orthogonal scale to # example in minibatch for memory
    ignore_validation_inference_recurrence: bool  # will make sparser influence tensors that ignore validation inference
    readout_uses_input_data: bool
    test_batch_size: int  # for test dataloader, how many examples to use in a batch
