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
class DelayAddConfig:
    t1: int
    t2: int
    numVal: int
    numTr: int
    numTe: int


@dataclass(frozen=True)
class RFLOConfig:
    time_constant: float


@dataclass(frozen=True)
class LearnConfig:
    train_percent: float
    batch_size: int
    log_influence_tensor: bool
    log_immediate_influence_tensor: bool
    learning_rate: float
    num_examples_in_minibatch: int  # for online its num parallel in a batch, for offline its num ex
    num_steps_in_timeseries: int  # for online its 1, for offline its n (could be whole if not BPTT)
    num_steps_to_avg_in_timeseries: int  # for BPTT offline if you want to consume the whole sequence, this better be num_steps_to_avg_in_timeseries = data_length / num_steps_in_timeseries. Otherwise it will partially update. For online this can be whatever, however much you want to update
    learner: Union[Literal["rtrl", "uoro", "identity", "bptt"], RFLOConfig]
    optimizer: Literal["sgd", "sgd_positive", "adam", "sgd_normalized", "sgd_clipped"]
    hyperparameter_parametrization: Literal["identity", "softplus"]


@dataclass(frozen=True)
class RnnConfig:
    activation_fn: Literal["tanh", "relu", "identity"]
    n_h: int


@dataclass(frozen=True)
class FeedForwardConfig:
    ffn_layers: tuple[tuple[int, Literal["tanh", "relu", "sigmoid", "identity", "softmax"]], ...]


@dataclass(frozen=True)
class GodConfig:
    data_root_dir: str
    dataset: Union[Literal["mnist", "fashionmnist"], DelayAddConfig]
    num_base_epochs: int
    checkpoint_every_n_minibatches: int
    seed: SeedConfig
    lossFn: Literal["cross_entropy", "cross_entropy_with_integer_labels"]
    transition_function: tuple[Union[RnnConfig], ...]  # if len()>1 creates stacked recurrence. LSTM/GRU TBD
    readout_function: Union[FeedForwardConfig]

    tau_task: bool
    inner_uoro_std: float
    outer_uoro_std: float
    initialization_std: float
    inner_log_special: bool
    outer_log_special: bool
    inner_lanczos_iterations: int
    outer_lanczos_iterations: int
    inner_clip: float
    inner_clip_sharpness: float
    outer_clip: float
    outer_clip_sharpness: float
    inner_log_expensive: Optional[bool] = None
    outer_log_expensive: Optional[bool] = None
