from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Union


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
class MnistConfig:
    n_in: int
    add_spurious_pixel_to_train: bool


@dataclass(frozen=True)
class FashionMnistConfig:
    n_in: int
    add_spurious_pixel_to_train: bool


@dataclass(frozen=True)
class CIFAR10Config:
    n_in: int


@dataclass(frozen=True)
class CIFAR100Config:
    n_in: int


@dataclass(frozen=True)
class DelayAddOnlineConfig:
    t1: int
    t2: int
    tau_task: bool
    n: int  # length of entire online sequence
    nTest: int


@dataclass(frozen=True)
class RTRLConfig:
    start_at_step: int
    momentum1: float
    momentum2: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class RTRLHessianDecompConfig:
    epsilon: float
    start_at_step: int
    momentum1: float
    momentum2: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class RTRLFiniteHvpConfig:
    epsilon: float
    start_at_step: int
    momentum1: float
    momentum2: float
    use_reverse_mode: bool


@dataclass(frozen=True)
class BPTTConfig: ...


@dataclass(frozen=True)
class IdentityConfig: ...


@dataclass(frozen=True)
class RFLOConfig:
    time_constant: float
    use_reverse_mode: bool


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

    value: float
    learnable: bool
    hyperparameter_parametrization: Union[identity, softplus, relu, softrelu, silu_positive, squared, softclip]
    min_value: float
    max_value: float


@dataclass(frozen=True)
class Clip:
    threshold: float
    sharpness: float


@dataclass(frozen=True)
class SGDConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float
    add_clip: Clip | None


@dataclass(frozen=True)
class SGDNormalizedConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    add_clip: Clip | None


@dataclass(frozen=True)
class ExponentiatedGradientAdamConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    momentum: float
    add_clip: Clip | None


@dataclass(frozen=True)
class ExponentiatedGradientConfig:
    learning_rate: HyperparameterConfig
    weight_decay: HyperparameterConfig
    add_clip: Clip | None


@dataclass(frozen=True)
class RecurrenceConfig:
    recurrent_optimizer: Union[
        SGDConfig, SGDNormalizedConfig, AdamConfig, ExponentiatedGradientConfig, ExponentiatedGradientAdamConfig
    ]
    readout_optimizer: Union[
        SGDConfig, SGDNormalizedConfig, AdamConfig, ExponentiatedGradientConfig, ExponentiatedGradientAdamConfig
    ]


type Optimizer = Union[
    SGDConfig,
    SGDNormalizedConfig,
    AdamConfig,
    ExponentiatedGradientConfig,
    ExponentiatedGradientAdamConfig,
    RecurrenceConfig,
]


@dataclass(frozen=True)
class LearnConfig:
    learner: Union[
        RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig, RTRLHessianDecompConfig, RTRLFiniteHvpConfig
    ]
    optimizer: Optimizer
    lanczos_iterations: int
    track_logs: bool
    track_special_logs: bool
    num_virtual_minibatches_per_turn: int  # for data loading purposes. iterate over minibatches. orthogonal scale to # example in minibatch for memory. need for hierarchy


@dataclass(frozen=True)
class LayerNorm:
    epsilon: float
    use_weight: bool
    use_bias: bool


@dataclass(frozen=True)
class NNLayer:
    n: int
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"]
    use_bias: bool
    use_in_readout: bool
    layer_norm: Optional[LayerNorm]
    use_random_init: bool


@dataclass(frozen=True)
class GRULayer:
    n: int
    use_bias: bool
    use_in_readout: bool
    use_random_init: bool


@dataclass(frozen=True)
class LSTMLayer:
    n: int
    use_bias: bool
    use_in_readout: bool
    use_random_init: bool


@dataclass(frozen=True)
class IdentityLayer:
    # no learnable parameters, just applies the activation function
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"]


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
    seed: SeedConfig
    clearml_run: bool
    data_root_dir: str
    epochs: int
    checkpoint_every_n_minibatches: int
    log_dir: str
    logger_config: dict[int, Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig]]
    dataset_configs: dict[str, DatasetConfig]
    datasets: dict[str, Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig, CIFAR10Config, CIFAR100Config]]
    model_dag: dict[str, list[str]]
    model_configs: dict[str, Union[NNLayer, GRULayer, LSTMLayer, IdentityLayer, FeedForwardConfig]]

    learners: dict[int, LearnConfig]

    loss_fn: Literal["cross_entropy", "cross_entropy_with_integer_labels", "mse"]


"""
Use the DAG way of doing inference. This solves
1. allows choosing which functions rereference the data source.
2. allows which functions get to be readout so even rnns can be readout.


1. how to set up easily clearml hpo configs?
2. how to model configs for DAG based inference?

"""


"""
Datasets will now be reorganized so that at each meta level I can specify the dataset that I want which
could be different from validation vs training potentially. additionally instead of a split percentage,
I will just say give me X examples for this meta level, and then based on if I repeatedely sampled from
this dataset, I will select random examples until it is exhausted. 


"""

"""
RL
1. add FeedForwardConfig to transition function
    - basically n_in is now whatever previous n_h was along with output size of final layer
2. add option for transition functions to not output a hidden state. so zeros gets passed around
3. add an identity readout where nontrainable just returns the n_h
3.5. add an option to determine whether readout will use a transition's n_h or if it will just be ()
4. add two new transition functions
    - RL Get State --- will ignore previous n_h and output the state as n_h+. Then the next transition function is the policy that acts on the state.
    - RL Step Env --- will take env state, previous policy's output, and output the next state and update it. 
    so an example looks like
    transition: (RL Get State, NNLayer, NNLayer, FeedForwardConfig (nonrecurrent), RL Step Env)
    readout: should concatenate (t, (), (), (), (), reward) to be (t, reward) which then it can be identity operation on or gamma^t*reward or something
    where t comes from actual data stream of integers that are timesteps. 


"""
