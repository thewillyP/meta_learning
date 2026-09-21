from meta_learn_lib.construct.term import Term
from meta_learn_lib.data_source.source import DataConfig
from meta_learn_lib.log.config import LoggerConfig

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedConfig:
    global_seed: int
    data_seed: int
    parameter_seed: int
    task_seed: int
    sample_seed: int


@dataclass(frozen=True)
class GodConfig[S, X, Y, HP, P]:
    term: Term[S, X, Y, HP, P]
    data: DataConfig
    seed: SeedConfig
    epochs: int
    checkpoint_every_n_minibatches: int
    checkpoint_every_n_epochs: int
    prefetch_buffer_size: int
    clearml_run: bool
    log_title: str
    loggers: tuple[LoggerConfig, ...]
    scalar_queue_size: int
    sample_queue_size: int
