from meta_learn_lib.experiment import (
    ClearMLLoggerConfig,
    ConsoleLoggerConfig,
    HDF5LoggerConfig,
    LoggerConfig,
    MatplotlibLoggerConfig,
    SQLiteLoggerConfig,
)
from meta_learn_lib.logger import ClearMLLogger, ConsoleLogger, HDF5Logger, Logger, MatplotlibLogger, SQLiteLogger

from dataclasses import dataclass
from typing import overload
import clearml
from plum import dispatch


@dataclass(frozen=True)
class Run:
    log_dir: str
    task_id: str
    checkpoint_every: int


@dataclass(frozen=True)
class ClearMLRun(Run):
    task: clearml.Task


@overload
def logger(t: ConsoleLoggerConfig, run: Run) -> Logger:
    return ConsoleLogger()


@overload
def logger(t: MatplotlibLoggerConfig, run: Run) -> Logger:
    return MatplotlibLogger(t.save_dir)


@overload
def logger(t: SQLiteLoggerConfig, run: Run) -> Logger:
    return SQLiteLogger(run.log_dir, run.task_id, run.checkpoint_every)


@overload
def logger(t: HDF5LoggerConfig, run: Run) -> Logger:
    return HDF5Logger(run.log_dir, run.task_id, run.checkpoint_every)


@overload
def logger(t: ClearMLLoggerConfig, run: ClearMLRun) -> Logger:
    return ClearMLLogger(run.task)


@overload
def logger(t: LoggerConfig, run: Run) -> Logger:
    raise NotImplementedError


@dispatch
def logger(t: LoggerConfig, run: Run) -> Logger:
    raise NotImplementedError
