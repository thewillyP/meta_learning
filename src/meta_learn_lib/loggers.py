from meta_learn_lib.experiment import (
    ClearMLLoggerConfig,
    ConsoleLoggerConfig,
    HDF5LoggerConfig,
    LoggerConfig,
    MatplotlibLoggerConfig,
    SQLiteLoggerConfig,
)
from meta_learn_lib.logger import ClearMLLogger, ConsoleLogger, HDF5Logger, Logger, MatplotlibLogger, SQLiteLogger

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, overload, runtime_checkable
import clearml
from plum import dispatch


@runtime_checkable
class NoDeps(Protocol): ...


@runtime_checkable
class HasRunId(Protocol):
    def run_id(self) -> str: ...


@runtime_checkable
class HasClearML(Protocol):
    def task(self) -> clearml.Task: ...


@dataclass(frozen=True)
class LocalRun:
    run_id: Callable[[], str]


@dataclass(frozen=True)
class ClearMLRun:
    run_id: Callable[[], str]
    task: Callable[[], clearml.Task]


@overload
def logger(t: ConsoleLoggerConfig, run: NoDeps) -> Logger:
    return ConsoleLogger()


@overload
def logger(t: MatplotlibLoggerConfig, run: NoDeps) -> Logger:
    return MatplotlibLogger(t.save_dir)


@overload
def logger(t: SQLiteLoggerConfig, run: HasRunId) -> Logger:
    return SQLiteLogger(t.log_dir, run.run_id(), t.checkpoint_every)


@overload
def logger(t: HDF5LoggerConfig, run: HasRunId) -> Logger:
    return HDF5Logger(t.log_dir, run.run_id(), t.checkpoint_every)


@overload
def logger(t: ClearMLLoggerConfig, run: HasClearML) -> Logger:
    return ClearMLLogger(run.task())


@overload
def logger(t: LoggerConfig, run: NoDeps) -> Logger:
    raise NotImplementedError


@dispatch
def logger(t: LoggerConfig, run: NoDeps) -> Logger:
    raise NotImplementedError
