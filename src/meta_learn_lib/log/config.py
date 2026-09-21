from dataclasses import dataclass


@dataclass(frozen=True)
class LoggerConfig: ...


@dataclass(frozen=True)
class ClearMLLoggerConfig(LoggerConfig): ...


@dataclass(frozen=True)
class HDF5LoggerConfig(LoggerConfig):
    log_dir: str
    checkpoint_every: int


@dataclass(frozen=True)
class SQLiteLoggerConfig(LoggerConfig):
    log_dir: str
    checkpoint_every: int


@dataclass(frozen=True)
class ConsoleLoggerConfig(LoggerConfig): ...


@dataclass(frozen=True)
class MatplotlibLoggerConfig(LoggerConfig):
    save_dir: str
