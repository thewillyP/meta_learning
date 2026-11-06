from dataclasses import dataclass

from ..category.paralens import ParaLens
from .env import Env


@dataclass(frozen=True)
class Optimizer:
    rule: ParaLens[Env, Env, Env, Env, Env, Env]
    targets: frozenset[str]
    state: frozenset[str]
