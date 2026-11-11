from dataclasses import dataclass

from ..category.paralens import ParaLens
from .env import HP, Log, Par


@dataclass(frozen=True)
class Optimizer:
    reparam: ParaLens[Log, HP, HP, Par, Par, Par, Par]
    targets: frozenset[str]
    state: frozenset[str]
