from dataclasses import dataclass

from ..category.paralens import ParaLens
from .env import HP, Log, Par, St


@dataclass(frozen=True)
class Optimizer:
    reparam: ParaLens[Log, HP, HP, St, St, Par, Par]
    targets: frozenset[str]
    state: frozenset[str]
