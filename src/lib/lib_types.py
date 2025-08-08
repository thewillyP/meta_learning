from typing import NewType
from dataclasses import dataclass
import jax

PRNG = NewType("PRNG", jax.Array)
FractionalList = NewType("FractionalList", list[float])


@dataclass(frozen=True)
class PaddedData:
    X: jax.Array  # (# examples, # virtual minibatches, # timesteps, feature_dim)
    Y: jax.Array
    mask: jax.Array  # (# examples, # virtual minibatches, 1) last is in int
