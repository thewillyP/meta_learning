from typing import NewType
from dataclasses import dataclass
import jax
from jaxtyping import PyTree

PRNG = NewType("PRNG", jax.Array)
FractionalList = NewType("FractionalList", list[float])

ACTIVATION = NewType("ACTIVATION", jax.Array)
PREDICTION = NewType("PREDICTION", jax.Array)
GRADIENT = NewType("GRADIENT", jax.Array)  # is a vector
JACOBIAN = NewType("JACOBIAN", jax.Array)  # is a matrix
INPUT = NewType("INPUT", jax.Array)  # is a vector
LABEL = NewType("LABEL", jax.Array)  # is a vector
REC_STATE = NewType("REC_STATE", jax.Array)  # is a vector
REC_PARAM = NewType("REC_PARAM", jax.Array)  # is a vector
LOSS = NewType("LOSS", jax.Array)  # is a scalar


@dataclass(frozen=True)
class traverse[DATA]:
    d: DATA


@dataclass(frozen=True)
class batched[DATA]:
    b: DATA


@dataclass(frozen=True)
class PaddedData:
    X: jax.Array  # (# examples, # virtual minibatches, # timesteps, feature_dim)
    Y: jax.Array
    mask: jax.Array  # (# examples, # virtual minibatches, 1) last is in int
