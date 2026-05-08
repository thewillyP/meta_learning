from typing import Literal, NewType
import equinox as eqx
import jax

PRNG = NewType("PRNG", jax.Array)

ACTIVATION = NewType("ACTIVATION", jax.Array)
GRADIENT = NewType("GRADIENT", jax.Array)  # is a vector
JACOBIAN = NewType("JACOBIAN", jax.Array)  # is a matrix
INPUT = NewType("INPUT", jax.Array)  # is a vector
LABEL = NewType("LABEL", jax.Array)  # is a vector
REC_STATE = NewType("REC_STATE", jax.Array)  # is a vector
REC_PARAM = NewType("REC_PARAM", jax.Array)  # is a vector
LOSS = NewType("LOSS", jax.Array)  # is a scalar

type Tag = Literal["scan", "batch", "feature", "time"]


class NamedStat(eqx.Module):
    data: jax.Array
    axes: tuple[Tag, ...] = eqx.field(static=True)


type STAT = dict[str, NamedStat]

LOGITS = NewType("LOGITS", jax.Array)
PREDICTION = NewType("PREDICTION", jax.Array)

type ACTIVATION_FN = Literal["tanh", "relu", "sigmoid", "identity", "softmax"]

type PixelTransform = Literal["normalize", "binarize", "raw"]

type Category = Literal["param", "state", "learning_state"]

type S_ID = tuple[str, int | None]
