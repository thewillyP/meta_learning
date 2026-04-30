from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field
from pyrsistent.typing import PMap, PVector

from meta_learn_lib.lib_types import *
from meta_learn_lib.util import deep_serialize


@dataclass(frozen=True)
class ParamMeta:
    learnable: bool
    min_value: float
    max_value: float
    parametrizes_transition: bool


@dataclass(frozen=True)
class StateMeta:
    is_stateful: frozenset[int]


type Meta = ParamMeta | StateMeta


class Tagged[T](PClass):
    value: T = field(serializer=deep_serialize)
    meta: Meta = field()


class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    largest_eigenvalue: Optional[jax.Array] = field(initial=None)
    influence_tensor_norm: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class RNN(PClass):
    w_rec: Tagged[jax.Array] = field(initial=None, serializer=deep_serialize)
    b_rec: Tagged[jax.Array] = field(initial=None, serializer=deep_serialize)
    layer_norm: Tagged[eqx.Module] = field(initial=None, serializer=deep_serialize)


class VanillaRecurrentState(PClass):
    activation: jax.Array = field(serializer=deep_serialize)
    activation_fn: ACTIVATION_FN = field(serializer=deep_serialize)


class LSTMState(PClass):
    h: jax.Array = field(serializer=deep_serialize)
    c: jax.Array = field(serializer=deep_serialize)


class UOROState(PClass):
    A: jax.Array = field(serializer=deep_serialize)
    B: jax.Array = field(serializer=deep_serialize)


class MidpointBuffer(PClass):
    P_prev: JACOBIAN = field(serializer=deep_serialize)
    predictor: JACOBIAN = field(serializer=deep_serialize)


class Parameters(PClass):
    mlps: PMap[int, Tagged[eqx.nn.Sequential]] = field(serializer=deep_serialize)
    rnns: PMap[int, RNN] = field(serializer=deep_serialize)
    grus: PMap[int, Tagged[eqx.nn.GRUCell]] = field(serializer=deep_serialize)
    lstms: PMap[int, Tagged[eqx.nn.LSTMCell]] = field(serializer=deep_serialize)
    learning_rates: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)
    weight_decays: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)
    time_constants: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)
    momentums: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)
    kl_regularizer_betas: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)


class LearningStates(PClass):
    influence_tensors: PMap[int, Tagged[JACOBIAN]] = field(serializer=deep_serialize)
    uoros: PMap[int, Tagged[UOROState]] = field(serializer=deep_serialize)
    midpoint_buffers: PMap[int, Tagged[MidpointBuffer]] = field(serializer=deep_serialize)
    opt_states: PMap[int, Tagged[optax.OptState]] = field(serializer=deep_serialize)


class ModelStates(PClass):
    recurrent_states: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PMap[int, Tagged[VanillaRecurrentState]] = field(serializer=deep_serialize)
    lstm_states: PMap[int, Tagged[LSTMState]] = field(serializer=deep_serialize)
    autoregressive_predictions: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)


class LevelMeta(PClass):
    log: Tagged[Logs] = field(serializer=deep_serialize)
    prngs: PMap[int, Tagged[PRNG]] = field(serializer=deep_serialize)
    ticks: PMap[int, Tagged[jax.Array]] = field(serializer=deep_serialize)


class GodState(PClass):
    model_states: PVector[ModelStates] = field(serializer=deep_serialize)
    learning_states: PVector[LearningStates] = field(serializer=deep_serialize)
    meta_parameters: PVector[Parameters] = field(serializer=deep_serialize)
    level_meta: PVector[LevelMeta] = field(serializer=deep_serialize)
    prng: PRNG = field(serializer=deep_serialize)


class Outputs(PClass):
    prediction: PREDICTION | None = field(serializer=deep_serialize, initial=None)
    mu: jax.Array | None = field(serializer=deep_serialize, initial=None)
    log_var: jax.Array | None = field(serializer=deep_serialize, initial=None)
    z: jax.Array | None = field(serializer=deep_serialize, initial=None)
    log_q_z: jax.Array | None = field(serializer=deep_serialize, initial=None)
