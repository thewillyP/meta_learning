from typing import Optional

import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field
from pyrsistent.typing import PMap, PVector

from meta_learn_lib.lib_types import *
from meta_learn_lib.util import deep_serialize


class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    largest_eigenvalue: Optional[jax.Array] = field(initial=None)
    influence_tensor_norm: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class RNN(PClass):
    w_rec: jax.Array = field(initial=None, serializer=deep_serialize)
    b_rec: jax.Array = field(initial=None, serializer=deep_serialize)
    layer_norm: eqx.Module = field(initial=None, serializer=deep_serialize)


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
    mlps: PMap[int, eqx.nn.Sequential] = field(serializer=deep_serialize)
    rnns: PMap[int, RNN] = field(serializer=deep_serialize)
    grus: PMap[int, eqx.nn.GRUCell] = field(serializer=deep_serialize)
    lstms: PMap[int, eqx.nn.LSTMCell] = field(serializer=deep_serialize)
    learning_rates: PMap[int, jax.Array] = field(serializer=deep_serialize)
    weight_decays: PMap[int, jax.Array] = field(serializer=deep_serialize)
    time_constants: PMap[int, jax.Array] = field(serializer=deep_serialize)
    momentums: PMap[int, jax.Array] = field(serializer=deep_serialize)
    kl_regularizer_betas: PMap[int, jax.Array] = field(serializer=deep_serialize)


class LearningStates(PClass):
    influence_tensors: PMap[int, JACOBIAN] = field(serializer=deep_serialize)
    uoros: PMap[int, UOROState] = field(serializer=deep_serialize)
    midpoint_buffers: PMap[int, MidpointBuffer] = field(serializer=deep_serialize)
    opt_states: PMap[int, optax.OptState] = field(serializer=deep_serialize)


class ModelStates(PClass):
    recurrent_states: PMap[int, jax.Array] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PMap[int, VanillaRecurrentState] = field(serializer=deep_serialize)
    lstm_states: PMap[int, LSTMState] = field(serializer=deep_serialize)
    autoregressive_predictions: PMap[int, jax.Array] = field(serializer=deep_serialize)


class LevelMeta(PClass):
    log: Logs = field(serializer=deep_serialize)
    prngs: PMap[int, PRNG] = field(serializer=deep_serialize)
    ticks: PMap[int, jax.Array] = field(serializer=deep_serialize)


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
