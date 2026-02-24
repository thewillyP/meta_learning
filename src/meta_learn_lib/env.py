from typing import Optional

import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field
from pyrsistent.typing import PMap, PVector

from meta_learn_lib.lib_types import *
from meta_learn_lib.util import deep_serialize


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================


# My parameter and states are always batched per virtue of opt setup.
class Parameter[T](PClass):
    value: T = field(serializer=deep_serialize)
    is_learnable: bool = field()
    min_value: float = field()
    max_value: float = field()
    parametrizes_transition: bool = field()


class State[T](PClass):
    value: T = field(serializer=deep_serialize)
    is_stateful: frozenset[int] = field()


class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    largest_eigenvalue: Optional[jax.Array] = field(initial=None)
    influence_tensor: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class MLP(PClass):
    model: Parameter[eqx.nn.Sequential] = field(serializer=deep_serialize)


class RNN(PClass):
    w_rec: Parameter[jax.Array] = field(serializer=deep_serialize)
    b_rec: Parameter[jax.Array] = field(serializer=deep_serialize)
    layer_norm: Parameter[eqx.Module] = field(serializer=deep_serialize)


class RecurrentState(PClass):
    activation: State[jax.Array] = field(serializer=deep_serialize)


class VanillaRecurrentState(RecurrentState):
    activation_fn: ACTIVATION_FN = field(serializer=deep_serialize)


class LSTMState(PClass):
    h: State[jax.Array] = field(serializer=deep_serialize)
    c: State[jax.Array] = field(serializer=deep_serialize)


class UOROState(PClass):
    A: State[jax.Array] = field(serializer=deep_serialize)
    B: State[jax.Array] = field(serializer=deep_serialize)


class Parameters(PClass):
    mlps: PMap[int, MLP] = field(serializer=deep_serialize)
    rnns: PMap[int, RNN] = field(serializer=deep_serialize)
    grus: PMap[int, Parameter[eqx.nn.GRUCell]] = field(serializer=deep_serialize)
    lstms: PMap[int, Parameter[eqx.nn.LSTMCell]] = field(serializer=deep_serialize)
    learning_rates: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    weight_decays: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    time_constants: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    momentums: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    kl_regularizer_betas: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)


class LearningStates(PClass):
    influence_tensors: PMap[int, State[JACOBIAN]] = field(serializer=deep_serialize)
    uoros: PMap[int, UOROState] = field(serializer=deep_serialize)
    opt_states: PMap[int, State[optax.OptState]] = field(serializer=deep_serialize)


class ModelStates(PClass):
    recurrent_states: PMap[int, RecurrentState] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PMap[int, VanillaRecurrentState] = field(serializer=deep_serialize)
    lstm_states: PMap[int, LSTMState] = field(serializer=deep_serialize)
    autoregressive_predictions: PMap[int, State[jax.Array]] = field(serializer=deep_serialize)


class LevelMeta(PClass):
    tick: jax.Array = field(serializer=deep_serialize)
    log: State[Logs] = field(serializer=deep_serialize)
    prngs: PMap[int, State[PRNG]] = field(serializer=deep_serialize)


class GodState(PClass):
    model_states: PVector[ModelStates] = field(serializer=deep_serialize)
    learning_states: PVector[LearningStates] = field(serializer=deep_serialize)
    meta_parameters: PVector[Parameters] = field(serializer=deep_serialize)
    level_meta: PVector[LevelMeta] = field(serializer=deep_serialize)
    prng: PRNG = field(serializer=deep_serialize)


class Outputs(PClass):
    prediction: PREDICTION = field(serializer=deep_serialize)
    logit: LOGITS = field(serializer=deep_serialize)
