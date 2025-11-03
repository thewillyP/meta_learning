from typing import Callable, Literal, Optional
import equinox as eqx
import jax
import optax
import jax.numpy as jnp
from pyrsistent import PClass, field, pmap, thaw
from pyrsistent.typing import PMap
from pyrsistent._pmap import PMap as PMapClass

from meta_learn_lib.lib_types import *


def deep_serialize(_, obj):
    """Recursively serialize pyrsistent objects to Python built-ins"""
    if isinstance(obj, PClass):
        serialized = obj.serialize()
        return {k: deep_serialize(_, v) for k, v in serialized.items()}
    elif isinstance(obj, PMapClass):
        thawed = thaw(obj)
        return {k: deep_serialize(_, v) for k, v in thawed.items()}
    elif isinstance(obj, dict):
        return {k: deep_serialize(_, v) for k, v in obj.items()}
    else:
        return obj


# Register PMap as PyTree
def _pmap_tree_flatten(pm):
    keys = tuple(sorted(pm.keys()))
    values = [pm[k] for k in keys]
    return values, keys


def _pmap_tree_unflatten(keys, values):
    return pmap(zip(keys, values))


jax.tree_util.register_pytree_node(PMapClass, _pmap_tree_flatten, _pmap_tree_unflatten)


# PyTree registration helpers
def register_pytree(cls, static_fields):
    """Register a class as a PyTree"""

    def tree_flatten(obj):
        all_fields = set(cls._pclass_fields.keys())
        dynamic_fields = all_fields - static_fields
        static_field_values = {name: getattr(obj, name) for name in static_fields if hasattr(obj, name)}
        dynamic_values = [getattr(obj, name) for name in sorted(dynamic_fields) if hasattr(obj, name)]
        return dynamic_values, (sorted(dynamic_fields), static_field_values)

    def tree_unflatten(aux_data, values):
        dynamic_fields, static_field_values = aux_data
        kwargs = dict(zip(dynamic_fields, values))
        kwargs.update(static_field_values)
        return cls(**kwargs)

    jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================


class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    immediate_influence_contains_nans: Optional[bool] = field(initial=None)


class SpecialLogs(PClass):
    influence_tensor: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class CustomSequential(eqx.Module):
    model: eqx.nn.Sequential

    def __init__(
        self,
        layer_defs: list[tuple[int, bool, Callable[[jax.Array], jax.Array], Optional[eqx.nn.LayerNorm]]],
        input_size: int,
        key: PRNG,
    ):
        layers = []
        in_size = input_size
        layer_keys = jax.random.split(key, len(layer_defs))

        for (out_size, use_bias, activation, layer_norm), k in zip(layer_defs, layer_keys):
            linear = eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=k)
            new_weight = jax.random.normal(k, (out_size, in_size)) * jnp.sqrt(1 / in_size)
            new_bias = jnp.zeros((out_size,)) if use_bias else None
            where = lambda l: (l.weight, l.bias)
            new_linear = eqx.tree_at(where, linear, (new_weight, new_bias))
            layers.append(new_linear)
            if layer_norm is not None:
                layers.append(layer_norm)
            layers.append(eqx.nn.Lambda(activation))
            in_size = out_size

        self.model = eqx.nn.Sequential(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)


class RNNState(PClass):
    activation: jax.Array = field()
    n_h: int = field()
    n_in: int = field()
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"] = field()


class LSTMState(PClass):
    h: jax.Array = field()
    c: jax.Array = field()
    n_h: int = field()
    n_in: int = field()


class RNN(PClass):
    w_rec: jax.Array = field()
    b_rec: Optional[jax.Array] = field()
    layer_norm: Optional[eqx.nn.LayerNorm] = field()


class UOROState(PClass):
    A: jax.Array = field()
    B: jax.Array = field()


class Hyperparameter(PClass):
    value: jax.Array = field()
    learnable: bool = field()


class LearningParameter(PClass):
    learning_rate: Optional[Hyperparameter] = field()
    weight_decay: Optional[Hyperparameter] = field()
    rflo_timeconstant: Optional[float] = field()


class LearningState(PClass):
    influence_tensor: Optional[JACOBIAN] = field()
    uoro: Optional[UOROState] = field(serializer=deep_serialize)
    opt_state: Optional[optax.OptState] = field()
    rflo_t: Optional[jax.Array] = field()


class InferenceParameter(PClass):
    rnn: Optional[RNN] = field(serializer=deep_serialize, initial=None)
    gru: Optional[eqx.nn.GRUCell] = field(initial=None)
    lstm: Optional[eqx.nn.LSTMCell] = field(initial=None)


class InferenceState(PClass):
    rnn: Optional[RNNState] = field(serializer=deep_serialize)
    lstm: Optional[LSTMState] = field(serializer=deep_serialize)


class General(PClass):
    current_virtual_minibatch: jax.Array = field()
    logs: Optional[Logs] = field(serializer=deep_serialize)
    special_logs: Optional[SpecialLogs] = field(serializer=deep_serialize)


class Parameter(PClass):
    transition_parameter: Optional[PMap[int, InferenceParameter]] = field(serializer=deep_serialize)
    readout_fn: Optional[CustomSequential] = field()
    learning_parameter: Optional[LearningParameter] = field(serializer=deep_serialize)


class GodState(PClass):
    learning_states: PMap[int, LearningState] = field(serializer=deep_serialize)
    inference_states: PMap[int, PMap[int, InferenceState]] = field(serializer=deep_serialize)
    validation_learning_states: PMap[int, LearningState] = field(serializer=deep_serialize)
    parameters: PMap[int, Parameter] = field(serializer=deep_serialize)
    general: PMap[int, General] = field(serializer=deep_serialize)
    prng: PMap[int, batched[PRNG]] = field(serializer=deep_serialize)
    prng_learning: PMap[int, PRNG] = field(serializer=deep_serialize)
    start_epoch: jax.Array = field()


# ============================================================================
# PYTREE REGISTRATIONS
# ============================================================================

# Register leaf types first
register_pytree(Hyperparameter, {"learnable"})
register_pytree(Logs, set())
register_pytree(SpecialLogs, set())
register_pytree(RNNState, {"n_h", "n_in", "activation_fn"})
register_pytree(LSTMState, {"n_h", "n_in"})
register_pytree(RNN, set())
register_pytree(UOROState, set())
register_pytree(LearningParameter, {"rflo_timeconstant"})

# Register container types that depend on leaf types
register_pytree(InferenceParameter, set())
register_pytree(InferenceState, set())
register_pytree(LearningState, set())
register_pytree(General, set())
register_pytree(Parameter, set())

# Register top-level container last
register_pytree(GodState, set())
