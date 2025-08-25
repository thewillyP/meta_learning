from typing import Callable, Literal, Optional
import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap, thaw
from pyrsistent.typing import PMap
from pyrsistent._pmap import PMap as PMapClass

from lib.lib_types import *


def deep_serialize(_, obj):
    """Recursively serialize pyrsistent objects to Python built-ins"""
    if isinstance(obj, PClass):
        serialized = obj.serialize()
        return {k: deep_serialize(_, v) for k, v in serialized.items()}
    elif isinstance(obj, PMapClass):
        thawed = thaw(obj)
        return {k: deep_serialize(_, v) for k, v in thawed.items()}
    elif isinstance(obj, dict):  # Handle regular dicts too
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


# Generic PClass flatten/unflatten for classes without static fields
def _pclass_tree_flatten(obj):
    field_names = tuple(sorted(obj._pclass_fields.keys()))
    values = [getattr(obj, name) for name in field_names]
    return values, (type(obj), field_names)


def _pclass_tree_unflatten(aux_data, values):
    cls, field_names = aux_data
    kwargs = dict(zip(field_names, values))
    return cls(**kwargs)


# Specific flatten/unflatten for RNNState (has static fields)
def _rnnstate_tree_flatten(obj):
    # Only activation is dynamic, rest are static
    dynamic_values = [obj.activation]
    static_fields = {"n_h": obj.n_h, "n_in": obj.n_in, "activation_fn": obj.activation_fn}
    return dynamic_values, static_fields


def _rnnstate_tree_unflatten(static_fields, values):
    kwargs = {"activation": values[0]}
    kwargs.update(static_fields)
    return RNNState(**kwargs)


# Specific flatten/unflatten for LearningParameter (has static fields)
def _learningparam_tree_flatten(obj):
    # Only learning_rate is dynamic, rflo_timeconstant is static
    dynamic_values = [obj.learning_rate]
    static_fields = {"rflo_timeconstant": obj.rflo_timeconstant}
    return dynamic_values, static_fields


def _learningparam_tree_unflatten(static_fields, values):
    kwargs = {"learning_rate": values[0]}
    kwargs.update(static_fields)
    return LearningParameter(**kwargs)


# Specific flatten/unflatten for RNNState (has static fields)
def _rnnstate_tree_flatten(obj):
    # Only activation is dynamic, rest are static
    dynamic_values = [obj.activation]
    static_fields = {"n_h": obj.n_h, "n_in": obj.n_in, "activation_fn": obj.activation_fn}
    return dynamic_values, static_fields


def _rnnstate_tree_unflatten(static_fields, values):
    kwargs = {"activation": values[0]}
    kwargs.update(static_fields)
    return RNNState(**kwargs)


# Specific flatten/unflatten for LearningParameter (has static fields)
def _learningparam_tree_flatten(obj):
    # Only learning_rate is dynamic, rflo_timeconstant is static
    dynamic_values = [obj.learning_rate]
    static_fields = {"rflo_timeconstant": obj.rflo_timeconstant}
    return dynamic_values, static_fields


def _learningparam_tree_unflatten(static_fields, values):
    kwargs = {"learning_rate": values[0]}
    kwargs.update(static_fields)
    return LearningParameter(**kwargs)


class Logs(PClass):
    gradient: Optional[jax.Array] = field()


class SpecialLogs(PClass):
    influence_tensor: Optional[jax.Array] = field()
    immediate_influence_tensor: Optional[jax.Array] = field()
    largest_jac_eigenvalue: Optional[jax.Array] = field()
    jacobian: Optional[jax.Array] = field()


class CustomSequential(eqx.Module):
    model: eqx.nn.Sequential

    def __init__(
        self, layer_defs: list[tuple[int, bool, Callable[[jax.Array], jax.Array]]], input_size: int, key: PRNG
    ):
        layers = []
        in_size = input_size
        layer_keys = jax.random.split(key, len(layer_defs))

        for (out_size, use_bias, activation), k in zip(layer_defs, layer_keys):
            layers.append(eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=k))
            layers.append(eqx.nn.Lambda(activation))
            in_size = out_size

        self.model = eqx.nn.Sequential(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)


class RNNState(PClass):
    activation: jax.Array = field()
    n_h: int = field()  # Static field
    n_in: int = field()  # Static field
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"] = field()  # Static field


class RNN(PClass):
    w_rec: jax.Array = field()
    b_rec: Optional[jax.Array] = field()


class UOROState(PClass):
    A: jax.Array = field()
    B: jax.Array = field()


class General(PClass):
    current_virtual_minibatch: int = field()
    logs: Optional[Logs] = field(serializer=deep_serialize)
    special_logs: Optional[SpecialLogs] = field(serializer=deep_serialize)


class InferenceState(PClass):
    rnn: Optional[RNNState] = field(serializer=deep_serialize)


class LearningState(PClass):
    influence_tensor: Optional[JACOBIAN] = field()
    uoro: Optional[UOROState] = field(serializer=deep_serialize)
    opt_state: Optional[optax.OptState] = field()


class InferenceParameter(PClass):
    rnn: Optional[RNN] = field(serializer=deep_serialize)


class LearningParameter(PClass):
    learning_rate: Optional[jax.Array] = field()
    rflo_timeconstant: Optional[float] = field()  # Static field


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
    start_epoch: int = field()


# Register all PClass objects as PyTrees
jax.tree_util.register_pytree_node(Logs, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(SpecialLogs, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(RNNState, _rnnstate_tree_flatten, _rnnstate_tree_unflatten)
jax.tree_util.register_pytree_node(RNN, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(UOROState, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(General, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(InferenceState, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(LearningState, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(InferenceParameter, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(LearningParameter, _learningparam_tree_flatten, _learningparam_tree_unflatten)
jax.tree_util.register_pytree_node(Parameter, _pclass_tree_flatten, _pclass_tree_unflatten)
jax.tree_util.register_pytree_node(GodState, _pclass_tree_flatten, _pclass_tree_unflatten)
