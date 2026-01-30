from typing import Optional
import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap, pvector, thaw
from pyrsistent.typing import PVector
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
    elif isinstance(obj, PVector):
        return [deep_serialize(_, v) for v in obj]
    elif isinstance(obj, dict):
        return {k: deep_serialize(_, v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_serialize(_, v) for v in obj)
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


# Register PVector as PyTree
def _pvector_tree_flatten(pv):
    return list(pv), None


def _pvector_tree_unflatten(_, values):
    return pvector(values)


jax.tree_util.register_pytree_node(PVector, _pvector_tree_flatten, _pvector_tree_unflatten)


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


class Parameter[T](PClass):
    value: T = field(serializer=deep_serialize)
    learnable: bool = field()
    min_value: float = field()
    max_value: float = field()


class State[T](PClass):
    value: T = field(serializer=deep_serialize)
    is_batched: bool = field()


# rather, put in configs a boolean for each field whether to allocate or not
class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    immediate_influence_contains_nans: Optional[bool] = field(initial=None)
    largest_eigenvalue: Optional[jax.Array] = field(initial=None)
    influence_tensor: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class MLP(PClass):
    model: Parameter[eqx.nn.Sequential] = field(serializer=deep_serialize)


class RNN(PClass):
    w_rec: Parameter[jax.Array] = field(serializer=deep_serialize)
    b_rec: Optional[Parameter[jax.Array]] = field(serializer=deep_serialize)
    layer_norm: Optional[Parameter[eqx.nn.LayerNorm]] = field(serializer=deep_serialize)
    time_constant: Parameter[jax.Array] = field(serializer=deep_serialize)


type Model = MLP | RNN | eqx.nn.GRUCell | eqx.nn.LSTMCell


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
    models: PVector[Model] = field(serializer=deep_serialize)
    learning_rates: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)
    weight_decays: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)
    rflo_timeconstants: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)


class States(PClass):
    influence_tensors: PVector[JACOBIAN] = field()
    uoros: PVector[UOROState] = field(serializer=deep_serialize)
    opt_states: PVector[optax.OptState] = field(serializer=deep_serialize)
    ticks: PVector[jax.Array] = field()
    logs: PVector[Logs] = field(serializer=deep_serialize)
    recurrent_states: PVector[RecurrentState] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PVector[VanillaRecurrentState] = field(serializer=deep_serialize)
    lstm_states: PVector[LSTMState] = field(serializer=deep_serialize)
    prngs: PVector[State[PRNG]] = field(serializer=deep_serialize)


class GodState(PClass):
    meta_levels: PVector[tuple[States, Parameters]] = field(serializer=deep_serialize)
    validation_levels: PVector[tuple[States, Parameters]] = field(serializer=deep_serialize)


# Idea: we let the interface take care of the index mappings. Just make it so that each level has access to all the info it needs

# ============================================================================
# PYTREE REGISTRATIONS
# ============================================================================

# Register leaf types first
register_pytree(Parameter, {"learnable", "min_value", "max_value"})
register_pytree(State, {"is_batched"})
register_pytree(Logs, set())
register_pytree(RecurrentState, set())
register_pytree(VanillaRecurrentState, {"activation_fn"})
register_pytree(LSTMState, set())
register_pytree(RNN, set())
register_pytree(MLP, set())
register_pytree(UOROState, set())

# Register container types that depend on leaf types
register_pytree(Parameters, set())
register_pytree(States, set())

# Register top-level container last
register_pytree(GodState, set())
