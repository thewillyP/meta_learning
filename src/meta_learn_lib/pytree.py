import jax
from pyrsistent import pmap, pvector
from pyrsistent._pmap import PMap as PMapClass
from pyrsistent._pvector import PythonPVector

from meta_learn_lib.env import *


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


jax.tree_util.register_pytree_node(PythonPVector, _pvector_tree_flatten, _pvector_tree_unflatten)


# PyTree registration helper
def register_pytree(cls, static_fields):
    """Register a PClass as a JAX PyTree"""

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
# PYTREE REGISTRATIONS
# ============================================================================

# Register leaf types first
register_pytree(Tagged, {"meta"})
register_pytree(Logs, set())
register_pytree(VanillaRecurrentState, {"activation_fn"})
register_pytree(LSTMState, set())
register_pytree(RNN, set())
register_pytree(UOROState, set())
register_pytree(MidpointBuffer, set())

# Register container types
register_pytree(Parameters, set())
register_pytree(LearningStates, set())
register_pytree(ModelStates, set())
register_pytree(LevelMeta, set())
register_pytree(Outputs, set())

# Register top-level container last
register_pytree(GodState, set())
