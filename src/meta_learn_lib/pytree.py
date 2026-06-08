import jax
from pyrsistent import pmap, pvector
from pyrsistent._pmap import PMap as PMapClass
from pyrsistent._pvector import PythonPVector

from meta_learn_lib.env import *


# Register PMap as PyTree
def _pmap_tree_flatten_with_keys(pm):
    keys = tuple(sorted(pm.keys()))
    children = [(jax.tree_util.DictKey(k), pm[k]) for k in keys]
    return children, keys


def _pmap_tree_unflatten(keys, values):
    return pmap(zip(keys, values))


jax.tree_util.register_pytree_with_keys(PMapClass, _pmap_tree_flatten_with_keys, _pmap_tree_unflatten)


# Register PVector as PyTree
def _pvector_tree_flatten_with_keys(pv):
    return [(jax.tree_util.SequenceKey(i), v) for i, v in enumerate(pv)], None


def _pvector_tree_unflatten(_, values):
    return pvector(values)


jax.tree_util.register_pytree_with_keys(PythonPVector, _pvector_tree_flatten_with_keys, _pvector_tree_unflatten)


# PyTree registration helper
def register_pytree(cls, static_fields):
    """Register a PClass as a JAX PyTree"""

    def tree_flatten_with_keys(obj):
        all_fields = set(cls._pclass_fields.keys())
        dynamic_fields = [name for name in sorted(all_fields - static_fields) if hasattr(obj, name)]
        static_field_values = {name: getattr(obj, name) for name in static_fields if hasattr(obj, name)}
        children = [(jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in dynamic_fields]
        return children, (dynamic_fields, static_field_values)

    def tree_unflatten(aux_data, values):
        dynamic_fields, static_field_values = aux_data
        kwargs = dict(zip(dynamic_fields, values))
        kwargs.update(static_field_values)
        return cls(**kwargs)

    jax.tree_util.register_pytree_with_keys(cls, tree_flatten_with_keys, tree_unflatten)


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
register_pytree(ExternalMemory, set())

# Register container types
register_pytree(Parameters, set())
register_pytree(LearningStates, set())
register_pytree(ModelStates, set())
register_pytree(LevelMeta, set())
register_pytree(Outputs, set())

# Register top-level container last
register_pytree(GodState, set())
