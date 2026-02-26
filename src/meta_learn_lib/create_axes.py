import jax

from meta_learn_lib.env import *


def create_axes[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    def to_axis(x):
        return 0 if isinstance(x, (Parameter, State)) else None

    return jax.tree.map(to_axis, env, is_leaf=is_leaf)


def diff_axes(old_env, new_env):
    is_leaf = lambda x: isinstance(x, (Parameter, State))
    old_leaves = {id(leaf) for leaf in jax.tree.leaves(old_env, is_leaf=is_leaf)}

    def to_axis(x):
        if isinstance(x, (Parameter, State)):
            return None if id(x) in old_leaves else 0
        return None

    return jax.tree.map(to_axis, new_env, is_leaf=is_leaf)


from pyrsistent import pmap, pvector, PVector
from pyrsistent import PMap as PMapType

_FIELDS = {
    "GodState": ["learning_states", "level_meta", "meta_parameters", "model_states", "prng"],
    "LearningStates": ["influence_tensors", "opt_states", "uoros"],
    "LevelMeta": ["log", "prngs", "tick"],
    "Parameters": [
        "grus",
        "kl_regularizer_betas",
        "learning_rates",
        "lstms",
        "mlps",
        "momentums",
        "rnns",
        "time_constants",
        "weight_decays",
    ],
    "ModelStates": ["autoregressive_predictions", "lstm_states", "recurrent_states", "vanilla_recurrent_states"],
    "VanillaRecurrentState": ["activation"],
    "RNN": ["b_rec", "layer_norm", "w_rec"],
    "MLP": ["model"],
}


def diff_axes_new(old_env, new_env):
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    new_flat, new_treedef = jax.tree.flatten(new_env, is_leaf=is_leaf)

    # build a map from new_env leaves to their axes
    axes_map = {}

    def _mark(old_node, new_node):
        cls_name = type(new_node).__name__

        if isinstance(new_node, (Parameter, State)):
            old_arrays = (
                jax.tree.leaves(old_node) if old_node is not None and isinstance(old_node, type(new_node)) else []
            )
            new_arrays = jax.tree.leaves(new_node)
            same = len(old_arrays) == len(new_arrays) and all(
                getattr(o, "shape", None) == getattr(n, "shape", None) for o, n in zip(old_arrays, new_arrays)
            )
            axes_map[id(new_node)] = None if same else 0
            return

        if isinstance(new_node, jax.Array) or cls_name == "PRNGKeyArray":
            return

        if cls_name in _FIELDS:
            for f in _FIELDS[cls_name]:
                old_child = getattr(old_node, f, None) if old_node is not None else None
                _mark(old_child, getattr(new_node, f))
            return

        if isinstance(new_node, PMapType):
            old_pmap = old_node if isinstance(old_node, PMapType) else pmap()
            for k in new_node.keys():
                _mark(old_pmap.get(k, None), new_node[k])
            return

        if isinstance(new_node, PVector):
            old_vec = old_node if isinstance(old_node, PVector) else []
            for i, child in enumerate(new_node):
                old_child = old_vec[i] if i < len(old_vec) else None
                _mark(old_child, child)
            return

        if isinstance(new_node, (list, tuple)):
            old_list = old_node if isinstance(old_node, (list, tuple)) else []
            for i, child in enumerate(new_node):
                old_child = old_list[i] if i < len(old_list) else None
                _mark(old_child, child)
            return

    _mark(old_env, new_env)

    def to_axis(x):
        if isinstance(x, (Parameter, State)):
            return axes_map.get(id(x), 0)
        return None

    return jax.tree.map(to_axis, new_env, is_leaf=is_leaf)
