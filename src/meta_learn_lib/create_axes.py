import jax

from meta_learn_lib.env import *


def create_axes[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    def to_axis(x):
        return 0 if isinstance(x, (Parameter, State)) else None

    return jax.tree.map(to_axis, env, is_leaf=is_leaf)


def diff_axes(old_env, new_env):
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    # Collect ids of ALL leaves reachable from old_env
    old_leaf_ids = set()
    for leaf in jax.tree.leaves(old_env, is_leaf=is_leaf):
        if isinstance(leaf, (Parameter, State)):
            # Use ids of the *inner* arrays, not the dataclass wrapper,
            # because pyrsistent may copy the wrapper but share arrays
            for arr in jax.tree.leaves(leaf):
                old_leaf_ids.add(id(arr))

    def to_axis(x):
        if isinstance(x, (Parameter, State)):
            inner_arrays = jax.tree.leaves(x)
            if not inner_arrays:
                return None
            # If ALL inner arrays existed before, this leaf is unchanged
            all_old = all(id(a) in old_leaf_ids for a in inner_arrays)
            return None if all_old else 0
        return None

    return jax.tree.map(to_axis, new_env, is_leaf=is_leaf)
