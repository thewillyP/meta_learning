import jax

from meta_learn_lib.env import Tagged


def create_axes[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, Tagged)
    return jax.tree.map(lambda x: 0 if isinstance(x, Tagged) else None, env, is_leaf=is_leaf)


def diff_axes[ENV](old_env: ENV, new_env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, Tagged)

    old_leaf_ids: set[int] = set()
    for leaf in jax.tree.leaves(old_env, is_leaf=is_leaf):
        if isinstance(leaf, Tagged):
            for arr in jax.tree.leaves(leaf):
                old_leaf_ids.add(id(arr))

    def to_axis(x):
        if isinstance(x, Tagged):
            inner_arrays = jax.tree.leaves(x)
            if not inner_arrays:
                return None
            all_old = all(id(a) in old_leaf_ids for a in inner_arrays)
            return None if all_old else 0
        return None

    return jax.tree.map(to_axis, new_env, is_leaf=is_leaf)
