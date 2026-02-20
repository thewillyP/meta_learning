import jax

from meta_learn_lib.env import *


def create_axes[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    def to_axis(x):
        return 0 if isinstance(x, (Parameter, State)) else None

    return jax.tree.map(to_axis, env, is_leaf=is_leaf)


def diff_axes[ENV](old_env: ENV, new_env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, (Parameter, State))
    old_paths = {p for p, _ in jax.tree_util.tree_leaves_with_path(old_env, is_leaf=is_leaf)}

    def to_axis(path, x):
        if not isinstance(x, (Parameter, State)):
            return None
        return None if path in old_paths else 0

    return jax.tree_util.tree_map_with_path(to_axis, new_env, is_leaf=is_leaf)
