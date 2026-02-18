import jax

from meta_learn_lib.env import *


def create_axes[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: isinstance(x, (Parameter, State))

    def to_axis(x):
        if isinstance(x, Parameter):
            return 0
        if isinstance(x, State):
            return 0
        return None

    return jax.tree.map(to_axis, env, is_leaf=is_leaf)
