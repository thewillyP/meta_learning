from typing import Any, Callable
import jax
import equinox as eqx

from lib.config import GodConfig
from lib.interface import InferenceInterface


def get_batched[ENV](env: ENV, interfaces: dict[int, InferenceInterface[ENV]]) -> tuple[Any, ...]:
    return tuple(interface.get_rnn_state(env) for interface in interfaces.values()) + (interfaces[0]._get_prng(env),)


def create_axes[ENV](env: ENV, interfaces: dict[int, dict[int, InferenceInterface[ENV]]]) -> dict[int, ENV]:
    get_axes: dict[int, ENV] = {}
    for i, interface in interfaces.items():
        _batched_env: ENV = jax.tree.map(lambda _: None, env)
        batched_env = eqx.tree_at(
            lambda env, interface=interface: get_batched(env, interface),
            _batched_env,
            replace=tuple(0 for _ in get_batched(env, interface)),
            is_leaf=lambda x: x is None,
        )
        get_axes[i] = batched_env

    return get_axes
