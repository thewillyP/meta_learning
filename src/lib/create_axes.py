from typing import Any, Callable
import jax
import equinox as eqx

from lib.config import GodConfig
from lib.interface import InferenceInterface


def get_batched[ENV](env: ENV, interfaces: dict[int, InferenceInterface[ENV]]) -> tuple[Any, ...]:
    return tuple(interface.get_rnn_state(env) for interface in interfaces.values()) + (interfaces[0]._get_prng(env),)


def create_axes[ENV](interfaces: dict[int, dict[int, InferenceInterface[ENV]]]) -> dict[int, Callable[[ENV], ENV]]:
    get_axes: dict[int, Callable[[ENV], ENV]] = {}
    for i, interface in interfaces.items():

        def _create_axes(env: ENV, inter=interface) -> ENV:
            _batched_env: ENV = jax.tree.map(lambda _: None, env)
            batched_env = eqx.tree_at(
                lambda env, inter=inter: get_batched(env, inter),
                _batched_env,
                replace=tuple(0 for _ in get_batched(env, inter)),
                is_leaf=lambda x: x is None,
            )
            return batched_env

        get_axes[i] = _create_axes

    return get_axes
