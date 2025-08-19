from typing import Any, Callable
import jax
import equinox as eqx

from lib.config import GodConfig
from lib.interface import InferenceInterface


def get_batched[ENV](env: ENV, interface: InferenceInterface[ENV]) -> tuple[Any, ...]:
    return (
        interface.get_rnn_state(env),
        interface._get_prng(env),
    )


def create_axes[ENV](env: ENV, interfaces: dict[int, InferenceInterface[ENV]]) -> dict[int, Callable[[ENV], ENV]]:
    _batched_env: ENV = jax.tree.map(lambda _: None, env)
    get_axes: dict[int, Callable[[ENV], ENV]] = {}
    for i, interface in interfaces.items():
        batched_env = eqx.tree_at(
            lambda env, interface=interface: get_batched(env, interface),
            _batched_env,
            replace=tuple(lambda _: 0 for _ in get_batched(env, interface)),
            is_leaf=lambda x: x is None,
        )
        get_axes[i] = batched_env
    return get_axes
