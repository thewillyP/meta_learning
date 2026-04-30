from dataclasses import dataclass
from typing import Callable

import jax
import optax
import equinox as eqx

from meta_learn_lib.env import (
    LSTMState,
    Logs,
    MidpointBuffer,
    Tagged,
    UOROState,
    VanillaRecurrentState,
)
from meta_learn_lib.lib_types import JACOBIAN, PRNG


@dataclass(frozen=True)
class Accessor[ENV, T]:
    get: Callable[[ENV], T]
    put: Callable[[ENV, T], ENV]
    put_tagged: Callable[[ENV, Tagged[T]], ENV]


# ============================================================================
# INTERFACE
# ============================================================================


@dataclass(frozen=True)
class GodInterface[ENV]:
    prng: Accessor[ENV, PRNG]
    tick: Accessor[ENV, jax.Array]
    logs: Accessor[ENV, Logs]
    state: Accessor[ENV, jax.Array]
    param: Accessor[ENV, jax.Array]
    mlp_model: Accessor[ENV, eqx.nn.Sequential]
    rnn_w_rec: Accessor[ENV, jax.Array]
    rnn_b_rec: Accessor[ENV, jax.Array]
    rnn_layer_norm: Accessor[ENV, eqx.Module]
    vanilla_rnn_state: Accessor[ENV, VanillaRecurrentState]
    gru_cell: Accessor[ENV, eqx.nn.GRUCell]
    gru_activation: Accessor[ENV, jax.Array]
    lstm_cell: Accessor[ENV, eqx.nn.LSTMCell]
    lstm_state: Accessor[ENV, LSTMState]
    autoregressive_predictions: Accessor[ENV, jax.Array]
    time_constant: Accessor[ENV, jax.Array]
    opt_state: Accessor[ENV, optax.OptState]
    forward_mode_jacobian: Accessor[ENV, JACOBIAN]
    uoro_state: Accessor[ENV, UOROState]
    midpoint_buffer: Accessor[ENV, MidpointBuffer]
    learning_rate: Accessor[ENV, jax.Array]
    weight_decay: Accessor[ENV, jax.Array]
    momentum: Accessor[ENV, jax.Array]
    kl_regularizer_beta: Accessor[ENV, jax.Array]

    def advance_tick(self: "GodInterface[ENV]", env: ENV) -> ENV:
        return self.tick.put(env, self.tick.get(env) + 1)

    def take_prng(self: "GodInterface[ENV]", env: ENV) -> tuple[PRNG, ENV]:
        sub, new = jax.random.split(self.prng.get(env))
        return sub, self.prng.put(env, new)

    def merge_logs(self: "GodInterface[ENV]", env: ENV, logs: Logs) -> ENV:
        current = self.logs.get(env)
        merged = current.set(**{k: v for k, v in logs.serialize().items() if v is not None})
        return self.logs.put(env, merged)


def default_god_interface[ENV]() -> GodInterface[ENV]:

    def noop[T]() -> Accessor[ENV, T]:
        return Accessor(
            get=lambda env: None,
            put=lambda env, v: env,
            put_tagged=lambda env, v: env,
        )

    return GodInterface[ENV](
        prng=noop(),
        tick=noop(),
        logs=noop(),
        state=noop(),
        param=noop(),
        mlp_model=noop(),
        rnn_w_rec=noop(),
        rnn_b_rec=noop(),
        rnn_layer_norm=noop(),
        vanilla_rnn_state=noop(),
        gru_cell=noop(),
        gru_activation=noop(),
        lstm_cell=noop(),
        lstm_state=noop(),
        autoregressive_predictions=noop(),
        time_constant=noop(),
        opt_state=noop(),
        forward_mode_jacobian=noop(),
        uoro_state=noop(),
        midpoint_buffer=noop(),
        learning_rate=noop(),
        weight_decay=noop(),
        momentum=noop(),
        kl_regularizer_beta=noop(),
    )
