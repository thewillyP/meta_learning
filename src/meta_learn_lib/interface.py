from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import jax
import math
import optax
import equinox as eqx

from meta_learn_lib.env import (
    LSTMState,
    Logs,
    MidpointBuffer,
    UOROState,
    VanillaRecurrentState,
)
from meta_learn_lib.lib_types import JACOBIAN, PRNG, Category


@dataclass(frozen=True)
class ParamMeta:
    learnable: bool
    min_value: float
    max_value: float
    parametrizes_transition: bool


@dataclass(frozen=True)
class StateMeta:
    is_stateful: frozenset[int]


type AccessorMeta = ParamMeta | StateMeta


@dataclass(frozen=True)
class Accessor[ENV, T]:
    get: Callable[[ENV], T]
    put: Callable[[ENV, T], ENV]
    meta: AccessorMeta
    category: Category


def build_mask[ENV](
    env: ENV,
    accessors: Iterable[Accessor[ENV, bool]],
    predicate: Callable[[AccessorMeta], bool],
) -> ENV:
    mask = jax.tree.map(lambda _: False, env)
    for acc in accessors:
        if predicate(acc.meta):
            mask = acc.put(mask, True)
    return mask


# ============================================================================
# INTERFACE
# ============================================================================


@dataclass(frozen=True)
class GodInterface[ENV]:
    take_prng: Callable[[ENV], tuple[PRNG, ENV]]
    put_prng: Callable[[ENV, PRNG], ENV]
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


def noop_meta() -> ParamMeta:
    return ParamMeta(
        learnable=False,
        min_value=-math.inf,
        max_value=math.inf,
        parametrizes_transition=False,
    )


def default_god_interface[ENV]() -> GodInterface[ENV]:

    def noop[T]() -> Accessor[ENV, T]:
        return Accessor(get=lambda env: None, put=lambda env, v: env, meta=noop_meta(), category=None)

    return GodInterface[ENV](
        take_prng=lambda env: (None, env),
        put_prng=lambda env, v: env,
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


def interface_to_accessors[ENV, T](interface: GodInterface[ENV]) -> list[Accessor[ENV, T]]:
    return [
        interface.tick,
        interface.logs,
        interface.state,
        interface.param,
        interface.mlp_model,
        interface.rnn_w_rec,
        interface.rnn_b_rec,
        interface.rnn_layer_norm,
        interface.vanilla_rnn_state,
        interface.gru_cell,
        interface.gru_activation,
        interface.lstm_cell,
        interface.lstm_state,
        interface.autoregressive_predictions,
        interface.time_constant,
        interface.opt_state,
        interface.forward_mode_jacobian,
        interface.uoro_state,
        interface.midpoint_buffer,
        interface.learning_rate,
        interface.weight_decay,
        interface.momentum,
        interface.kl_regularizer_beta,
    ]
