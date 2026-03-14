from dataclasses import dataclass
from typing import Callable
import jax
import optax
import equinox as eqx

from meta_learn_lib.env import (
    MLP,
    RNN,
    LSTMState,
    Logs,
    Parameter,
    RecurrentState,
    State,
    UOROState,
    VanillaRecurrentState,
)
from meta_learn_lib.lib_types import JACOBIAN, PRNG


@dataclass(frozen=True)
class GodInterface[ENV]:
    take_prng: Callable[[ENV], tuple[PRNG, ENV]]
    put_prng: Callable[[ENV, PRNG], ENV]
    get_tick: Callable[[ENV], jax.Array]
    put_tick: Callable[[ENV, jax.Array], ENV]
    advance_tick: Callable[[ENV], ENV]
    put_logs: Callable[[ENV, Logs], ENV]
    get_logs: Callable[[ENV], Logs]
    get_state: Callable[[ENV], jax.Array]
    put_state: Callable[[ENV, jax.Array], ENV]
    get_param: Callable[[ENV], jax.Array]
    put_param: Callable[[ENV, jax.Array], ENV]
    get_mlp_param: Callable[[ENV], MLP]
    put_mlp_param: Callable[[ENV, MLP], ENV]
    get_vanilla_rnn_state: Callable[[ENV], VanillaRecurrentState]
    put_vanilla_rnn_state: Callable[[ENV, VanillaRecurrentState], ENV]
    get_vanilla_rnn_param: Callable[[ENV], RNN]
    put_vanilla_rnn_param: Callable[[ENV, RNN], ENV]
    get_gru_state: Callable[[ENV], RecurrentState]
    put_gru_state: Callable[[ENV, RecurrentState], ENV]
    get_gru_param: Callable[[ENV], Parameter[eqx.nn.GRUCell]]
    put_gru_param: Callable[[ENV, Parameter[eqx.nn.GRUCell]], ENV]
    get_lstm_state: Callable[[ENV], LSTMState]
    put_lstm_state: Callable[[ENV, LSTMState], ENV]
    get_lstm_param: Callable[[ENV], Parameter[eqx.nn.LSTMCell]]
    put_lstm_param: Callable[[ENV, Parameter[eqx.nn.LSTMCell]], ENV]
    get_autoregressive_predictions: Callable[[ENV], State[jax.Array]]
    put_autoregressive_predictions: Callable[[ENV, State[jax.Array]], ENV]
    get_time_constant: Callable[[ENV], Parameter[jax.Array]]
    put_time_constant: Callable[[ENV, Parameter[jax.Array]], ENV]
    get_opt_state: Callable[[ENV], State[optax.OptState]]
    put_opt_state: Callable[[ENV, State[optax.OptState]], ENV]
    get_forward_mode_jacobian: Callable[[ENV], State[JACOBIAN]]
    put_forward_mode_jacobian: Callable[[ENV, State[JACOBIAN]], ENV]
    get_uoro_state: Callable[[ENV], UOROState]
    put_uoro_state: Callable[[ENV, UOROState], ENV]
    get_learning_rate: Callable[[ENV], Parameter[jax.Array]]
    put_learning_rate: Callable[[ENV, Parameter[jax.Array]], ENV]
    get_weight_decay: Callable[[ENV], Parameter[jax.Array]]
    put_weight_decay: Callable[[ENV, Parameter[jax.Array]], ENV]
    get_momentum: Callable[[ENV], Parameter[jax.Array]]
    put_momentum: Callable[[ENV, Parameter[jax.Array]], ENV]
    get_kl_regularizer_beta: Callable[[ENV], Parameter[jax.Array]]
    put_kl_regularizer_beta: Callable[[ENV, Parameter[jax.Array]], ENV]


def default_god_interface[ENV]() -> GodInterface[ENV]:
    return GodInterface[ENV](
        take_prng=lambda env: (None, env),
        put_prng=lambda env, prng: env,
        get_tick=lambda env: None,
        put_tick=lambda env, tick: env,
        advance_tick=lambda env: env,
        put_logs=lambda env, logs: env,
        get_logs=lambda env: None,
        get_state=lambda env: None,
        put_state=lambda env, state: env,
        get_param=lambda env: None,
        put_param=lambda env, param: env,
        get_mlp_param=lambda env: None,
        put_mlp_param=lambda env, mlp_param: env,
        get_vanilla_rnn_state=lambda env: None,
        put_vanilla_rnn_state=lambda env, state: env,
        get_vanilla_rnn_param=lambda env: None,
        put_vanilla_rnn_param=lambda env, param: env,
        get_gru_state=lambda env: None,
        put_gru_state=lambda env, state: env,
        get_gru_param=lambda env: None,
        put_gru_param=lambda env, param: env,
        get_lstm_state=lambda env: None,
        put_lstm_state=lambda env, state: env,
        get_lstm_param=lambda env: None,
        put_lstm_param=lambda env, param: env,
        get_autoregressive_predictions=lambda env: None,
        put_autoregressive_predictions=lambda env, predictions: env,
        get_time_constant=lambda env: None,
        put_time_constant=lambda env, time_constant: env,
        get_opt_state=lambda env: None,
        put_opt_state=lambda env, opt_state: env,
        get_forward_mode_jacobian=lambda env: None,
        put_forward_mode_jacobian=lambda env, jacobian: env,
        get_uoro_state=lambda env: None,
        put_uoro_state=lambda env, uoro_state: env,
        get_learning_rate=lambda env: None,
        put_learning_rate=lambda env, learning_rate: env,
        get_weight_decay=lambda env: None,
        put_weight_decay=lambda env, weight_decay: env,
        get_momentum=lambda env: None,
        put_momentum=lambda env, momentum: env,
        get_kl_regularizer_beta=lambda env: None,
        put_kl_regularizer_beta=lambda env, kl_regularizer_beta: env,
    )
