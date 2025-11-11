from dataclasses import dataclass
from typing import Callable
import jax
from jaxtyping import PyTree
import optax
import equinox as eqx

from meta_learn_lib.config import LearnConfig
from meta_learn_lib.env import RNN, CustomSequential, LSTMState, Logs, RNNState, SpecialLogs, UOROState
from meta_learn_lib.lib_types import JACOBIAN, PRNG, batched


@dataclass(frozen=True)
class ClassificationInterface[DATA]:
    get_input: Callable[[DATA], jax.Array]
    get_target: Callable[[DATA], jax.Array]
    get_sequence: Callable[[DATA], jax.Array]


@dataclass(frozen=True)
class GeneralInterface[ENV]:
    get_current_virtual_minibatch: Callable[[ENV], int]
    put_current_virtual_minibatch: Callable[[ENV, int], ENV]
    put_logs: Callable[[ENV, Logs], ENV]


@dataclass(frozen=True)
class InferenceInterface[ENV]:
    get_state: Callable[[ENV], jax.Array]
    get_readout_param: Callable[[ENV], CustomSequential]
    get_rnn_state: Callable[[ENV], RNNState]
    put_rnn_state: Callable[[ENV, RNNState], ENV]
    get_rnn_param: Callable[[ENV], RNN]
    get_prng: Callable[[ENV], tuple[PRNG, ENV]]
    _get_prng: Callable[[ENV], batched[PRNG]]
    get_rflo_timeconstant: Callable[[ENV], float]
    get_lstm_state: Callable[[ENV], LSTMState]
    put_lstm_state: Callable[[ENV, LSTMState], ENV]
    get_gru_param: Callable[[ENV], eqx.nn.GRUCell]
    get_lstm_param: Callable[[ENV], eqx.nn.LSTMCell]


@dataclass(frozen=True)
class LearnInterface[ENV]:
    get_state_pytree: Callable[[ENV], PyTree]
    get_state: Callable[[ENV], jax.Array]
    put_state: Callable[[ENV, jax.Array], ENV]
    get_param: Callable[[ENV], jax.Array]
    put_param: Callable[[ENV, jax.Array], ENV]
    get_sgd_param: Callable[[ENV], jax.Array]
    get_optimizer: Callable[[ENV], optax.GradientTransformation]
    get_updater: Callable[[optax.Params, optax.Updates], optax.Params]
    get_opt_state: Callable[[ENV], optax.OptState]
    put_opt_state: Callable[[ENV, optax.OptState], ENV]
    get_rflo_timeconstant: Callable[[ENV], float]
    get_influence_tensor: Callable[[ENV], JACOBIAN]
    put_influence_tensor: Callable[[ENV, JACOBIAN], ENV]
    get_influence_tensor_squared: Callable[[ENV], JACOBIAN]
    put_influence_tensor_squared: Callable[[ENV, JACOBIAN], ENV]
    get_uoro: Callable[[ENV], UOROState]
    put_uoro: Callable[[ENV, UOROState], ENV]
    learn_config: LearnConfig
    put_logs: Callable[[ENV, Logs], ENV]
    put_special_logs: Callable[[ENV, SpecialLogs], ENV]
    get_prng: Callable[[ENV], tuple[PRNG, ENV]]
    get_rflo_t: Callable[[ENV], int]
    put_rflo_t: Callable[[ENV, int], ENV]
    get_rtrl_t: Callable[[ENV], int]
    put_rtrl_t: Callable[[ENV, int], ENV]


def get_default_inference_interface[ENV]() -> InferenceInterface[ENV]:
    return InferenceInterface[ENV](
        get_state=lambda env: None,
        get_readout_param=lambda env: None,
        get_rnn_state=lambda env: None,
        put_rnn_state=lambda env, _: env,
        get_rnn_param=lambda env: None,
        get_prng=lambda env: (None, env),
        _get_prng=lambda env: None,
        get_rflo_timeconstant=lambda env: None,
        get_lstm_state=lambda env: None,
        put_lstm_state=lambda env, _: env,
        get_gru_param=lambda env: None,
        get_lstm_param=lambda env: None,
    )


def get_default_learn_interface[ENV]() -> LearnInterface[ENV]:
    return LearnInterface[ENV](
        get_state_pytree=lambda env: None,
        get_param=lambda env: None,
        put_param=lambda env, _: env,
        get_state=lambda env: None,
        put_state=lambda env, _: env,
        get_sgd_param=lambda env: None,
        get_optimizer=lambda env: None,
        get_updater=lambda opt_params, updates: None,
        get_opt_state=lambda env: None,
        put_opt_state=lambda env, _: env,
        get_rflo_timeconstant=lambda env: None,
        get_influence_tensor=lambda env: None,
        put_influence_tensor=lambda env, _: env,
        get_influence_tensor_squared=lambda env: None,
        put_influence_tensor_squared=lambda env, _: env,
        get_uoro=lambda env: None,
        put_uoro=lambda env, _: env,
        put_logs=lambda env, _: env,
        put_special_logs=lambda env, _: env,
        learn_config=None,
        get_prng=lambda env: (None, env),
        get_rflo_t=lambda env: None,
        put_rflo_t=lambda env, _: env,
        get_rtrl_t=lambda env: None,
        put_rtrl_t=lambda env, _: env,
    )


def get_default_general_interface[ENV]() -> GeneralInterface[ENV]:
    return GeneralInterface[ENV](
        get_current_virtual_minibatch=lambda env: None,
        put_current_virtual_minibatch=lambda env, value: env,
        put_logs=lambda env, logs: env,
    )
