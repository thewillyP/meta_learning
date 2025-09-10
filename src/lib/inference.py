from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from lib.config import GRULayer, GodConfig, LSTMLayer, NNLayer
from lib.interface import ClassificationInterface, GeneralInterface, InferenceInterface, LearnInterface
from lib.lib_types import PRNG, batched, traverse
from lib.util import filter_cond, get_activation_fn


def create_inferences[ENV, DATA](
    config: GodConfig,
    inference_interfaces: dict[int, dict[int, InferenceInterface[ENV]]],
    data_interface: ClassificationInterface[DATA],
    axeses: dict[int, ENV],
) -> dict[int, Callable[[ENV, batched[traverse[DATA]]], tuple[ENV, batched[traverse[jax.Array]]]]]:
    inferences: dict[int, Callable[[ENV, batched[traverse[DATA]]], tuple[ENV, batched[traverse[jax.Array]]]]] = {}
    for i, ((_, interfaces), (_, axes)) in enumerate(sorted(zip(inference_interfaces.items(), axeses.items()))):
        inference = create_inference(config, interfaces, data_interface, axes)
        inferences[i] = inference
    return inferences


def create_inference[ENV, DATA](
    config: GodConfig,
    inference_interfaces: dict[int, InferenceInterface[ENV]],
    data_interface: ClassificationInterface[DATA],
    axes: ENV,
) -> Callable[[ENV, batched[traverse[DATA]]], tuple[ENV, batched[traverse[jax.Array]]]]:
    transition_functions: list[Callable[[ENV, jax.Array], tuple[ENV, jax.Array]]] = []
    readout_functions: list[Callable[[ENV], jax.Array]] = []
    for i, (_, fn) in enumerate(sorted(config.transition_function.items())):
        match fn:
            case NNLayer():
                # If I really wanted to, I would separate this with a agnostic interface as input
                def rnn_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    rnn = inference_interfaces[i].get_rnn_param(env)
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    a = rnn_state.activation
                    a_rec = rnn.w_rec @ jnp.concat((a, data)) + (rnn.b_rec if rnn.b_rec is not None else 0)
                    a_new = (1 - alpha) * a + alpha * get_activation_fn(rnn_state.activation_fn)(a_rec)
                    new_env = inference_interfaces[i].put_rnn_state(env, rnn_state.set(activation=a_new))
                    return new_env, a_new

                def rnn_readout(env: ENV, i=i) -> jax.Array:
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    return rnn_state.activation

                transition_functions.append(rnn_transition)
                readout_functions.append(rnn_readout)
            case GRULayer():

                def gru_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    gru = inference_interfaces[i].get_gru_param(env)
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    a = rnn_state.activation
                    a_rec = gru(data, a)
                    a_new = (1 - alpha) * a + alpha * a_rec
                    new_env = inference_interfaces[i].put_rnn_state(env, rnn_state.set(activation=a_new))
                    return new_env, a_new

                def gru_readout(env: ENV, i=i) -> jax.Array:
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    return rnn_state.activation

                transition_functions.append(gru_transition)
                readout_functions.append(gru_readout)

            case LSTMLayer():

                def lstm_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    lstm = inference_interfaces[i].get_lstm_param(env)
                    lstm_state = inference_interfaces[i].get_lstm_state(env)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    h, c = lstm_state.h, lstm_state.c
                    h_rec, c_rec = lstm(data, (h, c))
                    h_new = (1 - alpha) * h + alpha * h_rec
                    c_new = (1 - alpha) * c + alpha * c_rec
                    new_env = inference_interfaces[i].put_lstm_state(env, lstm_state.set(h=h_new, c=c_new))
                    return new_env, h_new

                def lstm_readout(env: ENV, i=i) -> jax.Array:
                    lstm_state = inference_interfaces[i].get_lstm_state(env)
                    return lstm_state.h

                transition_functions.append(lstm_transition)
                readout_functions.append(lstm_readout)

    def inference_step(env: ENV, data: DATA) -> tuple[ENV, jax.Array]:
        x = data_interface.get_input(data)
        x_history = [x] if config.readout_uses_input_data else [jnp.zeros_like(x)]

        for transition_fn, readout_fn in zip(transition_functions, readout_functions):
            env, x = transition_fn(env, x)
            x_history.append(readout_fn(env))

        readout_param = inference_interfaces[0].get_readout_param(env)
        y = readout_param(jnp.concatenate(x_history, axis=-1))
        return env, y

    def inference(env: ENV, data: batched[traverse[DATA]]) -> tuple[ENV, batched[traverse[jax.Array]]]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(carry, x):
            _env = eqx.combine(carry, static)
            _env, out = inference_step(_env, x)
            carry, _ = eqx.partition(_env, eqx.is_array)
            return carry, out

        f = eqx.filter_vmap(eqx.Partial(jax.lax.scan, step), in_axes=(axes, 0), out_axes=(axes, 0))
        _env, outputs = f(arr, data.b.d)
        env = eqx.combine(_env, static)
        return env, batched(traverse(outputs))

    return inference


def reset_inference_env[ENV](env0: ENV, env: ENV, inference_interfaces: dict[int, InferenceInterface[ENV]]) -> ENV:
    for inference_interface in inference_interfaces.values():
        rnn = inference_interface.get_rnn_state(env0)
        lstm = inference_interface.get_lstm_state(env0)
        env = inference_interface.put_rnn_state(env, rnn)
        env = inference_interface.put_lstm_state(env, lstm)
    return env


def reset_validation_learn_env[ENV](env0: ENV, env: ENV, learn_interface: LearnInterface[ENV]) -> ENV:
    influence_tensor = learn_interface.get_influence_tensor(env0)
    uoro = learn_interface.get_uoro(env0)
    env = learn_interface.put_influence_tensor(env, influence_tensor)
    env = learn_interface.put_uoro(env, uoro)
    return env


def hard_reset_inference[ENV](
    get_env: Callable[[], ENV],
    inference_interface: dict[int, InferenceInterface[ENV]],
) -> Callable[[ENV], ENV]:
    def reset(env: ENV) -> ENV:
        env0 = get_env()
        env = reset_inference_env(env0, env, inference_interface)
        return env

    return reset


def make_resets[ENV, DATA](
    get_env: Callable[[PRNG], ENV],
    inferences: dict[int, Callable[[ENV, batched[traverse[DATA]]], tuple[ENV, batched[traverse[jax.Array]]]]],
    inference_interfaces: dict[int, dict[int, InferenceInterface[ENV]]],
    general_interfaces: dict[int, GeneralInterface[ENV]],
    validation_interfaces: dict[int, LearnInterface[ENV]],
    virtual_minibatches: dict[int, int],
) -> dict[int, Callable[[ENV], ENV]]:
    _inferences: dict[int, Callable[[ENV], ENV]] = {}
    for k, (j, _) in enumerate(sorted(inferences.items())):

        def reset_inference(env: ENV, i=j, k=k) -> ENV:
            current_virtual_minibatch = general_interfaces[i].get_current_virtual_minibatch(env)

            def do_reset(env: ENV, i=i) -> ENV:
                prng, env = validation_interfaces[min(validation_interfaces.keys())].get_prng(env)
                env0 = get_env(prng)
                return reset_inference_env(env0, env, inference_interfaces[i])

            env = filter_cond(
                current_virtual_minibatch % virtual_minibatches[i] == 0,
                do_reset,
                lambda e: e,
                env,
            )
            env = general_interfaces[i].put_current_virtual_minibatch(env, current_virtual_minibatch + 1)

            if k > 0:
                # validation learn env must always reset since their learning is purely a readout
                prng, env = validation_interfaces[min(validation_interfaces.keys())].get_prng(env)
                env0 = get_env(prng)
                env = reset_validation_learn_env(env0, env, validation_interfaces[i])

            return env

        _inferences[j] = reset_inference
    return _inferences
