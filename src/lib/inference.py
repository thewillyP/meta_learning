from typing import Callable
import copy
import jax
import jax.numpy as jnp
import equinox as eqx

from lib.config import GodConfig, NNLayer
from lib.env import RNNState
from lib.interface import ClassificationInterface, InferenceInterface, LearnInterface
from lib.lib_types import batched, traverse
from lib.util import get_activation_fn


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
                    new_env = inference_interfaces[i].put_rnn_state(env, copy.replace(rnn_state, activation=a_new))
                    return new_env, a_new

                transition_functions.append(rnn_transition)

    def inference_step(env: ENV, data: DATA) -> tuple[ENV, jax.Array]:
        x = data_interface.get_input(data)
        x_history = [x] if config.readout_uses_input_data else [jnp.zeros_like(x)]

        for transition_fn in transition_functions:
            env, x = transition_fn(env, x)
            x_history.append(x)

        readout_param = inference_interfaces[0].get_readout_param(env)
        y = readout_param(jnp.concatenate(x_history, axis=-1))
        return env, y

    def inference(env: ENV, data: batched[traverse[DATA]]) -> tuple[ENV, batched[traverse[jax.Array]]]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(carry, x):
            _env = eqx.combine(carry, static)
            _env, out = inference_step(_env, x.b.d)
            carry, _ = eqx.partition(_env, eqx.is_array)
            return carry, out

        f = eqx.filter_vmap(eqx.Partial(jax.lax.scan, step), in_axes=(axes, 0), out_axes=(axes, 0))
        _env, outputs = f(arr, data)
        env = eqx.combine(_env, static)
        return env, traverse(outputs)

    return inference


def reset_env[ENV](
    env: ENV, inference_interface: InferenceInterface[ENV], learn_interface: LearnInterface[ENV]
) -> ENV: ...


"""
Things to reset

1. every virtual cycle, reset the inference state since we completed an example
2. every virtual validation cycle, reset the validation inference state. 
    - typically this should never carry state but I researve the option to start the next cycle with the last state of the previous cycle
3. the learning state must be initialized newly every call to validation inference. this is mandatory
4. I need a learning state for validation since 

"""
