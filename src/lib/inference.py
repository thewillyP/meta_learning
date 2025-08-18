from typing import Callable
import copy
import jax
import jax.numpy as jnp
import equinox as eqx

from lib.config import GodConfig, NNLayer
from lib.env import RNNState
from lib.interface import ClassificationInterface, InferenceInterface
from lib.lib_types import batched, traverse
from lib.util import get_activation_fn


def create_inference[ENV, DATA](
    config: GodConfig,
    inference_interfaces: dict[int, InferenceInterface[ENV]],
    data_interface: ClassificationInterface[DATA],
) -> Callable[[ENV, batched[traverse[DATA]]], tuple[ENV, batched[traverse[jax.Array]]]]:
    transition_functions: list[Callable[[ENV, jax.Array], tuple[jax.Array, ENV]]] = []
    for i, (_, fn) in enumerate(sorted(config.transition_function.items())):
        match fn:
            case NNLayer():

                def rnn_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    rnn = inference_interfaces[i].get_rnn_param(env)
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    activation_fn = get_activation_fn(rnn_state.activation_fn)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    a = rnn_state.activation

                    a_rec = rnn.w_rec @ jnp.concat((a, data)) + (rnn.b_rec if rnn.b_rec is not None else 0)
                    a_new = (1 - alpha) * a + alpha * activation_fn(a_rec)
                    new_env = inference_interfaces[i].put_rnn_state(env, copy.replace(rnn_state, activation=a_new))
                    return new_env, a_new

                transition_functions.append(rnn_transition)

    def inference_step(env: ENV, data: DATA) -> tuple[ENV, jax.Array]:
        x = data_interface.get_input(data)
        x_history = [x]

        for transition_fn in transition_functions:
            env, x = transition_fn(env, x)
            x_history.append(x)

        readout_param = inference_interfaces[0].get_readout_param(env)
        y = readout_param(jnp.concatenate(x_history, axis=-1))
        return env, y

    def inference(env: ENV, data: batched[traverse[DATA]]) -> tuple[ENV, batched[traverse[jax.Array]]]:
        axes = None
        f = eqx.filter_vmap(jax.tree_util.Partial(jax.lax.scan, inference_step), in_axes=(axes, 0), out_axes=(axes, 0))
        env, outputs = f(env, data)
        return env, traverse(outputs)

    return inference
