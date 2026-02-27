from graphlib import TopologicalSorter
from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from pyrsistent import pmap
from pyrsistent.typing import PMap

from meta_learn_lib.config import *
from meta_learn_lib.env import *
from meta_learn_lib.interface import *
from meta_learn_lib.util import get_activation_fn, to_vector


def make_vanilla_rnn_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        rnn_state = interface.get_vanilla_rnn_state(env)
        return Outputs(prediction=rnn_state.activation.value)

    return read


def make_gru_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        gru_state = interface.get_gru_state(env)
        return Outputs(prediction=gru_state.activation.value)

    return read


def make_lstm_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        lstm_state = interface.get_lstm_state(env)
        return Outputs(prediction=lstm_state.h.value)

    return read


def make_scan_reader[ENV](sub_from_env: PMap[str, Callable[[ENV], Outputs]]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        x = to_vector([f(env) for f in sub_from_env.values()]).vector
        return Outputs(prediction=x)

    return read


def get_reader[ENV](
    node_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
) -> PMap[str, Callable[[ENV], Outputs]]:
    from_env: PMap[str, Callable[[ENV], Outputs]] = pmap()
    for node_name in TopologicalSorter(node_graph).static_order():
        match nodes[node_name]:
            case NNLayer():
                from_env = from_env.set(node_name, lambda env: Outputs())
            case VanillaRNNLayer():
                from_env = from_env.set(node_name, make_vanilla_rnn_reader(meta_interface[node_name]))
            case GRULayer():
                from_env = from_env.set(node_name, make_gru_reader(meta_interface[node_name]))
            case LSTMLayer():
                from_env = from_env.set(node_name, make_lstm_reader(meta_interface[node_name]))
            case Scan(graph, autoregressive_mask, pred_source, _start_token):
                sub_from_env = get_reader(graph, nodes, meta_interface)
                from_env = from_env.set(node_name, make_scan_reader(sub_from_env))
            case Repeat() | Concat() | ToEmpty() | UnlabeledSource() | LabeledSource():
                from_env = from_env.set(node_name, lambda env: Outputs())
            case _:
                from_env = from_env.set(node_name, lambda env: Outputs())
    return from_env


def get_inference[ENV](
    node_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
    from_env: PMap[str, Callable[[ENV], Outputs]],
) -> Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, PMap[str, Outputs]]]:
    def infer(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, PMap[str, Outputs]]:
        outputs: PMap[str, Outputs] = pmap()
        x_data, y_data = data
        for node_name in TopologicalSorter(node_graph).static_order():
            match nodes[node_name]:
                case NNLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    mlp = meta_interface[node_name].get_mlp_param(env).model.value
                    y = mlp(x)
                    outputs = outputs.set(node_name, Outputs(prediction=y))
                case VanillaRNNLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    w_rec = meta_interface[node_name].get_vanilla_rnn_param(env).w_rec.value
                    b_rec = meta_interface[node_name].get_vanilla_rnn_param(env).b_rec.value
                    rnn_state = meta_interface[node_name].get_vanilla_rnn_state(env)
                    a = rnn_state.activation.value
                    activation_fn = rnn_state.activation_fn
                    alpha = meta_interface[node_name].get_time_constant(env).value
                    layer_norm = meta_interface[node_name].get_vanilla_rnn_param(env).layer_norm.value

                    a_rec = w_rec @ jnp.concat((a, x)) + b_rec
                    a_rec = layer_norm(a_rec)
                    a_new = (1 - alpha) * a + alpha * get_activation_fn(activation_fn)(a_rec)
                    env = meta_interface[node_name].put_vanilla_rnn_state(
                        env,
                        rnn_state.set(activation=rnn_state.activation.set(value=a_new)),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=a_new))
                case GRULayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    gru = meta_interface[node_name].get_gru_param(env).value
                    gru_state = meta_interface[node_name].get_gru_state(env)
                    alpha = meta_interface[node_name].get_time_constant(env).value
                    a = gru_state.activation.value

                    a_rec = gru(x, a)
                    a_new = (1 - alpha) * a + alpha * a_rec
                    env = meta_interface[node_name].put_gru_state(
                        env,
                        gru_state.set(activation=gru_state.activation.set(value=a_new)),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=a_new))
                case LSTMLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    lstm = meta_interface[node_name].get_lstm_param(env).value
                    lstm_state = meta_interface[node_name].get_lstm_state(env)
                    alpha = meta_interface[node_name].get_time_constant(env).value
                    h, c = lstm_state.h.value, lstm_state.c.value
                    h_rec, c_rec = lstm(x, (h, c))
                    h_new = (1 - alpha) * h + alpha * h_rec
                    c_new = (1 - alpha) * c + alpha * c_rec
                    env = meta_interface[node_name].put_lstm_state(
                        env,
                        lstm_state.set(h=lstm_state.h.set(value=h_new), c=lstm_state.c.set(value=c_new)),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=h_new))
                case Scan(graph, autoregressive_mask, pred_source, _start_token):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env and n != pred_source],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs and n != pred_source],
                    ]
                    x = to_vector(deps).vector
                    y = from_env[pred_source](env) if pred_source in from_env else outputs[pred_source].prediction
                    last_sub_node = list(TopologicalSorter(graph).static_order())[-1]
                    token = meta_interface[node_name].get_autoregressive_predictions(env).value

                    fn = get_inference(graph, nodes, meta_interface)

                    def step(
                        carry: tuple[ENV, jax.Array],
                        x__y: tuple[jax.Array, jax.Array],
                    ) -> tuple[tuple[ENV, jax.Array], jax.Array]:
                        env, autoregress = carry
                        x, y = x__y
                        z = to_vector((autoregress, x)).vector
                        env, outs = fn(env, (z, y))
                        out = outs[last_sub_node].prediction
                        match autoregressive_mask:
                            case "teacher_forcing":
                                new_autoregress = y
                            case "identity":
                                new_autoregress = out
                            case "erase":
                                new_autoregress = jnp.zeros_like(out)
                        return (env, new_autoregress), out

                    (env, new_token), predictions = jax.lax.scan(step, (env, token), (x, y))
                    env = meta_interface[node_name].put_autoregressive_predictions(
                        env,
                        meta_interface[node_name].get_autoregressive_predictions(env).set(value=new_token),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=predictions))

                case Repeat(i):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    outputs = outputs.set(node_name, Outputs(prediction=jnp.broadcast_to(x[None], (i, *x.shape))))
                case Concat():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    outputs = outputs.set(node_name, Outputs(prediction=x))
                case ToEmpty():
                    outputs = outputs.set(node_name, Outputs())
                case UnlabeledSource():
                    outputs = outputs.set(node_name, Outputs(prediction=x_data))
                case LabeledSource():
                    outputs = outputs.set(node_name, Outputs(prediction=y_data))
                case _:
                    outputs = outputs.set(node_name, Outputs())
        return env, outputs

    return infer


def create_inference_and_readout[ENV](
    config: GodConfig,
    meta_interface: dict[str, GodInterface[ENV]],
):
    env_readout = get_reader(config.transition_graph, config.nodes, meta_interface)
    _transition_inference = get_inference(config.transition_graph, config.nodes, meta_interface, pmap())
    _readout_inference = get_inference(config.readout_graph, config.nodes, meta_interface, env_readout)

    def transition_inference(env: ENV, data: tuple[jax.Array, jax.Array]) -> ENV:
        env, _ = _transition_inference(env, data)
        return env

    def readout_inference(env: ENV, data: tuple[jax.Array, jax.Array]) -> Outputs:
        last_node = list(TopologicalSorter(config.readout_graph).static_order())[-1]
        _, outputs = _readout_inference(env, data)
        return outputs[last_node]

    return transition_inference, readout_inference


def create_inference[ENV, DATA](
    config: GodConfig,
    inference_interfaces: dict[int, InferenceInterface[ENV]],
    data_interface: ClassificationInterface[DATA],
    axes: ENV,
):
    transition_functions: list[Callable[[ENV, jax.Array], tuple[ENV, jax.Array]]] = []
    readout_functions: list[Callable[[ENV], jax.Array]] = []
    for i, (_, fn) in enumerate(sorted(config.transition_function.items())):
        match fn:
            case NNLayer(n_h, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
                # If I really wanted to, I would separate this with a agnostic interface as input
                def rnn_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    rnn = inference_interfaces[i].get_rnn_param(env)
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    a = rnn_state.activation
                    a_rec = rnn.w_rec @ jnp.concat((a, data)) + (rnn.b_rec if rnn.b_rec is not None else 0)
                    if rnn.layer_norm is not None:
                        a_rec = rnn.layer_norm(a_rec)
                    a_new = (1 - alpha) * a + alpha * get_activation_fn(rnn_state.activation_fn)(a_rec)
                    new_env = inference_interfaces[i].put_rnn_state(env, rnn_state.set(activation=a_new))
                    return new_env, a_new

                if use_in_readout:

                    def rnn_readout(env: ENV, i=i) -> jax.Array:
                        rnn_state = inference_interfaces[i].get_rnn_state(env)
                        return rnn_state.activation
                else:

                    def rnn_readout(env: ENV, i=i) -> jax.Array:
                        return jnp.array([])

                transition_functions.append(rnn_transition)
                readout_functions.append(rnn_readout)
            case GRULayer(n_h, use_bias, use_in_readout, use_random_init):

                def gru_transition(env: ENV, data: jax.Array, i=i) -> tuple[ENV, jax.Array]:
                    gru = inference_interfaces[i].get_gru_param(env)
                    rnn_state = inference_interfaces[i].get_rnn_state(env)
                    alpha = inference_interfaces[i].get_rflo_timeconstant(env)
                    a = rnn_state.activation
                    a_rec = gru(data, a)
                    a_new = (1 - alpha) * a + alpha * a_rec
                    new_env = inference_interfaces[i].put_rnn_state(env, rnn_state.set(activation=a_new))
                    return new_env, a_new

                if use_in_readout:

                    def gru_readout(env: ENV, i=i) -> jax.Array:
                        rnn_state = inference_interfaces[i].get_rnn_state(env)
                        return rnn_state.activation
                else:

                    def gru_readout(env: ENV, i=i) -> jax.Array:
                        return jnp.array([])

                transition_functions.append(gru_transition)
                readout_functions.append(gru_readout)

            case LSTMLayer(n_h, use_bias, use_in_readout, use_random_init):

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

                if use_in_readout:

                    def lstm_readout(env: ENV, i=i) -> jax.Array:
                        lstm_state = inference_interfaces[i].get_lstm_state(env)
                        return lstm_state.h

                else:

                    def lstm_readout(env: ENV, i=i) -> jax.Array:
                        return jnp.array([])

                transition_functions.append(lstm_transition)
                readout_functions.append(lstm_readout)

            case IdentityLayer(activation_fn):

                def identity_transition(
                    env: ENV, data: jax.Array, activation_fn=activation_fn, i=i
                ) -> tuple[ENV, jax.Array]:
                    x = get_activation_fn(activation_fn)(data)
                    return env, x

                def identity_readout(env: ENV, i=i) -> jax.Array:
                    return jnp.array([])

                transition_functions.append(identity_transition)
                readout_functions.append(identity_readout)

    def transition_inference_step(env: ENV, data: DATA) -> ENV:
        x = data_interface.get_input(data)
        for transition_fn in transition_functions:
            env, x = transition_fn(env, x)
        return env

    def readout_inference_step(env: ENV, data: DATA) -> jax.Array:
        x = data_interface.get_input(data)
        x_history = [x] if config.readout_uses_input_data else [jnp.zeros_like(x)]
        for readout_fn in readout_functions:
            x_history.append(readout_fn(env))

        readout_param = inference_interfaces[0].get_readout_param(env)
        y = readout_param(jnp.concatenate(x_history, axis=-1))
        return y

    transition_inference = eqx.filter_vmap(
        lambda e, d: transition_inference_step(e, d.b), in_axes=(axes, 0), out_axes=axes
    )
    readout_inference = eqx.filter_vmap(
        lambda e, d: batched(readout_inference_step(e, d.b)), in_axes=(axes, 0), out_axes=0
    )

    return transition_inference, readout_inference


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


def make_resets[ENV](
    get_env: Callable[[PRNG], ENV],
    inference_interfaces: dict[int, dict[int, InferenceInterface[ENV]]],
    general_interfaces: dict[int, GeneralInterface[ENV]],
    validation_interfaces: dict[int, LearnInterface[ENV]],
    virtual_minibatches: dict[int, int],
    learn_interface: LearnInterface[ENV],
) -> dict[int, Callable[[ENV], ENV]]:
    _inferences: dict[int, Callable[[ENV], ENV]] = {}
    for j, _ in sorted(inference_interfaces.items()):

        def reset_inference(env: ENV, i=j) -> ENV:
            current_virtual_minibatch = general_interfaces[i].get_current_virtual_minibatch(env)

            def do_reset(env: ENV, i=i) -> ENV:
                prng, env = validation_interfaces[min(validation_interfaces.keys())].get_prng(env)
                env0 = get_env(prng)
                env = reset_validation_learn_env(env0, env, learn_interface)
                return reset_inference_env(env0, env, inference_interfaces[i])

            env = filter_cond(
                current_virtual_minibatch % virtual_minibatches[i] == 0,
                do_reset,
                lambda e: e,
                env,
            )
            env = general_interfaces[i].put_current_virtual_minibatch(env, current_virtual_minibatch + 1)

            return env

        _inferences[j] = reset_inference
    return _inferences
