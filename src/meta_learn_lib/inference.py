from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from pyrsistent import pmap
from pyrsistent.typing import PMap
from toposort import toposort_flatten

from meta_learn_lib.config import *
from meta_learn_lib.env import *
from meta_learn_lib.interface import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.constants import *
from meta_learn_lib.util import get_activation_fn, to_vector


def make_vanilla_rnn_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        rnn_state = interface.vanilla_rnn_state.get(env)
        return Outputs(prediction=rnn_state.activation)

    return read


def make_gru_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        a = interface.gru_activation.get(env)
        return Outputs(prediction=a)

    return read


def make_lstm_reader[ENV](interface: GodInterface[ENV]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        lstm_state = interface.lstm_state.get(env)
        return Outputs(prediction=lstm_state.h)

    return read


def make_scan_reader[ENV](sub_from_env: PMap[str, Callable[[ENV], Outputs]]) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        x = to_vector([f(env) for f in sub_from_env.values()]).vector
        return Outputs(prediction=x)

    return read


def get_reader[ENV](
    node_graph: dict[str, set[str]],
    nodes: dict[str, Node],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
) -> PMap[str, Callable[[ENV], Outputs]]:
    from_env: PMap[str, Callable[[ENV], Outputs]] = pmap()
    for node_name in toposort_flatten(node_graph):
        match nodes[node_name]:
            case NNLayer():
                from_env = from_env.set(node_name, lambda env: Outputs())
            case VanillaRNNLayer():
                from_env = from_env.set(node_name, make_vanilla_rnn_reader(interfaces[(node_name, level)]))
            case GRULayer():
                from_env = from_env.set(node_name, make_gru_reader(interfaces[(node_name, level)]))
            case LSTMLayer():
                from_env = from_env.set(node_name, make_lstm_reader(interfaces[(node_name, level)]))
            case Scan(graph, autoregressive_mask, pred_source, _start_token):
                sub_from_env = get_reader(graph, nodes, interfaces, level)
                from_env = from_env.set(node_name, make_scan_reader(sub_from_env))
            case (
                Repeat()
                | Concat()
                | ToEmpty()
                | UnlabeledSource()
                | LabeledSource()
                | ReparameterizeLayer()
                | MergeOutputs()
                | ExtractZ()
                | Reshape()
            ):
                from_env = from_env.set(node_name, lambda env: Outputs())
            case _:
                from_env = from_env.set(node_name, lambda env: Outputs())
    return from_env


def get_inference[ENV](
    node_graph: dict[str, set[str]],
    nodes: dict[str, Node],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    from_env: PMap[str, Callable[[ENV], Outputs]],
) -> Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, PMap[str, Outputs]]]:
    def infer(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, PMap[str, Outputs]]:
        outputs: PMap[str, Outputs] = pmap()
        x_data, y_data = data
        for node_name in (n for n in toposort_flatten(node_graph) if n in node_graph):
            interface = interfaces[(node_name, level)]
            match nodes[node_name]:
                case NNLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    mlp = interface.mlp_model.get(env)
                    y = mlp(x)
                    outputs = outputs.set(node_name, Outputs(prediction=y))
                case VanillaRNNLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    w_rec = interface.rnn_w_rec.get(env)
                    b_rec = interface.rnn_b_rec.get(env)
                    layer_norm = interface.rnn_layer_norm.get(env)
                    rnn_state = interface.vanilla_rnn_state.get(env)
                    a = rnn_state.activation
                    activation_fn = rnn_state.activation_fn
                    alpha = interface.time_constant.get(env)

                    a_rec = w_rec @ jnp.concat((a, x)) + b_rec
                    a_rec = layer_norm(a_rec)
                    a_new = (1 - alpha) * a + alpha * get_activation_fn(activation_fn)(a_rec)
                    env = interface.vanilla_rnn_state.put(
                        env,
                        rnn_state.set(activation=a_new),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=a_new))
                case GRULayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    gru = interface.gru_cell.get(env)
                    alpha = interface.time_constant.get(env)
                    a = interface.gru_activation.get(env)

                    a_rec = gru(x, a)
                    a_new = (1 - alpha) * a + alpha * a_rec
                    env = interface.gru_activation.put(env, a_new)
                    outputs = outputs.set(node_name, Outputs(prediction=a_new))
                case LSTMLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    lstm = interface.lstm_cell.get(env)
                    alpha = interface.time_constant.get(env)
                    lstm_st = interface.lstm_state.get(env)
                    h, c = lstm_st.h, lstm_st.c
                    h_rec, c_rec = lstm(x, (h, c))
                    h_new = (1 - alpha) * h + alpha * h_rec
                    c_new = (1 - alpha) * c + alpha * c_rec
                    env = interface.lstm_state.put(
                        env,
                        lstm_st.set(h=h_new, c=c_new),
                    )
                    outputs = outputs.set(node_name, Outputs(prediction=h_new))
                case Scan(graph, autoregressive_mask, pred_source, _start_token):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env and n != pred_source],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs and n != pred_source],
                    ]
                    x = to_vector(deps).vector
                    y = from_env[pred_source](env) if pred_source in from_env else outputs[pred_source].prediction
                    last_sub_node = toposort_flatten(graph)[-1]
                    token = interface.autoregressive_predictions.get(env)

                    fn = get_inference(graph, nodes, interfaces, level, pmap())

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
                    env = interface.autoregressive_predictions.put(env, new_token)
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
                case ReparameterizeLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    n = x.shape[0] // 2
                    mu, log_var = x[:n], x[n:]
                    key, env = interface.take_prng(env)
                    std = jnp.exp(0.5 * log_var)
                    z = mu + std * jax.random.normal(key, mu.shape)
                    outputs = outputs.set(node_name, Outputs(mu=mu, log_var=log_var, z=z))
                case MergeOutputs():
                    upstream = [outputs[n] for n in node_graph[node_name] if n in outputs]
                    merged = eqx.combine(*upstream, is_leaf=lambda x: x is None)
                    outputs = outputs.set(node_name, merged)
                case ExtractZ(i):
                    dep = [outputs[dep_name] for dep_name in node_graph[node_name] if dep_name in outputs][0]
                    outputs = outputs.set(node_name, Outputs(prediction=dep.z))
                case Reshape(target_shape):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    outputs = outputs.set(node_name, Outputs(prediction=x.reshape(target_shape)))
                case _:
                    outputs = outputs.set(node_name, Outputs())
        return env, outputs

    return infer


type TransitionFn[ENV] = Callable[[ENV, tuple[jax.Array, jax.Array]], ENV]
type ReadoutFn[ENV] = Callable[[ENV, tuple[jax.Array, jax.Array]], Outputs]


def create_raw_inference[ENV](
    transition_graph: dict[str, set[str]],
    readout_graph: dict[str, set[str]],
    nodes: dict[str, Node],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
) -> tuple[TransitionFn[ENV], ReadoutFn[ENV]]:
    env_readout = get_reader(transition_graph, nodes, interfaces, level)
    raw_transition = get_inference(transition_graph, nodes, interfaces, level, pmap())
    raw_readout = get_inference(readout_graph, nodes, interfaces, level, env_readout)
    last_node = toposort_flatten(readout_graph)[-1]

    def transition_fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> ENV:
        env, _ = raw_transition(env, data)
        return env

    def readout_fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> Outputs:
        _, outputs = raw_readout(env, data)
        return outputs[last_node]

    return transition_fn, readout_fn


def create_inference_and_readout[ENV](
    config: GodConfig,
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    axes: ENV,
) -> tuple[TransitionFn[ENV], ReadoutFn[ENV]]:
    transition_fn, readout_fn = create_raw_inference(
        config.transition_graph,
        config.readout_graph,
        config.nodes,
        interfaces,
        level,
    )

    transition_inference: TransitionFn[ENV] = eqx.filter_vmap(
        eqx.filter_vmap(lambda e, d: transition_fn(e, d), in_axes=(axes, 0), out_axes=axes),
        in_axes=(axes, 0),
        out_axes=axes,
    )
    readout_inference: ReadoutFn[ENV] = eqx.filter_vmap(
        eqx.filter_vmap(lambda e, d: readout_fn(e, d), in_axes=(axes, 0), out_axes=0),
        in_axes=(axes, 0),
        out_axes=0,
    )

    return transition_inference, readout_inference
