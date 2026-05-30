import math
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
from meta_learn_lib.create_env import create_inference_state, get_output_shapes


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


def make_const_reader[ENV](value: jax.Array) -> Callable[[ENV], Outputs]:
    def read(env: ENV) -> Outputs:
        return Outputs(prediction=value)

    return read


def get_reader[ENV](
    node_graph: dict[Uncanon, set[Uncanon]],
    nodes: dict[Canon, Node],
    aliases: dict[Uncanon, Canon],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
) -> PMap[Uncanon, Callable[[ENV], Outputs]]:
    from_env: PMap[Uncanon, Callable[[ENV], Outputs]] = pmap()
    for node_name in toposort_flatten(node_graph):
        canon = aliases.get(node_name, node_name)
        match nodes[canon]:
            case NNLayer():
                from_env = from_env.set(node_name, lambda env: Outputs())
            case VanillaRNNLayer():
                from_env = from_env.set(node_name, make_vanilla_rnn_reader(interfaces[(canon, level)]))
            case GRULayer():
                from_env = from_env.set(node_name, make_gru_reader(interfaces[(canon, level)]))
            case LSTMLayer():
                from_env = from_env.set(node_name, make_lstm_reader(interfaces[(canon, level)]))
            case Scan(graph, _, _, _, _) | MemoryScan(graph, _, _):
                sub_from_env = get_reader(graph, nodes, aliases, interfaces, level)
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
                | ExtractMu()
                | Reshape()
                | Take()
                | Interpolate()
            ):
                from_env = from_env.set(node_name, lambda env: Outputs())
            case _:
                from_env = from_env.set(node_name, lambda env: Outputs())
    return from_env


def get_inference[ENV](
    node_graph: dict[Uncanon, set[Uncanon]],
    nodes: dict[Canon, Node],
    aliases: dict[Uncanon, Canon],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    from_env: PMap[Uncanon, Callable[[ENV], Outputs]],
    dataset_source: Task,
) -> Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, PMap[Uncanon, Outputs]]]:
    def infer(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, PMap[Uncanon, Outputs]]:
        outputs: PMap[Uncanon, Outputs] = pmap()
        x_data, y_data = data
        for node_name in (n for n in toposort_flatten(node_graph) if n in node_graph):
            canon = aliases.get(node_name, node_name)
            interface = interfaces[(canon, level)]
            match nodes[canon]:
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
                case Scan(graph, autoregressive_mask, carry_transform, pred_source, _start_token):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env and n != pred_source],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs and n != pred_source],
                    ]
                    x = deps[-1].prediction
                    y = (from_env[pred_source](env) if pred_source in from_env else outputs[pred_source]).prediction
                    last_sub_node = toposort_flatten(graph)[-1]
                    token = interface.autoregressive_predictions.get(env)

                    fn = get_inference(graph, nodes, aliases, interfaces, level, pmap(), dataset_source)

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
                                source = y
                            case "identity":
                                source = out
                            case "erase":
                                source = jnp.zeros_like(out)
                        match carry_transform:
                            case "identity":
                                new_autoregress = source
                            case "take_last":
                                new_autoregress = source[-1]
                            case "take_first":
                                new_autoregress = source[0]
                        return (env, new_autoregress), out

                    (env, new_token), predictions = jax.lax.scan(step, (env, token), (x, y))
                    env = interface.autoregressive_predictions.put(env, new_token)
                    outputs = outputs.set(node_name, Outputs(prediction=predictions))

                case MemoryScan(graph, K, cell_shape):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x_t = deps[-1].prediction
                    M = interface.external_memory.get(env).buffer
                    ys = jnp.broadcast_to(x_t, (K, *x_t.shape))

                    env = interface.autoregressive_predictions.put(env, x_t)

                    inner_scan = Scan(
                        graph=graph,
                        autoregressive_mask="teacher_forcing",
                        carry_transform="identity",
                        pred_source="_ys_src",
                        start_token="zeros",
                    )
                    inner_graph = {node_name: frozenset({"_M_src", "_ys_src"})}
                    inner_nodes = {**nodes, node_name: inner_scan}
                    inner_from_env = pmap(
                        {
                            "_M_src": make_const_reader(M),
                            "_ys_src": make_const_reader(ys),
                        }
                    )

                    inner_fn = get_inference(
                        inner_graph,
                        inner_nodes,
                        aliases,
                        interfaces,
                        level,
                        inner_from_env,
                        dataset_source,
                    )
                    env, inner_outputs = inner_fn(env, (None, None))

                    M_new = inner_outputs[node_name].prediction
                    env = interface.external_memory.put(env, ExternalMemory(buffer=M_new))
                    outputs = outputs.set(node_name, Outputs(prediction=M_new[-1]))

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
                case ExtractMu(i):
                    dep = [outputs[dep_name] for dep_name in node_graph[node_name] if dep_name in outputs][0]
                    outputs = outputs.set(node_name, Outputs(prediction=dep.mu))
                case Reshape(target_shape):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    outputs = outputs.set(node_name, Outputs(prediction=x.reshape(target_shape)))
                case Take(start, length):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = to_vector(deps).vector
                    outputs = outputs.set(node_name, Outputs(prediction=x[start : start + length]))
                case Interpolate(n_steps, start, end):
                    z_start = (outputs[start] if start in outputs else from_env[start](env)).prediction
                    z_end = (outputs[end] if end in outputs else from_env[end](env)).prediction
                    alphas = jnp.linspace(0.0, 1.0, n_steps).reshape((n_steps,) + (1,) * z_start.ndim)
                    interp = (1 - alphas) * z_start + alphas * z_end
                    outputs = outputs.set(node_name, Outputs(prediction=interp))
                case Activation(activation_fn):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    outputs = outputs.set(node_name, Outputs(prediction=get_activation_fn(activation_fn)(x)))
                case LayerNorm() | GroupNorm():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    norm = interface.norm_module.get(env)
                    outputs = outputs.set(node_name, Outputs(prediction=norm(x)))
                case Conv2dLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    conv = interface.conv2d.get(env)
                    outputs = outputs.set(node_name, Outputs(prediction=conv(x)))
                case ConvTranspose2dLayer():
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    conv_t = interface.conv_transpose2d.get(env)
                    outputs = outputs.set(node_name, Outputs(prediction=conv_t(x)))
                case MaxPool2dLayer(kernel_size, stride):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    pool = eqx.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                    outputs = outputs.set(node_name, Outputs(prediction=pool(x)))
                case AvgPool2dLayer(kernel_size, stride):
                    deps = [
                        *[from_env[n](env) for n in node_graph[node_name] if n in from_env],
                        *[outputs[n] for n in node_graph[node_name] if n in outputs],
                    ]
                    x = deps[-1].prediction
                    pool = eqx.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
                    outputs = outputs.set(node_name, Outputs(prediction=pool(x)))
                case _:
                    outputs = outputs.set(node_name, Outputs())
        return env, outputs

    return infer


type TransitionFn[ENV] = Callable[[ENV, tuple[jax.Array, jax.Array]], ENV]
type ReadoutFn[ENV] = Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, Outputs]]


def create_raw_inference[ENV](
    transition_graph: dict[Uncanon, set[Uncanon]],
    readout_graph: dict[Uncanon, set[Uncanon]],
    nodes: dict[Canon, Node],
    aliases: dict[Uncanon, Canon],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    dataset_source: Task,
) -> tuple[TransitionFn[ENV], ReadoutFn[ENV]]:
    env_readout = get_reader(transition_graph, nodes, aliases, interfaces, level)
    raw_transition = get_inference(transition_graph, nodes, aliases, interfaces, level, pmap(), dataset_source)
    raw_readout = get_inference(readout_graph, nodes, aliases, interfaces, level, env_readout, dataset_source)
    last_node = toposort_flatten(readout_graph)[-1]

    def transition_fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> ENV:
        env, _ = raw_transition(env, data)
        return env

    def readout_fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, Outputs]:
        env, outputs = raw_readout(env, data)
        return env, outputs[last_node]

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
        config.aliases,
        interfaces,
        level,
        config.levels[level].dataset_source,
    )

    transition_inference: TransitionFn[ENV] = eqx.filter_vmap(
        eqx.filter_vmap(lambda e, d: transition_fn(e, d), in_axes=(axes, 0), out_axes=axes),
        in_axes=(axes, 0),
        out_axes=axes,
    )
    readout_inference: ReadoutFn[ENV] = eqx.filter_vmap(
        eqx.filter_vmap(lambda e, d: readout_fn(e, d), in_axes=(axes, 0), out_axes=(axes, 0)),
        in_axes=(axes, 0),
        out_axes=(axes, 0),
    )

    return transition_inference, readout_inference
