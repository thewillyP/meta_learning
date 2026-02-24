import math
from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from pyrsistent import pmap
from graphlib import TopologicalSorter
import functools
from pyrsistent import pmap, pvector


from meta_learn_lib.config import *
from meta_learn_lib.create_axes import create_axes, diff_axes
from meta_learn_lib.env import *
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import *
from meta_learn_lib.optimizer import get_opt_state
from meta_learn_lib.util import get_activation_fn, hyperparameter_reparametrization


def get_output_shapes(
    node_features: dict[str, tuple[int, ...]],
    node_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    data_shape: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    x_shape, y_shape = data_shape
    for node_name in TopologicalSorter(node_graph).static_order():
        match nodes[node_name]:
            case NNLayer(n, activation_fn, use_bias, layer_norm):
                node_features[node_name] = (n,)
            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                node_features[node_name] = (nn_layer.n,)
            case GRULayer(n, use_bias, use_random_init):
                node_features[node_name] = (n,)
            case LSTMLayer(n, use_bias, use_random_init):
                node_features[node_name] = (n,)
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                # Because I dont know how to concatenate your multiple dependencies, im just going to take the last one
                scan_x_shape = [node_features[n] for n in node_graph[node_name] if n != pred_source][-1]
                scan_y_shape = node_features[pred_source]
                last_sub_node = list(TopologicalSorter(graph).static_order())[-1]
                node_features = get_output_shapes(node_features, graph, nodes, (scan_x_shape[1:], scan_y_shape[1:]))
                n_out = (scan_x_shape[0],) + node_features[last_sub_node]
                node_features[node_name] = n_out
            case Repeat(n):
                # same issue as scan, just take the last one
                n_in = [node_features[n] for n in node_graph[node_name]][-1]
                node_features[node_name] = (n,) + n_in
            case Concat():
                n_in = sum([math.prod(node_features[n]) for n in node_graph[node_name]])
                node_features[node_name] = (n_in,)
            case ToEmpty():
                node_features[node_name] = ()
            case UnlabeledSource():
                node_features[node_name] = x_shape
            case LabeledSource():
                node_features[node_name] = y_shape
            case _:
                node_features[node_name] = ()

    return node_features


def create_inference_state[ENV](
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
    node_features: dict[str, tuple[int, ...]],
    track_influence_in: frozenset[int],
    is_init: bool,
    env: ENV,
    prng: PRNG,
) -> ENV:

    for node_name, node in nodes.items():
        interface = meta_interface[node_name]
        match node:
            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (nn_layer.n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((nn_layer.n,)))

                rnn_state = VanillaRecurrentState(
                    activation=State(
                        value=activation,
                        is_stateful=track_influence_in,
                    ),
                    activation_fn=nn_layer.activation_fn,
                )
                env = interface.put_vanilla_rnn_state(env, rnn_state)
                env = interface.put_prng(env, k2)
            case GRULayer(n, use_bias, use_random_init):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))

                gru_state = RecurrentState(activation=State(value=activation, is_stateful=track_influence_in))
                env = interface.put_gru_state(env, gru_state)
                env = interface.put_prng(env, k2)
            case LSTMLayer(n, use_bias, use_random_init):
                k1, k2, k3, prng = jax.random.split(prng, 4)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                    cell = ACTIVATION(jax.random.normal(k2, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))
                    cell = ACTIVATION(jnp.zeros((n,)))

                lstm_state = LSTMState(
                    h=State(value=activation, is_stateful=track_influence_in),
                    c=State(value=cell, is_stateful=track_influence_in),
                )
                env = interface.put_lstm_state(env, lstm_state)
                env = interface.put_prng(env, k3)
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                k1, prng = jax.random.split(prng, 2)
                shape = node_features[node_name][1:]  # remove time dimension
                match start_token:
                    case "zeros":
                        token = jnp.zeros(shape)
                env = interface.put_autoregressive_predictions(env, State(value=token, is_stateful=track_influence_in))
                env = interface.put_prng(env, k1)
            case _:
                # could save space here on deterministic nodes
                k1, prng = jax.random.split(prng, 2)
                env = interface.put_prng(env, k1)

    return env


def create_inference_parameters[ENV](
    nodes: dict[str, Node],
    transition_graph: dict[str, list[str]],
    readout_graph: dict[str, list[str]],
    meta_interface: dict[str, GodInterface[ENV]],
    node_features: dict[str, tuple[int, ...]],
    learnables: frozenset[str],
    env: ENV,
    prng: PRNG,
) -> ENV:

    node_graph = transition_graph | readout_graph

    for node_name, node in nodes.items():
        interface = meta_interface[node_name]
        is_learnable = node_name in learnables
        match node:
            case NNLayer(n, activation_fn, use_bias, layer_norm):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                linear = eqx.nn.Linear(n_in, n, use_bias=use_bias, key=k1)
                new_weight = jax.random.normal(k1, (n, n_in)) * jnp.sqrt(1 / n_in)
                new_bias = jnp.zeros((n,)) if use_bias else None
                where = lambda l: (l.weight, l.bias)
                new_linear: eqx.Module = eqx.tree_at(where, linear, (new_weight, new_bias))

                match layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
                    case None:
                        layer = eqx.nn.Identity()

                nn_layer = eqx.nn.Sequential([new_linear, layer, eqx.nn.Lambda(get_activation_fn(activation_fn))])
                env = interface.put_mlp_param(
                    env,
                    MLP(
                        model=Parameter[eqx.nn.Sequential](
                            value=nn_layer,
                            is_learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=node_name not in readout_graph,
                        )
                    ),
                )
                env = interface.put_prng(env, k2)

            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, k3, prng = jax.random.split(prng, 4)

                W_in = jax.random.normal(k1, (nn_layer.n, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(k2, (nn_layer.n, nn_layer.n)))[0]
                w_rec = Parameter(
                    value=jnp.hstack([W_rec, W_in]),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                    parametrizes_transition=node_name not in readout_graph,
                )
                b_rec: jax.Array = Parameter(
                    value=jnp.zeros((nn_layer.n,)),
                    is_learnable=nn_layer.use_bias,
                    min_value=-math.inf,
                    max_value=math.inf,
                    parametrizes_transition=node_name not in readout_graph,
                )

                match nn_layer.layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = Parameter(
                            value=eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias),
                            is_learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=node_name not in readout_graph,
                        )
                    case None:
                        layer = Parameter(
                            value=eqx.nn.Identity(),
                            is_learnable=False,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=node_name not in readout_graph,
                        )

                rnn_param = RNN(w_rec=w_rec, b_rec=b_rec, layer_norm=layer)
                env = interface.put_vanilla_rnn_param(env, rnn_param)
                env = interface.put_prng(env, k3)
            case GRULayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                gru = Parameter(
                    value=eqx.nn.GRUCell(n_in, n, use_bias=use_bias, key=k1),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                    parametrizes_transition=node_name not in readout_graph,
                )
                env = interface.put_gru_param(env, gru)
                env = interface.put_prng(env, k2)

            case LSTMLayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                lstm = Parameter(
                    value=eqx.nn.LSTMCell(n_in, n, use_bias=use_bias, key=k1),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                    parametrizes_transition=node_name not in readout_graph,
                )
                env = interface.put_lstm_param(env, lstm)
                env = interface.put_prng(env, k2)
            case _:
                # could save space here on deterministic nodes
                k1, prng = jax.random.split(prng, 2)
                env = interface.put_prng(env, k1)

    return env


def create_hyperparameters[ENV](
    hps: dict[HP, HyperparameterConfig],
    meta_interface: dict[str, GodInterface[ENV]],
    learnables: frozenset[str],
    env: ENV,
    prng: PRNG,
) -> ENV:
    # make sure to loop through current meta interface's hyperparameters only. not ALL hyperparameters
    for hp_name, interface in meta_interface.items():
        if hp_name in hps:
            hp_config = hps[hp_name]
            is_learnable = hp_name in learnables
            _, invert = hyperparameter_reparametrization(hp_config.hyperparameter_parametrization)
            match hp_config.kind:
                case "learning_rate":
                    k1, prng = jax.random.split(prng, 2)
                    lrs = Parameter(
                        value=jnp.full((hp_config.count,), invert(hp_config.value)),
                        is_learnable=is_learnable,
                        min_value=hp_config.min_value,
                        max_value=hp_config.max_value,
                        parametrizes_transition=hp_config.parametrizes_transition,
                    )
                    env = interface.put_learning_rate(env, lrs)
                    env = interface.put_prng(env, k1)
                case "weight_decay":
                    k1, prng = jax.random.split(prng, 2)
                    wds = Parameter(
                        value=jnp.full((hp_config.count,), invert(hp_config.value)),
                        is_learnable=is_learnable,
                        min_value=hp_config.min_value,
                        max_value=hp_config.max_value,
                        parametrizes_transition=hp_config.parametrizes_transition,
                    )
                    env = interface.put_weight_decay(env, wds)
                    env = interface.put_prng(env, k1)
                case "momentum":
                    k1, prng = jax.random.split(prng, 2)
                    ms = Parameter(
                        value=jnp.full((hp_config.count,), invert(hp_config.value)),
                        is_learnable=is_learnable,
                        min_value=hp_config.min_value,
                        max_value=hp_config.max_value,
                        parametrizes_transition=hp_config.parametrizes_transition,
                    )
                    env = interface.put_momentum(env, ms)
                    env = interface.put_prng(env, k1)
                case "time_constant":
                    k1, prng = jax.random.split(prng, 2)
                    tcs = Parameter(
                        value=jnp.full((hp_config.count,), invert(hp_config.value)),
                        is_learnable=is_learnable,
                        min_value=hp_config.min_value,
                        max_value=hp_config.max_value,
                        parametrizes_transition=hp_config.parametrizes_transition,
                    )
                    env = interface.put_time_constant(env, tcs)
                    env = interface.put_prng(env, k1)
                case "kl_regularizer_beta":
                    k1, prng = jax.random.split(prng, 2)
                    kl_betas = Parameter(
                        value=jnp.full((hp_config.count,), invert(hp_config.value)),
                        is_learnable=is_learnable,
                        min_value=hp_config.min_value,
                        max_value=hp_config.max_value,
                        parametrizes_transition=hp_config.parametrizes_transition,
                    )
                    env = interface.put_kl_regularizer_beta(env, kl_betas)
                    env = interface.put_prng(env, k1)

    return env


def create_learner_states[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    method: GradientMethod,
    interface: GodInterface[ENV],
    track_influence_in: frozenset[int],
    env: ENV,
    prng: PRNG,
) -> ENV:
    match method:
        case RTRLConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig() | RFLOConfig():
            k1, k2, prng = jax.random.split(prng, 3)
            new_env = factory(env, k1)
            param = interface.get_param(new_env)
            state = interface.get_state(new_env)

            def infl_fn(p: jax.Array) -> jax.Array:
                _env = interface.put_param(new_env, p)
                _env = factory(_env, k1)
                s = interface.get_state(_env)
                return s

            if state.shape[0] > param.shape[0]:
                dhdp = eqx.filter_jacfwd(infl_fn)(param)
            else:
                dhdp = eqx.filter_jacrev(infl_fn)(param)
            env = interface.put_forward_mode_jacobian(new_env, State(value=dhdp, is_stateful=track_influence_in))
            env = interface.put_prng(env, k2)
        case UOROConfig():
            k0, k1, k2, k3, prng = jax.random.split(prng, 5)
            env = factory(env, k0)

            param = interface.get_param(env)
            state = interface.get_state(env)

            # A: random init, shape = (|h|,)
            a = jax.random.normal(k1, state.shape)

            # B: start fully random, then zero out nonrecurrent params
            b_env = interface.put_param(env, jax.random.normal(k2, param.shape))

            def zero_readout(x):
                if isinstance(x, Parameter) and not x.parametrizes_transition:
                    arrays, non_arrays = eqx.partition(x, eqx.is_inexact_array)
                    zeroed = jax.tree.map(lambda v: jnp.zeros_like(v), arrays)
                    return eqx.combine(zeroed, non_arrays)
                return x

            b_env = jax.tree.map(zero_readout, b_env, is_leaf=lambda x: isinstance(x, Parameter))

            b = interface.get_param(b_env)

            uoro = UOROState(
                A=State(value=a, is_stateful=track_influence_in),
                B=State(value=b, is_stateful=track_influence_in),
            )
            env = interface.put_uoro_state(env, uoro)
            env = interface.put_prng(env, k3)
        case _:
            k1, k2, prng = jax.random.split(prng, 3)
            env = factory(env, k1)
            env = interface.put_prng(env, k2)

    return env


def vmap_factory[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    batch: list[int],
) -> Callable[[ENV, PRNG], ENV]:
    def vmap_env(env: ENV, prng: PRNG) -> ENV:
        k1, k2, prng = jax.random.split(prng, 3)
        old_axes = create_axes(env)
        new_axes = create_axes(factory(env, k1))
        axes = diff_axes(old_axes, new_axes)
        batched_keys = jax.random.split(k2, math.prod(batch)).reshape(*batch)
        b_fn = functools.reduce(
            lambda f, _: eqx.filter_vmap(f, in_axes=(None, 0), out_axes=axes),
            batch,
            lambda e, k: factory(e, k),
        )
        return b_fn(env, batched_keys)

    return vmap_env


def reset_validation[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    meta_interface: dict[str, GodInterface[ENV]],
    vl_learner: GodInterface[ENV],
    meta_config: MetaConfig,
    nodes: dict[str, Node],
    node_features: dict[str, tuple[int, ...]],
    is_init: bool,  # if need to reset things like RL ENV
) -> Callable[[ENV, PRNG], ENV]:

    def create_env(env: ENV, prng: PRNG) -> ENV:

        # 1. Create inference state
        k1, prng = jax.random.split(prng, 2)
        f1 = lambda e, k: create_inference_state(
            nodes, meta_interface, node_features, meta_config.dataset_validation.track_influence_in, is_init, e, k
        )
        batch_size = [
            meta_config.dataset_validation.task_batch_size,
            meta_config.dataset_validation.num_examples_in_minibatch,
        ]
        # env = vmap_factory(f1, batch_size)(env, k1)
        env = f1(env, k1)

        # 2. use it for learning states
        k1, k2, prng = jax.random.split(prng, 3)

        env = create_learner_states(
            factory,
            meta_config.learner.model_learner.method,
            vl_learner,
            meta_config.dataset_validation.track_influence_in,
            env,
            k2,
        )

        return env

    return create_env


def reset_states[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    meta_interface: dict[str, GodInterface[ENV]],
    meta_learner: GodInterface[ENV],
    meta_config: MetaConfig,
    hyperparameters: dict[HP, HyperparameterConfig],
    nodes: dict[str, Node],
    transition_graph: dict[str, list[str]],
    readout_graph: dict[str, list[str]],
    node_features: dict[str, tuple[int, ...]],
    create_inference_param: bool,
) -> Callable[[ENV, PRNG], ENV]:

    learnables: frozenset[str] = frozenset().union(*[v.target for v in meta_config.learner.optimizer.values()])

    def create_env(env: ENV, prng: PRNG) -> ENV:

        # 1. Create parameters
        if create_inference_param:
            k1, prng = jax.random.split(prng, 2)
            env = create_inference_parameters(
                nodes, transition_graph, readout_graph, meta_interface, node_features, learnables, env, k1
            )

        # 2. Create hyperparameters
        k2, prng = jax.random.split(prng, 2)
        env = create_hyperparameters(hyperparameters, meta_interface, learnables, env, k2)

        # 3. use it for optimizer states
        env = get_opt_state(
            meta_config.learner.optimizer,
            meta_interface,
            env,
            hyperparameters,
            meta_config.meta_opt.track_influence_in,
        )

        # 3. use it for learning states
        k3, prng = jax.random.split(prng, 2)

        env = create_learner_states(
            factory,
            meta_config.learner.optimizer_learner.method,
            meta_learner,
            meta_config.meta_opt.track_influence_in,
            env,
            k3,
        )

        # 4. Create the logs
        logs = Logs(
            gradient=jnp.zeros_like(meta_learner.get_param(env)) if meta_config.track_logs.gradient else None,
            hessian_contains_nans=jnp.array(False) if meta_config.track_logs.hessian_contains_nans else None,
            largest_eigenvalue=jnp.array(0.0) if meta_config.track_logs.largest_eigenvalue else None,
            influence_tensor=jnp.zeros((meta_learner.get_state(env).shape[0], meta_learner.get_param(env).shape[0]))
            if meta_config.track_logs.influence_tensor
            else None,
            immediate_influence_tensor=jnp.zeros(
                (meta_learner.get_state(env).shape[0], meta_learner.get_param(env).shape[0])
            )
            if meta_config.track_logs.immediate_influence_tensor
            else None,
            largest_jac_eigenvalue=jnp.array(0.0) if meta_config.track_logs.largest_jac_eigenvalue else None,
            jacobian=jnp.zeros((meta_learner.get_state(env).shape[0], meta_learner.get_state(env).shape[0]))
            if meta_config.track_logs.jacobian
            else None,
        )

        env = meta_learner.put_logs(env, logs)

        return env

    return create_env


def create_empty_env(config: GodConfig, prng: PRNG) -> GodState:

    k1, prng = jax.random.split(prng, 2)
    env = GodState(
        model_states=pvector(
            [
                ModelStates(
                    recurrent_states=pmap({}),
                    vanilla_recurrent_states=pmap({}),
                    lstm_states=pmap({}),
                    autoregressive_predictions=pmap({}),
                )
                for _ in range(len(config.levels))
            ]
        ),
        learning_states=pvector(
            [
                LearningStates(
                    influence_tensors=pmap({}),
                    uoros=pmap({}),
                    opt_states=pmap({}),
                )
                for _ in range(len(config.levels))
            ]
        ),
        meta_parameters=pvector(
            [
                Parameters(
                    mlps=pmap({}),
                    rnns=pmap({}),
                    grus=pmap({}),
                    lstms=pmap({}),
                    learning_rates=pmap({}),
                    weight_decays=pmap({}),
                    time_constants=pmap({}),
                    momentums=pmap({}),
                    kl_regularizer_betas=pmap({}),
                )
                for _ in range(len(config.levels))
            ]
        ),
        level_meta=pvector(
            [
                LevelMeta(
                    tick=jnp.array(0),
                    log=State(value=Logs(), is_stateful=frozenset()),
                    prngs=pmap({}),
                )
                for _ in range(len(config.levels) + 1)
            ]
        ),
        prng=k1,
    )

    return env


def create_env(
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    meta_interfaces: list[dict[str, GodInterface[GodState]]],
    learn_interfaces: list[tuple[GodInterface[GodState], GodInterface[GodState]]],
    prng: PRNG,
) -> GodState:
    is_inits = [True] * len(config.levels)
    create_inference_params = [True] + [False] * (len(config.levels) - 1)

    k1, k2, prng = jax.random.split(prng, 3)
    env = create_empty_env(config, k1)
    creator = env_creator(
        config,
        shapes,
        meta_interfaces,
        learn_interfaces,
        is_inits,
        create_inference_params,
    )
    return creator(env, k2)


def env_creator(
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    meta_interfaces: list[dict[str, GodInterface[GodState]]],
    learn_interfaces: list[tuple[GodInterface[GodState], GodInterface[GodState]]],
    is_inits: list[bool],
    create_inference_params: list[bool],
) -> Callable[[GodState, PRNG], GodState]:

    factory = lambda e, k: e

    def fold(
        accum: Callable[[GodState, PRNG], GodState],
        shape: tuple[tuple[int, ...], tuple[int, ...]],
        meta_interface: dict[str, GodInterface[GodState]],
        learn_interface: tuple[GodInterface[GodState], GodInterface[GodState]],
        meta_config: MetaConfig,
        create_inference_param: bool,
        is_init: bool,
    ) -> GodState:
        node_features = get_output_shapes({}, config.readout_graph | config.transition_graph, config.nodes, shape)
        val_interface, nest_interface = learn_interface

        generator = reset_states(
            reset_validation(
                accum,
                meta_interface,
                val_interface,
                meta_config,
                config.nodes,
                node_features,
                is_init,
            ),
            meta_interface,
            nest_interface,
            meta_config,
            config.hyperparameters,
            config.nodes,
            config.transition_graph,
            config.readout_graph,
            node_features,
            create_inference_param,
        )
        return generator
        # generatored = vmap_factory(generator, [meta_config.meta_opt.batch])
        # return generatored

    creator = functools.reduce(
        lambda acc, args: fold(acc, *args),
        zip(shapes, meta_interfaces, learn_interfaces, config.levels, create_inference_params, is_inits),
        factory,
    )

    return creator
