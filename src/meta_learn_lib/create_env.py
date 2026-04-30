import math
from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from pyrsistent import pmap, pvector
from toposort import toposort_flatten


from meta_learn_lib.config import *
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.env import *
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import *
from meta_learn_lib.constants import *
from meta_learn_lib.optimizer import init_opt_state
from meta_learn_lib.util import filter_cond, get_activation_fn, hyperparameter_reparametrization


def get_output_shapes(
    node_features: dict[str, tuple[int, ...]],
    node_graph: dict[str, set[str]],
    nodes: dict[str, Node],
    data_shape: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    x_shape, y_shape = data_shape
    for node_name in toposort_flatten(node_graph):
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
                last_sub_node = toposort_flatten(graph)[-1]
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
            case ReparameterizeLayer():
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                node_features[node_name] = (n_in // 2,)
            case MergeOutputs():
                node_features[node_name] = (0,)  # is an end node
            case ExtractZ(n):
                node_features[node_name] = (n,)
            case Reshape(target_shape):
                node_features[node_name] = target_shape
            case _:
                node_features[node_name] = ()

    return node_features


def create_inference_state[ENV](
    nodes: dict[str, Node],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    node_features: dict[str, tuple[int, ...]],
    track_influence_in: frozenset[int],
    is_init: bool,
    env: ENV,
    prng: PRNG,
) -> ENV:

    for node_name, node in nodes.items():
        interface = interfaces[(node_name, level)]
        match node:
            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (nn_layer.n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((nn_layer.n,)))

                rnn_state = VanillaRecurrentState(activation=activation, activation_fn=nn_layer.activation_fn)
                env = interface.vanilla_rnn_state.put_tagged(
                    env, Tagged(value=rnn_state, meta=StateMeta(is_stateful=track_influence_in))
                )
                env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))
            case GRULayer(n, use_bias, use_random_init):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))

                env = interface.gru_activation.put_tagged(
                    env, Tagged(value=activation, meta=StateMeta(is_stateful=track_influence_in))
                )
                env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))
            case LSTMLayer(n, use_bias, use_random_init):
                k1, k2, k3, prng = jax.random.split(prng, 4)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                    cell = ACTIVATION(jax.random.normal(k2, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))
                    cell = ACTIVATION(jnp.zeros((n,)))

                lstm_state = LSTMState(h=activation, c=cell)
                env = interface.lstm_state.put_tagged(
                    env, Tagged(value=lstm_state, meta=StateMeta(is_stateful=track_influence_in))
                )
                env = interface.prng.put_tagged(env, Tagged(value=k3, meta=StateMeta(is_stateful=frozenset())))
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                k1, prng = jax.random.split(prng, 2)
                shape = node_features[node_name][1:]
                match start_token:
                    case "zeros":
                        token = jnp.zeros(shape)
                env = interface.autoregressive_predictions.put_tagged(
                    env, Tagged(value=token, meta=StateMeta(is_stateful=track_influence_in))
                )
                env = interface.prng.put_tagged(env, Tagged(value=k1, meta=StateMeta(is_stateful=frozenset())))
            case _:
                k1, prng = jax.random.split(prng, 2)
                env = interface.prng.put_tagged(env, Tagged(value=k1, meta=StateMeta(is_stateful=frozenset())))

    return env


def create_inference_parameters[ENV](
    nodes: dict[str, Node],
    transition_graph: dict[str, set[str]],
    readout_graph: dict[str, set[str]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    node_features: dict[str, tuple[int, ...]],
    learnables: frozenset[str],
    env: ENV,
    prng: PRNG,
) -> ENV:

    node_graph = transition_graph | readout_graph

    for node_name, node in nodes.items():
        interface = interfaces[(node_name, level)]
        is_learnable = node_name in learnables
        is_transition = node_name not in readout_graph
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

                nn_model = eqx.nn.Sequential([new_linear, layer, eqx.nn.Lambda(get_activation_fn(activation_fn))])
                env = interface.mlp_model.put_tagged(
                    env,
                    Tagged(
                        value=nn_model,
                        meta=ParamMeta(
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))

            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, k3, prng = jax.random.split(prng, 4)

                W_in = jax.random.normal(k1, (nn_layer.n, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(k2, (nn_layer.n, nn_layer.n)))[0]
                w_rec = jnp.hstack([W_rec, W_in])
                b_rec = jnp.zeros((nn_layer.n,))

                match nn_layer.layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = eqx.nn.LayerNorm(nn_layer.n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
                    case None:
                        layer = eqx.nn.Identity()

                env = interface.rnn_w_rec.put_tagged(
                    env,
                    Tagged(
                        value=w_rec,
                        meta=ParamMeta(
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.rnn_b_rec.put_tagged(
                    env,
                    Tagged(
                        value=b_rec,
                        meta=ParamMeta(
                            learnable=is_learnable and nn_layer.use_bias,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.rnn_layer_norm.put_tagged(
                    env,
                    Tagged(
                        value=layer,
                        meta=ParamMeta(
                            learnable=is_learnable and nn_layer.layer_norm is not None,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.prng.put_tagged(env, Tagged(value=k3, meta=StateMeta(is_stateful=frozenset())))

            case GRULayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                gru = eqx.nn.GRUCell(n_in, n, use_bias=use_bias, key=k1)
                env = interface.gru_cell.put_tagged(
                    env,
                    Tagged(
                        value=gru,
                        meta=ParamMeta(
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))

            case LSTMLayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                lstm = eqx.nn.LSTMCell(n_in, n, use_bias=use_bias, key=k1)
                env = interface.lstm_cell.put_tagged(
                    env,
                    Tagged(
                        value=lstm,
                        meta=ParamMeta(
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                        ),
                    ),
                )
                env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))
            case _:
                k1, prng = jax.random.split(prng, 2)
                env = interface.prng.put_tagged(env, Tagged(value=k1, meta=StateMeta(is_stateful=frozenset())))

    return env


def create_hyperparameters[ENV](
    hps: dict[HP, HyperparameterConfig],
    interfaces: dict[S_ID, GodInterface[ENV]],
    learnables_per_level: list[frozenset[str]],
    env: ENV,
    prng: PRNG,
) -> ENV:
    for hp_name, hp_config in hps.items():
        if (hp_name, hp_config.level) not in interfaces:
            continue
        interface = interfaces[(hp_name, hp_config.level)]
        _, invert = hyperparameter_reparametrization(hp_config.hyperparameter_parametrization)
        val = jnp.full((hp_config.count,), invert(hp_config.value))
        meta = ParamMeta(
            learnable=hp_name in learnables_per_level[hp_config.level],
            min_value=hp_config.min_value,
            max_value=hp_config.max_value,
            parametrizes_transition=hp_config.parametrizes_transition,
        )
        k1, prng = jax.random.split(prng, 2)

        match hp_config.kind:
            case "learning_rate":
                env = interface.learning_rate.put_tagged(env, Tagged(value=val, meta=meta))
            case "weight_decay":
                env = interface.weight_decay.put_tagged(env, Tagged(value=val, meta=meta))
            case "momentum":
                env = interface.momentum.put_tagged(env, Tagged(value=val, meta=meta))
            case "time_constant":
                env = interface.time_constant.put_tagged(env, Tagged(value=val, meta=meta))
            case "kl_regularizer_beta":
                env = interface.kl_regularizer_beta.put_tagged(env, Tagged(value=val, meta=meta))

        env = interface.prng.put_tagged(env, Tagged(value=k1, meta=StateMeta(is_stateful=frozenset())))

    return env


def create_learner_states[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    method: GradientMethod,
    interfaces: dict[S_ID, GodInterface[ENV]],
    learner_key: S_ID,
    track_influence_in: frozenset[int],
    env: ENV,
    prng: PRNG,
) -> ENV:
    interface = interfaces[learner_key]
    state_meta = StateMeta(is_stateful=track_influence_in)

    match method:
        case (
            RTRLConfig()
            | TikhonovRTRLConfig()
            | PadeRTRLConfig()
            | MidpointRTRLConfig()
            | HeunRTRLConfig()
            | ImplicitEulerRTRLConfig()
            | RFLOConfig() as gradient_method
        ):
            k1, k2, prng = jax.random.split(prng, 3)
            new_env = factory(env, k1)
            param = interface.param.get(new_env)
            state = interface.state.get(new_env)

            def infl_fn(p: jax.Array) -> jax.Array:
                _env = interface.param.put(new_env, p)
                _env = factory(_env, k1)
                s = interface.state.get(_env)
                return s

            if state.shape[0] > param.shape[0]:
                dhdp = eqx.filter_jacfwd(infl_fn)(param)
            else:
                dhdp = eqx.filter_jacrev(infl_fn)(param)
            env = interface.forward_mode_jacobian.put_tagged(new_env, Tagged(value=dhdp, meta=state_meta))
            match gradient_method:
                case MidpointRTRLConfig() | HeunRTRLConfig():
                    env = interface.midpoint_buffer.put_tagged(
                        env,
                        Tagged(value=MidpointBuffer(P_prev=dhdp, predictor=dhdp), meta=state_meta),
                    )
            env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))
        case UOROConfig():
            k0, k1, k2, k3, prng = jax.random.split(prng, 5)
            env = factory(env, k0)

            param = interface.param.get(env)
            state = interface.state.get(env)

            # A: random init, shape = (|h|,)
            a = jax.random.normal(k1, state.shape)

            # B: start fully random, then zero out nonrecurrent params
            b_env = interface.param.put(env, jax.random.normal(k2, param.shape))

            def zero_readout(x):
                if isinstance(x, Tagged) and isinstance(x.meta, ParamMeta) and not x.meta.parametrizes_transition:
                    arrays, non_arrays = eqx.partition(x, eqx.is_inexact_array)
                    zeroed = jax.tree.map(lambda v: jnp.zeros_like(v), arrays)
                    return eqx.combine(zeroed, non_arrays)
                return x

            b_env = jax.tree.map(zero_readout, b_env, is_leaf=lambda x: isinstance(x, Tagged))
            b = interface.param.get(b_env)

            uoro = UOROState(A=a, B=b)
            env = interface.uoro_state.put_tagged(env, Tagged(value=uoro, meta=state_meta))
            env = interface.prng.put_tagged(env, Tagged(value=k3, meta=StateMeta(is_stateful=frozenset())))
        case _:
            k1, k2, prng = jax.random.split(prng, 3)
            env = factory(env, k1)
            env = interface.prng.put_tagged(env, Tagged(value=k2, meta=StateMeta(is_stateful=frozenset())))

    env = interface.tick.put_tagged(env, Tagged(value=jnp.array(0), meta=StateMeta(is_stateful=frozenset())))
    return env


def vmap_factory[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    batches: list[int],
) -> Callable[[ENV, PRNG], ENV]:

    def vmap_env(env: ENV, prng: PRNG) -> ENV:
        k1, k2, prng = jax.random.split(prng, 3)
        new_env = factory(env, k1)
        axes = diff_axes(env, new_env)
        _, static = eqx.partition(new_env, eqx.is_array)

        def f_init(e, k):
            e = factory(e, k)
            arr, _ = eqx.partition(e, eqx.is_array)
            return arr

        batched_keys = jax.random.split(k2, math.prod(batches)).reshape(*batches)
        b_fn = functools.reduce(
            lambda f, _: eqx.filter_vmap(f, in_axes=(None, 0), out_axes=axes),
            batches,
            f_init,
        )
        arr = b_fn(env, batched_keys)
        return eqx.combine(arr, static)

    return vmap_env


def reset_validation[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    interfaces: dict[S_ID, GodInterface[ENV]],
    meta_config: MetaConfig,
    level: int,
    nodes: dict[str, Node],
    node_features: dict[str, tuple[int, ...]],
    is_init: bool,
) -> Callable[[ENV, PRNG], ENV]:

    track_influence_in = meta_config.validation.track_influence_in

    def create_env(env: ENV, prng: PRNG) -> ENV:
        k1, prng = jax.random.split(prng, 2)
        f1 = lambda e, k: create_inference_state(
            nodes, interfaces, level, node_features, track_influence_in, is_init, e, k
        )
        batch_size = [
            meta_config.validation.batch,
            meta_config.dataset.num_examples_in_minibatch,
        ]
        env = vmap_factory(f1, batch_size)(env, k1)

        k1, k2, prng = jax.random.split(prng, 3)
        env = create_learner_states(
            factory,
            meta_config.learner.model_learner.method,
            interfaces,
            (MODEL_LEARNER, level),
            track_influence_in,
            env,
            k2,
        )

        return env

    return create_env


def reset_params_hyperparams_optimizer[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    config: GodConfig,
    interfaces: dict[S_ID, GodInterface[ENV]],
    meta_config: MetaConfig,
    level: int,
    hyperparameters: dict[HP, HyperparameterConfig],
    nodes: dict[str, Node],
    transition_graph: dict[str, set[str]],
    readout_graph: dict[str, set[str]],
    node_features: dict[str, tuple[int, ...]],
    create_inference_param: bool,
) -> Callable[[ENV, PRNG], ENV]:

    learnables_per_level = [
        frozenset().union(*[v.target for v in mc.learner.optimizer.values()]) for mc in config.levels
    ]

    def create_env(env: ENV, prng: PRNG) -> ENV:
        if create_inference_param:
            k1, prng = jax.random.split(prng, 2)
            env = create_inference_parameters(
                nodes,
                transition_graph,
                readout_graph,
                interfaces,
                level,
                node_features,
                learnables_per_level[0],
                env,
                k1,
            )

        k2, prng = jax.random.split(prng, 2)
        env = create_hyperparameters(hyperparameters, interfaces, learnables_per_level, env, k2)

        for assignment_name, assignment in meta_config.learner.optimizer.items():
            opt_value = init_opt_state(assignment_name, assignment, interfaces, level, env, hyperparameters)
            env = interfaces[(assignment_name, level)].opt_state.put_tagged(
                env,
                Tagged(value=opt_value, meta=StateMeta(is_stateful=meta_config.nested.track_influence_in)),
            )

        k3, prng = jax.random.split(prng, 2)
        env = factory(env, k3)

        return env

    return create_env


def reset_nested_learner[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    interfaces: dict[S_ID, GodInterface[ENV]],
    meta_config: MetaConfig,
    level: int,
) -> Callable[[ENV, PRNG], ENV]:

    nest_interface = interfaces[(OPTIMIZER_LEARNER, level)]
    track_influence_in = meta_config.nested.track_influence_in

    def create_env(env: ENV, prng: PRNG) -> ENV:
        k1, k2, prng = jax.random.split(prng, 3)
        env = factory(env, k1)

        env = create_learner_states(
            factory,
            meta_config.learner.optimizer_learner.method,
            interfaces,
            (OPTIMIZER_LEARNER, level),
            track_influence_in,
            env,
            k2,
        )

        logs = Logs(
            gradient=jnp.zeros_like(nest_interface.param.get(env)) if meta_config.track_logs.gradient else None,
            hessian_contains_nans=jnp.array(False) if meta_config.track_logs.hessian_contains_nans else None,
            largest_eigenvalue=jnp.array(0.0) if meta_config.track_logs.largest_eigenvalue else None,
            influence_tensor_norm=jnp.array(0.0) if meta_config.track_logs.influence_tensor_norm else None,
            immediate_influence_tensor=jnp.zeros(
                (nest_interface.state.get(env).shape[0], nest_interface.param.get(env).shape[0])
            )
            if meta_config.track_logs.immediate_influence_tensor
            else None,
            largest_jac_eigenvalue=jnp.array(0.0) if meta_config.track_logs.largest_jac_eigenvalue else None,
            jacobian=jnp.zeros((nest_interface.state.get(env).shape[0], nest_interface.state.get(env).shape[0]))
            if meta_config.track_logs.jacobian
            else None,
        )
        env = nest_interface.logs.put(env, logs)

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
                    midpoint_buffers=pmap({}),
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
                    log=Tagged(value=Logs(), meta=StateMeta(is_stateful=frozenset())),
                    prngs=pmap({}),
                    ticks=pmap({}),
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
    interfaces: dict[S_ID, GodInterface[GodState]],
    prng: PRNG,
) -> GodState:
    k1, k2, prng = jax.random.split(prng, 3)
    env = create_empty_env(config, k1)
    creator = env_creator(
        config,
        shapes,
        interfaces,
        is_inits=[True] * len(config.levels),
    )
    return creator(env, k2)


def env_validation_resetters[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
) -> list[Callable[[ENV, PRNG], ENV]]:

    resetters: list[Callable[[ENV, PRNG], ENV]] = []
    is_inits = [False] * len(config.levels)

    for level, (shape, meta_config, is_init) in enumerate(
        zip(
            shapes,
            config.levels,
            is_inits,
        )
    ):
        node_features = get_output_shapes({}, config.readout_graph | config.transition_graph, config.nodes, shape)

        resetter = reset_validation(
            lambda e, k: e,
            interfaces,
            meta_config,
            level,
            config.nodes,
            node_features,
            is_init,
        )
        resetters.append(resetter)

    return resetters


def env_resetters[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    is_inits: list[bool],
) -> list[tuple[Callable[[ENV, PRNG], ENV], Callable[[ENV, PRNG], ENV]]]:
    create_inference_params = [True] + [False] * (len(config.levels) - 1)
    factory: Callable[[ENV, PRNG], ENV] = lambda e, k: e

    def fold(
        accum: Callable[[ENV, PRNG], ENV],
        shape: tuple[tuple[int, ...], tuple[int, ...]],
        level: int,
        meta_config: MetaConfig,
        create_inference_param: bool,
        is_init: bool,
    ) -> tuple[Callable[[ENV, PRNG], ENV], Callable[[ENV, PRNG], ENV]]:
        node_features = get_output_shapes({}, config.readout_graph | config.transition_graph, config.nodes, shape)

        _reset_validation = reset_validation(
            accum,
            interfaces,
            meta_config,
            level,
            config.nodes,
            node_features,
            is_init,
        )
        _reset_nested_learner = reset_nested_learner(
            _reset_validation,
            interfaces,
            meta_config,
            level,
        )

        inner = vmap_factory(_reset_nested_learner, [meta_config.nested.batch])

        full = reset_params_hyperparams_optimizer(
            inner,
            config,
            interfaces,
            meta_config,
            level,
            config.hyperparameters,
            config.nodes,
            config.transition_graph,
            config.readout_graph,
            node_features,
            create_inference_param,
        )
        return inner, full

    nested_resetters: list[tuple[Callable[[ENV, PRNG], ENV], Callable[[ENV, PRNG], ENV]]] = []
    accum: Callable[[ENV, PRNG], ENV] = factory
    for level, (shape, meta_config, create_inference_param, is_init) in enumerate(
        zip(
            shapes,
            config.levels,
            create_inference_params,
            is_inits,
        )
    ):
        inner, full = fold(accum, shape, level, meta_config, create_inference_param, is_init)
        nested_resetters.append((inner, full))
        accum = full

    val_resetters = env_validation_resetters(config, shapes, interfaces)
    batches = [mc.nested.batch for mc in reversed(config.levels)]
    batched_val_resetters = [vmap_factory(vr, batches) for vr in val_resetters]

    def compose(
        nested_inner: Callable[[ENV, PRNG], ENV],
        nested_full: Callable[[ENV, PRNG], ENV],
        batched_vrs: list[Callable[[ENV, PRNG], ENV]],
    ) -> tuple[Callable[[ENV, PRNG], ENV], Callable[[ENV, PRNG], ENV]]:
        def composed_inner(env: ENV, prng: PRNG) -> ENV:
            k1, prng = jax.random.split(prng, 2)
            env = nested_inner(env, k1)
            for bvr in batched_vrs:
                k, prng = jax.random.split(prng, 2)
                env = bvr(env, k)
            return env

        def composed_full(env: ENV, prng: PRNG) -> ENV:
            k1, prng = jax.random.split(prng, 2)
            env = nested_full(env, k1)
            for bvr in batched_vrs:
                k, prng = jax.random.split(prng, 2)
                env = bvr(env, k)
            return env

        return composed_inner, composed_full

    return [
        compose(inner, full, batched_val_resetters[: level + 1]) for level, (inner, full) in enumerate(nested_resetters)
    ]


def env_creator[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    is_inits: list[bool],
) -> Callable[[ENV, PRNG], ENV]:
    return env_resetters(config, shapes, interfaces, is_inits)[-1][1]


def make_tick_advancer[ENV](interface: GodInterface[ENV]):
    def advance(env: ENV) -> ENV:
        return interface.advance_tick(env)

    return advance


def make_reset_checker[ENV](
    interface: GodInterface[ENV],
    resetter: Callable[[ENV, PRNG], ENV],
    reset_t: int | None,
):
    def check_reset(env: ENV) -> ENV:
        if reset_t is None:
            return env
        tick = interface.tick.get(env)

        def do_reset(e):
            prng, e = interface.take_prng(e)
            return resetter(e, prng)

        def no_reset(e):
            return e

        return filter_cond(tick % reset_t == 0, do_reset, no_reset, env)

    return check_reset


def create_transition_fns[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    transitions: list[Callable[[ENV, tuple[jax.Array, jax.Array]], ENV]],
) -> list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, STAT]]]:

    def make_composed(check, advance, transition):
        def composed(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, STAT]:
            env = check(env)
            env = advance(env)
            return transition(env, data), {}

        return composed

    resetters = env_validation_resetters(config, shapes, interfaces)

    return [
        make_composed(
            make_reset_checker(interfaces[(MODEL_LEARNER, level)], resetter, meta_config.validation.reset_t),
            make_tick_advancer(interfaces[(MODEL_LEARNER, level)]),
            transition,
        )
        for level, (resetter, meta_config, transition) in enumerate(zip(resetters, config.levels, transitions))
    ]


def create_inference_axes[ENV](
    env: ENV,
    config: GodConfig,
    interfaces: dict[S_ID, GodInterface[ENV]],
    shape: tuple[tuple[int, ...], tuple[int, ...]],
    level: int,
) -> ENV:
    node_features = get_output_shapes({}, config.readout_graph | config.transition_graph, config.nodes, shape)
    track_influence_in = config.levels[level].validation.track_influence_in
    f = lambda e, k: create_inference_state(
        config.nodes, interfaces, level, node_features, track_influence_in, False, e, k
    )
    return diff_axes(env, f(env, jax.random.key(0)))
