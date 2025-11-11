import jax
import jax.numpy as jnp
import equinox as eqx
from itertools import islice
from pyrsistent import pmap
from pyrsistent.typing import PMap

from meta_learn_lib.config import *
from meta_learn_lib.create_interface import filter_hyperparam
from meta_learn_lib.env import (
    RNN,
    CustomSequential,
    General,
    GodState,
    Hyperparameter,
    InferenceParameter,
    InferenceState,
    LSTMState,
    LearningParameter,
    LearningState,
    Logs,
    Parameter,
    RNNState,
    SpecialLogs,
    TransitionParameter,
    UOROState,
)
from meta_learn_lib.interface import LearnInterface
from meta_learn_lib.lib_types import *
from meta_learn_lib.util import (
    get_activation_fn,
    hyperparameter_reparametrization,
)


def create_state(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> PMap[int, InferenceState]:
    transition_state: PMap[int, InferenceState] = pmap({})
    n_in, *_ = n_in_shape
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
                prng1, prng = jax.random.split(prng, 2)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n_h,)))
                transition_state = transition_state.set(
                    i,
                    InferenceState(
                        rnn=RNNState(
                            activation=activation,
                            n_h=n_h,
                            n_in=n_in,
                            activation_fn=activation_fn,
                        ),
                        lstm=None,
                    ),
                )
                n_in = n_h  # Update n_in for the next layer
            case GRULayer(n_h, use_bias, use_in_readout, use_random_init):
                prng1, prng = jax.random.split(prng, 2)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n_h,)))

                transition_state = transition_state.set(
                    i,
                    InferenceState(
                        rnn=RNNState(
                            activation=activation,
                            n_h=n_h,
                            n_in=n_in,
                            activation_fn="",
                        ),
                        lstm=None,
                    ),
                )
                n_in = n_h
            case LSTMLayer(n_h, use_bias, use_in_readout, use_random_init):
                prng1, prng2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
                    cell = ACTIVATION(jax.random.normal(prng2, (n_h,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n_h,)))
                    cell = ACTIVATION(jnp.zeros((n_h,)))

                transition_state = transition_state.set(
                    i,
                    InferenceState(
                        lstm=LSTMState(h=activation, c=cell, n_h=n_h, n_in=n_in),
                        rnn=None,
                    ),
                )
                n_in = n_h
            case IdentityLayer():
                transition_state = transition_state.set(
                    i,
                    InferenceState(
                        rnn=None,
                        lstm=None,
                    ),
                )
            case _:
                raise ValueError("Unsupported transition function")
    return transition_state


def create_inference_parameter(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> Parameter:
    transition_parameter: PMap[int, InferenceParameter] = pmap({})
    n_in_size = 0
    n_in, *_ = n_in_shape
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
                prgn1, prng2, prng = jax.random.split(prng, 3)
                W_in = jax.random.normal(prgn1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(prng2, (n_h, n_h)))[0]
                w_rec = jnp.hstack([W_rec, W_in])
                b_rec: jax.Array | None = jnp.zeros((n_h,)) if use_bias else None
                match layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = eqx.nn.LayerNorm(n_h, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
                    case None:
                        layer = None

                transition_parameter = transition_parameter.set(
                    i,
                    InferenceParameter(
                        rnn=RNN(
                            w_rec=w_rec,
                            b_rec=b_rec,
                            layer_norm=layer,
                        ),
                        gru=None,
                        lstm=None,
                    ),
                )
                if use_in_readout:
                    n_in_size += n_h
                n_in = n_h  # Update n_in for the next layer
            case GRULayer(n_h, use_bias, use_in_readout, use_random_init):
                prng1, prng = jax.random.split(prng, 2)
                gru = eqx.nn.GRUCell(n_in, n_h, use_bias=use_bias, key=prng1)
                transition_parameter = transition_parameter.set(i, InferenceParameter(gru=gru, rnn=None, lstm=None))
                if use_in_readout:
                    n_in_size += n_h
                n_in = n_h
            case LSTMLayer(n_h, use_bias, use_in_readout, use_random_init):
                prng1, prng = jax.random.split(prng, 2)
                lstm = eqx.nn.LSTMCell(n_in, n_h, use_bias=use_bias, key=prng1)
                transition_parameter = transition_parameter.set(i, InferenceParameter(lstm=lstm, rnn=None, gru=None))
                if use_in_readout:
                    n_in_size += n_h
                n_in = n_h
            case IdentityLayer():
                pass
            case _:
                raise ValueError("Unsupported transition function")

    match config.readout_function:
        case FeedForwardConfig(ffw_layers):
            layers = []
            data_in, *_ = n_in_shape
            n_in_size += data_in
            for i, layer in ffw_layers.items():
                match layer:  # purely for pattern matching, no other case should actually exist
                    case NNLayer(n, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
                        match layer_norm:
                            case LayerNorm(epsilon, use_weight, use_bias):
                                layer = eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
                            case None:
                                layer = None

                        layers.append((n, use_bias, get_activation_fn(activation_fn), layer))
            prng1, prng = jax.random.split(prng, 2)
            readout_fn = CustomSequential(layers, n_in_size, prng1)
        case _:
            raise ValueError("Unsupported readout function")

    return Parameter(
        transition_parameter=TransitionParameter(param=transition_parameter),
        readout_fn=readout_fn,
        learning_parameter=None,
    )


def create_optimizer_parameter(optimizer: Optimizer) -> LearningParameter:
    parameter = LearningParameter(
        learning_rate=None, weight_decay=None, rflo_timeconstant=None, multiple_parameters=None
    )
    match optimizer:
        case SGDConfig() | SGDNormalizedConfig() | AdamConfig() | ExponentiatedGradientConfig() as opt:
            _, lr_backward = hyperparameter_reparametrization(opt.learning_rate.hyperparameter_parametrization)
            _, wd_backward = hyperparameter_reparametrization(opt.weight_decay.hyperparameter_parametrization)
            parameter = parameter.set(
                learning_rate=Hyperparameter(
                    value=lr_backward(jnp.array([opt.learning_rate.value])),
                    learnable=opt.learning_rate.learnable,
                ),
                weight_decay=Hyperparameter(
                    value=wd_backward(jnp.array([opt.weight_decay.value])),
                    learnable=opt.weight_decay.learnable,
                ),
            )
        case RecurrenceConfig(recurrence_optimizer, out_optimizer):
            recurrence_param = create_optimizer_parameter(recurrence_optimizer)
            out_param = create_optimizer_parameter(out_optimizer)
            parameter = parameter.set(multiple_parameters=(recurrence_param, out_param))

    return parameter


def create_learning_parameter(learn_config: LearnConfig) -> LearningParameter:
    parameter = create_optimizer_parameter(learn_config.optimizer)

    match learn_config.learner:
        case RFLOConfig(time_constant):
            parameter = parameter.set(rflo_timeconstant=time_constant)

    return parameter


def create_learning_state(
    learn_config: LearnConfig,
    env: GodState,
    parameter: Parameter,
    learn_interface: LearnInterface[GodState],
    prng: PRNG,
) -> LearningState:
    state = LearningState(
        influence_tensor=None, influence_tensor_squared=None, uoro=None, opt_state=None, rflo_t=None, rtrl_t=None
    )
    flat_state = learn_interface.get_state(env)
    flat_param = filter_hyperparam(parameter)
    match learn_config.learner:
        case RTRLConfig() | RFLOConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig():
            # prng1, prng = jax.random.split(prng, 2)
            influence_tensor = jnp.zeros((flat_state.size, flat_param.size))
            influence_tensor_squared = jnp.zeros((flat_state.size, flat_param.size))
            state = state.set(
                influence_tensor=JACOBIAN(influence_tensor), influence_tensor_squared=JACOBIAN(influence_tensor_squared)
            )
        case UOROConfig():
            prng1, prng2, prng = jax.random.split(prng, 3)

            # Step 1: Get the tree structure and leaves
            leaves = jax.tree_util.tree_leaves(parameter, is_leaf=lambda x: isinstance(x, CustomSequential))
            treedef = jax.tree_util.tree_structure(parameter, is_leaf=lambda x: isinstance(x, CustomSequential))

            # Step 2: Split keys for each leaf
            keys = jax.random.split(prng1, len(leaves))
            keys_tree = jax.tree_util.tree_unflatten(treedef, keys)

            # Step 3: Function to edit each leaf
            def edit_fn(leaf, key):
                if isinstance(leaf, CustomSequential):
                    return jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, leaf)
                else:
                    return jax.random.normal(key, leaf.shape) if eqx.is_array(leaf) else leaf

            _parameter = jax.tree.map(edit_fn, parameter, keys_tree, is_leaf=lambda x: isinstance(x, CustomSequential))

            a = jax.random.normal(prng2, (flat_state.size,))
            b = filter_hyperparam(_parameter)
            uoro = UOROState(A=a, B=b)
            state = state.set(uoro=uoro)
        case BPTTConfig():
            ...
        case IdentityConfig():
            ...

    match learn_config.learner:
        case RFLOConfig():
            state = state.set(rflo_t=jax.lax.stop_gradient(jnp.array(1)))
        case RTRLConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig():
            state = state.set(rtrl_t=jax.lax.stop_gradient(jnp.array(1)))

    _opt = learn_interface.get_optimizer(env)
    if _opt is not None:
        opt_state = _opt.init(flat_param)
        state = state.set(opt_state=opt_state)

    return state


def create_env(
    config: GodConfig,
    n_in_shape: tuple[int, ...],
    learn_interfaces: dict[int, LearnInterface[GodState]],
    validation_learn_interfaces: dict[int, LearnInterface[GodState]],
    prng: PRNG,
) -> GodState:
    prng1, prng2, prng3, prng = jax.random.split(prng, 4)
    env = GodState(
        learning_states=pmap({}),
        inference_states=pmap({}),
        validation_learning_states=pmap({}),
        parameters=pmap({}),
        general=pmap({}),
        prng=pmap(
            {
                i: batched(jax.random.split(key, v.num_examples_in_minibatch))
                for i, (v, key) in enumerate(zip(config.data.values(), jax.random.split(prng1, len(config.data))))
            }
        ),
        prng_learning=pmap({i: key for i, key in enumerate(jax.random.split(prng2, len(config.learners)))}),
        start_epoch=jnp.array(0),
    )
    general: PMap[int, General] = pmap({})

    # Create inference states
    load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
    for _, data_config in sorted(config.data.items()):
        prng1, prng = jax.random.split(prng, 2)
        vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
        transition_state_vl: PMap[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
        env: GodState = env.transform(["inference_states", len(env.inference_states)], lambda _: transition_state_vl)

    parameter = create_inference_parameter(config, n_in_shape, prng3)
    env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
    for i, (_, learn_config) in enumerate(sorted(config.learners.items())):
        prng1, prng = jax.random.split(prng, 2)

        # Create learning state and new parameter
        prev_parameter = parameter
        learning_parameter = create_learning_parameter(learn_config)
        parameter = Parameter(transition_parameter=None, readout_fn=None, learning_parameter=learning_parameter)
        env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
        learning_state_vl = create_learning_state(learn_config, env, prev_parameter, learn_interfaces[i], prng1)

        if learn_config.track_logs:
            logs = Logs(
                gradient=jnp.zeros_like(filter_hyperparam(prev_parameter)),
                hessian_contains_nans=jnp.array(False),
                immediate_influence_contains_nans=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
        else:
            logs = None

        if learn_config.track_special_logs:
            special_logs = SpecialLogs(
                influence_tensor=(
                    jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
                ),
                immediate_influence_tensor=(
                    jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
                ),
                largest_jac_eigenvalue=jnp.array(0.0),
                jacobian=jnp.zeros((filter_hyperparam(env).size, filter_hyperparam(prev_parameter).size)),
            )
        else:
            special_logs = None

        general = general.set(
            len(general),
            General(
                current_virtual_minibatch=jnp.array(0),
                logs=logs,
                special_logs=special_logs,
            ),
        )

        env = env.transform(["learning_states", len(env.learning_states)], lambda _: learning_state_vl)

    env = env.set(general=general)

    # create learning states for validation
    validation_learning_states: PMap[int, LearningState] = pmap({})
    for i, _ in enumerate(islice(config.learners.items(), 1, None), 1):
        prng1, prng = jax.random.split(prng, 2)
        learning_state_vl = create_learning_state(
            config.learners[min(config.learners.keys())],
            env,
            learn_interfaces[i].get_state_pytree(env),
            validation_learn_interfaces[i],
            prng1,
        )
        validation_learning_states = validation_learning_states.set(i, learning_state_vl)
    env = env.set(validation_learning_states=validation_learning_states)

    return env


def reinitialize_env(
    env: GodState,
    config: GodConfig,
    n_in_shape: tuple[int, ...],
    prng: PRNG,
) -> GodState:
    # Create inference states
    inference_states: PMap[int, PMap[int, InferenceState]] = pmap({})
    load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
    for i, (_, data_config) in enumerate(sorted(config.data.items())):
        prng1, prng = jax.random.split(prng, 2)
        vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
        transition_state_vl: PMap[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
        inference_states = inference_states.set(i, transition_state_vl)
    env = env.set(inference_states=inference_states)

    return env
