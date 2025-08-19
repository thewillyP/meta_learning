import jax
import jax.numpy as jnp
import copy
import equinox as eqx

from lib.config import *
from lib.env import (
    RNN,
    CustomSequential,
    General,
    GodState,
    InferenceParameter,
    InferenceState,
    LearningParameter,
    LearningState,
    Logs,
    Parameter,
    RNNState,
    SpecialLogs,
    UOROState,
)
from lib.interface import LearnInterface
from lib.lib_types import *
from lib.util import (
    get_activation_fn,
    hyperparameter_reparametrization,
    to_vector,
)


def create_state(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> dict[int, InferenceState]:
    transition_state: dict[int, InferenceState] = {}
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias):
                n_in, *_ = n_in_shape
                prng1, prng = jax.random.split(prng, 2)
                activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
                transition_state[i] = InferenceState(
                    rnn=RNNState(
                        activation=activation,
                        n_h=n_h,
                        n_in=n_in,
                        activation_fn=activation_fn,
                    )
                )
            case _:
                raise ValueError("Unsupported transition function")
    return transition_state


def create_inference_parameter(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> Parameter:
    transition_parameter: dict[int, InferenceParameter] = {}
    n_in_size = 0
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias):
                n_in, *_ = n_in_shape
                prgn1, prng2, prng = jax.random.split(prng, 3)
                W_in = jax.random.normal(prgn1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(prng2, (n_h, n_h)))[0]
                w_rec = jnp.hstack([W_rec, W_in])
                b_rec: jax.Array | None = jnp.zeros((n_h, 1)) if use_bias else None
                transition_parameter[i] = InferenceParameter(
                    rnn=RNN(
                        w_rec=w_rec,
                        b_rec=b_rec,
                    )
                )
                n_in_size += n_h
            case _:
                raise ValueError("Unsupported transition function")

    match config.readout_function:
        case FeedForwardConfig(ffw_layers):
            layers = []
            data_in, *_ = n_in_shape
            n_in_size += data_in
            for i, layer in ffw_layers.items():
                match layer:  # purely for pattern matching, no other case should actually exist
                    case NNLayer(n, activation_fn, use_bias):
                        layers.append((n, use_bias, get_activation_fn(activation_fn)))
            prng1, prng = jax.random.split(prng, 2)
            readout_fn = CustomSequential(layers, n_in_size, prng1)
        case _:
            raise ValueError("Unsupported readout function")

    return Parameter(transition_parameter=transition_parameter, readout_fn=readout_fn)


def create_learning_parameter(
    learn_config: LearnConfig,
) -> LearningParameter:
    _, backward = hyperparameter_reparametrization(learn_config.hyperparameter_parametrization)

    parameter = LearningParameter()
    match learn_config.optimizer:
        case SGDConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=backward(jnp.array([learning_rate])))
        case SGDNormalizedConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=backward(jnp.array([learning_rate])))
        case SGDClipConfig(learning_rate, threshold, sharpness):
            parameter = copy.replace(parameter, learning_rate=backward(jnp.array([learning_rate])))
        case AdamConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=backward(jnp.array([learning_rate])))

    match learn_config.learner:
        case RFLOConfig(time_constant):
            parameter = copy.replace(parameter, rflo_timeconstant=time_constant)

    return parameter


def create_learning_state(
    learn_config: LearnConfig,
    env: GodState,
    parameter: Parameter,
    learn_interface: LearnInterface[GodState],
    prng: PRNG,
) -> LearningState:
    state = LearningState()
    flat_state = learn_interface.get_state(env)
    flat_param = to_vector(parameter)
    match learn_config.learner:
        case RTRLConfig() | RFLOConfig():
            prng1, prng = jax.random.split(prng, 2)
            influence_tensor = jax.random.normal(prng1, (flat_state.size, flat_param.vector.size))
            state = copy.replace(state, influence_tensor=JACOBIAN(influence_tensor))
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
            b = to_vector(_parameter).vector
            uoro = UOROState(A=a, B=b)
            state = copy.replace(state, uoro=uoro)
        case BPTTConfig():
            ...
        case IdentityConfig():
            ...

    _opt = learn_interface.get_optimizer(env)
    opt_state = _opt.init(flat_param.vector)
    state = copy.replace(state, opt_state=opt_state)

    return state


def create_env(
    config: GodConfig, n_in_shape: tuple[int, ...], learn_interfaces: dict[int, LearnInterface[GodState]], prng: PRNG
) -> GodState:
    prng1, prng2, prng = jax.random.split(prng, 3)
    env = GodState(
        learning_states={},
        inference_states={},
        parameters={},
        general={},
        prng={
            i: jax.random.split(key, v.num_examples_in_minibatch)
            for i, (v, key) in enumerate(zip(config.data.values(), jax.random.split(prng1, len(config.data))))
        },
        start_epoch=0,
    )
    general: dict[int, General] = {}

    # Create inference states
    for _, data_config in sorted(config.data.items()):
        prng1, prng = jax.random.split(prng, 2)
        load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
        vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
        transition_state_vl: dict[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
        env = copy.replace(
            env, inference_states={**env.inference_states, len(env.inference_states): transition_state_vl}
        )

    parameter = create_inference_parameter(config, n_in_shape, prng2)
    env = copy.replace(env, parameters={**env.parameters, len(env.parameters): parameter})

    for i, (_, learn_config) in enumerate(sorted(config.learners.items())):
        prng1, prng = jax.random.split(prng, 2)

        # Create learning state and new parameter
        prev_parameter = parameter
        learning_parameter = create_learning_parameter(learn_config)
        parameter = Parameter(learning_parameter=learning_parameter)
        env = copy.replace(env, parameters={**env.parameters, len(env.parameters): parameter})
        learning_state_vl = create_learning_state(learn_config, env, prev_parameter, learn_interfaces[i], prng1)

        if learn_config.track_logs:
            logs = Logs(gradient=jnp.zeros_like(to_vector(prev_parameter).vector))
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
                largest_jac_eigenvalue=0.0,
                jacobian=jnp.zeros((to_vector(env).vector.size, to_vector(prev_parameter).vector.size)),
            )
        else:
            special_logs = None

        general[len(env.general)] = General(
            current_virtual_minibatch=0,
            logs=logs,
            special_logs=special_logs,
        )

        # Add final state
        env = copy.replace(
            env,
            learning_states={
                **env.learning_states,
                len(env.learning_states): learning_state_vl,
            },
        )

    return env
