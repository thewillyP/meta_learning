import jax
import jax.numpy as jnp
import copy
import equinox as eqx
import optax

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
    State,
    UOROState,
)
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
                n_in, _ = n_in_shape
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
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias):
                n_in, _ = n_in_shape
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
            case _:
                raise ValueError("Unsupported transition function")

    match config.readout_function:
        case FeedForwardConfig(ffw_layers):
            layers = []
            n_in, _ = n_in_shape
            for i, layer in ffw_layers.items():
                match layer:
                    case NNLayer(n, activation_fn, use_bias):
                        layers.append((n, use_bias, get_activation_fn(activation_fn)))
            prng1, prng = jax.random.split(prng, 2)
            readout_fn = CustomSequential(layers, n_in, prng1)
        case _:
            raise ValueError("Unsupported readout function")

    return Parameter(transition_parameter=transition_parameter, readout_fn=readout_fn)


def create_learning(
    learn_config: LearnConfig,
    env: GodState,
    parameter: Parameter,
    prng: PRNG,
) -> tuple[LearningState, LearningParameter]:
    state = LearningState()
    match learn_config.learner:
        case RTRLConfig() | RFLOConfig() | UOROConfig():
            flat_state = to_vector(env)
            flat_param = to_vector(parameter)
            prng1, prng = jax.random.split(prng, 2)
            influence_tensor = jax.random.normal(prng1, (flat_state.vector.size, flat_param.vector.size))
            state = copy.replace(state, influence_tensor=JACOBIAN(influence_tensor))
            match learn_config.learner:
                case UOROConfig():
                    prng1, prng2, prng3, prng = jax.random.split(prng, 4)

                    def edit_fn(leaf):
                        if isinstance(leaf, RNN):
                            w_rec = jax.random.normal(prng1, leaf.w_rec.shape)
                            b_rec = jax.random.normal(prng2, leaf.b_rec.shape) if leaf.b_rec is not None else None
                            return eqx.tree_at(lambda r: (r.w_rec, r.b_rec), leaf, (w_rec, b_rec))
                        else:
                            return jnp.zeros_like(leaf)

                    _parameter = jax.tree.map(edit_fn, parameter, is_leaf=lambda x: isinstance(x, RNN))

                    a = jax.random.normal(prng3, (flat_state.vector.size,))
                    b = to_vector(_parameter).vector
                    uoro = UOROState(A=a, B=b)
                    state = copy.replace(state, uoro=uoro)
        case BPTTConfig():
            ...
        case IdentityConfig():
            ...

    forward, backward = hyperparameter_reparametrization(learn_config.hyperparameter_parametrization)

    match learn_config.optimizer:
        case SGDConfig(learning_rate):
            _opt = optax.sgd(backward(learning_rate))
            get_optimizer = lambda pr: optax.sgd(forward(pr.learning_parameter.learning_rate))
        case SGDNormalizedConfig(learning_rate):
            _opt = optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.sgd(backward(learning_rate)),
            )
            get_optimizer = lambda pr: optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.sgd(forward(pr.learning_parameter.learning_rate)),
            )

        case SGDClipConfig(learning_rate, threshold, sharpness):

            def update_fn(updates, state, _):
                grads_flat, _ = jax.flatten_util.ravel_pytree(updates)
                grad_norm = jnp.linalg.norm(grads_flat)
                clipped_norm = grad_norm - jax.nn.softplus(sharpness * (grad_norm - threshold)) / sharpness
                scale = clipped_norm / (grad_norm + 1e-6)
                updates_scaled = jax.tree.map(lambda g: g * scale, updates)
                return updates_scaled, state

            _opt = optax.chain(
                optax.GradientTransformation(lambda _: (), update_fn), optax.sgd(backward(learning_rate))
            )
            get_optimizer = lambda pr: optax.chain(
                optax.GradientTransformation(lambda _: (), update_fn),
                optax.sgd(forward(pr.learning_parameter.learning_rate)),
            )

        case AdamConfig(learning_rate):
            _opt = optax.adam(backward(learning_rate))
            get_optimizer = lambda pr: optax.adam(forward(pr.learning_parameter.learning_rate))

    opt_state = _opt.init(eqx.filter(parameter, eqx.is_inexact_array))
    state = copy.replace(state, opt_state=opt_state, get_optimizer=get_optimizer)

    parameter = LearningParameter()
    match learn_config.optimizer:
        case SGDConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=jnp.array(learning_rate))
        case SGDNormalizedConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=jnp.array(learning_rate))
        case SGDClipConfig(learning_rate, threshold, sharpness):
            parameter = copy.replace(parameter, learning_rate=jnp.array(learning_rate))
        case AdamConfig(learning_rate):
            parameter = copy.replace(parameter, learning_rate=jnp.array(learning_rate))

    match learn_config.learner:
        case RFLOConfig(time_constant):
            parameter = copy.replace(parameter, rflo_timeconstant=time_constant)

    return state, parameter


def create_env(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> GodState:
    prng1, prng2, prng = jax.random.split(prng, 3)
    env = GodState(
        states={},
        parameters={},
        logs={},
        prng=prng1,
        start_epoch=0,
    )
    parameter = create_inference_parameter(config, n_in_shape, prng2)

    for _, learn_config in sorted(config.learners.items()):
        prng1, prng2, prng = jax.random.split(prng, 3)

        # Create inference state
        load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
        vl_prngs = jax.random.split(prng1, learn_config.num_examples_in_minibatch)
        transition_state_vl = load_state(config, n_in_shape, vl_prngs)

        # Add inference state to env
        env = copy.replace(env, states={**env.states, len(env.states): State(inference_state=transition_state_vl)})

        # Create learning state and new parameter
        prev_parameter = parameter
        learning_state_vl, _parameter = create_learning(learn_config, env, prev_parameter, prng2)
        parameter = Parameter(learning_parameter=_parameter)

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

        general = General(current_virtual_minibatch=0, logs=logs, special_logs=special_logs)

        # Add final state and prev parameter
        env = copy.replace(
            env,
            states={
                **env.states,
                len(env.states): State(inference_state=transition_state_vl, learning_state=learning_state_vl),
            },
            parameters={**env.parameters, len(env.parameters): prev_parameter},
            general={**env.logs, len(env.logs): general},
        )

    # Add final parameter
    env = copy.replace(env, parameters={**env.parameters, len(env.parameters): parameter})

    return env
