import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from meta_learn_lib.config import *
from meta_learn_lib.env import *
from meta_learn_lib.interface import *
from meta_learn_lib.util import Vector, hyperparameter_reparametrization, to_vector


def project_parameters[ENV](env: ENV) -> ENV:
    is_leaf = lambda x: x is None or isinstance(x, Parameter)

    def clamp(x):
        if isinstance(x, Parameter) and x.is_learnable:
            return jax.tree.map(lambda v: jnp.clip(v, x.min_value, x.max_value), x)
        return x

    return jax.tree.map(clamp, env, is_leaf=is_leaf)


def get_parameters[ENV](
    assignment: OptimizerAssignment,
    meta_interfaces: dict[str, GodInterface[ENV]],
    env: ENV,
) -> Vector[ENV]:
    mask = jax.tree.map(lambda _: False, env)

    for name in assignment.target:
        interface = meta_interfaces[name]
        for get_fn, put_fn in [
            (interface.get_mlp_param, interface.put_mlp_param),
            (interface.get_vanilla_rnn_param, interface.put_vanilla_rnn_param),
            (interface.get_gru_param, interface.put_gru_param),
            (interface.get_lstm_param, interface.put_lstm_param),
            (interface.get_learning_rate, interface.put_learning_rate),
            (interface.get_weight_decay, interface.put_weight_decay),
            (interface.get_momentum, interface.put_momentum),
            (interface.get_time_constant, interface.put_time_constant),
            (interface.get_kl_regularizer_beta, interface.put_kl_regularizer_beta),
        ]:
            if get_fn(env) is not None:
                mask = put_fn(mask, True)

    params, _ = eqx.partition(env, mask, is_leaf=lambda x: x is None)
    return to_vector(params)


def put_parameters[ENV](env: ENV, new_env: ENV) -> ENV:
    return eqx.combine(new_env, env)


def scale_by_sign_of_param() -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        return None

    def update_fn(grad, state, params):
        return grad * jnp.sign(params), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def add_scalar_wd(wd_value) -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        return None

    def update_fn(grad, state, params):
        return grad + wd_value, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_opt[ENV](
    assignment: OptimizerAssignment,
    env: ENV,
    interface: GodInterface[ENV],
    hps: dict[HP, HyperparameterConfig],
) -> optax.GradientTransformation:
    match assignment.optimizer:
        case SGDConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.get_learning_rate(env).value
            wd = interface.get_weight_decay(env).value
            m = interface.get_momentum(env).value
            return optax.chain(
                optax.add_decayed_weights(wd_forward(wd)),
                optax.sgd(lr_forward(lr), momentum=m_forward(m)),
            )
        case SGDNormalizedConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.get_learning_rate(env).value
            wd = interface.get_weight_decay(env).value
            m = interface.get_momentum(env).value
            return optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.add_decayed_weights(wd_forward(wd)),
                optax.sgd(lr_forward(lr), momentum=m_forward(m)),
            )
        case AdamConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.get_learning_rate(env).value
            wd = interface.get_weight_decay(env).value
            m = interface.get_momentum(env).value
            return optax.chain(
                optax.adamw(
                    lr_forward(lr),
                    weight_decay=wd_forward(wd),
                    b1=m_forward(m),
                ),
            )
        case ExponentiatedGradientConfig(learning_rate, weight_decay, momentum, use_adam):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.get_learning_rate(env).value
            wd = interface.get_weight_decay(env).value
            m = interface.get_momentum(env).value
            return optax.chain(
                scale_by_sign_of_param(),
                add_scalar_wd(wd_forward(wd)),
                optax.scale_by_adam(b1=m_forward(m)) if use_adam else optax.identity(),
                optax.scale(-lr_forward(lr)),
            )


def get_batched_env_and_axes[ENV](
    assignment: OptimizerAssignment,
    env: ENV,
    interface: GodInterface[ENV],
) -> tuple[int, ENV, ENV]:
    match assignment.optimizer:
        case SGDConfig() | SGDNormalizedConfig() | AdamConfig() | ExponentiatedGradientConfig():
            hp_values = [
                jnp.atleast_1d(interface.get_learning_rate(env).value),
                jnp.atleast_1d(interface.get_weight_decay(env).value),
                jnp.atleast_1d(interface.get_momentum(env).value),
            ]
            K = min(v.shape[0] for v in hp_values)
            batched_env = interface.put_learning_rate(env, interface.get_learning_rate(env).set(value=hp_values[0][:K]))
            batched_env = interface.put_weight_decay(
                batched_env, interface.get_weight_decay(env).set(value=hp_values[1][:K])
            )
            batched_env = interface.put_momentum(batched_env, interface.get_momentum(env).set(value=hp_values[2][:K]))

            env_axes = jax.tree.map(lambda _: None, batched_env)
            env_axes = interface.put_learning_rate(env_axes, 0)
            env_axes = interface.put_weight_decay(env_axes, 0)
            env_axes = interface.put_momentum(env_axes, 0)

    return K, batched_env, env_axes


def get_opt_state[ENV](
    assignments: dict[str, OptimizerAssignment],
    meta_interfaces: dict[str, GodInterface[ENV]],
    env: ENV,
    hps: dict[HP, HyperparameterConfig],
    track_influence_in: frozenset[int],
) -> ENV:

    for assignment_name, assignment in assignments.items():
        interface = meta_interfaces[assignment_name]

        param_vec: Vector = get_parameters(assignment, meta_interfaces, env)
        flat_params = param_vec.vector  # (N,)

        K, batched_env, env_axes = get_batched_env_and_axes(assignment, env, interface)

        chunk_size = flat_params.shape[0] // K
        param_chunks = flat_params[: K * chunk_size].reshape(K, chunk_size)

        def init_one(env_slice: ENV, param_chunk: jax.Array) -> optax.OptState:
            return get_opt(assignment, env_slice, interface, hps).init(param_chunk)

        batched_opt_state = jax.vmap(init_one, in_axes=(env_axes, 0))(batched_env, param_chunks)

        env = interface.put_opt_state(env, State(value=batched_opt_state, is_stateful=track_influence_in))

    return env


def get_opt_step[ENV](
    assignments: dict[str, OptimizerAssignment],
    meta_interfaces: dict[str, GodInterface[ENV]],
    env: ENV,
    gr_env: ENV,
    hps: dict[HP, HyperparameterConfig],
) -> ENV:
    for assignment_name, assignment in assignments.items():
        interface = meta_interfaces[assignment_name]

        param_vec: Vector = get_parameters(assignment, meta_interfaces, env)
        gr_vec: Vector = get_parameters(assignment, meta_interfaces, gr_env)
        flat_params = param_vec.vector  # (N,)
        flat_gr = gr_vec.vector  # (N,)

        K, batched_env, env_axes = get_batched_env_and_axes(assignment, env, interface)

        chunk_size = flat_params.shape[0] // K
        param_chunks = flat_params[: K * chunk_size].reshape(K, chunk_size)
        gr_chunks = flat_gr[: K * chunk_size].reshape(K, chunk_size)
        batched_opt_state = interface.get_opt_state(env).value

        def update_one(
            env_slice: ENV, gr_chunk: jax.Array, param_chunk: jax.Array, opt_state
        ) -> tuple[jax.Array, optax.OptState]:
            updates, new_opt_state = get_opt(assignment, env_slice, interface, hps).update(
                gr_chunk, opt_state, param_chunk
            )
            match assignment.optimizer:
                case ExponentiatedGradientConfig():
                    new_param = param_chunk * jnp.exp(updates)
                case _:
                    new_param = optax.apply_updates(param_chunk, updates)
            return new_param, new_opt_state

        new_param_chunks, new_opt_state = jax.vmap(update_one, in_axes=(env_axes, 0, 0, 0))(
            batched_env, gr_chunks, param_chunks, batched_opt_state
        )

        new_params = new_param_chunks.reshape(-1)
        env = interface.put_opt_state(env, interface.get_opt_state(env).set(value=new_opt_state))
        env = put_parameters(env, param_vec.to_param(new_params))

    return project_parameters(env)
