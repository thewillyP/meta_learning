import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from meta_learn_lib.config import *
from meta_learn_lib.env import *
from meta_learn_lib.interface import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.constants import *
from meta_learn_lib.util import Vector, hyperparameter_reparametrization, to_vector


def project_parameters[ENV](
    env: ENV,
    interfaces: dict[S_ID, GodInterface[ENV]],
) -> ENV:
    all_accs = [
        acc
        for iface in interfaces.values()
        for acc in interface_to_accessors(iface)
        if acc.category is not None and isinstance(acc.meta, ParamMeta) and acc.meta.learnable
    ]
    for acc in all_accs:
        val = acc.get(env)
        if val is not None:
            env = acc.put(env, jax.tree.map(lambda v: jnp.clip(v, acc.meta.min_value, acc.meta.max_value), val))
    return env


def get_parameters[ENV](
    assignment: OptimizerAssignment,
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    env: ENV,
) -> Vector[ENV]:
    mask = jax.tree.map(lambda _: False, env)

    for name in assignment.target:
        interface = interfaces[(name, level)]
        for acc in interface_to_accessors(interface):
            if acc.category == "param" and acc.get(env) is not None:
                mask = acc.put(mask, True)

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
            lr = interface.learning_rate.get(env)
            wd = interface.weight_decay.get(env)
            m = interface.momentum.get(env)
            return optax.chain(
                optax.add_decayed_weights(wd_forward(wd)),
                optax.sgd(lr_forward(lr), momentum=m_forward(m)),
            )
        case SGDNormalizedConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.learning_rate.get(env)
            wd = interface.weight_decay.get(env)
            m = interface.momentum.get(env)
            return optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.add_decayed_weights(wd_forward(wd)),
                optax.sgd(lr_forward(lr), momentum=m_forward(m)),
            )
        case AdamConfig(learning_rate, weight_decay, momentum, eps, eps_root):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.learning_rate.get(env)
            wd = interface.weight_decay.get(env)
            m = interface.momentum.get(env)
            return optax.chain(
                optax.adamw(
                    lr_forward(lr),
                    weight_decay=wd_forward(wd),
                    b1=m_forward(m),
                    eps=eps,
                    eps_root=eps_root,
                ),
            )
        case ExponentiatedGradientConfig(learning_rate, weight_decay, momentum, use_adam):
            lr_forward, _ = hyperparameter_reparametrization(hps[learning_rate].hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(hps[weight_decay].hyperparameter_parametrization)
            m_forward, _ = hyperparameter_reparametrization(hps[momentum].hyperparameter_parametrization)
            lr = interface.learning_rate.get(env)
            wd = interface.weight_decay.get(env)
            m = interface.momentum.get(env)
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
                jnp.atleast_1d(interface.learning_rate.get(env)),
                jnp.atleast_1d(interface.weight_decay.get(env)),
                jnp.atleast_1d(interface.momentum.get(env)),
            ]
            K = min(v.shape[0] for v in hp_values)
            batched_env = interface.learning_rate.put(env, hp_values[0][:K])
            batched_env = interface.weight_decay.put(batched_env, hp_values[1][:K])
            batched_env = interface.momentum.put(batched_env, hp_values[2][:K])

            env_axes = jax.tree.map(lambda _: None, batched_env)
            env_axes = interface.learning_rate.put(env_axes, 0)
            env_axes = interface.weight_decay.put(env_axes, 0)
            env_axes = interface.momentum.put(env_axes, 0)

    return K, batched_env, env_axes


def get_opt_state[ENV](
    assignments: dict[str, OptimizerAssignment],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    env: ENV,
    hps: dict[HP, HyperparameterConfig],
) -> ENV:

    for assignment_name, assignment in assignments.items():
        interface = interfaces[(assignment_name, level)]

        param_vec: Vector = get_parameters(assignment, interfaces, level, env)
        flat_params = param_vec.vector

        K, batched_env, env_axes = get_batched_env_and_axes(assignment, env, interface)

        chunk_size = flat_params.shape[0] // K
        param_chunks = flat_params[: K * chunk_size].reshape(K, chunk_size)

        def init_one(env_slice: ENV, param_chunk: jax.Array) -> optax.OptState:
            return get_opt(assignment, env_slice, interface, hps).init(param_chunk)

        batched_opt_state = jax.vmap(init_one, in_axes=(env_axes, 0))(batched_env, param_chunks)

        env = interface.opt_state.put(env, batched_opt_state)

    return env


def get_opt_step[ENV](
    assignments: dict[str, OptimizerAssignment],
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
    env: ENV,
    gr_env: ENV,
    hps: dict[HP, HyperparameterConfig],
) -> ENV:
    for assignment_name, assignment in assignments.items():
        interface = interfaces[(assignment_name, level)]

        param_vec: Vector = get_parameters(assignment, interfaces, level, env)
        gr_vec: Vector = get_parameters(assignment, interfaces, level, gr_env)
        flat_params = param_vec.vector
        flat_gr = gr_vec.vector

        K, batched_env, env_axes = get_batched_env_and_axes(assignment, env, interface)

        chunk_size = flat_params.shape[0] // K
        param_chunks = flat_params[: K * chunk_size].reshape(K, chunk_size)
        gr_chunks = flat_gr[: K * chunk_size].reshape(K, chunk_size)
        batched_opt_state = interface.opt_state.get(env)

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
        env = interface.opt_state.put(env, new_opt_state)
        env = put_parameters(env, param_vec.to_param(new_params))

    return project_parameters(env, interfaces)
