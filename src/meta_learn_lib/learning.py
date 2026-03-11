import copy
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.create_env import env_resetters, env_validation_resetters, make_reset_checker, make_tick_advancer
from meta_learn_lib.interface import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.optimizer import get_opt_step
from meta_learn_lib.util import filter_cond, jacobian_matrix_product, softclip


def process_gradient(grad: GRADIENT, grad_config: GradientConfig) -> GRADIENT:
    g = grad * grad_config.scale
    match grad_config.add_clip:
        case HardClip(threshold):
            g = jnp.clip(g, -threshold, threshold)
        case SoftClip(threshold, sharpness):
            g = softclip(g, a=None, b=threshold, sharpness=sharpness)
        case None:
            pass
    return GRADIENT(g)


def rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    _config: RTRLConfig | RTRLFiniteHvpConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    match _config:
        case RTRLConfig():
            config = _config
        case RTRLFiniteHvpConfig():
            config = _config.rtrl_config

    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            t = learn_interface.get_tick(env)
            mu = config.damping
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def state_fn(state: jax.Array) -> tuple[jax.Array, None]:
                _env = learn_interface.put_state(env, state)
                _env, _ = transition(_env, tr_data)
                state = learn_interface.get_state(_env)
                return state, None

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, tr_data)
                state = learn_interface.get_state(_env)
                _arr, _ = eqx.partition(_env, eqx.is_array)
                return state, (_arr, stat)

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn, has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn, has_aux=True)(p)
            env = eqx.combine(arr, static)

            hmp: JACOBIAN
            match _config:
                case RTRLConfig():
                    _primals, hmp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor_s.value)
                case RTRLFiniteHvpConfig(epsilon, ___):

                    def finite_hvp(v: jax.Array) -> jax.Array:
                        return (state_fn(s + epsilon * v) - state_fn(s - epsilon * v)) / (2 * epsilon)

                    hmp = eqx.filter_vmap(finite_hvp, in_axes=1, out_axes=1)(influence_tensor_s.value)

            influence_tensor: JACOBIAN
            influence_tensor = filter_cond(
                t >= config.start_at_step,
                lambda _: hmp + dhdp - mu * influence_tensor_s.value,
                lambda _: influence_tensor_s.value,
                None,
            )
            env, credit_gr, readout_stat = readout_gr(env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            env = learn_interface.put_forward_mode_jacobian(env, influence_tensor_s.set(value=influence_tensor))

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

    return gradient_fn


def uoro[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: UOROConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        std = config.std
        state_shape = learn_interface.get_state(env_init).shape
        match config.distribution:
            case "uniform":
                distribution = lambda key: jax.random.uniform(key, state_shape, minval=-std, maxval=std)
            case "normal":
                distribution = lambda key: jax.random.normal(key, state_shape) * std

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            key, env = learn_interface.take_prng(env)
            random_vector = distribution(key)
            mu = config.damping
            uoro_state = learn_interface.get_uoro_state(env)
            A_s = uoro_state.A
            B_s = uoro_state.B
            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)

            def state_fn(state: jax.Array) -> jax.Array:
                _env = learn_interface.put_state(env, state)
                _env, _ = transition(_env, tr_data)
                state = learn_interface.get_state(_env)
                return state

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, tr_data)
                state = learn_interface.get_state(_env)
                _arr, _ = eqx.partition(_env, eqx.is_array)
                return state, (_arr, stat)

            immediateJacobian__A_projection = eqx.filter_jvp(state_fn, (s,), (A_s.value,))[1] - mu * A_s.value
            _, vjp_func, (arr, trans_stat) = eqx.filter_vjp(param_fn, p, has_aux=True)
            (immediateInfluence__random_projection,) = vjp_func(random_vector)
            env = eqx.combine(arr, static)

            rho0 = jnp.sqrt(jnp.linalg.norm(B_s.value) / jnp.linalg.norm(immediateJacobian__A_projection))
            rho1 = jnp.sqrt(jnp.linalg.norm(immediateInfluence__random_projection) / jnp.linalg.norm(random_vector))

            A_new: jax.Array = rho0 * immediateJacobian__A_projection + rho1 * random_vector
            B_new: jax.Array = B_s.value / rho0 + immediateInfluence__random_projection / rho1

            env, credit_gr, readout_stat = readout_gr(env, vl_data)
            grad = (credit_gr[: A_new.shape[0]] @ A_new) * B_new + credit_gr[A_new.shape[0] :]

            env = learn_interface.put_uoro_state(
                env,
                UOROState(A=A_s.set(value=A_new), B=B_s.set(value=B_new)),
            )

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

    return gradient_fn


def rflo[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: RFLOConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            mu = config.damping
            alpha = learn_interface.get_time_constant(env).value
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, tr_data)
                state = learn_interface.get_state(_env)
                _arr, _ = eqx.partition(_env, eqx.is_array)
                return state, (_arr, stat)

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn, has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn, has_aux=True)(p)
            env = eqx.combine(arr, static)

            influence_tensor: JACOBIAN
            influence_tensor = (1 - alpha) * influence_tensor_s.value + dhdp - mu * influence_tensor_s.value

            env, credit_gr, readout_stat = readout_gr(env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            env = learn_interface.put_forward_mode_jacobian(env, influence_tensor_s.set(value=influence_tensor))

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

    return gradient_fn


def bptt[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout: Callable[[ENV, VL_DATA], tuple[ENV, LOSS, STAT]],
    learn_interface: GodInterface[ENV],
    config: BPTTConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        param = learn_interface.get_param(env_init)

        def loss_fn(param: jax.Array, ds: tuple[TR_DATA, VL_DATA]) -> tuple[LOSS, tuple[ENV, STAT]]:
            env = learn_interface.put_param(env_init, param)
            arr_init, static = eqx.partition(env, eqx.is_array)

            def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[LOSS, STAT]]:
                _env = eqx.combine(arr, static)
                tr_data, vl_data = data

                if config.truncate_at is not None:
                    t = learn_interface.get_tick(_env)
                    s = filter_cond(
                        t % config.truncate_at == 0,
                        lambda _: jax.lax.stop_gradient(learn_interface.get_state(_env)),
                        lambda _: learn_interface.get_state(_env),
                        None,
                    )
                    _env = learn_interface.put_state(_env, s)

                _env, trans_stat = transition(_env, tr_data)
                _env, loss, readout_stat = readout(_env, vl_data)
                arr, _ = eqx.partition(_env, eqx.is_array)
                return arr, (loss, trans_stat | readout_stat)

            arr, (losses, stats) = jax.lax.scan(lambda x, y: vmap_this(inference_fn)(x, y), arr_init, ds, length=length)
            env = eqx.combine(arr, static)
            env = learn_interface.put_param(env, param)
            return jnp.sum(losses), (env, stats)

        grad, (env, stats) = eqx.filter_grad(loss_fn, has_aux=True)(param, ds_init)
        env = learn_interface.put_param(env, param)
        return env, process_gradient(GRADIENT(grad), grad_config), stats

    return gradient_fn


def identity_loss[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout: Callable[[ENV, VL_DATA], tuple[ENV, LOSS, STAT]],
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]:

    def loss_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, LOSS, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[LOSS, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            env, trans_stat = transition(env, tr_data)
            env, loss, readout_stat = readout(env, vl_data)
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (loss, trans_stat | readout_stat)

        arr, (losses, stats) = jax.lax.scan(
            lambda x, y: vmap_this(inference_fn)(x, y), arr_init, ds_init, length=length
        )
        env = eqx.combine(arr, static)
        return env, jnp.sum(losses), stats

    return loss_fn


def identity[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout: Callable[[ENV, VL_DATA], tuple[ENV, LOSS, STAT]],
    learn_interface: GodInterface[ENV],
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[LOSS, STAT]]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    _loss_fn = identity_loss(transition, readout, length, vmap_this)

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        env, loss, stats = _loss_fn(env_init, ds_init)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return env, process_gradient(GRADIENT(grad), grad_config), stats

    return gradient_fn


def create_validation_learners[ENV, TR_DATA, VL_DATA](
    transition_fns: list[Callable[[ENV, TR_DATA], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, VL_DATA], tuple[ENV, LOSS, STAT]]],
    val_learn_interfaces: list[GodInterface[ENV]],
    config: GodConfig,
) -> tuple[
    list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]],
    list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]],
]:

    def identity_transition(env: ENV, data: TR_DATA) -> tuple[ENV, STAT]:
        return env, {}

    def shim_expand_time(
        grad_fn: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]],
    ) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
        def wrapper(env: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
            data_with_time = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), (data, data))
            _, gradient, stat = grad_fn(env, data_with_time)
            stat = jax.tree.map(lambda x: x[0], stat)
            return env, gradient, stat

        return wrapper

    def make_readout_interface(interface: GodInterface[ENV]) -> GodInterface[ENV]:
        def get_param(env: ENV) -> jax.Array:
            state = interface.get_state(env)
            p = interface.get_param(env)
            return jnp.concatenate([state, p])

        def put_param(env: ENV, param: jax.Array) -> ENV:
            state_size = interface.get_state(env).shape[0]
            env = interface.put_state(env, param[:state_size])
            env = interface.put_param(env, param[state_size:])
            return env

        return copy.replace(
            interface,
            get_param=get_param,
            put_param=put_param,
            get_state=lambda env: jnp.empty(0),
            put_state=lambda env, s: env,
        )

    gradient_fns: list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]] = []
    loss_fns: list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]] = []

    for transition, readout_fn, interface, meta_config in zip(
        transition_fns,
        readout_fns,
        val_learn_interfaces,
        config.levels,
    ):
        method = meta_config.learner.model_learner.method
        model_grad_config = meta_config.learner.model_learner
        length = meta_config.validation.num_steps
        readout_interface = make_readout_interface(interface)
        readout_gr = shim_expand_time(
            bptt(
                identity_transition,
                readout_fn,
                readout_interface,
                BPTTConfig(truncate_at=None),
                GradientConfig(method=BPTTConfig(truncate_at=None), add_clip=None, scale=1.0),
                1,
                lambda f: f,
            )
        )

        readout_loss = identity_loss(transition, readout_fn, length, lambda f: f)

        match method:
            case BPTTConfig():
                fn = bptt(transition, readout_fn, interface, method, model_grad_config, length, lambda f: f)
            case IdentityLearnerConfig():
                fn = identity(transition, readout_fn, interface, model_grad_config, length, lambda f: f)
            case RTRLConfig() | RTRLFiniteHvpConfig():
                fn = rtrl(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f)
            case UOROConfig():
                fn = uoro(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f)
            case RFLOConfig():
                fn = rflo(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f)

        gradient_fns.append(fn)
        loss_fns.append(readout_loss)

    return gradient_fns, loss_fns


def restore_broadcast[ENV, X](
    fn: Callable[[ENV, tuple], tuple[ENV, X, STAT]],
    axes: ENV,
) -> Callable[[ENV, tuple], tuple[ENV, X, STAT]]:
    is_leaf = lambda x: x is None

    def wrapper(env: ENV, data: tuple) -> tuple[ENV, X, STAT]:
        out_env, x, stat = fn(env, data)
        merged_env = jax.tree.map(
            lambda ax, inp, out: inp if ax is None else out,
            axes,
            env,
            out_env,
            is_leaf=is_leaf,
        )
        return merged_env, x, stat

    return wrapper


def create_meta_learner[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    transition_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]],
    val_learn_interfaces: list[GodInterface[ENV]],
    nest_learn_interfaces: list[GodInterface[ENV]],
    meta_interfaces: list[dict[str, GodInterface[ENV]]],
    env: ENV,
) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

    validation_learners, validation_losses = create_validation_learners(
        transition_fns, readout_fns, val_learn_interfaces, config
    )
    learn_interface_pairs = list(zip(val_learn_interfaces, nest_learn_interfaces))
    resetters = env_resetters(config, shapes, meta_interfaces, learn_interface_pairs, [False] * len(config.levels))

    # Per-level validation axes: marks only that level's validation states with 0
    val_resetters = env_validation_resetters(config, shapes, meta_interfaces, val_learn_interfaces)
    per_level_val_axes = [diff_axes(env, vr(env, jax.random.key(0))) for vr in val_resetters]

    def make_optimized_transition[X](
        inner: Callable[[ENV, tuple], tuple[ENV, STAT]],
        readout_gr: Callable[[ENV, tuple], tuple[ENV, GRADIENT, STAT]],
        readout: Callable[[ENV, tuple], tuple[ENV, LOSS, STAT]],
        resetter: Callable[[ENV, PRNG], ENV],
        reset_t: int | None,
        nest_interface: GodInterface[ENV],
        assignments: dict[str, OptimizerAssignment],
        interfaces: dict[str, GodInterface[ENV]],
        method: GradientMethod,
        grad_config: GradientConfig,
        length: int,
        vmap_this: Callable[
            [Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]]],
            Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]],
        ],
    ) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

        check = make_reset_checker(nest_interface, resetter, reset_t)
        advance = make_tick_advancer(nest_interface)

        def composed_inner(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env = check(env)
            env = advance(env)
            return inner(env, data)

        match method:
            case RTRLConfig() | RTRLFiniteHvpConfig():
                grad_fn = rtrl(composed_inner, readout_gr, nest_interface, method, grad_config, length, vmap_this)
            case BPTTConfig():
                grad_fn = bptt(composed_inner, readout, nest_interface, method, grad_config, length, vmap_this)
            case IdentityLearnerConfig():
                grad_fn = identity(composed_inner, readout, nest_interface, grad_config, length, vmap_this)
            case UOROConfig():
                grad_fn = uoro(composed_inner, readout_gr, nest_interface, method, grad_config, length, vmap_this)
            case RFLOConfig():
                grad_fn = rflo(composed_inner, readout_gr, nest_interface, method, grad_config, length, vmap_this)

        def optimized_transition(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env, gradient, stat = grad_fn(env, data)
            gr_env = nest_interface.put_param(env, gradient)
            env = get_opt_step(assignments, interfaces, env, gr_env, config.hyperparameters)
            return env, stat

        return optimized_transition

    # Collect axes for all levels
    all_axes: list[ENV] = []
    current_transition: Callable[[ENV, tuple], tuple[ENV, STAT]] = lambda env, data: (env, {})
    current_resetter: Callable[[ENV, PRNG], ENV] = lambda env, prng: env

    for level in range(len(config.levels)):
        meta_config = config.levels[level]
        nest_interface = nest_learn_interfaces[level]
        inner_resetter, full_resetter = resetters[level]
        vl_learner = validation_learners[level]
        vl_loss = validation_losses[level]
        interfaces = meta_interfaces[level]

        axes = diff_axes(env, inner_resetter(env, jax.random.key(0)))
        all_axes.append(axes)

        # N extra vmaps on readout for LOWER levels' nested dims
        # Combine each lower level's axes with current level's validation axes
        for ax in all_axes[:level]:
            combined = eqx.combine(ax, per_level_val_axes[level])
            vl_learner = restore_broadcast(vl_learner, combined)
            vl_loss = restore_broadcast(vl_loss, combined)
            vl_learner = eqx.filter_vmap(vl_learner, in_axes=(combined, 0), out_axes=(combined, 0, 0))
            vl_loss = eqx.filter_vmap(vl_loss, in_axes=(combined, 0), out_axes=(combined, 0, 0))

        # vmap_this wraps the scan step inside learners, peeling this level's nested dim
        vmap_this = lambda f, a=axes: eqx.filter_vmap(f, in_axes=(a, 0), out_axes=(a, 0))

        current_transition = make_optimized_transition(
            current_transition,
            vl_learner,
            vl_loss,
            current_resetter,
            meta_config.nested.reset_t,
            nest_interface,
            meta_config.learner.optimizer,
            interfaces,
            meta_config.learner.optimizer_learner.method,
            meta_config.learner.optimizer_learner,
            meta_config.nested.num_steps,
            vmap_this,
        )

        current_resetter = full_resetter

    return current_transition
