from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.create_env import env_resetters, make_reset_checker, make_tick_advancer
from meta_learn_lib.interface import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.optimizer import get_opt_step
from meta_learn_lib.util import filter_cond, jacobian_matrix_product


def rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    _config: RTRLConfig | RTRLFiniteHvpConfig,
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
            credit_gr, readout_stat = readout_gr(env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            env = learn_interface.put_forward_mode_jacobian(env, influence_tensor_s.set(value=influence_tensor))

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(step, arr_init, ds)
        env = eqx.combine(arr, static)
        return env, GRADIENT(jnp.sum(grads, axis=0)), stats

    return gradient_fn


def uoro[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: UOROConfig,
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

            rho0 = jnp.sqrt(jnp.linalg.norm(B_s.value) / jnp.linalg.norm(immediateJacobian__A_projection))
            rho1 = jnp.sqrt(jnp.linalg.norm(immediateInfluence__random_projection) / jnp.linalg.norm(random_vector))

            A_new: jax.Array = rho0 * immediateJacobian__A_projection + rho1 * random_vector
            B_new: jax.Array = B_s.value / rho0 + immediateInfluence__random_projection / rho1

            credit_gr, readout_stat = readout_gr(env, vl_data)
            grad = (credit_gr[: A_new.shape[0]] @ A_new) * B_new + credit_gr[A_new.shape[0] :]

            env = learn_interface.put_uoro_state(
                env,
                UOROState(A=A_s.set(value=A_new), B=B_s.set(value=B_new)),
            )

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(step, arr_init, ds)
        env = eqx.combine(arr, static)
        return env, GRADIENT(jnp.sum(grads, axis=0)), stats

    return gradient_fn


def rflo[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: RFLOConfig,
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

            credit_gr, readout_stat = readout_gr(env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            env = learn_interface.put_forward_mode_jacobian(env, influence_tensor_s.set(value=influence_tensor))

            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(step, arr_init, ds)
        env = eqx.combine(arr, static)
        return env, GRADIENT(jnp.sum(grads, axis=0)), stats

    return gradient_fn


def bptt[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout: Callable[[ENV, VL_DATA], tuple[LOSS, STAT]],
    learn_interface: GodInterface[ENV],
    config: BPTTConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        def loss_fn(param: jax.Array, ds: tuple[TR_DATA, VL_DATA]) -> tuple[LOSS, tuple[ENV, STAT]]:
            env = learn_interface.put_param(env_init, param)
            arr_init, static = eqx.partition(env, eqx.is_array)

            def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[STAT, LOSS]]:
                _env = eqx.combine(arr, static)
                tr_data, vl_data = data

                t = learn_interface.get_tick(_env)
                s = filter_cond(
                    t % config.truncate_at == 0,
                    lambda _: jax.lax.stop_gradient(learn_interface.get_state(_env)),
                    lambda _: learn_interface.get_state(_env),
                    None,
                )
                _env = learn_interface.put_state(_env, s)

                _env, trans_stat = transition(_env, tr_data)
                loss, readout_stat = readout(_env, vl_data)
                arr, _ = eqx.partition(_env, eqx.is_array)
                return arr, (trans_stat | readout_stat, loss)

            arr, (stats, losses) = jax.lax.scan(inference_fn, arr_init, ds)
            env = eqx.combine(arr, static)
            return jnp.sum(losses), (env, stats)

        param = learn_interface.get_param(env_init)
        grad, (env, stats) = eqx.filter_grad(loss_fn, has_aux=True)(param, ds_init)
        return env, GRADIENT(grad), stats

    return gradient_fn


def identity[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout: Callable[[ENV, VL_DATA], tuple[LOSS, STAT]],
    learn_interface: GodInterface[ENV],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[STAT, LOSS]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            env, trans_stat = transition(env, tr_data)
            loss, readout_stat = readout(env, vl_data)
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (trans_stat | readout_stat, loss)

        arr, (stats, losses) = jax.lax.scan(inference_fn, arr_init, ds_init)
        env = eqx.combine(arr, static)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return env, GRADIENT(grad), stats

    return gradient_fn


def create_validation_learners[ENV, TR_DATA, VL_DATA](
    transition_fns: list[Callable[[ENV, TR_DATA], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, VL_DATA], tuple[LOSS, STAT]]],
    val_learn_interfaces: list[GodInterface[ENV]],
    config: GodConfig,
) -> list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[GRADIENT, STAT]]]:

    def identity_transition(env: ENV, data: TR_DATA) -> tuple[ENV, STAT]:
        return env, {}

    def shim_expand_time(
        grad_fn: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]],
    ) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
        def wrapper(env: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
            data_with_time = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), data)
            env, gradient, stat = grad_fn(env, data_with_time)
            stat = jax.tree.map(lambda x: x[0], stat)
            return env, gradient, stat

        return wrapper

    def drop_env(
        fn: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]],
    ) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[GRADIENT, STAT]]:
        def wrapper(env: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[GRADIENT, STAT]:
            _, gradient, stat = fn(env, data)
            return gradient, stat

        return wrapper

    transitions = [identity_transition] + transition_fns[1:]

    gradient_fns: list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]] = []
    for transition, readout_fn, interface, meta_config in zip(
        transitions,
        readout_fns,
        val_learn_interfaces,
        config.levels,
    ):
        method = meta_config.learner.model_learner.method
        readout_gr = drop_env(
            shim_expand_time(bptt(identity_transition, readout_fn, interface, BPTTConfig(truncate_at=None)))
        )

        match method:
            case BPTTConfig():
                fn = bptt(transition, readout_fn, interface, method)
            case IdentityLearnerConfig():
                fn = identity(transition, readout_fn, interface)
            case RTRLConfig() | RTRLFiniteHvpConfig():
                fn = rtrl(transition, readout_gr, interface, method)
            case UOROConfig():
                fn = uoro(transition, readout_gr, interface, method)
            case RFLOConfig():
                fn = rflo(transition, readout_gr, interface, method)

        gradient_fns.append(fn)

    gradient_fns[0] = shim_expand_time(gradient_fns[0])

    return [drop_env(fn) for fn in gradient_fns]


def create_meta_learner[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    transition_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]]],
    val_learn_interfaces: list[GodInterface[ENV]],
    nest_learn_interfaces: list[GodInterface[ENV]],
    meta_interfaces: list[dict[str, GodInterface[ENV]]],
    env: ENV,
) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

    validation_learners = create_validation_learners(transition_fns, readout_fns, val_learn_interfaces, config)
    learn_interface_pairs = list(zip(val_learn_interfaces, nest_learn_interfaces))
    resetters = env_resetters(config, shapes, meta_interfaces, learn_interface_pairs, [False] * len(config.levels))

    def make_optimized_transition(
        inner: Callable[[ENV, tuple], tuple[ENV, STAT]],
        readout_gr: Callable[[ENV, tuple], tuple[GRADIENT, STAT]],
        readout: Callable[[ENV, tuple], tuple[LOSS, STAT]],
        check: Callable[[ENV], ENV],
        advance: Callable[[ENV], ENV],
        assignments: dict[str, OptimizerAssignment],
        interfaces: dict[str, GodInterface[ENV]],
        nest_interface: GodInterface[ENV],
        method: GradientMethod,
    ) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

        match method:
            case RTRLConfig() | RTRLFiniteHvpConfig():
                grad_fn = rtrl(inner, readout_gr, nest_interface, method)
            case BPTTConfig():
                grad_fn = bptt(inner, readout, nest_interface, method)
            case IdentityLearnerConfig():
                grad_fn = identity(inner, readout, nest_interface)
            case UOROConfig():
                grad_fn = uoro(inner, readout_gr, nest_interface, method)
            case RFLOConfig():
                grad_fn = rflo(inner, readout_gr, nest_interface, method)

        def optimized_transition(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env = check(env)
            env = advance(env)
            env, gradient, stat = grad_fn(env, data)
            gr_env = nest_interface.put_param(env, gradient)
            env = get_opt_step(assignments, interfaces, env, gr_env, config.hyperparameters)
            return env, stat

        return optimized_transition

    current_transition = transition_fns[0]

    for level in range(len(config.levels)):
        meta_config = config.levels[level]
        nest_interface = nest_learn_interfaces[level]
        resetter = resetters[level]
        vl_learner = validation_learners[level]
        readout_fn = readout_fns[level]
        interfaces = meta_interfaces[level]

        check = make_reset_checker(nest_interface, resetter, meta_config.meta_opt.reset_t)
        advance = make_tick_advancer(nest_interface)

        current_transition = make_optimized_transition(
            current_transition,
            vl_learner,
            readout_fn,
            check,
            advance,
            meta_config.learner.optimizer,
            interfaces,
            nest_interface,
            meta_config.learner.optimizer_learner.method,
        )

        axes = diff_axes(env, resetter(env, jax.random.key(0)))
        current_transition = eqx.filter_vmap(current_transition, in_axes=(axes, 0), out_axes=(axes, 0))

    return current_transition
