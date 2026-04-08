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
from meta_learn_lib.util import (
    filter_cond,
    finite_difference_jmp,
    finite_difference_jvp,
    jacobian_matrix_product,
    softclip,
)


def process_gradient(grad: GRADIENT, grad_config: GradientConfig) -> GRADIENT:
    g = grad * grad_config.scale
    match grad_config.add_clip:
        case HardClip(threshold):
            norm = jnp.linalg.norm(g)
            g = GRADIENT(g * jnp.minimum(1.0, threshold / jnp.maximum(norm, 1e-8)))
        case SoftClip(threshold, sharpness):
            g = softclip(g, a=None, b=threshold, sharpness=sharpness)
        case None:
            pass
    return GRADIENT(g)


def rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: RTRLConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            t = learn_interface.get_tick(env)
            mu = config.damping
            beta = config.beta
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def state_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, None]]:
                def fn(state: jax.Array) -> tuple[jax.Array, None]:
                    _env = learn_interface.put_state(e, state)
                    _env, _ = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    return state, None

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = learn_interface.put_param(e, param)
                    _env, stat = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn(env), has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn(env), has_aux=True)(p)
            new_env = eqx.combine(arr, static)

            hmp: JACOBIAN
            match config.finite_hvp:
                case None:
                    _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn(env), s, influence_tensor_s.value)
                    hmp = hmp_jvp - mu * influence_tensor_s.value
                case RTRLFiniteHvpConfig(eps):
                    f = lambda x: state_fn(env)(x)[0]
                    hmp = finite_difference_jmp(f, s, influence_tensor_s.value, eps) - mu * influence_tensor_s.value

            influence_tensor: JACOBIAN
            influence_tensor = filter_cond(
                t >= config.start_at_step,
                lambda _: beta * (hmp + dhdp) + (1 - beta) * influence_tensor_s.value,
                lambda _: influence_tensor_s.value,
                None,
            )
            new_env, credit_gr, readout_stat = readout_gr(new_env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            new_env = learn_interface.put_forward_mode_jacobian(new_env, influence_tensor_s.set(value=influence_tensor))
            if track_logs.influence_tensor_norm:
                influence_tensor_norm = jnp.linalg.norm(influence_tensor)
                new_env = learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

            arr, _ = eqx.partition(new_env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

    return gradient_fn


def tikhonov_rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    _config: TikhonovRTRLConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

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
            beta = config.beta
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def state_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, None]]:
                def fn(state: jax.Array) -> tuple[jax.Array, None]:
                    _env = learn_interface.put_state(e, state)
                    _env, _ = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    return state, None

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = learn_interface.put_param(e, param)
                    _env, stat = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn(env), has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn(env), has_aux=True)(p)
            new_env = eqx.combine(arr, static)

            # D_tau = dF/dz @ Gamma + dF/dphi (forward sensitivity)
            match config.finite_hvp:
                case None:
                    _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn(env), s, influence_tensor_s.value)
                case RTRLFiniteHvpConfig(eps):
                    f = lambda x: state_fn(env)(x)[0]
                    hmp_jvp = finite_difference_jmp(f, s, influence_tensor_s.value, eps)

            d_tau = hmp_jvp + dhdp
            error = influence_tensor_s.value - d_tau

            # (dF/dz)^T @ (Gamma - D) via vmapped VJPs (adjoint error correction)
            _, vjp_fn = eqx.filter_vjp(lambda x: state_fn(env)(x)[0], s)
            correction = eqx.filter_vmap(lambda col: vjp_fn(col)[0], in_axes=1, out_axes=1)(error)

            # Gamma_new = (1 - beta) * Gamma + beta * (D + (dF/dz)^T(Gamma - D) - mu * Gamma)
            target = d_tau + correction - mu * influence_tensor_s.value
            updated = beta * target + (1 - beta) * influence_tensor_s.value

            influence_tensor: JACOBIAN
            influence_tensor = filter_cond(
                t >= config.start_at_step,
                lambda _: updated,
                lambda _: influence_tensor_s.value,
                None,
            )
            new_env, credit_gr, readout_stat = readout_gr(new_env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            new_env = learn_interface.put_forward_mode_jacobian(new_env, influence_tensor_s.set(value=influence_tensor))
            if track_logs.influence_tensor_norm:
                influence_tensor_norm = jnp.linalg.norm(influence_tensor)
                new_env = learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

            arr, _ = eqx.partition(new_env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

    return gradient_fn


def pade_rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    _config: PadeRTRLConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    config = _config.rtrl_config

    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            t = learn_interface.get_tick(env)
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def state_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, None]]:
                def fn(state: jax.Array) -> tuple[jax.Array, None]:
                    _env = learn_interface.put_state(e, state)
                    _env, _ = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    return state, None

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = learn_interface.put_param(e, param)
                    _env, stat = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn(env), has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn(env), has_aux=True)(p)
            new_env = eqx.combine(arr, static)

            # JVP 1: dF/dz @ Gamma (for D_tau)
            # JVP 2: dF/dz @ dF/dphi (extra cost for Pade)
            match config.finite_hvp:
                case None:
                    _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn(env), s, influence_tensor_s.value)
                    _primals2, dhdz_dhdp, _aux2 = jacobian_matrix_product(state_fn(env), s, dhdp)
                case RTRLFiniteHvpConfig(eps):
                    f = lambda x: state_fn(env)(x)[0]
                    hmp_jvp = finite_difference_jmp(f, s, influence_tensor_s.value, eps)
                    dhdz_dhdp = finite_difference_jmp(f, s, dhdp, eps)

            # Pade [1,1]: Gamma_{t+1} = 1/2 * D_tau + 1/2 * (I + dF/dz) * dF/dphi
            d_tau = hmp_jvp + dhdp
            pade_update = 0.5 * d_tau + 0.5 * (dhdp + dhdz_dhdp)

            influence_tensor: JACOBIAN
            influence_tensor = filter_cond(
                t >= config.start_at_step,
                lambda _: pade_update,
                lambda _: influence_tensor_s.value,
                None,
            )
            new_env, credit_gr, readout_stat = readout_gr(new_env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            new_env = learn_interface.put_forward_mode_jacobian(new_env, influence_tensor_s.set(value=influence_tensor))
            if track_logs.influence_tensor_norm:
                influence_tensor_norm = jnp.linalg.norm(influence_tensor)
                new_env = learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

            arr, _ = eqx.partition(new_env, eqx.is_array)
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
    _config: UOROConfig | UOROFiniteDiffConfig,
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    match _config:
        case UOROConfig():
            config = _config
        case UOROFiniteDiffConfig():
            config = _config.uoro_config

    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        std = config.std
        match config.distribution:
            case "uniform":
                distribution = lambda key, shape: jax.random.uniform(key, shape, minval=-std, maxval=std)
            case "normal":
                distribution = lambda key, shape: jax.random.normal(key, shape) * std

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            key, env = learn_interface.take_prng(env)
            mu = config.damping
            beta = config.beta
            uoro_state = learn_interface.get_uoro_state(env)
            A_s = uoro_state.A
            B_s = uoro_state.B
            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            state_shape = s.shape
            random_vector = distribution(key, state_shape)

            def state_fn(e: ENV) -> Callable[[jax.Array], jax.Array]:
                def fn(state: jax.Array) -> jax.Array:
                    _env = learn_interface.put_state(e, state)
                    _env, _ = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    return state

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = learn_interface.put_param(e, param)
                    _env, stat = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            match _config:
                case UOROConfig():
                    damped_jvp = eqx.filter_jvp(state_fn(env), (s,), (A_s.value,))[1] - mu * A_s.value
                case UOROFiniteDiffConfig(eps, _):
                    damped_jvp = finite_difference_jvp(state_fn(env), s, A_s.value, eps) - mu * A_s.value
            A_propagated = beta * damped_jvp + (1 - beta) * A_s.value
            _, vjp_func, (arr, trans_stat) = eqx.filter_vjp(param_fn(env), p, has_aux=True)
            (immediateInfluence__random_projection,) = vjp_func(random_vector)
            scaled_immediate = beta * immediateInfluence__random_projection
            new_env = eqx.combine(arr, static)

            rho0 = jnp.sqrt(jnp.linalg.norm(B_s.value) / jnp.linalg.norm(A_propagated))
            rho1 = jnp.sqrt(jnp.linalg.norm(scaled_immediate) / jnp.linalg.norm(random_vector))

            A_new: jax.Array = rho0 * A_propagated + rho1 * random_vector
            B_new: jax.Array = B_s.value / rho0 + scaled_immediate / rho1

            new_env, credit_gr, readout_stat = readout_gr(new_env, vl_data)
            grad = (credit_gr[..., : A_new.shape[0]] @ A_new) * B_new + credit_gr[..., A_new.shape[0] :]

            new_env = learn_interface.put_uoro_state(
                new_env,
                UOROState(A=A_s.set(value=A_new), B=B_s.set(value=B_new)),
            )
            if track_logs.influence_tensor_norm:
                influence_tensor_norm = jnp.linalg.norm(A_new) * jnp.linalg.norm(B_new)
                new_env = learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

            arr, _ = eqx.partition(new_env, eqx.is_array)
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
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            mu = config.damping
            beta = config.beta
            alpha = learn_interface.get_time_constant(env).value
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = learn_interface.put_param(e, param)
                    _env, stat = transition(_env, tr_data)
                    state = learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            if s.shape[0] > p.shape[0]:
                dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn(env), has_aux=True)(p)
            else:
                dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn(env), has_aux=True)(p)
            new_env = eqx.combine(arr, static)

            influence_tensor: JACOBIAN
            naive = (1 - alpha) * influence_tensor_s.value + dhdp - mu * influence_tensor_s.value
            influence_tensor = beta * naive + (1 - beta) * influence_tensor_s.value

            new_env, credit_gr, readout_stat = readout_gr(new_env, vl_data)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            grad: GRADIENT = credit_gr @ state_jacobian
            new_env = learn_interface.put_forward_mode_jacobian(new_env, influence_tensor_s.set(value=influence_tensor))
            if track_logs.influence_tensor_norm:
                influence_tensor_norm = jnp.linalg.norm(influence_tensor)
                new_env = learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

            arr, _ = eqx.partition(new_env, eqx.is_array)
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
    track_logs: TrackLogs,
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
    track_logs: TrackLogs,
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
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    _loss_fn = identity_loss(transition, readout, length, vmap_this, track_logs)

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        env, loss, stats = _loss_fn(env_init, ds_init)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return env, process_gradient(GRADIENT(grad), grad_config), stats

    return gradient_fn


def immediate[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    grad_config: GradientConfig,
    length: int,
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ],
    track_logs: TrackLogs,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data
            env, trans_stat = transition(env, tr_data)
            env, credit_gr, readout_stat = readout_gr(env, vl_data)
            n_s = learn_interface.get_state(env).shape[0]
            grad = GRADIENT(credit_gr[..., n_s:])
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: vmap_this(step)(x, y), arr_init, ds, length=length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, grad_config)
        return env, total_grad, stats

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
        track_logs = meta_config.track_logs
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
                track_logs,
            )
        )

        readout_loss = identity_loss(transition, readout_fn, length, lambda f: f, track_logs)

        match method:
            case BPTTConfig():
                fn = bptt(transition, readout_fn, interface, method, model_grad_config, length, lambda f: f, track_logs)
            case IdentityLearnerConfig():
                fn = identity(transition, readout_fn, interface, model_grad_config, length, lambda f: f, track_logs)
            case RTRLConfig():
                fn = rtrl(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs)
            case TikhonovRTRLConfig():
                fn = tikhonov_rtrl(
                    transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs
                )
            case PadeRTRLConfig():
                fn = pade_rtrl(
                    transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs
                )
            case UOROConfig() | UOROFiniteDiffConfig():
                fn = uoro(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs)
            case RFLOConfig():
                fn = rflo(transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs)
            case ImmediateLearnerConfig():
                fn = immediate(transition, readout_gr, interface, model_grad_config, length, lambda f: f, track_logs)

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
        level: int,
        length: int,
        vmap_this: Callable[
            [Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]]],
            Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]],
        ],
        track_logs: TrackLogs,
    ) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

        check = make_reset_checker(nest_interface, resetter, reset_t)
        advance = make_tick_advancer(nest_interface)

        def composed_inner(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env = check(env)
            env = advance(env)
            return inner(env, data)

        match method:
            case RTRLConfig():
                grad_fn = rtrl(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case TikhonovRTRLConfig():
                grad_fn = tikhonov_rtrl(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case PadeRTRLConfig():
                grad_fn = pade_rtrl(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case BPTTConfig():
                grad_fn = bptt(
                    composed_inner,
                    readout,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case IdentityLearnerConfig():
                grad_fn = identity(
                    composed_inner,
                    readout,
                    nest_interface,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case UOROConfig() | UOROFiniteDiffConfig():
                grad_fn = uoro(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case RFLOConfig():
                grad_fn = rflo(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case ImmediateLearnerConfig():
                grad_fn = immediate(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )

        def optimized_transition(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env, gradient, stat = grad_fn(env, data)
            gr_env = nest_interface.put_param(env, gradient)
            env = get_opt_step(assignments, interfaces, env, gr_env, config.hyperparameters)
            stat[f"level{level}/meta_gradient_norm"] = jax.lax.stop_gradient(jnp.linalg.norm(gradient))
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
            level,
            meta_config.nested.num_steps,
            vmap_this,
            meta_config.track_logs,
        )

        current_resetter = full_resetter

    return current_transition
