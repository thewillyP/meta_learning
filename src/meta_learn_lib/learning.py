import copy
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from matfree import decomp as matfree_decomp

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
    jvp,
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


def compute_dhdp[ENV](
    param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
    s: jax.Array,
    p: jax.Array,
    static: ENV,
) -> tuple[JACOBIAN, ENV, STAT]:
    if s.shape[0] > p.shape[0]:
        dhdp, (arr, trans_stat) = eqx.filter_jacfwd(param_fn, has_aux=True)(p)
    else:
        dhdp, (arr, trans_stat) = eqx.filter_jacrev(param_fn, has_aux=True)(p)
    new_env = eqx.combine(arr, static)
    return dhdp, new_env, trans_stat


@dataclass(frozen=True)
class LearningArg[ENV, TR_DATA, VL_DATA]:
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]]
    readout_gr: Callable[[ENV, VL_DATA], tuple[ENV, GRADIENT, STAT]]
    learn_interface: GodInterface[ENV]
    grad_config: GradientConfig
    length: int
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[GRADIENT, STAT]]],
    ]
    track_logs: TrackLogs


def get_forward_mode[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    update_influence: Callable[
        [
            Callable[[jax.Array], tuple[jax.Array, None]],
            Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
            ENV,
        ],
        tuple[ENV, STAT],
    ],
    credit_gr_fn: Callable[[GRADIENT, GodInterface[ENV], ENV], GRADIENT],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            def state_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, None]]:
                def fn(state: jax.Array) -> tuple[jax.Array, None]:
                    _env = args.learn_interface.put_state(e, state)
                    _env, _ = args.transition(_env, tr_data)
                    state = args.learn_interface.get_state(_env)
                    return state, None

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = args.learn_interface.put_param(e, param)
                    _env, stat = args.transition(_env, tr_data)
                    state = args.learn_interface.get_state(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            new_env, trans_stat = update_influence(state_fn(env), param_fn(env), env)
            new_env, credit_gr, readout_stat = args.readout_gr(new_env, vl_data)
            grad = credit_gr_fn(credit_gr, args.learn_interface, new_env)

            arr, _ = eqx.partition(new_env, eqx.is_array)
            return arr, (grad, trans_stat | readout_stat)

        arr, (grads, stats) = jax.lax.scan(lambda x, y: args.vmap_this(step)(x, y), arr_init, ds, length=args.length)
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad = process_gradient(total_grad, args.grad_config)
        return env, total_grad, stats

    return gradient_fn


def rtrl_like[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    update_tensor: Callable[
        [
            Callable[[jax.Array], tuple[jax.Array, None]],
            jax.Array,
            JACOBIAN,
            JACOBIAN,
            ENV,
        ],
        JACOBIAN,
    ],
    start_at_step: int,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.get_state(env)
        p = args.learn_interface.get_param(env)
        influence_tensor_s = args.learn_interface.get_forward_mode_jacobian(env)

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        new_influence_tensor = update_tensor(state_fn, s, dhdp, influence_tensor_s.value, env)

        new_influence_tensor = filter_cond(
            args.learn_interface.get_tick(env) >= start_at_step,
            lambda _: new_influence_tensor,
            lambda _: influence_tensor_s.value,
            None,
        )

        new_env = args.learn_interface.put_forward_mode_jacobian(
            new_env,
            influence_tensor_s.set(value=new_influence_tensor),
        )
        if args.track_logs.influence_tensor_norm:
            influence_tensor_norm: jax.Array = jnp.linalg.norm(new_influence_tensor)
            new_env = args.learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)
        state_jacobian = jnp.vstack([influence_tensor_s.value, jnp.eye(influence_tensor_s.value.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: RTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
    ) -> jax.Array:
        mu = config.damping
        beta = config.beta

        hmp: JACOBIAN
        match config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor)
                hmp = hmp_jvp - mu * influence_tensor
            case eps:
                hmp = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor, eps) - mu * influence_tensor

        updated = beta * (hmp + dhdp) + (1 - beta) * influence_tensor
        return updated

    return rtrl_like(args, update_tensor, config.start_at_step)


def tikhonov_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: TikhonovRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
    ) -> jax.Array:
        mu = config.rtrl_config.damping
        beta = config.rtrl_config.beta

        hmp_jvp: JACOBIAN
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor)
            case eps:
                hmp_jvp = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor, eps)

        d_tau = hmp_jvp + dhdp
        error = influence_tensor - d_tau

        _, vjp_fn = eqx.filter_vjp(lambda x: state_fn(x)[0], s)
        correction = eqx.filter_vmap(lambda col: vjp_fn(col)[0], in_axes=1, out_axes=1)(error)

        target = d_tau + correction - mu * influence_tensor
        updated = beta * target + (1 - beta) * influence_tensor
        return updated

    return rtrl_like(args, update_tensor, config.rtrl_config.start_at_step)


def pade_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: PadeRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
    ) -> jax.Array:
        # JVP 1: dF/dz @ Gamma (for D_tau)
        # JVP 2: dF/dz @ dF/dphi (extra cost for Pade)
        hmp_jvp: JACOBIAN
        dhdz_dhdp: JACOBIAN
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor)
                _primals2, dhdz_dhdp, _aux2 = jacobian_matrix_product(state_fn, s, dhdp)
            case eps:
                hmp_jvp = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor, eps)
                dhdz_dhdp = finite_difference_jmp(lambda x: state_fn(x)[0], s, dhdp, eps)

        # Pade [1,1]: Gamma_{t+1} = 1/2 * D_tau + 1/2 * (I + dF/dz) * dF/dphi
        d_tau = hmp_jvp + dhdp
        return 0.5 * d_tau + 0.5 * (dhdp + dhdz_dhdp)

    return rtrl_like(args, update_tensor, config.rtrl_config.start_at_step)


def midpoint_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: MidpointRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.get_state(env)
        p = args.learn_interface.get_param(env)
        influence_tensor_s = args.learn_interface.get_forward_mode_jacobian(env)
        midpoint_buffer_s = args.learn_interface.get_midpoint_buffer(env)

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        # JVP 1: J_t @ P_t (for next step's predictor)
        # JVP 2: J_t @ predictor (for corrector of interval ending at t+1)
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp_current, _aux = jacobian_matrix_product(state_fn, s, influence_tensor_s.value)
                _primals2, hmp_jvp_predictor, _aux2 = jacobian_matrix_product(
                    state_fn, s, midpoint_buffer_s.predictor.value
                )
            case eps:
                hmp_jvp_current = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor_s.value, eps)
                hmp_jvp_predictor = finite_difference_jmp(
                    lambda x: state_fn(x)[0], s, midpoint_buffer_s.predictor.value, eps
                )

        # Forward Euler from P_t (used as bootstrap and as next step's predictor)
        fe_update = hmp_jvp_current + dhdp
        # Midpoint corrector: P_new = P_prev + 2*((J_t - I) @ predictor + B_t)
        midpoint_update = midpoint_buffer_s.P_prev.value + 2 * (
            hmp_jvp_predictor - midpoint_buffer_s.predictor.value + dhdp
        )
        # First active step: forward Euler bootstrap. Subsequent: midpoint corrector.
        is_bootstrap = args.learn_interface.get_tick(env) <= config.rtrl_config.start_at_step
        active_update = jnp.where(is_bootstrap, fe_update, midpoint_update)
        influence_tensor: JACOBIAN = filter_cond(
            args.learn_interface.get_tick(env) >= config.rtrl_config.start_at_step,
            lambda _: active_update,
            lambda _: influence_tensor_s.value,
            None,
        )

        # Buffer update: P_prev = current P_t, predictor = forward Euler from P_t
        new_P_prev = filter_cond(
            args.learn_interface.get_tick(env) >= config.rtrl_config.start_at_step,
            lambda _: influence_tensor_s.value,
            lambda _: midpoint_buffer_s.P_prev.value,
            None,
        )

        new_predictor = filter_cond(
            args.learn_interface.get_tick(env) >= config.rtrl_config.start_at_step,
            lambda _: fe_update,
            lambda _: midpoint_buffer_s.predictor.value,
            None,
        )

        new_env = args.learn_interface.put_forward_mode_jacobian(
            new_env, influence_tensor_s.set(value=influence_tensor)
        )
        new_env = args.learn_interface.put_midpoint_buffer(
            new_env,
            MidpointBuffer(
                P_prev=midpoint_buffer_s.P_prev.set(value=new_P_prev),
                predictor=midpoint_buffer_s.predictor.set(value=new_predictor),
            ),
        )

        if args.track_logs.influence_tensor_norm:
            readout_tensor = 0.5 * (influence_tensor + influence_tensor_s.value)
            influence_tensor_norm = jnp.linalg.norm(readout_tensor)
            new_env = args.learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))

        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)
        # 2-tap boxcar: average P_t and P_{t-1} for readout to kill parity mode
        readout_tensor = 0.5 * (influence_tensor_s.value + learn_interface.get_midpoint_buffer(env).P_prev.value)
        state_jacobian = jnp.vstack([readout_tensor, jnp.eye(readout_tensor.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def heun_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: HeunRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    rtrl_config = config.rtrl_config

    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.get_state(env)
        p = args.learn_interface.get_param(env)
        t = args.learn_interface.get_tick(env)
        influence_tensor_s = args.learn_interface.get_forward_mode_jacobian(env)
        midpoint_buffer_s = args.learn_interface.get_midpoint_buffer(env)

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        # JVP 1: J_t @ predictor_stored (corrector slope at current tick)
        predictor_stored = midpoint_buffer_s.predictor.value
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, corrector_jvp, _aux = jacobian_matrix_product(state_fn, s, predictor_stored)
            case eps:
                f = lambda x: state_fn(x)[0]
                corrector_jvp = finite_difference_jmp(f, s, predictor_stored, eps)

        # Heun: P_new = 0.5 * (P_prev + J_t @ predictor + B_t)
        heun_update = 0.5 * (influence_tensor_s.value + corrector_jvp + dhdp)

        # Forward Euler (bootstrap, first active step only)
        # At bootstrap, predictor_stored == P_prev, so corrector_jvp == J_t @ P_prev
        fe_update = corrector_jvp + dhdp

        is_bootstrap = t <= rtrl_config.start_at_step
        active_update = jnp.where(is_bootstrap, fe_update, heun_update)

        influence_tensor: JACOBIAN
        influence_tensor = filter_cond(
            t >= rtrl_config.start_at_step,
            lambda _: active_update,
            lambda _: influence_tensor_s.value,
            None,
        )

        # JVP 2: J_t @ P_new (for next step's predictor)
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals2, pred_jvp, _aux2 = jacobian_matrix_product(state_fn, s, influence_tensor)
            case eps:
                f = lambda x: state_fn(x)[0]
                pred_jvp = finite_difference_jmp(f, s, influence_tensor, eps)

        new_predictor = filter_cond(
            t >= rtrl_config.start_at_step,
            lambda _: pred_jvp + dhdp,
            lambda _: midpoint_buffer_s.predictor.value,
            None,
        )

        new_env = args.learn_interface.put_forward_mode_jacobian(
            new_env, influence_tensor_s.set(value=influence_tensor)
        )
        new_env = args.learn_interface.put_midpoint_buffer(
            new_env,
            MidpointBuffer(
                P_prev=midpoint_buffer_s.P_prev,
                predictor=midpoint_buffer_s.predictor.set(value=new_predictor),
            ),
        )
        if args.track_logs.influence_tensor_norm:
            influence_tensor_norm = jnp.linalg.norm(influence_tensor)
            new_env = args.learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)
        state_jacobian = jnp.vstack([influence_tensor_s.value, jnp.eye(influence_tensor_s.value.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def implicit_euler_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: ImplicitEulerRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    rtrl_config = config.rtrl_config
    num_arnoldi_iters = config.num_arnoldi_iters

    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
    ) -> jax.Array:
        mu = rtrl_config.damping

        # JVP oracle: v -> J_t @ v (vector, not matrix)
        f_eval = lambda x: state_fn(x)[0]
        match rtrl_config.use_finite_hvp:
            case None:

                def jvp_Jt(v: jax.Array) -> jax.Array:
                    _primals, tangent, _aux = jvp(f_eval, s, v)
                    return tangent
            case eps:

                def jvp_Jt(v: jax.Array) -> jax.Array:
                    return finite_difference_jvp(f_eval, s, v, eps)

        # Implicit Euler: solve ((2+mu)I - J_t) P_t = P_{t-1} + B_t per column
        # via GMRES = Arnoldi (matfree) + small least-squares solve.
        # mu shifts A's spectrum away from zero (regularization).
        # custom_vjp=False uses standard JAX backprop, supports any order of
        # differentiation. jax.checkpoint avoids storing K * state_dim per
        # column persistently across the scan: forward saves only the solution,
        # backward recomputes the GMRES on demand.
        def A_fn(v: jax.Array) -> jax.Array:
            return (2.0 + mu) * v - jvp_Jt(v)

        arnoldi = matfree_decomp.hessenberg(num_arnoldi_iters, reortho="full", custom_vjp=False)

        rhs = influence_tensor + dhdp  # (state_dim, param_dim)

        @jax.checkpoint
        def solve_column(rhs_col: jax.Array, x0_col: jax.Array) -> jax.Array:
            # Initial residual
            r0 = rhs_col - A_fn(x0_col)
            # Arnoldi factorization of A starting from r0
            result = arnoldi(A_fn, r0)
            Q = result.Q_tall  # (state_dim, k)
            H = result.J_small  # (k, k)
            beta = 1.0 / result.init_length_inv  # ||r0||
            h_kp1_k = jnp.linalg.norm(result.residual)  # h_{k+1,k}
            k = H.shape[0]
            # Build (k+1, k) upper Hessenberg H_bar
            H_bar = jnp.zeros((k + 1, k), dtype=H.dtype)
            H_bar = H_bar.at[:k, :].set(H)
            H_bar = H_bar.at[k, k - 1].set(h_kp1_k)
            # Least-squares: min ||H_bar y - beta e_1||
            rhs_lstsq = jnp.zeros(k + 1, dtype=H.dtype).at[0].set(beta)
            y, _, _, _ = jnp.linalg.lstsq(H_bar, rhs_lstsq)
            return x0_col + Q @ y

        return eqx.filter_vmap(solve_column, in_axes=(1, 1), out_axes=1)(rhs, influence_tensor)

    return rtrl_like(args, update_tensor, rtrl_config.start_at_step)


def uoro[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: UOROConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    std = config.std
    match config.distribution:
        case "uniform":
            distribution = lambda key, shape: jax.random.uniform(key, shape, minval=-std, maxval=std)
        case "normal":
            distribution = lambda key, shape: jax.random.normal(key, shape) * std

    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        key, env = args.learn_interface.take_prng(env)
        mu = config.damping
        beta = config.beta
        uoro_state = args.learn_interface.get_uoro_state(env)
        A_s = uoro_state.A
        B_s = uoro_state.B
        s = args.learn_interface.get_state(env)
        p = args.learn_interface.get_param(env)
        random_vector = distribution(key, s.shape)

        # state_fn from get_forward_mode returns (state, None); UORO wants just state for JVP
        state_only = lambda x: state_fn(x)[0]
        match config.rtrl_config.use_finite_hvp:
            case None:
                damped_jvp = eqx.filter_jvp(state_only, (s,), (A_s.value,))[1] - mu * A_s.value
            case eps:
                damped_jvp = finite_difference_jvp(state_only, s, A_s.value, eps) - mu * A_s.value
        A_propagated = beta * damped_jvp + (1 - beta) * A_s.value
        _, vjp_func, (arr, trans_stat) = eqx.filter_vjp(param_fn, p, has_aux=True)
        (immediateInfluence__random_projection,) = vjp_func(random_vector)
        scaled_immediate = beta * immediateInfluence__random_projection
        new_env = eqx.combine(arr, static)

        rho0 = jnp.sqrt(optax.safe_norm(B_s.value, 1e-12) / optax.safe_norm(A_propagated, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(scaled_immediate, 1e-12) / optax.safe_norm(random_vector, 1e-12))

        A_new: jax.Array = rho0 * A_propagated + rho1 * random_vector
        B_new: jax.Array = B_s.value / rho0 + scaled_immediate / rho1

        new_env = args.learn_interface.put_uoro_state(
            new_env,
            UOROState(A=A_s.set(value=A_new), B=B_s.set(value=B_new)),
        )
        if args.track_logs.influence_tensor_norm:
            influence_tensor_norm = jnp.linalg.norm(A_new) * jnp.linalg.norm(B_new)
            new_env = args.learn_interface.put_logs(new_env, Logs(influence_tensor_norm=influence_tensor_norm))
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        uoro_state = learn_interface.get_uoro_state(env)
        A = uoro_state.A.value
        B = uoro_state.B.value
        return (credit_gr[..., : A.shape[0]] @ A) * B + credit_gr[..., A.shape[0] :]

    return get_forward_mode(args, update_influence, credit_gr_fn)


def rflo[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA],
    config: RFLOConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        env: ENV,
    ) -> jax.Array:
        mu = config.damping
        beta = config.beta
        alpha = args.learn_interface.get_time_constant(env).value
        naive = (1 - alpha) * influence_tensor + dhdp - mu * influence_tensor
        return beta * naive + (1 - beta) * influence_tensor

    return rtrl_like(args, update_tensor, config.rtrl_config.start_at_step)


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
            case MidpointRTRLConfig():
                fn = midpoint_rtrl(
                    transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs
                )
            case HeunRTRLConfig():
                fn = heun_rtrl(
                    transition, readout_gr, interface, method, model_grad_config, length, lambda f: f, track_logs
                )
            case ImplicitEulerRTRLConfig():
                fn = implicit_euler_rtrl(
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
            case MidpointRTRLConfig():
                grad_fn = midpoint_rtrl(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case HeunRTRLConfig():
                grad_fn = heun_rtrl(
                    composed_inner,
                    readout_gr,
                    nest_interface,
                    method,
                    grad_config,
                    length,
                    vmap_this,
                    track_logs,
                )
            case ImplicitEulerRTRLConfig():
                grad_fn = implicit_euler_rtrl(
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
