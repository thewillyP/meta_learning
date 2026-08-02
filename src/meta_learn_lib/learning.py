import copy
import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from matfree import decomp as matfree_decomp

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.create_env import env_resetters, env_validation_resetters, make_reset_checker, make_tick_advancer
from meta_learn_lib.env import Logs, Outputs
from meta_learn_lib.interface import *
from meta_learn_lib.loss_function import create_objective_fns
from meta_learn_lib.lib_types import *
from meta_learn_lib.constants import *
from meta_learn_lib.optimizer import get_opt_step, make_grad_transform
from meta_learn_lib.util import *


def process_gradient[ENV](
    grad: GRADIENT, grad_config: GradientConfig, interface: GodInterface[ENV], env: ENV
) -> tuple[GRADIENT, ENV]:
    tx = make_grad_transform(grad_config)
    state = interface.opt_state.get(env)
    g, new_state = tx.update(grad, state)
    env = interface.opt_state.put(env, new_state)
    return GRADIENT(g), env


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
class LearningArg[ENV, TR_DATA, VL_DATA, READOUT]:
    transition: Callable[[ENV, TR_DATA], tuple[ENV, STAT]]
    readout: Callable[[ENV, VL_DATA], tuple[ENV, READOUT, STAT]]
    # The f/l split for Gauss-Newton. readout_outputs is f (env -> Outputs), objective is l
    # (Outputs -> scalar). None on learners where GN is unsupported (the optimizer_learner).
    readout_outputs: Optional[Callable[[ENV, VL_DATA], tuple[ENV, Outputs]]]
    objective: Optional[Callable[[ENV, Outputs, VL_DATA], LOSS]]
    learn_interface: GodInterface[ENV]
    grad_config: GradientConfig
    length: int
    vmap_this: Callable[
        [Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[READOUT, STAT]]]],
        Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, tuple[READOUT, STAT]]],
    ]
    track_logs: TrackLogs
    scan_tag: Tag
    log_prefix: str


def influence_column_norm_stats(
    column_norms: jax.Array,
    layout: list[tuple[str, int]],
    prefix: str,
) -> STAT:
    """Split a (param_dim,) vector of per-column norms into one named scalar per parameter leaf."""
    stat: STAT = {}
    start = 0
    for name, size in layout:
        block = column_norms[start : start + size]
        stat[f"{prefix}/influence_column_norm/{name}"] = scalar(jax.lax.stop_gradient(jnp.linalg.norm(block)))
        start += size
    return stat


def influence_column_cosine_stats(
    new_tensor: jax.Array,
    old_tensor: jax.Array,
    layout: list[tuple[str, int]],
    prefix: str,
) -> STAT:
    """Per-parameter cosine similarity between consecutive influence tensors.
    cos = ⟨J_t^{[block]}, J_{t-1}^{[block]}⟩ / (‖·‖·‖·‖). 1.0 = same direction, 0 = orthogonal, -1 = flipped."""
    stat: STAT = {}
    start = 0
    for name, size in layout:
        new_block = new_tensor[:, start : start + size].reshape(-1)
        old_block = old_tensor[:, start : start + size].reshape(-1)
        new_norm = jnp.linalg.norm(new_block)
        old_norm = jnp.linalg.norm(old_block)
        denom = jnp.maximum(new_norm * old_norm, 1e-30)
        cos = jnp.dot(new_block, old_block) / denom
        stat[f"{prefix}/influence_column_cosine/{name}"] = scalar(jax.lax.stop_gradient(cos))
        start += size
    return stat


def parameter_norm_stats(param: jax.Array, layout: list[tuple[str, int]], prefix: str) -> STAT:
    stat: STAT = {}
    start = 0
    for name, size in layout:
        block = param[start : start + size]
        stat[f"{prefix}/parameter_norm/{name}"] = scalar(jax.lax.stop_gradient(jnp.linalg.norm(block)))
        start += size
    return stat


def influence_column_value_stats(
    values: jax.Array,
    layout: list[tuple[str, int]],
    prefix: str,
    name: str,
) -> STAT:
    """Split a per-column diagnostic (broadcastable to (1, param_dim)) into one named scalar per parameter leaf."""
    stat: STAT = {}
    total = sum(size for _, size in layout)
    vals = jnp.broadcast_to(values, (1, total)).reshape(-1)
    start = 0
    for pname, size in layout:
        block = vals[start : start + size]
        stat[f"{prefix}/{name}/{pname}"] = scalar(jax.lax.stop_gradient(jnp.mean(block)))
        start += size
    return stat


def lanczos_ritz(matvec: Callable[[jax.Array], jax.Array], init_vec: jax.Array, num_matvecs: int) -> jax.Array:
    lanczos = matfree_decomp.tridiag_sym(num_matvecs, reortho="full", custom_vjp=False)
    result = lanczos(matvec, init_vec)
    return jnp.linalg.eigvalsh(result.J_small)


def lanczos_ritz_pairs(
    matvec: Callable[[jax.Array], jax.Array], init_vec: jax.Array, num_matvecs: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    lanczos = matfree_decomp.tridiag_sym(num_matvecs, reortho="full", custom_vjp=False)
    result = lanczos(matvec, init_vec)
    theta, small_vecs = jnp.linalg.eigh(result.J_small)
    ritz_vectors = result.Q_tall @ small_vecs
    residuals = jnp.linalg.norm(result.residual) * jnp.abs(small_vecs[-1, :])
    return theta, ritz_vectors, residuals


def spectral_matvec(
    state_fn: Callable[[jax.Array], tuple[jax.Array, None]], s: jax.Array, use_finite_hvp: jax.Array | None
) -> Callable[[jax.Array], jax.Array]:
    match use_finite_hvp:
        case None:
            return lambda v: jvp(state_fn, s, v)[1]
        case eps:
            return lambda v: finite_difference_jvp(lambda x: state_fn(x)[0], s, v, eps)


def spectral_clip_jmp(
    clip: SpectralClip,
    matvec: Callable[[jax.Array], jax.Array],
    working: jax.Array,
    hmp_jvp: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    num_matvecs = min(clip.num_matvecs, working.shape[0])
    k_mix, k_fallback = jax.random.split(key)
    weights = jax.random.normal(k_mix, (working.shape[1],))
    seed = working @ weights
    fallback = jax.random.normal(k_fallback, (working.shape[0],))
    seed = jnp.where(jnp.linalg.norm(seed) > 1e-30, seed, fallback)
    theta, ritz_vectors, residuals = lanczos_ritz_pairs(matvec, seed, num_matvecs)
    converged = residuals < clip.residual_tol * jnp.maximum(jnp.abs(theta), 1e-30)
    match clip.ends:
        case "both":
            explosive = jnp.abs(theta) > clip.margin
        case "negative":
            explosive = theta < -clip.margin
    mask = converged & explosive
    excess = jnp.where(mask, theta - jnp.sign(theta) * clip.margin, 0.0)
    proj = ritz_vectors.T @ working
    corrected = hmp_jvp - ritz_vectors @ (excess[:, None] * proj)
    growth = jnp.linalg.norm(corrected, axis=0) / jnp.maximum(jnp.linalg.norm(working, axis=0), 1e-30)
    k = jnp.broadcast_to(jnp.sum(mask).astype(corrected.dtype), (growth.shape[0],))
    top = jnp.broadcast_to(jnp.max(jnp.abs(theta)), (growth.shape[0],))
    diag = {
        "spectral_k": k[None, :],
        "spectral_top_ritz": top[None, :],
        "spectral_growth": growth[None, :],
    }
    return corrected, diag


def unit_circle_scale(clip: UnitCircleClip, g: jax.Array, g_ema: jax.Array) -> tuple[jax.Array, jax.Array]:
    match clip.ema_decay:
        case None:
            g_ema_new = jnp.broadcast_to(g, g_ema.shape)
        case decay:
            g_ema_new = jnp.maximum(g, decay * g_ema)
    scale = jnp.minimum(1.0, clip.margin / jnp.maximum(g_ema_new, 1e-30))
    return scale, g_ema_new


def ritz_danger_and_log(
    clip: UnitCircleClip | None,
    track_largest: bool,
    use_finite_hvp: jax.Array | None,
    state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
    s: jax.Array,
) -> tuple[jax.Array, Logs]:
    match clip:
        case UnitCircleClip(measure="eigenvalue"):
            needs_eigenvalue = True
        case _:
            needs_eigenvalue = track_largest
    if not needs_eigenvalue:
        return jnp.array(0.0), Logs()
    init_vec = jax.random.normal(jax.random.key(0), s.shape)
    match use_finite_hvp:
        case None:
            matvec = lambda v: jvp(state_fn, s, v)[1]
        case eps:
            matvec = lambda v: finite_difference_jvp(lambda x: state_fn(x)[0], s, v, eps)
    smallest = jnp.min(lanczos_ritz(matvec, init_vec, min(30, s.shape[0])))
    danger = jnp.maximum(0.0, -smallest)
    log = Logs(largest_eigenvalue=jax.lax.stop_gradient(1.0 - smallest)) if track_largest else Logs()
    return danger, log


def get_forward_mode[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
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

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            def state_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, None]]:
                def fn(state: jax.Array) -> tuple[jax.Array, None]:
                    _env = args.learn_interface.state.put(e, state)
                    _env, _ = args.transition(_env, tr_data)
                    state = args.learn_interface.state.get(_env)
                    return state, None

                return fn

            def param_fn(e: ENV) -> Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]]:
                def fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                    _env = args.learn_interface.param.put(e, param)
                    _env, stat = args.transition(_env, tr_data)
                    state = args.learn_interface.state.get(_env)
                    _arr, _ = eqx.partition(_env, eqx.is_array)
                    return state, (_arr, stat)

                return fn

            new_env, trans_stat = update_influence(state_fn(env), param_fn(env), env)
            new_env, credit_gr, readout_stat = args.readout(new_env, vl_data)
            grad = credit_gr_fn(credit_gr, args.learn_interface, new_env)

            arr, _ = eqx.partition(new_env, eqx.is_array)
            return arr, grad, trans_stat | readout_stat

        arr, grads, stats = tagged_scan(
            as_scan_body(args.vmap_this(step)),
            arr_init,
            ds,
            length=args.length,
            tag=args.scan_tag,
        )
        env = eqx.combine(arr, static)
        total_grad = GRADIENT(jnp.sum(grads, axis=tuple(range(grads.ndim - 1))))
        total_grad, env = process_gradient(total_grad, args.grad_config, args.learn_interface, env)
        return env, total_grad, stats

    return gradient_fn


def rtrl_like[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    update_tensor: Callable[
        [
            Callable[[jax.Array], tuple[jax.Array, None]],
            jax.Array,
            JACOBIAN,
            JACOBIAN,
            ENV,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        tuple[JACOBIAN, jax.Array, dict[str, jax.Array]],
    ],
    start_at_step: int,
    clip: UnitCircleClip | SpectralClip | None,
    use_finite_hvp: jax.Array | None,
    immediate_ema_decay: jax.Array | None,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.state.get(env)
        p = args.learn_interface.param.get(env)
        influence_tensor = args.learn_interface.forward_mode_jacobian.get(env)
        g_ema = args.learn_interface.unit_circle_ema.get(env)

        danger, log_fragment = ritz_danger_and_log(
            clip, args.track_logs.largest_eigenvalue, use_finite_hvp, state_fn, s
        )

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        # Low-pass the driving term B_t = dhdp before it enters the recursion; the propagation
        # (I-αG) is left untouched, so this de-noises the input without changing the dynamics.
        if immediate_ema_decay is not None:
            b_ema = args.learn_interface.immediate_ema.get(env)
            dhdp = immediate_ema_decay * b_ema + (1.0 - immediate_ema_decay) * dhdp
            new_env = args.learn_interface.immediate_ema.put(new_env, dhdp)

        match clip:
            case SpectralClip():
                key, new_env = args.learn_interface.take_prng(new_env)
            case _:
                key = jax.random.key(0)

        new_influence_tensor, g_ema_new, clip_diag = update_tensor(
            state_fn, s, dhdp, influence_tensor, env, danger, g_ema, key
        )

        new_influence_tensor = filter_cond(
            args.learn_interface.tick.get(env) >= start_at_step,
            lambda _: new_influence_tensor,
            lambda _: influence_tensor,
            None,
        )

        new_env = args.learn_interface.forward_mode_jacobian.put(new_env, new_influence_tensor)
        new_env = args.learn_interface.unit_circle_ema.put(new_env, g_ema_new)
        if args.track_logs.influence_tensor_norm:
            column_norms = jnp.linalg.norm(new_influence_tensor, axis=0)
            layout = args.learn_interface.param_layout(new_env)
            trans_stat = trans_stat | influence_column_norm_stats(column_norms, layout, args.log_prefix)
            trans_stat = trans_stat | influence_column_cosine_stats(
                new_influence_tensor, influence_tensor, layout, args.log_prefix
            )
        if args.track_logs.spectral_clip:
            layout = args.learn_interface.param_layout(new_env)
            for diag_name, diag_values in clip_diag.items():
                trans_stat = trans_stat | influence_column_value_stats(diag_values, layout, args.log_prefix, diag_name)
        new_env = args.learn_interface.merge_logs(new_env, log_fragment)
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor = learn_interface.forward_mode_jacobian.get(env)
        state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: RTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
        danger: jax.Array,
        g_ema: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        mu = config.damping
        beta = config.beta

        col_norm = jnp.linalg.norm(influence_tensor, axis=0, keepdims=True)
        safe = jnp.where(col_norm == 0.0, 1.0, col_norm)
        if config.influence_clip is None:
            down = 1.0 / safe
        else:
            down = jnp.minimum(1.0, config.influence_clip.threshold / safe)
        working = influence_tensor * down

        hmp_jvp: JACOBIAN
        match config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, working)
            case eps:
                hmp_jvp = finite_difference_jmp(lambda x: state_fn(x)[0], s, working, eps)

        match config.unit_circle_clip:
            case None:
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case SpectralClip() as clip:
                matvec = spectral_matvec(state_fn, s, config.use_finite_hvp)
                hmp_jvp, clip_diag = spectral_clip_jmp(clip, matvec, working, hmp_jvp, key)
                scale, g_ema_new = jnp.array(1.0), g_ema
            case UnitCircleClip(measure="growth") as clip:
                growth = jnp.linalg.norm(hmp_jvp, axis=0, keepdims=True) / jnp.maximum(
                    jnp.linalg.norm(working, axis=0, keepdims=True), 1e-30
                )
                scale, g_ema_new = unit_circle_scale(clip, growth, g_ema)
                clip_diag = {"unit_circle_scale": scale, "unit_circle_growth": growth}
            case UnitCircleClip(measure="eigenvalue") as clip:
                scale, g_ema_new = unit_circle_scale(clip, danger, g_ema)
                clip_diag = {"unit_circle_scale": scale}
        hmp = scale * hmp_jvp / down - mu * influence_tensor

        if config.propagation_clip is not None:
            hmp_norm = jnp.linalg.norm(hmp, axis=0, keepdims=True)
            alpha = jnp.minimum(1.0, config.propagation_clip / jnp.where(hmp_norm == 0.0, 1.0, hmp_norm))
            hmp = alpha * hmp
            dhdp = alpha * dhdp

        updated = beta * (hmp + dhdp) + (1 - beta) * influence_tensor
        return updated, g_ema_new, clip_diag

    return rtrl_like(
        args,
        update_tensor,
        config.start_at_step,
        config.unit_circle_clip,
        config.use_finite_hvp,
        config.immediate_ema_decay,
    )


def tikhonov_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: TikhonovRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
        danger: jax.Array,
        g_ema: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        mu = config.rtrl_config.damping
        beta = config.rtrl_config.beta

        match config.rtrl_config.unit_circle_clip:
            case None:
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case SpectralClip():
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case UnitCircleClip() as clip:
                scale, g_ema_new = unit_circle_scale(clip, danger, g_ema)
                clip_diag = {"unit_circle_scale": scale}

        hmp_jvp: JACOBIAN
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor * scale)
            case eps:
                hmp_jvp = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor * scale, eps)

        match config.rtrl_config.unit_circle_clip:
            case SpectralClip() as clip:
                matvec = spectral_matvec(state_fn, s, config.rtrl_config.use_finite_hvp)
                key_forward, key_correction = jax.random.split(key)
                hmp_jvp, clip_diag = spectral_clip_jmp(clip, matvec, influence_tensor, hmp_jvp, key_forward)

        d_tau = hmp_jvp + dhdp
        error = influence_tensor - d_tau

        _, vjp_fn = eqx.filter_vjp(lambda x: state_fn(x)[0], s)
        correction = eqx.filter_vmap(lambda col: vjp_fn(col)[0], in_axes=1, out_axes=1)(error)
        match config.rtrl_config.unit_circle_clip:
            case SpectralClip() as clip:
                correction, _ = spectral_clip_jmp(clip, matvec, error, correction, key_correction)

        target = d_tau + correction - mu * influence_tensor
        updated = beta * target + (1 - beta) * influence_tensor
        return updated, g_ema_new, clip_diag

    return rtrl_like(
        args,
        update_tensor,
        config.rtrl_config.start_at_step,
        config.rtrl_config.unit_circle_clip,
        config.rtrl_config.use_finite_hvp,
        config.rtrl_config.immediate_ema_decay,
    )


def pade_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: PadeRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        _env: ENV,
        danger: jax.Array,
        g_ema: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        match config.rtrl_config.unit_circle_clip:
            case None:
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case SpectralClip():
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case UnitCircleClip() as clip:
                scale, g_ema_new = unit_circle_scale(clip, danger, g_ema)
                clip_diag = {"unit_circle_scale": scale}

        # JVP 1: dF/dz @ Gamma (for D_tau)
        # JVP 2: dF/dz @ dF/dphi (extra cost for Pade)
        hmp_jvp: JACOBIAN
        dhdz_dhdp: JACOBIAN
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp, _aux = jacobian_matrix_product(state_fn, s, influence_tensor * scale)
                _primals2, dhdz_dhdp, _aux2 = jacobian_matrix_product(state_fn, s, dhdp)
            case eps:
                hmp_jvp = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor * scale, eps)
                dhdz_dhdp = finite_difference_jmp(lambda x: state_fn(x)[0], s, dhdp, eps)

        # Pade [1,1]: Gamma_{t+1} = 1/2 * D_tau + 1/2 * (I + dF/dz) * dF/dphi
        match config.rtrl_config.unit_circle_clip:
            case SpectralClip() as clip:
                matvec = spectral_matvec(state_fn, s, config.rtrl_config.use_finite_hvp)
                key_gamma, key_driver = jax.random.split(key)
                hmp_jvp, clip_diag = spectral_clip_jmp(clip, matvec, influence_tensor, hmp_jvp, key_gamma)
                dhdz_dhdp, _ = spectral_clip_jmp(clip, matvec, dhdp, dhdz_dhdp, key_driver)

        d_tau = hmp_jvp + dhdp
        return 0.5 * d_tau + 0.5 * (dhdp + dhdz_dhdp), g_ema_new, clip_diag

    return rtrl_like(
        args,
        update_tensor,
        config.rtrl_config.start_at_step,
        config.rtrl_config.unit_circle_clip,
        config.rtrl_config.use_finite_hvp,
        config.rtrl_config.immediate_ema_decay,
    )


def midpoint_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: MidpointRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.state.get(env)
        p = args.learn_interface.param.get(env)
        influence_tensor = args.learn_interface.forward_mode_jacobian.get(env)
        mb = args.learn_interface.midpoint_buffer.get(env)

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        # JVP 1: J_t @ P_t (for next step's predictor)
        # JVP 2: J_t @ predictor (for corrector of interval ending at t+1)
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, hmp_jvp_current, _aux = jacobian_matrix_product(state_fn, s, influence_tensor)
                _primals2, hmp_jvp_predictor, _aux2 = jacobian_matrix_product(state_fn, s, mb.predictor)
            case eps:
                hmp_jvp_current = finite_difference_jmp(lambda x: state_fn(x)[0], s, influence_tensor, eps)
                hmp_jvp_predictor = finite_difference_jmp(lambda x: state_fn(x)[0], s, mb.predictor, eps)
        # Forward Euler from P_t (used as bootstrap and as next step's predictor)
        fe_update = hmp_jvp_current + dhdp
        # Midpoint corrector: P_new = P_prev + 2*((J_t - I) @ predictor + B_t)
        midpoint_update = mb.P_prev + 2 * (hmp_jvp_predictor - mb.predictor + dhdp)
        # First active step: forward Euler bootstrap. Subsequent: midpoint corrector.
        is_bootstrap = args.learn_interface.tick.get(env) <= config.rtrl_config.start_at_step
        active_update = jnp.where(is_bootstrap, fe_update, midpoint_update)
        new_influence_tensor: JACOBIAN = filter_cond(
            args.learn_interface.tick.get(env) >= config.rtrl_config.start_at_step,
            lambda _: active_update,
            lambda _: influence_tensor,
            None,
        )

        new_P_prev = filter_cond(
            args.learn_interface.tick.get(env) >= config.rtrl_config.start_at_step,
            lambda _: influence_tensor,
            lambda _: mb.P_prev,
            None,
        )

        new_predictor = filter_cond(
            args.learn_interface.tick.get(env) >= config.rtrl_config.start_at_step,
            lambda _: fe_update,
            lambda _: mb.predictor,
            None,
        )

        new_env = args.learn_interface.forward_mode_jacobian.put(new_env, new_influence_tensor)
        new_env = args.learn_interface.midpoint_buffer.put(
            new_env,
            MidpointBuffer(P_prev=new_P_prev, predictor=new_predictor),
        )

        if args.track_logs.influence_tensor_norm:
            readout_tensor = 0.5 * (new_influence_tensor + influence_tensor)
            column_norms = jnp.linalg.norm(readout_tensor, axis=0)
            layout = args.learn_interface.param_layout(new_env)
            trans_stat = trans_stat | influence_column_norm_stats(column_norms, layout, args.log_prefix)
            trans_stat = trans_stat | influence_column_cosine_stats(
                new_influence_tensor, influence_tensor, layout, args.log_prefix
            )

        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor = learn_interface.forward_mode_jacobian.get(env)
        # 2-tap boxcar: average P_t and P_{t-1} for readout to kill parity mode
        readout_tensor = 0.5 * (influence_tensor + learn_interface.midpoint_buffer.get(env).P_prev)
        state_jacobian = jnp.vstack([readout_tensor, jnp.eye(readout_tensor.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def heun_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: HeunRTRLConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    rtrl_config = config.rtrl_config

    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        s = args.learn_interface.state.get(env)
        p = args.learn_interface.param.get(env)
        t = args.learn_interface.tick.get(env)
        influence_tensor = args.learn_interface.forward_mode_jacobian.get(env)
        mb = args.learn_interface.midpoint_buffer.get(env)

        dhdp, new_env, trans_stat = compute_dhdp(param_fn, s, p, static)

        # JVP 1: J_t @ predictor_stored (corrector slope at current tick)
        predictor_stored = mb.predictor
        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals, corrector_jvp, _aux = jacobian_matrix_product(state_fn, s, predictor_stored)
            case eps:
                f = lambda x: state_fn(x)[0]
                corrector_jvp = finite_difference_jmp(f, s, predictor_stored, eps)

        # Heun: P_new = 0.5 * (P_prev + J_t @ predictor + B_t)
        heun_update = 0.5 * (influence_tensor + corrector_jvp + dhdp)

        # Forward Euler (bootstrap, first active step only)
        # At bootstrap, predictor_stored == P_prev, so corrector_jvp == J_t @ P_prev
        fe_update = corrector_jvp + dhdp

        is_bootstrap = t <= rtrl_config.start_at_step
        active_update = jnp.where(is_bootstrap, fe_update, heun_update)

        new_influence_tensor: JACOBIAN
        new_influence_tensor = filter_cond(
            t >= rtrl_config.start_at_step,
            lambda _: active_update,
            lambda _: influence_tensor,
            None,
        )

        match config.rtrl_config.use_finite_hvp:
            case None:
                _primals2, pred_jvp, _aux2 = jacobian_matrix_product(state_fn, s, new_influence_tensor)
            case eps:
                f = lambda x: state_fn(x)[0]
                pred_jvp = finite_difference_jmp(f, s, new_influence_tensor, eps)

        new_predictor = filter_cond(
            t >= rtrl_config.start_at_step,
            lambda _: pred_jvp + dhdp,
            lambda _: mb.predictor,
            None,
        )

        new_env = args.learn_interface.forward_mode_jacobian.put(new_env, new_influence_tensor)
        new_env = args.learn_interface.midpoint_buffer.put(
            new_env,
            MidpointBuffer(P_prev=mb.P_prev, predictor=new_predictor),
        )
        if args.track_logs.influence_tensor_norm:
            column_norms = jnp.linalg.norm(new_influence_tensor, axis=0)
            layout = args.learn_interface.param_layout(new_env)
            trans_stat = trans_stat | influence_column_norm_stats(column_norms, layout, args.log_prefix)
            trans_stat = trans_stat | influence_column_cosine_stats(
                new_influence_tensor, influence_tensor, layout, args.log_prefix
            )
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        influence_tensor = learn_interface.forward_mode_jacobian.get(env)
        state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
        grad = credit_gr @ state_jacobian
        return grad

    return get_forward_mode(args, update_influence, credit_gr_fn)


def implicit_euler_rtrl[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
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
        danger: jax.Array,
        g_ema: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        mu = rtrl_config.damping

        match rtrl_config.unit_circle_clip:
            case None:
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case SpectralClip():
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case UnitCircleClip() as clip:
                scale, g_ema_new = unit_circle_scale(clip, danger, g_ema)
                clip_diag = {"unit_circle_scale": scale}

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

        rhs = influence_tensor * scale + dhdp  # (state_dim, param_dim)

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

        solved = eqx.filter_vmap(solve_column, in_axes=(1, 1), out_axes=1)(rhs, influence_tensor)
        match rtrl_config.unit_circle_clip:
            case SpectralClip() as clip:

                def phi_clip_column(rhs_col: jax.Array, p_col: jax.Array) -> jax.Array:
                    rnd = jax.random.normal(jax.random.key(0), rhs_col.shape)
                    norm = jnp.linalg.norm(rhs_col)
                    seed = jnp.where(norm > 1e-12, rhs_col + 1e-3 * norm * rnd, rnd)
                    theta, ritz_vectors, residuals = lanczos_ritz_pairs(
                        jvp_Jt, seed, min(clip.num_matvecs, rhs_col.shape[0])
                    )
                    phi = 1.0 / (2.0 + mu - theta)
                    converged = residuals < clip.residual_tol * jnp.maximum(jnp.abs(theta), 1e-30)
                    match clip.ends:
                        case "both":
                            explosive = jnp.abs(phi) > clip.margin
                        case "negative":
                            explosive = phi < -clip.margin
                    mask = converged & explosive
                    excess = jnp.where(mask, phi - jnp.sign(phi) * clip.margin, 0.0)
                    return p_col - ritz_vectors @ (excess * (ritz_vectors.T @ rhs_col))

                solved = eqx.filter_vmap(phi_clip_column, in_axes=(1, 1), out_axes=1)(rhs, solved)
        return solved, g_ema_new, clip_diag

    return rtrl_like(
        args,
        update_tensor,
        rtrl_config.start_at_step,
        rtrl_config.unit_circle_clip,
        rtrl_config.use_finite_hvp,
        rtrl_config.immediate_ema_decay,
    )


def uoro[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
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
        uoro_st = args.learn_interface.uoro_state.get(env)
        A = uoro_st.A
        B = uoro_st.B
        s = args.learn_interface.state.get(env)
        p = args.learn_interface.param.get(env)
        random_vector = distribution(key, s.shape)

        state_only = lambda x: state_fn(x)[0]
        match config.rtrl_config.use_finite_hvp:
            case None:
                damped_jvp = eqx.filter_jvp(state_only, (s,), (A,))[1] - mu * A
            case eps:
                damped_jvp = finite_difference_jvp(state_only, s, A, eps) - mu * A
        A_propagated = beta * damped_jvp + (1 - beta) * A
        _, vjp_func, (arr, trans_stat) = eqx.filter_vjp(param_fn, p, has_aux=True)
        (immediateInfluence__random_projection,) = vjp_func(random_vector)
        scaled_immediate = beta * immediateInfluence__random_projection
        new_env = eqx.combine(arr, static)

        rho0 = jnp.sqrt(optax.safe_norm(B, 1e-12) / optax.safe_norm(A_propagated, 1e-12))
        rho1 = jnp.sqrt(optax.safe_norm(scaled_immediate, 1e-12) / optax.safe_norm(random_vector, 1e-12))

        A_new: jax.Array = rho0 * A_propagated + rho1 * random_vector
        B_new: jax.Array = B / rho0 + scaled_immediate / rho1

        new_env = args.learn_interface.uoro_state.put(new_env, UOROState(A=A_new, B=B_new))
        if args.track_logs.influence_tensor_norm:
            # Rank-1 estimate P ≈ A Bᵀ, so column j of P is A * B_j and ‖P[:,j]‖ = ‖A‖·|B_j|.
            column_norms = jnp.linalg.norm(A_new) * jnp.abs(B_new)
            trans_stat = trans_stat | influence_column_norm_stats(
                column_norms, args.learn_interface.param_layout(new_env), args.log_prefix
            )
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        uoro_st = learn_interface.uoro_state.get(env)
        A = uoro_st.A
        B = uoro_st.B
        return (credit_gr[..., : A.shape[0]] @ A) * B + credit_gr[..., A.shape[0] :]

    return get_forward_mode(args, update_influence, credit_gr_fn)


def rflo[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    config: RFLOConfig,
    hyperparameters: dict[HP, HyperparameterConfig],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    tc_forward, _ = hyperparameter_reparametrization(
        hyperparameters[config.time_constant].hyperparameter_parametrization
    )

    def update_tensor(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        s: jax.Array,
        dhdp: jax.Array,
        influence_tensor: jax.Array,
        env: ENV,
        danger: jax.Array,
        g_ema: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        mu = config.damping
        beta = config.beta
        match config.rtrl_config.unit_circle_clip:
            case None:
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case SpectralClip():
                scale, g_ema_new, clip_diag = jnp.array(1.0), g_ema, {}
            case UnitCircleClip() as clip:
                scale, g_ema_new = unit_circle_scale(clip, danger, g_ema)
                clip_diag = {"unit_circle_scale": scale}
        alpha = tc_forward(args.learn_interface.time_constant.get(env))
        naive = (1 - alpha) * influence_tensor * scale + dhdp - mu * influence_tensor
        return beta * naive + (1 - beta) * influence_tensor, g_ema_new, clip_diag

    return rtrl_like(
        args,
        update_tensor,
        config.rtrl_config.start_at_step,
        config.rtrl_config.unit_circle_clip,
        config.rtrl_config.use_finite_hvp,
        config.rtrl_config.immediate_ema_decay,
    )


def immediate[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def update_influence(
        state_fn: Callable[[jax.Array], tuple[jax.Array, None]],
        param_fn: Callable[[jax.Array], tuple[jax.Array, tuple[ENV, STAT]]],
        env: ENV,
    ) -> tuple[ENV, STAT]:
        _, static = eqx.partition(env, eqx.is_array)
        p = args.learn_interface.param.get(env)
        _state, (arr, trans_stat) = param_fn(p)
        new_env = eqx.combine(arr, static)
        return new_env, trans_stat

    def credit_gr_fn(credit_gr: GRADIENT, learn_interface: GodInterface[ENV], env: ENV) -> GRADIENT:
        n_s = learn_interface.state.get(env).shape[0]
        return GRADIENT(credit_gr[..., n_s:])

    return get_forward_mode(args, update_influence, credit_gr_fn)


def get_backward_mode[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    truncate_at: Optional[int],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]:
    def loss_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, LOSS, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def inference_fn(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, LOSS, STAT]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            if truncate_at is not None:
                t = args.learn_interface.tick.get(env)
                s = filter_cond(
                    t % truncate_at == 0,
                    lambda _: jax.lax.stop_gradient(args.learn_interface.state.get(env)),
                    lambda _: args.learn_interface.state.get(env),
                    None,
                )
                env = args.learn_interface.state.put(env, s)

            env, trans_stat = args.transition(env, tr_data)
            env, loss, readout_stat = args.readout(env, vl_data)
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, loss, trans_stat | readout_stat

        arr, losses, stats = tagged_scan(
            as_scan_body(args.vmap_this(inference_fn)),
            arr_init,
            ds_init,
            length=args.length,
            tag=args.scan_tag,
        )
        env = eqx.combine(arr, static)
        return env, jnp.sum(losses), stats

    return loss_fn


def get_backward_mode_with_grad[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    truncate_at: Optional[int],
    loss_to_grad: Callable[
        [
            Callable[[jax.Array, tuple[TR_DATA, VL_DATA]], tuple[LOSS, tuple[ENV, STAT]]],
            jax.Array,
            tuple[TR_DATA, VL_DATA],
        ],
        tuple[jax.Array, tuple[ENV, STAT]],
    ],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    _loss_fn = get_backward_mode(args, truncate_at)

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        param = args.learn_interface.param.get(env_init)

        def loss_fn(p: jax.Array, ds: tuple[TR_DATA, VL_DATA]) -> tuple[LOSS, tuple[ENV, STAT]]:
            env_with_p = args.learn_interface.param.put(env_init, p)
            env, loss, stats = _loss_fn(env_with_p, ds)
            env = args.learn_interface.param.put(env, p)
            return loss, (env, stats)

        grad, (env, stats) = loss_to_grad(loss_fn, param, ds_init)
        env = args.learn_interface.param.put(env, param)
        grad, env = process_gradient(GRADIENT(grad), args.grad_config, args.learn_interface, env)
        return env, grad, stats

    return gradient_fn


def get_outputs_mode[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    truncate_at: Optional[int],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, Outputs, STAT]]:
    """Like get_backward_mode, but scans args.readout_outputs (f: env -> Outputs) and stacks the
    per-step Outputs along the scan axis instead of summing the loss. This is the `f` of the
    Gauss-Newton split, run over the full trajectory."""

    def outputs_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, Outputs, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def inference_fn(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, Outputs, STAT]:
            env = eqx.combine(arr, static)
            tr_data, vl_data = data

            if truncate_at is not None:
                t = args.learn_interface.tick.get(env)
                s = filter_cond(
                    t % truncate_at == 0,
                    lambda _: jax.lax.stop_gradient(args.learn_interface.state.get(env)),
                    lambda _: args.learn_interface.state.get(env),
                    None,
                )
                env = args.learn_interface.state.put(env, s)

            env, trans_stat = args.transition(env, tr_data)
            env, outputs = args.readout_outputs(env, vl_data)
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, outputs, trans_stat

        arr, outputs, stats = tagged_scan(
            as_scan_body(args.vmap_this(inference_fn)),
            arr_init,
            ds_init,
            length=args.length,
            tag=args.scan_tag,
        )
        env = eqx.combine(arr, static)
        return env, outputs, stats

    return outputs_fn


def ggn_vector_product(f, l, theta, v, env, ds):
    """Gauss-Newton vector product F_theta v = J_fᵀ H_l J_f v, where f(theta,env,ds) -> z and
    l(z,env,ds) -> scalar. A plain matvec at the current theta (no stop_gradient — unlike
    gauss_newton_value_and_grad, nothing differentiates this further; it's evaluated at the top
    meta level). env, ds are held fixed (closed over) since the meta step is the outermost level."""
    f_t = lambda t: f(t, env, ds)
    z, Jv = eqx.filter_jvp(f_t, (theta,), (v,))
    grad_l = lambda zz: eqx.filter_grad(lambda o: l(o, env, ds))(zz)
    _, HlJv = eqx.filter_jvp(grad_l, (z,), (Jv,))
    HlJv = eqx.combine(HlJv, jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else None, z))
    _, vjp_f = eqx.filter_vjp(f_t, theta)
    return vjp_f(HlJv)[0]


def natural_gradient_step(grad_phi, gamma, f, l, theta, env, ds, damping):
    """Gauss-Newton / natural-gradient step in hyperparameter space, reusing the RTRL influence
    tensor. grad_phi: (P,) meta-gradient (= gammaᵀ g_theta). gamma: (T, P) = ∂theta/∂phi (the
    influence tensor columns for the P target hyperparameters). f/l: the validation readout
    (theta -> z) and objective (z -> scalar). Returns (H_phi + damping·I)⁻¹ grad_phi with the
    pushforward Gauss-Newton curvature H_phi = gammaᵀ J_valᵀ H_l J_val gamma = Kᵀ A K (K = J_val gamma),
    which is PSD. Near an instability gamma blows up, so H_phi ~ ‖gamma‖² blows up faster than
    grad_phi ~ ‖gamma‖, and the step ~ 1/‖gamma‖ self-brakes."""
    fvp = lambda v: ggn_vector_product(f, l, theta, v, env, ds)
    w = eqx.filter_vmap(fvp, in_axes=1, out_axes=1)(gamma)  # (T, P) = F_theta gamma
    h_phi = gamma.T @ w  # (P, P), PSD
    p_dim = grad_phi.shape[0]
    return jnp.linalg.solve(h_phi + damping * jnp.eye(p_dim, dtype=h_phi.dtype), grad_phi)


def gauss_newton_value_and_grad(f, l, p, env, ds):
    """Gradient of L = l(f(p, env, ds), env, ds) w.r.t. p whose value is the exact gradient
    J_f^T grad_l(z) but whose *derivative* is the generalized Gauss-Newton matrix G = J_f^T H_l J_f
    (J_f = ∂f/∂p) — i.e. it drops the grad_l-weighted f'' term the true Hessian has. G is symmetric
    PSD whenever H_l is PSD (l convex in z).

    Built by the stop-gradient linearization identity: with the Jacobian/vjp points held at their
    stop-gradient'd value, g = J_0^T grad_l( f_0 + J_0 (delta_p, delta_env) ), where delta_p = p - sg p
    and delta_env = env - sg env carry whatever outer tangents reach p and the inner state in env. The
    linearization point is frozen so no grad_l-weighted f'' term enters, but the deltas keep BOTH p's
    and the inner state's first-order paths, so the derivative is the generalized Gauss-Newton of the
    FULL differentiated state: d g/d p = J_p^T H_l J_p and d g/d(state) = J_p^T H_l J_state. This holds
    for any architecture — a recurrent f's J_0 already includes the BPTT unroll, and the recurrent
    hidden state in env contributes its coupling through delta_env (nothing is feedforward-specific).
    Plain autodiff (no custom_jvp), so it composes with both a forward-mode outer (RTRL) and a
    reverse-mode outer (BPTT-over-BPTT). env and ds are passed explicitly (never closed over) so the
    internal jvp/vjp never capture outer tracers."""
    sg_leaf = lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x
    sg = lambda t: jax.tree.map(sg_leaf, t)
    p_sg, env_sg, ds_sg = jax.lax.stop_gradient(p), sg(env), sg(ds)
    # Deltas are numerically zero but carry the outer tangents on p and on the inner state in env.
    delta_p = p - p_sg
    delta_env = jax.tree.map(lambda x: (x - jax.lax.stop_gradient(x)) if eqx.is_inexact_array(x) else None, env)
    none_env = jax.tree.map(lambda _: None, env_sg)
    none_ds = jax.tree.map(lambda _: None, ds_sg)

    z0, jvp_delta = eqx.filter_jvp(f, (p_sg, env_sg, ds_sg), (delta_p, delta_env, none_ds))
    z_lin = jax.tree.map(lambda a, b: a + b, z0, jvp_delta)  # f_0 + J_0 (delta_p, delta_env)
    grad_l = eqx.filter_grad(lambda o: l(o, env_sg, ds_sg))(z_lin)
    _, vjp_f = eqx.filter_vjp(f, p_sg, env_sg, ds_sg)
    return vjp_f(grad_l)[0]


def gauss_newton_backward[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    truncate_at: Optional[int],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    """Inner gradient g(theta) = J_f^T grad_l(f(theta)) whose value is identical to exact mode but whose
    *derivative* (as seen by an outer learner) is the generalized Gauss-Newton G = J_f^T H_l J_f instead
    of the true Hessian. Delegated to gauss_newton_value_and_grad (stop-gradient linearization, no
    custom_jvp), so it works under both a forward-mode outer (RTRL) and a reverse-mode outer (BPTT).
    f scans transition + per-step readout (theta -> stacked Outputs, full BPTT unroll for recurrent
    models); l sums the per-step objective over those Outputs.

    GN only takes effect when the outer learner differentiates this with exact autodiff: a finite-diff
    outer (use_finite_hvp) evaluates the inner step at perturbed primals and never goes through this
    linearization, so it would recover the true Hessian instead."""
    forward = get_backward_mode(args, truncate_at)
    outputs_scan = get_outputs_mode(args, truncate_at)

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        param = args.learn_interface.param.get(env_init)

        def f(p: jax.Array, env: ENV, ds: tuple[TR_DATA, VL_DATA]) -> Outputs:
            env_p = args.learn_interface.param.put(env, p)
            _, outputs, _ = outputs_scan(env_p, ds)
            return outputs

        def l(outputs: Outputs, env: ENV, ds: tuple[TR_DATA, VL_DATA]) -> jax.Array:
            _, vl = ds

            def body(acc: jax.Array, out_t__d_t) -> tuple[jax.Array, None]:
                out_t, d_t = out_t__d_t
                return acc + args.objective(env, out_t, d_t), None

            total, _ = jax.lax.scan(body, jnp.array(0.0), (outputs, vl))
            return total

        grad = gauss_newton_value_and_grad(f, l, param, env_init, ds_init)
        env, _, stats = forward(env_init, ds_init)
        env = args.learn_interface.param.put(env, param)
        grad, env = process_gradient(GRADIENT(grad), args.grad_config, args.learn_interface, env)
        return env, grad, stats

    return gradient_fn


def bptt[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    config: BPTTConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    match args.grad_config.hessian_mode:
        case "gauss_newton":
            if args.readout_outputs is None or args.objective is None:
                raise ValueError(
                    "hessian_mode='gauss_newton' needs the f/l split (readout_outputs, objective) on "
                    "the LearningArg. It is wired for the model_learner / readout gradient only; the "
                    "optimizer_learner must use 'exact'."
                )
            return gauss_newton_backward(args, config.truncate_at)
        case _:

            def loss_to_grad(loss_fn, param, ds):
                return eqx.filter_grad(loss_fn, has_aux=True)(param, ds)

            return get_backward_mode_with_grad(args, config.truncate_at, loss_to_grad)


def identity[ENV, TR_DATA, VL_DATA](
    args: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    config: IdentityLearnerConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def loss_to_grad(loss_fn, param, ds):
        _loss, (env, stats) = loss_fn(param, ds)
        return jnp.zeros_like(param), (env, stats)

    return get_backward_mode_with_grad(args, config.bptt_config.truncate_at, loss_to_grad)


def dispatch_learner[ENV, TR_DATA, VL_DATA](
    method: GradientMethod,
    args_gr: LearningArg[ENV, TR_DATA, VL_DATA, GRADIENT],
    args_loss: LearningArg[ENV, TR_DATA, VL_DATA, LOSS],
    hyperparameters: dict[HP, HyperparameterConfig],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    match method:
        case RTRLConfig():
            return rtrl(args_gr, method)
        case TikhonovRTRLConfig():
            return tikhonov_rtrl(args_gr, method)
        case PadeRTRLConfig():
            return pade_rtrl(args_gr, method)
        case MidpointRTRLConfig():
            return midpoint_rtrl(args_gr, method)
        case HeunRTRLConfig():
            return heun_rtrl(args_gr, method)
        case ImplicitEulerRTRLConfig():
            return implicit_euler_rtrl(args_gr, method)
        case UOROConfig():
            return uoro(args_gr, method)
        case RFLOConfig():
            return rflo(args_gr, method, hyperparameters)
        case ImmediateLearnerConfig():
            return immediate(args_gr)
        case BPTTConfig():
            return bptt(args_loss, method)
        case IdentityLearnerConfig():
            return identity(args_loss, method)


def create_validation_learners[ENV, TR_DATA, VL_DATA](
    transition_fns: list[Callable[[ENV, TR_DATA], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, VL_DATA], tuple[ENV, LOSS, STAT]]],
    readout_output_fns: list[Callable[[ENV, VL_DATA], tuple[ENV, Outputs]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    config: GodConfig,
) -> tuple[
    list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]],
    list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]],
    list[tuple[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, Outputs, STAT]], Callable]],
]:

    def identity_transition(env: ENV, data: TR_DATA) -> tuple[ENV, STAT]:
        return env, {}

    def shim_expand_time(
        grad_fn: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]],
    ) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
        def wrapper(env: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
            data_with_time = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), (data, data))
            _, gradient, stat = grad_fn(env, data_with_time)
            return env, gradient, strip_leading_axis(stat)

        return wrapper

    def make_readout_interface(interface: GodInterface[ENV], grad_config: GradientConfig) -> GodInterface[ENV]:
        state_acc = interface.state
        param_acc = interface.param

        def get_param(env: ENV) -> jax.Array:
            return jnp.concatenate([state_acc.get(env), param_acc.get(env)])

        def put_param(env: ENV, param: jax.Array) -> ENV:
            state_size = state_acc.get(env).shape[0]
            env = state_acc.put(env, param[:state_size])
            env = param_acc.put(env, param[state_size:])
            return env

        noop_put_tagged = lambda env, v: env
        readout_state = make_grad_transform(grad_config).init(None)
        return copy.replace(
            interface,
            state=Accessor(get=lambda env: jnp.empty(0), put=lambda env, s: env, put_tagged=noop_put_tagged),
            param=Accessor(get=get_param, put=put_param, put_tagged=noop_put_tagged),
            opt_state=Accessor(get=lambda env: readout_state, put=lambda env, v: env, put_tagged=noop_put_tagged),
        )

    gradient_fns: list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]] = []
    loss_fns: list[Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, LOSS, STAT]]] = []
    curvature_pieces: list[tuple[Callable, Callable]] = []

    objective_fns = create_objective_fns(config, interfaces)

    for level, (transition, readout_fn, readout_output_fn, objective, meta_config) in enumerate(
        zip(
            transition_fns,
            readout_fns,
            readout_output_fns,
            objective_fns,
            config.levels,
        )
    ):
        interface = interfaces[(MODEL_LEARNER, level)]
        method = meta_config.learner.model_learner.method
        model_grad_config = meta_config.learner.model_learner
        length = meta_config.validation.num_steps
        track_logs = meta_config.track_logs
        readout_grad_config = GradientConfig(
            method=BPTTConfig(truncate_at=None),
            add_clip=None,
            scale=1.0,
            hessian_mode=model_grad_config.hessian_mode,
        )
        readout_interface = make_readout_interface(interface, readout_grad_config)
        readout_gr = shim_expand_time(
            bptt(
                LearningArg(
                    transition=identity_transition,
                    readout=readout_fn,
                    readout_outputs=readout_output_fn,
                    objective=objective,
                    learn_interface=readout_interface,
                    grad_config=readout_grad_config,
                    length=1,
                    vmap_this=lambda f: f,
                    track_logs=track_logs,
                    scan_tag="time",
                    log_prefix=f"level{level}",
                ),
                BPTTConfig(truncate_at=None),
            )
        )

        args_loss = LearningArg(
            transition=transition,
            readout=readout_fn,
            readout_outputs=readout_output_fn,
            objective=objective,
            learn_interface=interface,
            grad_config=model_grad_config,
            length=length,
            vmap_this=lambda f: f,
            track_logs=track_logs,
            scan_tag="time",
            log_prefix=f"level{level}",
        )
        args_gr = dataclasses.replace(args_loss, readout=readout_gr)

        gradient_fns.append(dispatch_learner(method, args_gr, args_loss, config.hyperparameters))
        loss_fns.append(get_backward_mode(args_loss, truncate_at=None))
        # Pieces for the outer Gauss-Newton curvature: the validation forward (theta -> stacked
        # Outputs over the window) and the per-step objective. Reused to build F_theta-vps.
        curvature_pieces.append((get_outputs_mode(args_loss, None), objective))

    return gradient_fns, loss_fns, curvature_pieces


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


def build_meta_curvature_fn[ENV](
    nest_interface: GodInterface[ENV],
    curvature_pieces: tuple[Callable, Callable],
    axes: ENV,
    lower_axes: list[ENV],
    val_axes: ENV,
) -> Callable[[ENV, tuple], jax.Array]:
    """Assemble the outer Gauss-Newton curvature H = gammaᵀ J_valᵀ H_l J_val gamma for one level's
    meta-step, where gamma = forward_mode_jacobian (the RTRL influence tensor = d(lower params)/d(this
    level's params)) and (J_val, H_l) come from the validation forward. Per meta-batch element, gamma is
    unbatched (state_dim, P); F_theta is applied to each of its P columns via ggn_vector_product, then
    contracted to a (P, P) block. The per-element block is vmapped over the meta batch axes exactly like
    the validation learner and summed (the meta-objective is a sum over those elements). Generic per
    level — theta is set through nest_interface.state so F_theta lives in gamma's row space."""
    out_scan, objective = curvature_pieces

    def curv_base(env: ENV, data: tuple) -> tuple[ENV, jax.Array, STAT]:
        _, vl_data = data
        gamma = nest_interface.forward_mode_jacobian.get(env)  # (state_dim, P)
        theta = nest_interface.state.get(env)

        def f(t, e, d):
            _, outs, _ = out_scan(nest_interface.state.put(e, t), d)
            return outs

        def l(z, e, d):
            _, vl = d

            def body(acc, ot_dt):
                return acc + objective(e, ot_dt[0], ot_dt[1]), None

            return jax.lax.scan(body, jnp.array(0.0), (z, vl))[0]

        fvp = lambda v: ggn_vector_product(f, l, theta, v, env, vl_data)
        w = eqx.filter_vmap(fvp, in_axes=1, out_axes=1)(gamma)  # (state_dim, P) = F_theta gamma
        return env, gamma.T @ w, {}

    curv_fn = curv_base
    for ax in lower_axes:
        combined = eqx.combine(ax, val_axes)
        curv_fn = restore_broadcast(curv_fn, combined)
        curv_fn = tagged_vmap(curv_fn, in_axes=(combined, 0), out_axes=(combined, 0, 0))
    curv_fn = tagged_vmap(curv_fn, in_axes=(axes, 0), out_axes=(axes, 0, 0))

    def meta_curvature_fn(env: ENV, data: tuple) -> jax.Array:
        data_step = jax.tree.map(lambda x: x[-1], data)  # the meta scan step that produced the final gamma
        _, h_batched, _ = curv_fn(env, data_step)
        return jnp.sum(h_batched, axis=tuple(range(h_batched.ndim - 2)))  # sum the per-element (P,P) blocks

    return meta_curvature_fn


def create_meta_learner[ENV](
    config: GodConfig,
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]],
    transition_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, STAT]]],
    readout_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]],
    readout_output_fns: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, Outputs]]],
    interfaces: dict[S_ID, GodInterface[ENV]],
    env: ENV,
) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

    validation_learners, validation_losses, validation_curvatures = create_validation_learners(
        transition_fns, readout_fns, readout_output_fns, interfaces, config
    )
    resetters = env_resetters(config, shapes, interfaces, [False] * len(config.levels))

    val_resetters = env_validation_resetters(config, shapes, interfaces)
    per_level_val_axes = [diff_axes(env, vr(env, jax.random.key(0))) for vr in val_resetters]

    def make_optimized_transition[X](
        inner: Callable[[ENV, tuple], tuple[ENV, STAT]],
        readout_gr: Callable[[ENV, tuple], tuple[ENV, GRADIENT, STAT]],
        readout: Callable[[ENV, tuple], tuple[ENV, LOSS, STAT]],
        resetter: Callable[[ENV, PRNG], ENV],
        reset_t: int | None,
        nest_interface: GodInterface[ENV],
        assignments: dict[str, OptimizerAssignment],
        method: GradientMethod,
        grad_config: GradientConfig,
        level: int,
        length: int,
        vmap_this: Callable[
            [Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]]],
            Callable[[ENV, tuple], tuple[ENV, tuple[X, STAT]]],
        ],
        track_logs: TrackLogs,
        natural_gradient: Optional[NaturalGradientConfig],
        meta_curvature_fn: Callable[[ENV, tuple], jax.Array],
    ) -> Callable[[ENV, tuple], tuple[ENV, STAT]]:

        check = make_reset_checker(nest_interface, resetter, reset_t)
        advance = make_tick_advancer(nest_interface)

        def composed_inner(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env = check(env)
            env = advance(env)
            return inner(env, data)

        args_loss = LearningArg(
            transition=composed_inner,
            readout=readout,
            readout_outputs=None,
            objective=None,
            learn_interface=nest_interface,
            grad_config=grad_config,
            length=length,
            vmap_this=vmap_this,
            track_logs=track_logs,
            scan_tag="scan",
            log_prefix=f"level{level}",
        )
        args_gr = dataclasses.replace(args_loss, readout=readout_gr)
        grad_fn = dispatch_learner(method, args_gr, args_loss, config.hyperparameters)

        match method:
            case RTRLConfig():
                edge_margin = method.lr_edge_margin
            case (
                TikhonovRTRLConfig()
                | PadeRTRLConfig()
                | MidpointRTRLConfig()
                | HeunRTRLConfig()
                | ImplicitEulerRTRLConfig()
                | RFLOConfig()
                | UOROConfig()
            ):
                edge_margin = method.rtrl_config.lr_edge_margin
            case _:
                edge_margin = None
        lr_targets = [
            hp
            for a in assignments.values()
            for hp in a.target
            if hp in config.hyperparameters
            and config.hyperparameters[hp].kind == "learning_rate"
            and edge_margin is not None
            and track_logs.largest_eigenvalue
        ]

        def optimized_transition(env: ENV, data: tuple) -> tuple[ENV, STAT]:
            env, gradient, stat = grad_fn(env, data)
            stat[f"level{level}/meta_gradient_norm"] = scalar(jax.lax.stop_gradient(jnp.linalg.norm(gradient)))
            if natural_gradient is not None:
                h_phi = meta_curvature_fn(env, data)  # (P, P), pushforward GN curvature over this level's params
                p_dim = gradient.shape[0]
                inverse_hessian = jnp.linalg.inv(h_phi + natural_gradient.damping * jnp.eye(p_dim, dtype=h_phi.dtype))
                gradient = GRADIENT(inverse_hessian @ gradient)
                for i in range(p_dim):
                    for j in range(p_dim):
                        stat[f"level{level}/inverse_hessian/{i}/{j}"] = scalar(
                            jax.lax.stop_gradient(inverse_hessian[i, j])
                        )
            lr_pre = {hp: interfaces[(hp, level)].learning_rate.get(env) for hp in lr_targets}
            gr_env = nest_interface.param.put(env, gradient)
            env = get_opt_step(assignments, interfaces, level, env, gr_env, config.hyperparameters)
            if track_logs.parameter_norm:
                stat = stat | parameter_norm_stats(
                    nest_interface.param.get(env), nest_interface.param_layout(env), f"level{level}"
                )
            for hp in lr_targets:
                forward, invert = hyperparameter_reparametrization(
                    config.hyperparameters[hp].hyperparameter_parametrization
                )
                metric = nest_interface.logs.get(env).largest_eigenvalue
                edge = jnp.where(
                    metric > 1e-12,
                    edge_margin * 2.0 * forward(lr_pre[hp]) / jnp.maximum(metric, 1e-30),
                    jnp.inf,
                )
                iface = interfaces[(hp, level)]
                env = iface.learning_rate.put(env, invert(jnp.minimum(forward(iface.learning_rate.get(env)), edge)))
            return env, stat

        return optimized_transition

    current_transition: Callable[[ENV, tuple], tuple[ENV, STAT]] = lambda env, data: (env, {})
    current_resetter: Callable[[ENV, PRNG], ENV] = lambda env, prng: env

    for level in range(len(config.levels)):
        meta_config = config.levels[level]
        nest_interface = interfaces[(OPTIMIZER_LEARNER, level)]
        inner_resetter, full_resetter = resetters[level]
        vl_learner = validation_learners[level]
        vl_loss = validation_losses[level]

        axes = diff_axes(env, inner_resetter(env, jax.random.key(0)))
        lower_axes = [diff_axes(env, resetters[l][0](env, jax.random.key(0))) for l in range(level)]

        for ax in lower_axes:
            combined = eqx.combine(ax, per_level_val_axes[level])
            vl_learner = restore_broadcast(vl_learner, combined)
            vl_loss = restore_broadcast(vl_loss, combined)
            vl_learner = tagged_vmap(vl_learner, in_axes=(combined, 0), out_axes=(combined, 0, 0))
            vl_loss = tagged_vmap(vl_loss, in_axes=(combined, 0), out_axes=(combined, 0, 0))

        vmap_this = lambda f, a=axes: tagged_vmap(f, in_axes=(a, 0), out_axes=(a, 0, 0))

        meta_curvature_fn = build_meta_curvature_fn(
            nest_interface,
            validation_curvatures[level],
            axes,
            lower_axes,
            per_level_val_axes[level],
        )

        current_transition = make_optimized_transition(
            current_transition,
            vl_learner,
            vl_loss,
            current_resetter,
            meta_config.nested.reset_t,
            nest_interface,
            meta_config.learner.optimizer,
            meta_config.learner.optimizer_learner.method,
            meta_config.learner.optimizer_learner,
            level,
            meta_config.nested.num_steps,
            vmap_this,
            meta_config.track_logs,
            meta_config.natural_gradient,
            meta_curvature_fn,
        )

        current_resetter = full_resetter

    return current_transition
