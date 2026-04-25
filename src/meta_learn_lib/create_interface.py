import copy
import jax
import equinox as eqx
import math

from meta_learn_lib.config import *
from meta_learn_lib.env import GodState
from meta_learn_lib.interface import *
from meta_learn_lib.util import to_vector
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import *


# ============================================================================
# PRNG / TICK / LOG — non-Accessor primitives (special signatures)
# ============================================================================


def take_prng(key: int, level: int):
    def fn(env: GodState) -> tuple[jax.Array, GodState]:
        prng, new_prng = jax.random.split(env.level_meta[level].prngs[key])
        return prng, env.transform(["level_meta", level, "prngs", key], lambda _: new_prng)

    return fn


def put_prng(key: int, level: int):
    def fn(env: GodState, prng: jax.Array) -> GodState:
        return env.transform(["level_meta", level, "prngs", key], lambda _: prng)

    return fn


def prng_fns(
    key: int, level: int
) -> tuple[
    Callable[[GodState], PRNG],
    Callable[[GodState, PRNG], GodState],
]:
    def get(env: GodState) -> PRNG:
        return env.level_meta[level].prngs.get(key)

    def put(env: GodState, val: PRNG) -> GodState:
        return env.transform(["level_meta", level, "prngs", key], lambda _: val)

    return get, put


def get_tick(i: int, level: int):
    def fn(env: GodState) -> jax.Array:
        return env.level_meta[level].ticks.get(i)

    return fn


def put_tick(i: int, level: int):
    def fn(env: GodState, tick: jax.Array) -> GodState:
        return env.transform(["level_meta", level, "ticks", i], lambda _: tick)

    return fn


def get_logs(level: int):
    def fn(env: GodState) -> Logs:
        return env.level_meta[level].log

    return fn


def put_logs(level: int):
    def fn(env: GodState, logs: Logs) -> GodState:
        return env.transform(
            ["level_meta", level, "log"],
            lambda old: old.set(**{k: v for k, v in logs.serialize().items() if v is not None}),
        )

    return fn


# ============================================================================
# PARAM ACCESSORS — (get, put) pairs for parameter slots
# ============================================================================


def mlp_model(
    i: int, level: int
) -> tuple[
    Callable[[GodState], eqx.nn.Sequential],
    Callable[[GodState, eqx.nn.Sequential], GodState],
]:
    def get(env: GodState) -> eqx.nn.Sequential:
        return env.meta_parameters[level].mlps.get(i)

    def put(env: GodState, val: eqx.nn.Sequential) -> GodState:
        return env.transform(["meta_parameters", level, "mlps", i], lambda _: val)

    return get, put


def rnn_w_rec(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].rnns.get(i).w_rec

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "w_rec"], lambda _: val)

    return get, put


def rnn_b_rec(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].rnns.get(i).b_rec

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "b_rec"], lambda _: val)

    return get, put


def rnn_layer_norm(
    i: int, level: int
) -> tuple[
    Callable[[GodState], eqx.Module],
    Callable[[GodState, eqx.Module], GodState],
]:
    def get(env: GodState) -> eqx.Module:
        return env.meta_parameters[level].rnns.get(i).layer_norm

    def put(env: GodState, val: eqx.Module) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "layer_norm"], lambda _: val)

    return get, put


def gru_cell(
    i: int, level: int
) -> tuple[
    Callable[[GodState], eqx.nn.GRUCell],
    Callable[[GodState, eqx.nn.GRUCell], GodState],
]:
    def get(env: GodState) -> eqx.nn.GRUCell:
        return env.meta_parameters[level].grus.get(i)

    def put(env: GodState, val: eqx.nn.GRUCell) -> GodState:
        return env.transform(["meta_parameters", level, "grus", i], lambda _: val)

    return get, put


def lstm_cell(
    i: int, level: int
) -> tuple[
    Callable[[GodState], eqx.nn.LSTMCell],
    Callable[[GodState, eqx.nn.LSTMCell], GodState],
]:
    def get(env: GodState) -> eqx.nn.LSTMCell:
        return env.meta_parameters[level].lstms.get(i)

    def put(env: GodState, val: eqx.nn.LSTMCell) -> GodState:
        return env.transform(["meta_parameters", level, "lstms", i], lambda _: val)

    return get, put


def learning_rate(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].learning_rates.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "learning_rates", i], lambda _: val)

    return get, put


def weight_decay(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].weight_decays.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "weight_decays", i], lambda _: val)

    return get, put


def momentum(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].momentums.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "momentums", i], lambda _: val)

    return get, put


def time_constant(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].time_constants.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "time_constants", i], lambda _: val)

    return get, put


def kl_regularizer_beta(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.meta_parameters[level].kl_regularizer_betas.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "kl_regularizer_betas", i], lambda _: val)

    return get, put


# ============================================================================
# STATE ACCESSORS — (get, put) pairs for state slots
# ============================================================================


def vanilla_rnn_state(
    i: int, level: int
) -> tuple[
    Callable[[GodState], VanillaRecurrentState],
    Callable[[GodState, VanillaRecurrentState], GodState],
]:
    def get(env: GodState) -> VanillaRecurrentState:
        return env.model_states[level].vanilla_recurrent_states.get(i)

    def put(env: GodState, state: VanillaRecurrentState) -> GodState:
        return env.transform(["model_states", level, "vanilla_recurrent_states", i], lambda _: state)

    return get, put


def gru_activation(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.model_states[level].recurrent_states.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["model_states", level, "recurrent_states", i], lambda _: val)

    return get, put


def lstm_state(
    i: int, level: int
) -> tuple[
    Callable[[GodState], LSTMState],
    Callable[[GodState, LSTMState], GodState],
]:
    def get(env: GodState) -> LSTMState:
        return env.model_states[level].lstm_states.get(i)

    def put(env: GodState, state: LSTMState) -> GodState:
        return env.transform(["model_states", level, "lstm_states", i], lambda _: state)

    return get, put


def autoregressive_predictions(
    i: int, level: int
) -> tuple[
    Callable[[GodState], jax.Array],
    Callable[[GodState, jax.Array], GodState],
]:
    def get(env: GodState) -> jax.Array:
        return env.model_states[level].autoregressive_predictions.get(i)

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["model_states", level, "autoregressive_predictions", i], lambda _: val)

    return get, put


# ============================================================================
# LEARNING STATE ACCESSORS — (get, put) pairs for learning state slots
# ============================================================================


def opt_state(
    i: int, level: int
) -> tuple[
    Callable[[GodState], optax.OptState],
    Callable[[GodState, optax.OptState], GodState],
]:
    def get(env: GodState) -> optax.OptState:
        return env.learning_states[level].opt_states.get(i)

    def put(env: GodState, val: optax.OptState) -> GodState:
        return env.transform(["learning_states", level, "opt_states", i], lambda _: val)

    return get, put


def forward_mode_jacobian(
    i: int, level: int
) -> tuple[
    Callable[[GodState], JACOBIAN],
    Callable[[GodState, JACOBIAN], GodState],
]:
    def get(env: GodState) -> JACOBIAN:
        return env.learning_states[level].influence_tensors.get(i)

    def put(env: GodState, val: JACOBIAN) -> GodState:
        return env.transform(["learning_states", level, "influence_tensors", i], lambda _: val)

    return get, put


def uoro_state(
    i: int, level: int
) -> tuple[
    Callable[[GodState], UOROState],
    Callable[[GodState, UOROState], GodState],
]:
    def get(env: GodState) -> UOROState:
        return env.learning_states[level].uoros.get(i)

    def put(env: GodState, val: UOROState) -> GodState:
        return env.transform(["learning_states", level, "uoros", i], lambda _: val)

    return get, put


def midpoint_buffer(
    i: int, level: int
) -> tuple[
    Callable[[GodState], MidpointBuffer],
    Callable[[GodState, MidpointBuffer], GodState],
]:
    def get(env: GodState) -> MidpointBuffer:
        return env.learning_states[level].midpoint_buffers.get(i)

    def put(env: GodState, val: MidpointBuffer) -> GodState:
        return env.transform(["learning_states", level, "midpoint_buffers", i], lambda _: val)

    return get, put


def build_id_map(config: GodConfig) -> dict[S_ID, int]:
    keys: list[S_ID] = []

    # shared params — one per node, level=None
    keys.extend((name, None) for name in config.nodes)

    # per-level states — one per (node, level)
    for level in range(len(config.levels)):
        keys.extend((name, level) for name in config.nodes)

    # HPs — each at its own level
    keys.extend((name, hp.level) for name, hp in config.hyperparameters.items())

    # per-level optimizers
    for level, mc in enumerate(config.levels):
        keys.extend((name, level) for name in mc.learner.optimizer)

    # per-level learners and tasks
    for level in range(len(config.levels)):
        keys.extend(
            [
                (MODEL_LEARNER, level),
                (OPTIMIZER_LEARNER, level),
                (TASK, level),
            ]
        )

    return {key: i for i, key in enumerate(keys)}


def param_accessor[ENV, T](
    get: Callable[[ENV], T],
    put: Callable[[ENV, T], ENV],
    learnable: bool,
    min_value: float,
    max_value: float,
    parametrizes_transition: bool,
    category: Category,
) -> Accessor[ENV, T]:
    return Accessor(
        get=get,
        put=put,
        meta=ParamMeta(
            learnable=learnable,
            min_value=min_value,
            max_value=max_value,
            parametrizes_transition=parametrizes_transition,
        ),
        category=category,
    )


def prng_accessor(key: int, level: int) -> Accessor[GodState, PRNG]:
    g, p = prng_fns(key, level)
    return Accessor(get=g, put=p, meta=noop_meta(), category=None)


def state_accessor[ENV, T](
    get: Callable[[ENV], T],
    put: Callable[[ENV, T], ENV],
    is_stateful: frozenset[int],
    category: Category,
) -> Accessor[ENV, T]:
    return Accessor(
        get=get,
        put=put,
        meta=StateMeta(
            is_stateful=is_stateful,
        ),
        category=category,
    )


@dataclass(frozen=True)
class Lens:
    get: Callable[[GodState], jax.Array]
    put: Callable[[GodState, jax.Array], GodState]


def make_lens(
    accessors: list[tuple[S_ID, Accessor]],
    model_states_sl: slice,
    learning_states_sl: slice,
    params_sl: slice,
) -> Lens:
    target_level = params_sl.stop

    relevant = [
        acc
        for (_, level), acc in accessors
        if level is not None
        and (
            (acc.category == "state" and model_states_sl.start <= level < model_states_sl.stop)
            or (acc.category == "learning_state" and learning_states_sl.start <= level < learning_states_sl.stop)
            or (acc.category == "param" and params_sl.start <= level < params_sl.stop)
        )
    ]

    def predicate(meta: AccessorMeta) -> bool:
        match meta:
            case ParamMeta():
                return meta.learnable
            case StateMeta():
                return target_level in meta.is_stateful

    def get(env: GodState) -> jax.Array:
        mask = build_mask(env, relevant, predicate)
        filtered, _ = eqx.partition(env, mask)
        return to_vector(filtered).vector

    def put(env: GodState, vector: jax.Array) -> GodState:
        mask = build_mask(env, relevant, predicate)
        filtered, static = eqx.partition(env, mask)
        return eqx.combine(to_vector(filtered).to_param(vector), static)

    return Lens(get=get, put=put)


def build_interfaces(
    config: GodConfig,
    id_map: dict[S_ID, int],
) -> dict[S_ID, GodInterface[GodState]]:

    interfaces: dict[S_ID, GodInterface[GodState]] = {}
    default = default_god_interface()
    num_levels = len(config.levels)
    learnables_per_level = [
        frozenset().union(*[v.target for v in mc.learner.optimizer.values()]) for mc in config.levels
    ]

    # 1. nodes
    for level in range(num_levels):
        val_track = config.levels[level].validation.track_influence_in

        for name, node in config.nodes.items():
            pi = id_map[(name, None)]
            si = id_map[(name, level)]
            is_learnable = name in learnables_per_level[0]
            is_transition = name not in config.readout_graph

            logs_acc = Accessor(get=get_logs(level), put=put_logs(level), meta=noop_meta(), category=None)

            match node:
                case NNLayer():
                    g, p = mlp_model(pi, 0)
                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        mlp_model=param_accessor(
                            g,
                            p,
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                    )

                case VanillaRNNLayer(nn_layer, _, tc):
                    g_wr, p_wr = rnn_w_rec(pi, 0)
                    g_br, p_br = rnn_b_rec(pi, 0)
                    g_ln, p_ln = rnn_layer_norm(pi, 0)
                    g_st, p_st = vanilla_rnn_state(si, level)

                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    g_tc, p_tc = time_constant(hi, hp_cfg.level)

                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        rnn_w_rec=param_accessor(
                            g_wr,
                            p_wr,
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                        rnn_b_rec=param_accessor(
                            g_br,
                            p_br,
                            learnable=is_learnable and nn_layer.use_bias,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                        rnn_layer_norm=param_accessor(
                            g_ln,
                            p_ln,
                            learnable=is_learnable and nn_layer.layer_norm is not None,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                        vanilla_rnn_state=state_accessor(g_st, p_st, is_stateful=val_track, category="state"),
                        time_constant=param_accessor(
                            g_tc,
                            p_tc,
                            learnable=tc in learnables_per_level[hp_cfg.level],
                            min_value=hp_cfg.min_value,
                            max_value=hp_cfg.max_value,
                            parametrizes_transition=hp_cfg.parametrizes_transition,
                            category=None,
                        ),
                    )

                case GRULayer(_, _, _, tc):
                    g_cell, p_cell = gru_cell(pi, 0)
                    g_act, p_act = gru_activation(si, level)

                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    g_tc, p_tc = time_constant(hi, hp_cfg.level)

                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        gru_cell=param_accessor(
                            g_cell,
                            p_cell,
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                        gru_activation=state_accessor(g_act, p_act, is_stateful=val_track, category="state"),
                        time_constant=param_accessor(
                            g_tc,
                            p_tc,
                            learnable=tc in learnables_per_level[hp_cfg.level],
                            min_value=hp_cfg.min_value,
                            max_value=hp_cfg.max_value,
                            parametrizes_transition=hp_cfg.parametrizes_transition,
                            category=None,
                        ),
                    )

                case LSTMLayer(_, _, _, tc):
                    g_cell, p_cell = lstm_cell(pi, 0)
                    g_st, p_st = lstm_state(si, level)

                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    g_tc, p_tc = time_constant(hi, hp_cfg.level)

                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        lstm_cell=param_accessor(
                            g_cell,
                            p_cell,
                            learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                            parametrizes_transition=is_transition,
                            category="param",
                        ),
                        lstm_state=state_accessor(g_st, p_st, is_stateful=val_track, category="state"),
                        time_constant=param_accessor(
                            g_tc,
                            p_tc,
                            learnable=tc in learnables_per_level[hp_cfg.level],
                            min_value=hp_cfg.min_value,
                            max_value=hp_cfg.max_value,
                            parametrizes_transition=hp_cfg.parametrizes_transition,
                            category=None,
                        ),
                    )

                case Scan():
                    g_ap, p_ap = autoregressive_predictions(si, level)
                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        autoregressive_predictions=state_accessor(g_ap, p_ap, is_stateful=val_track, category="state"),
                    )

                case ReparameterizeLayer():
                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(si, level),
                        put_prng=put_prng(si, level),
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                    )

                case _:
                    interfaces[(name, level)] = default

    # 2. HPs
    for name, hp in config.hyperparameters.items():
        hi = id_map[(name, hp.level)]
        is_learnable = name in learnables_per_level[hp.level]
        hp_meta_kw = dict(
            learnable=is_learnable,
            min_value=hp.min_value,
            max_value=hp.max_value,
            parametrizes_transition=hp.parametrizes_transition,
            category="param",
        )
        logs_acc = Accessor(get=get_logs(hp.level), put=put_logs(hp.level), meta=noop_meta(), category=None)

        match hp.kind:
            case "learning_rate":
                g, p = learning_rate(hi, hp.level)
                interfaces[(name, hp.level)] = copy.replace(
                    default,
                    take_prng=take_prng(hi, hp.level),
                    put_prng=put_prng(hi, hp.level),
                    prng=prng_accessor(hi, hp.level),
                    logs=logs_acc,
                    learning_rate=param_accessor(g, p, **hp_meta_kw),
                )
            case "weight_decay":
                g, p = weight_decay(hi, hp.level)
                interfaces[(name, hp.level)] = copy.replace(
                    default,
                    take_prng=take_prng(hi, hp.level),
                    put_prng=put_prng(hi, hp.level),
                    prng=prng_accessor(hi, hp.level),
                    logs=logs_acc,
                    weight_decay=param_accessor(g, p, **hp_meta_kw),
                )
            case "momentum":
                g, p = momentum(hi, hp.level)
                interfaces[(name, hp.level)] = copy.replace(
                    default,
                    take_prng=take_prng(hi, hp.level),
                    put_prng=put_prng(hi, hp.level),
                    prng=prng_accessor(hi, hp.level),
                    logs=logs_acc,
                    momentum=param_accessor(g, p, **hp_meta_kw),
                )
            case "time_constant":
                g, p = time_constant(hi, hp.level)
                interfaces[(name, hp.level)] = copy.replace(
                    default,
                    take_prng=take_prng(hi, hp.level),
                    put_prng=put_prng(hi, hp.level),
                    prng=prng_accessor(hi, hp.level),
                    logs=logs_acc,
                    time_constant=param_accessor(g, p, **hp_meta_kw),
                )
            case "kl_regularizer_beta":
                g, p = kl_regularizer_beta(hi, hp.level)
                interfaces[(name, hp.level)] = copy.replace(
                    default,
                    take_prng=take_prng(hi, hp.level),
                    put_prng=put_prng(hi, hp.level),
                    prng=prng_accessor(hi, hp.level),
                    logs=logs_acc,
                    kl_regularizer_beta=param_accessor(g, p, **hp_meta_kw),
                )

    # 3. optimizers
    for level, mc in enumerate(config.levels):
        nested_track = mc.nested.track_influence_in
        for name, assignment in mc.learner.optimizer.items():
            oi = id_map[(name, level)]
            logs_acc = Accessor(get=get_logs(level), put=put_logs(level), meta=noop_meta(), category=None)

            match assignment.optimizer:
                case SGDConfig() | SGDNormalizedConfig() | AdamConfig() | ExponentiatedGradientConfig() as opt:
                    g_os, p_os = opt_state(oi, level)

                    hp_lr = config.hyperparameters[opt.learning_rate]
                    g_lr, p_lr = learning_rate(id_map[(opt.learning_rate, hp_lr.level)], hp_lr.level)

                    hp_wd = config.hyperparameters[opt.weight_decay]
                    g_wd, p_wd = weight_decay(id_map[(opt.weight_decay, hp_wd.level)], hp_wd.level)

                    hp_m = config.hyperparameters[opt.momentum]
                    g_m, p_m = momentum(id_map[(opt.momentum, hp_m.level)], hp_m.level)

                    interfaces[(name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(oi, level),
                        put_prng=put_prng(oi, level),
                        prng=prng_accessor(oi, level),
                        logs=logs_acc,
                        opt_state=state_accessor(g_os, p_os, is_stateful=nested_track, category="learning_state"),
                        learning_rate=param_accessor(
                            g_lr,
                            p_lr,
                            learnable=opt.learning_rate in learnables_per_level[hp_lr.level],
                            min_value=hp_lr.min_value,
                            max_value=hp_lr.max_value,
                            parametrizes_transition=hp_lr.parametrizes_transition,
                            category=None,
                        ),
                        weight_decay=param_accessor(
                            g_wd,
                            p_wd,
                            learnable=opt.weight_decay in learnables_per_level[hp_wd.level],
                            min_value=hp_wd.min_value,
                            max_value=hp_wd.max_value,
                            parametrizes_transition=hp_wd.parametrizes_transition,
                            category=None,
                        ),
                        momentum=param_accessor(
                            g_m,
                            p_m,
                            learnable=opt.momentum in learnables_per_level[hp_m.level],
                            min_value=hp_m.min_value,
                            max_value=hp_m.max_value,
                            parametrizes_transition=hp_m.parametrizes_transition,
                            category=None,
                        ),
                    )

                case _:
                    interfaces[(name, level)] = default

    # 4. tasks
    for level, mc in enumerate(config.levels):
        ti = id_map[(TASK, level)]
        logs_acc = Accessor(get=get_logs(level), put=put_logs(level), meta=noop_meta(), category=None)

        match mc.objective_fn:
            case ELBOObjective(beta, _):
                hp_cfg = config.hyperparameters[beta]
                hi = id_map[(beta, hp_cfg.level)]
                g_kl, p_kl = kl_regularizer_beta(hi, hp_cfg.level)
                interfaces[(TASK, level)] = copy.replace(
                    default,
                    take_prng=take_prng(ti, level),
                    put_prng=put_prng(ti, level),
                    prng=prng_accessor(ti, level),
                    logs=logs_acc,
                    kl_regularizer_beta=param_accessor(
                        g_kl,
                        p_kl,
                        learnable=beta in learnables_per_level[hp_cfg.level],
                        min_value=hp_cfg.min_value,
                        max_value=hp_cfg.max_value,
                        parametrizes_transition=hp_cfg.parametrizes_transition,
                        category=None,
                    ),
                )
            case _:
                interfaces[(TASK, level)] = copy.replace(
                    default,
                    take_prng=take_prng(ti, level),
                    put_prng=put_prng(ti, level),
                    prng=prng_accessor(ti, level),
                    logs=logs_acc,
                )

    # 5. learner states (without state/param lenses)
    for level, mc in enumerate(config.levels):
        nested_track = mc.nested.track_influence_in

        for slot_name, learner in [
            (MODEL_LEARNER, mc.learner.model_learner),
            (OPTIMIZER_LEARNER, mc.learner.optimizer_learner),
        ]:
            li = id_map[(slot_name, level)]
            tick_acc = Accessor(get=get_tick(li, level), put=put_tick(li, level), meta=noop_meta(), category=None)
            logs_acc = Accessor(get=get_logs(level), put=put_logs(level), meta=noop_meta(), category=None)

            match learner.method:
                case RTRLConfig() | TikhonovRTRLConfig() | PadeRTRLConfig() | ImplicitEulerRTRLConfig():
                    g_fj, p_fj = forward_mode_jacobian(li, level)
                    interfaces[(slot_name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(li, level),
                        put_prng=put_prng(li, level),
                        prng=prng_accessor(li, level),
                        tick=tick_acc,
                        logs=logs_acc,
                        forward_mode_jacobian=state_accessor(
                            g_fj,
                            p_fj,
                            is_stateful=nested_track,
                            category="learning_state",
                        ),
                    )

                case MidpointRTRLConfig() | HeunRTRLConfig():
                    g_fj, p_fj = forward_mode_jacobian(li, level)
                    g_mb, p_mb = midpoint_buffer(li, level)
                    interfaces[(slot_name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(li, level),
                        put_prng=put_prng(li, level),
                        prng=prng_accessor(li, level),
                        tick=tick_acc,
                        logs=logs_acc,
                        forward_mode_jacobian=state_accessor(
                            g_fj,
                            p_fj,
                            is_stateful=nested_track,
                            category="learning_state",
                        ),
                        midpoint_buffer=state_accessor(g_mb, p_mb, is_stateful=nested_track, category="learning_state"),
                    )

                case RFLOConfig(tc):
                    g_fj, p_fj = forward_mode_jacobian(li, level)
                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    g_tc, p_tc = time_constant(hi, hp_cfg.level)
                    interfaces[(slot_name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(li, level),
                        put_prng=put_prng(li, level),
                        prng=prng_accessor(li, level),
                        tick=tick_acc,
                        logs=logs_acc,
                        forward_mode_jacobian=state_accessor(
                            g_fj,
                            p_fj,
                            is_stateful=nested_track,
                            category="learning_state",
                        ),
                        time_constant=param_accessor(
                            g_tc,
                            p_tc,
                            learnable=tc in learnables_per_level[hp_cfg.level],
                            min_value=hp_cfg.min_value,
                            max_value=hp_cfg.max_value,
                            parametrizes_transition=hp_cfg.parametrizes_transition,
                            category=None,
                        ),
                    )

                case UOROConfig():
                    g_uo, p_uo = uoro_state(li, level)
                    interfaces[(slot_name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(li, level),
                        put_prng=put_prng(li, level),
                        prng=prng_accessor(li, level),
                        tick=tick_acc,
                        logs=logs_acc,
                        uoro_state=state_accessor(g_uo, p_uo, is_stateful=nested_track, category="learning_state"),
                    )

                case _:
                    interfaces[(slot_name, level)] = copy.replace(
                        default,
                        take_prng=take_prng(li, level),
                        put_prng=put_prng(li, level),
                        prng=prng_accessor(li, level),
                        tick=tick_acc,
                        logs=logs_acc,
                    )

    # 6. compute all_tagged from everything built so far
    all_tagged: list[tuple[S_ID, Accessor]] = [
        ((name, level), acc)
        for (name, level), iface in interfaces.items()
        for acc in interface_to_accessors(iface)
        if acc.category is not None
    ]

    # 7. wire learner lenses onto learner interfaces
    for level, mc in enumerate(config.levels):
        for slot_name, learner, is_val in [
            (MODEL_LEARNER, mc.learner.model_learner, True),
            (OPTIMIZER_LEARNER, mc.learner.optimizer_learner, False),
        ]:
            if is_val:
                state_lens = make_lens(all_tagged, slice(level, level + 1), slice(level, level), slice(level, level))
                param_lens1 = make_lens(all_tagged, slice(0, level), slice(0, level), slice(0, level))
                param_lens2 = make_lens(all_tagged, slice(level, level), slice(level, level), slice(level, level + 1))

                def get_param(env, pl1=param_lens1, pl2=param_lens2):
                    return to_vector((pl1.get(env), pl2.get(env))).vector

                def put_param(env, vector, pl1=param_lens1, pl2=param_lens2):
                    new1, new2 = to_vector((pl1.get(env), pl2.get(env))).to_param(vector)
                    return pl2.put(pl1.put(env, new1), new2)

                param_lens = Lens(get=get_param, put=put_param)
            else:
                state_lens = make_lens(all_tagged, slice(0, level), slice(0, level), slice(0, level))
                param_lens = make_lens(all_tagged, slice(level, level), slice(level, level), slice(level, level + 1))

            iface = interfaces[(slot_name, level)]
            interfaces[(slot_name, level)] = copy.replace(
                iface,
                state=Accessor(get=state_lens.get, put=state_lens.put, meta=noop_meta(), category=None),
                param=Accessor(get=param_lens.get, put=param_lens.put, meta=noop_meta(), category=None),
            )

    return interfaces
