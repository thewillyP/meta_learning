import copy
from dataclasses import dataclass

import jax
import equinox as eqx
import optax

from meta_learn_lib.config import *
from meta_learn_lib.env import RNN, GodState, Tagged, ParamMeta, StateMeta, Logs
from meta_learn_lib.interface import *
from meta_learn_lib.util import to_vector
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import *


def read_only[ENV, T](acc: Accessor[ENV, T]) -> Accessor[ENV, T]:
    return Accessor(get=acc.get, put=lambda env, v: env, put_tagged=lambda env, v: env)


# ============================================================================
# PRNG primitives
# ============================================================================


def prng_accessor(key: int, level: int) -> Accessor[GodState, PRNG]:
    def get(env: GodState) -> PRNG:
        t = env.level_meta[level].prngs.get(key)
        return None if t is None else t.value

    def put(env: GodState, val: PRNG) -> GodState:
        return env.transform(["level_meta", level, "prngs", key, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[PRNG]) -> GodState:
        return env.transform(["level_meta", level, "prngs", key], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def tick_accessor(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.level_meta[level].ticks.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["level_meta", level, "ticks", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["level_meta", level, "ticks", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def logs_accessor(level: int) -> Accessor[GodState, Logs]:
    def get(env: GodState) -> Logs:
        t = env.level_meta[level].log
        return None if t is None else t.value

    def put(env: GodState, val: Logs) -> GodState:
        return env.transform(["level_meta", level, "log", "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[Logs]) -> GodState:
        return env.transform(["level_meta", level, "log"], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


# ============================================================================
# PARAMETER ACCESSORS
# ============================================================================


def mlp_model(i: int, level: int) -> Accessor[GodState, eqx.nn.Sequential]:
    def get(env: GodState) -> eqx.nn.Sequential:
        t = env.meta_parameters[level].mlps.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: eqx.nn.Sequential) -> GodState:
        return env.transform(["meta_parameters", level, "mlps", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[eqx.nn.Sequential]) -> GodState:
        return env.transform(["meta_parameters", level, "mlps", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def upsert_rnn[T](
    level: int,
    i: int,
    set_field: Callable[[RNN, T], RNN],
) -> Callable[[GodState, T], GodState]:
    def fn(env: GodState, val: T) -> GodState:
        def update(existing):
            base = existing if isinstance(existing, RNN) else RNN()
            return set_field(base, val)

        return env.transform(["meta_parameters", level, "rnns", i], update)

    return fn


def rnn_w_rec(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        rnn = env.meta_parameters[level].rnns.get(i)
        if rnn is None or rnn.w_rec is None:
            return None
        return rnn.w_rec.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "w_rec", "value"], lambda _: val)

    put_tagged = upsert_rnn(level, i, lambda r, v: r.set(w_rec=v))
    return Accessor(get=get, put=put, put_tagged=put_tagged)


def rnn_b_rec(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        rnn = env.meta_parameters[level].rnns.get(i)
        if rnn is None or rnn.b_rec is None:
            return None
        return rnn.b_rec.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "b_rec", "value"], lambda _: val)

    put_tagged = upsert_rnn(level, i, lambda r, v: r.set(b_rec=v))
    return Accessor(get=get, put=put, put_tagged=put_tagged)


def rnn_layer_norm(i: int, level: int) -> Accessor[GodState, eqx.Module]:
    def get(env: GodState) -> eqx.Module:
        rnn = env.meta_parameters[level].rnns.get(i)
        if rnn is None or rnn.layer_norm is None:
            return None
        return rnn.layer_norm.value

    def put(env: GodState, val: eqx.Module) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i, "layer_norm", "value"], lambda _: val)

    put_tagged = upsert_rnn(level, i, lambda r, v: r.set(layer_norm=v))
    return Accessor(get=get, put=put, put_tagged=put_tagged)


def gru_cell(i: int, level: int) -> Accessor[GodState, eqx.nn.GRUCell]:
    def get(env: GodState) -> eqx.nn.GRUCell:
        t = env.meta_parameters[level].grus.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: eqx.nn.GRUCell) -> GodState:
        return env.transform(["meta_parameters", level, "grus", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[eqx.nn.GRUCell]) -> GodState:
        return env.transform(["meta_parameters", level, "grus", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def lstm_cell(i: int, level: int) -> Accessor[GodState, eqx.nn.LSTMCell]:
    def get(env: GodState) -> eqx.nn.LSTMCell:
        t = env.meta_parameters[level].lstms.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: eqx.nn.LSTMCell) -> GodState:
        return env.transform(["meta_parameters", level, "lstms", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[eqx.nn.LSTMCell]) -> GodState:
        return env.transform(["meta_parameters", level, "lstms", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def learning_rate(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.meta_parameters[level].learning_rates.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "learning_rates", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "learning_rates", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def weight_decay(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.meta_parameters[level].weight_decays.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "weight_decays", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "weight_decays", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def momentum(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.meta_parameters[level].momentums.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "momentums", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "momentums", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def time_constant(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.meta_parameters[level].time_constants.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "time_constants", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "time_constants", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def kl_regularizer_beta(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.meta_parameters[level].kl_regularizer_betas.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["meta_parameters", level, "kl_regularizer_betas", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "kl_regularizer_betas", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


# ============================================================================
# STATE ACCESSORS
# ============================================================================


def vanilla_rnn_state(i: int, level: int) -> Accessor[GodState, VanillaRecurrentState]:
    def get(env: GodState) -> VanillaRecurrentState:
        t = env.model_states[level].vanilla_recurrent_states.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: VanillaRecurrentState) -> GodState:
        return env.transform(["model_states", level, "vanilla_recurrent_states", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[VanillaRecurrentState]) -> GodState:
        return env.transform(["model_states", level, "vanilla_recurrent_states", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def gru_activation(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.model_states[level].recurrent_states.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["model_states", level, "recurrent_states", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["model_states", level, "recurrent_states", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def lstm_state(i: int, level: int) -> Accessor[GodState, LSTMState]:
    def get(env: GodState) -> LSTMState:
        t = env.model_states[level].lstm_states.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: LSTMState) -> GodState:
        return env.transform(["model_states", level, "lstm_states", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[LSTMState]) -> GodState:
        return env.transform(["model_states", level, "lstm_states", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def autoregressive_predictions(i: int, level: int) -> Accessor[GodState, jax.Array]:
    def get(env: GodState) -> jax.Array:
        t = env.model_states[level].autoregressive_predictions.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: jax.Array) -> GodState:
        return env.transform(["model_states", level, "autoregressive_predictions", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[jax.Array]) -> GodState:
        return env.transform(["model_states", level, "autoregressive_predictions", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


# ============================================================================
# LEARNING STATE ACCESSORS
# ============================================================================


def opt_state(i: int, level: int) -> Accessor[GodState, optax.OptState]:
    def get(env: GodState) -> optax.OptState:
        t = env.learning_states[level].opt_states.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: optax.OptState) -> GodState:
        return env.transform(["learning_states", level, "opt_states", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[optax.OptState]) -> GodState:
        return env.transform(["learning_states", level, "opt_states", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def forward_mode_jacobian(i: int, level: int) -> Accessor[GodState, JACOBIAN]:
    def get(env: GodState) -> JACOBIAN:
        t = env.learning_states[level].influence_tensors.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: JACOBIAN) -> GodState:
        return env.transform(["learning_states", level, "influence_tensors", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[JACOBIAN]) -> GodState:
        return env.transform(["learning_states", level, "influence_tensors", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def uoro_state(i: int, level: int) -> Accessor[GodState, UOROState]:
    def get(env: GodState) -> UOROState:
        t = env.learning_states[level].uoros.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: UOROState) -> GodState:
        return env.transform(["learning_states", level, "uoros", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[UOROState]) -> GodState:
        return env.transform(["learning_states", level, "uoros", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


def midpoint_buffer(i: int, level: int) -> Accessor[GodState, MidpointBuffer]:
    def get(env: GodState) -> MidpointBuffer:
        t = env.learning_states[level].midpoint_buffers.get(i)
        return None if t is None else t.value

    def put(env: GodState, val: MidpointBuffer) -> GodState:
        return env.transform(["learning_states", level, "midpoint_buffers", i, "value"], lambda _: val)

    def put_tagged(env: GodState, tagged: Tagged[MidpointBuffer]) -> GodState:
        return env.transform(["learning_states", level, "midpoint_buffers", i], lambda _: tagged)

    return Accessor(get=get, put=put, put_tagged=put_tagged)


# ============================================================================
# ID MAP
# ============================================================================


def build_id_map(config: GodConfig) -> dict[S_ID, int]:
    keys: list[S_ID] = []

    # shared params — one per node, level=None
    keys.extend((name, None) for name in sorted(config.nodes))

    # per-level states — one per (node, level)
    for level in range(len(config.levels)):
        keys.extend((name, level) for name in sorted(config.nodes))

    # HPs — each at its own level
    keys.extend((name, hp.level) for name, hp in sorted(config.hyperparameters.items()))

    # per-level optimizers
    for level, mc in enumerate(config.levels):
        keys.extend((name, level) for name in sorted(mc.learner.optimizer))

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


# ============================================================================
# LENS — vectorized view over learnable params / stateful states in a slice
# ============================================================================


@dataclass(frozen=True)
class Lens:
    get: Callable[[GodState], jax.Array]
    put: Callable[[GodState, jax.Array], GodState]


def make_lens(model_states: slice, learning_states: slice, params: slice) -> Lens:
    is_leaf = lambda x: x is None or isinstance(x, Tagged)

    def predicate(x):
        if isinstance(x, Tagged):
            match x.meta:
                case ParamMeta():
                    return x.meta.learnable
                case StateMeta():
                    return params.stop in x.meta.is_stateful
        return True

    def get(env: GodState) -> jax.Array:
        combined = (
            env.model_states[model_states],
            env.learning_states[learning_states],
            env.meta_parameters[params],
        )
        filtered, _ = eqx.partition(combined, predicate, is_leaf=is_leaf)
        return to_vector(filtered).vector

    def put(env: GodState, vector: jax.Array) -> GodState:
        combined = (
            env.model_states[model_states],
            env.learning_states[learning_states],
            env.meta_parameters[params],
        )
        filtered, static = eqx.partition(combined, predicate, is_leaf=is_leaf)
        new_model, new_learning, new_params = eqx.combine(to_vector(filtered).to_param(vector), static, is_leaf=is_leaf)
        return env.set(
            model_states=env.model_states[: model_states.start] + new_model + env.model_states[model_states.stop :],
            learning_states=env.learning_states[: learning_states.start]
            + new_learning
            + env.learning_states[learning_states.stop :],
            meta_parameters=env.meta_parameters[: params.start] + new_params + env.meta_parameters[params.stop :],
        )

    return Lens(get=get, put=put)


# ============================================================================
# BUILD INTERFACES
# ============================================================================


def build_interfaces(
    config: GodConfig,
    id_map: dict[S_ID, int],
) -> dict[S_ID, GodInterface[GodState]]:

    interfaces: dict[S_ID, GodInterface[GodState]] = {}
    default = default_god_interface()
    num_levels = len(config.levels)

    # 1. nodes
    for level in range(num_levels):
        for name, node in config.nodes.items():
            pi = id_map[(name, None)]
            si = id_map[(name, level)]
            logs_acc = logs_accessor(level)

            match node:
                case NNLayer():
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        mlp_model=mlp_model(pi, 0),
                    )

                case VanillaRNNLayer(_, _, tc):
                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        rnn_w_rec=rnn_w_rec(pi, 0),
                        rnn_b_rec=rnn_b_rec(pi, 0),
                        rnn_layer_norm=rnn_layer_norm(pi, 0),
                        vanilla_rnn_state=vanilla_rnn_state(si, level),
                        time_constant=read_only(time_constant(hi, hp_cfg.level)),
                    )

                case GRULayer(_, _, _, tc):
                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        gru_cell=gru_cell(pi, 0),
                        gru_activation=gru_activation(si, level),
                        time_constant=read_only(time_constant(hi, hp_cfg.level)),
                    )

                case LSTMLayer(_, _, _, tc):
                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        lstm_cell=lstm_cell(pi, 0),
                        lstm_state=lstm_state(si, level),
                        time_constant=read_only(time_constant(hi, hp_cfg.level)),
                    )

                case Scan():
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                        autoregressive_predictions=autoregressive_predictions(si, level),
                    )

                case ReparameterizeLayer():
                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(si, level),
                        logs=logs_acc,
                    )

                case _:
                    interfaces[(name, level)] = default

    # 2. HPs
    for name, hp in config.hyperparameters.items():
        hi = id_map[(name, hp.level)]
        logs_acc = logs_accessor(hp.level)
        base = copy.replace(
            default,
            prng=prng_accessor(hi, hp.level),
            logs=logs_acc,
        )

        match hp.kind:
            case "learning_rate":
                interfaces[(name, hp.level)] = copy.replace(base, learning_rate=learning_rate(hi, hp.level))
            case "weight_decay":
                interfaces[(name, hp.level)] = copy.replace(base, weight_decay=weight_decay(hi, hp.level))
            case "momentum":
                interfaces[(name, hp.level)] = copy.replace(base, momentum=momentum(hi, hp.level))
            case "time_constant":
                interfaces[(name, hp.level)] = copy.replace(base, time_constant=time_constant(hi, hp.level))
            case "kl_regularizer_beta":
                interfaces[(name, hp.level)] = copy.replace(base, kl_regularizer_beta=kl_regularizer_beta(hi, hp.level))

    # 3. optimizers
    for level, mc in enumerate(config.levels):
        for name, assignment in mc.learner.optimizer.items():
            oi = id_map[(name, level)]
            logs_acc = logs_accessor(level)

            match assignment.optimizer:
                case SGDConfig() | SGDNormalizedConfig() | AdamConfig() | ExponentiatedGradientConfig() as opt:
                    hp_lr = config.hyperparameters[opt.learning_rate]
                    hp_wd = config.hyperparameters[opt.weight_decay]
                    hp_m = config.hyperparameters[opt.momentum]

                    interfaces[(name, level)] = copy.replace(
                        default,
                        prng=prng_accessor(oi, level),
                        logs=logs_acc,
                        opt_state=opt_state(oi, level),
                        learning_rate=learning_rate(id_map[(opt.learning_rate, hp_lr.level)], hp_lr.level),
                        weight_decay=weight_decay(id_map[(opt.weight_decay, hp_wd.level)], hp_wd.level),
                        momentum=momentum(id_map[(opt.momentum, hp_m.level)], hp_m.level),
                    )

                case _:
                    interfaces[(name, level)] = default

    # 4. tasks
    for level, mc in enumerate(config.levels):
        ti = id_map[(TASK, level)]
        logs_acc = logs_accessor(level)
        base = copy.replace(
            default,
            prng=prng_accessor(ti, level),
            logs=logs_acc,
        )

        match mc.objective_fn:
            case ELBOObjective(beta, _):
                hp_cfg = config.hyperparameters[beta]
                hi = id_map[(beta, hp_cfg.level)]
                interfaces[(TASK, level)] = copy.replace(
                    base,
                    kl_regularizer_beta=read_only(kl_regularizer_beta(hi, hp_cfg.level)),
                )
            case _:
                interfaces[(TASK, level)] = base

    # 5. learner states
    for level, mc in enumerate(config.levels):
        for slot_name, learner in [
            (MODEL_LEARNER, mc.learner.model_learner),
            (OPTIMIZER_LEARNER, mc.learner.optimizer_learner),
        ]:
            li = id_map[(slot_name, level)]
            tick_acc = tick_accessor(li, level)
            logs_acc = logs_accessor(level)
            base = copy.replace(
                default,
                prng=prng_accessor(li, level),
                tick=tick_acc,
                logs=logs_acc,
            )

            match learner.method:
                case RTRLConfig() | TikhonovRTRLConfig() | PadeRTRLConfig() | ImplicitEulerRTRLConfig():
                    interfaces[(slot_name, level)] = copy.replace(
                        base,
                        forward_mode_jacobian=forward_mode_jacobian(li, level),
                    )

                case MidpointRTRLConfig() | HeunRTRLConfig():
                    interfaces[(slot_name, level)] = copy.replace(
                        base,
                        forward_mode_jacobian=forward_mode_jacobian(li, level),
                        midpoint_buffer=midpoint_buffer(li, level),
                    )

                case RFLOConfig(tc):
                    hp_cfg = config.hyperparameters[tc]
                    hi = id_map[(tc, hp_cfg.level)]
                    interfaces[(slot_name, level)] = copy.replace(
                        base,
                        forward_mode_jacobian=forward_mode_jacobian(li, level),
                        time_constant=read_only(time_constant(hi, hp_cfg.level)),
                    )

                case UOROConfig():
                    interfaces[(slot_name, level)] = copy.replace(
                        base,
                        uoro_state=uoro_state(li, level),
                    )

                case _:
                    interfaces[(slot_name, level)] = base

    # 6. wire learner state/param lenses
    noop_put_tagged = lambda env, v: env

    for level in range(num_levels):
        for slot_name in (MODEL_LEARNER, OPTIMIZER_LEARNER):
            if slot_name == MODEL_LEARNER:
                state_lens = make_lens(slice(level, level + 1), slice(level, level), slice(level, level))
                param_lens1 = make_lens(slice(0, level), slice(0, level), slice(0, level))
                param_lens2 = make_lens(slice(level, level), slice(level, level), slice(level, level + 1))

                def get_param(env, pl1=param_lens1, pl2=param_lens2):
                    return to_vector((pl1.get(env), pl2.get(env))).vector

                def put_param(env, vector, pl1=param_lens1, pl2=param_lens2):
                    new1, new2 = to_vector((pl1.get(env), pl2.get(env))).to_param(vector)
                    return pl2.put(pl1.put(env, new1), new2)

                param_lens = Lens(get=get_param, put=put_param)
            else:
                state_lens = make_lens(slice(0, level), slice(0, level), slice(0, level))
                param_lens = make_lens(slice(level, level), slice(level, level), slice(level, level + 1))

            iface = interfaces[(slot_name, level)]
            interfaces[(slot_name, level)] = copy.replace(
                iface,
                state=Accessor(get=state_lens.get, put=state_lens.put, put_tagged=noop_put_tagged),
                param=Accessor(get=param_lens.get, put=param_lens.put, put_tagged=noop_put_tagged),
            )

    return interfaces
