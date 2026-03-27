import copy
import jax
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.env import GodState
from meta_learn_lib.interface import *
from meta_learn_lib.util import to_vector


def prng_factory(key: int, level: int):
    def get_prng(env: GodState) -> tuple[jax.Array, GodState]:
        prng, new_prng = jax.random.split(env.level_meta[level].prngs[key].value)
        return prng, env.transform(
            ["level_meta", level, "prngs", key], lambda _: State(value=new_prng, is_stateful=frozenset())
        )

    return get_prng


def put_prng(key: int, level: int):
    def put_prng_fn(env: GodState, prng: jax.Array) -> GodState:
        return env.transform(["level_meta", level, "prngs", key], lambda _: State(value=prng, is_stateful=frozenset()))

    return put_prng_fn


def get_tick(i: int, level: int):
    def get_tick_fn(env: GodState) -> jax.Array:
        return env.level_meta[level].ticks.get(i).value

    return get_tick_fn


def put_tick(i: int, level: int):
    def put_tick_fn(env: GodState, tick: jax.Array) -> GodState:
        return env.transform(["level_meta", level, "ticks", i], lambda _: State(value=tick, is_stateful=frozenset()))

    return put_tick_fn


def advance_tick(i: int, level: int):
    def advance_tick_fn(env: GodState) -> GodState:
        t = get_tick(i, level)(env)
        return put_tick(i, level)(env, t + 1)

    return advance_tick_fn


def get_mlp_param(i: int, level: int):
    def get_mlp_param_fn(env: GodState) -> MLP:
        return env.meta_parameters[level].mlps.get(i)

    return get_mlp_param_fn


def put_mlp_param(i: int, level: int):
    def put_mlp_param_fn(env: GodState, param: MLP) -> GodState:
        return env.transform(["meta_parameters", level, "mlps", i], lambda _: param)

    return put_mlp_param_fn


def get_vanilla_rnn_state(i: int, level: int):
    def get_vanilla_rnn_state_fn(env: GodState) -> VanillaRecurrentState:
        return env.model_states[level].vanilla_recurrent_states.get(i)

    return get_vanilla_rnn_state_fn


def put_vanilla_rnn_state(i: int, level: int):
    def put_vanilla_rnn_state_fn(env: GodState, state: VanillaRecurrentState) -> GodState:
        return env.transform(["model_states", level, "vanilla_recurrent_states", i], lambda _: state)

    return put_vanilla_rnn_state_fn


def get_vanilla_rnn_param(i: int, level: int):
    def get_vanilla_rnn_param_fn(env: GodState) -> RNN:
        return env.meta_parameters[level].rnns.get(i)

    return get_vanilla_rnn_param_fn


def put_vanilla_rnn_param(i: int, level: int):
    def put_vanilla_rnn_param_fn(env: GodState, param: RNN) -> GodState:
        return env.transform(["meta_parameters", level, "rnns", i], lambda _: param)

    return put_vanilla_rnn_param_fn


def get_gru_state(i: int, level: int):
    def get_gru_state_fn(env: GodState) -> RecurrentState:
        return env.model_states[level].recurrent_states.get(i)

    return get_gru_state_fn


def put_gru_state(i: int, level: int):
    def put_gru_state_fn(env: GodState, state: RecurrentState) -> GodState:
        return env.transform(["model_states", level, "recurrent_states", i], lambda _: state)

    return put_gru_state_fn


def get_gru_param(i: int, level: int):
    def get_gru_param_fn(env: GodState) -> Parameter[eqx.nn.GRUCell]:
        return env.meta_parameters[level].grus.get(i)

    return get_gru_param_fn


def put_gru_param(i: int, level: int):
    def put_gru_param_fn(env: GodState, param: Parameter[eqx.nn.GRUCell]) -> GodState:
        return env.transform(["meta_parameters", level, "grus", i], lambda _: param)

    return put_gru_param_fn


def get_lstm_state(i: int, level: int):
    def get_lstm_state_fn(env: GodState) -> LSTMState:
        return env.model_states[level].lstm_states.get(i)

    return get_lstm_state_fn


def put_lstm_state(i: int, level: int):
    def put_lstm_state_fn(env: GodState, state: LSTMState) -> GodState:
        return env.transform(["model_states", level, "lstm_states", i], lambda _: state)

    return put_lstm_state_fn


def get_lstm_param(i: int, level: int):
    def get_lstm_param_fn(env: GodState) -> Parameter[eqx.nn.LSTMCell]:
        return env.meta_parameters[level].lstms.get(i)

    return get_lstm_param_fn


def put_lstm_param(i: int, level: int):
    def put_lstm_param_fn(env: GodState, param: Parameter[eqx.nn.LSTMCell]) -> GodState:
        return env.transform(["meta_parameters", level, "lstms", i], lambda _: param)

    return put_lstm_param_fn


def get_autoregressive_predictions(i: int, level: int):
    def get_autoregressive_predictions_fn(env: GodState) -> State[jax.Array]:
        return env.model_states[level].autoregressive_predictions.get(i)

    return get_autoregressive_predictions_fn


def put_autoregressive_predictions(i: int, level: int):
    def put_autoregressive_predictions_fn(env: GodState, predictions: State[jax.Array]) -> GodState:
        return env.transform(["model_states", level, "autoregressive_predictions", i])

    return put_autoregressive_predictions_fn


def get_time_constant(i: int, level: int):
    def get_time_constant_fn(env: GodState) -> Parameter[jax.Array]:
        return env.meta_parameters[level].time_constants.get(i)

    return get_time_constant_fn


def put_time_constant(i: int, level: int):
    def put_time_constant_fn(env: GodState, time_constant: Parameter[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "time_constants", i], lambda _: time_constant)

    return put_time_constant_fn


def get_learning_rate(i: int, level: int):
    def get_learning_rate_fn(env: GodState) -> Parameter[jax.Array]:
        return env.meta_parameters[level].learning_rates.get(i)

    return get_learning_rate_fn


def put_learning_rate(i: int, level: int):
    def put_learning_rate_fn(env: GodState, learning_rate: Parameter[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "learning_rates", i], lambda _: learning_rate)

    return put_learning_rate_fn


def get_weight_decay(i: int, level: int):
    def get_weight_decay_fn(env: GodState) -> Parameter[jax.Array]:
        return env.meta_parameters[level].weight_decays.get(i)

    return get_weight_decay_fn


def put_weight_decay(i: int, level: int):
    def put_weight_decay_fn(env: GodState, weight_decay: Parameter[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "weight_decays", i], lambda _: weight_decay)

    return put_weight_decay_fn


def get_momentum(i: int, level: int):
    def get_momentum_fn(env: GodState) -> Parameter[jax.Array]:
        return env.meta_parameters[level].momentums.get(i)

    return get_momentum_fn


def put_momentum(i: int, level: int):
    def put_momentum_fn(env: GodState, momentum: Parameter[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "momentums", i], lambda _: momentum)

    return put_momentum_fn


def get_kl_regularizer_beta(i: int, level: int):
    def get_kl_regularizer_beta_fn(env: GodState) -> Parameter[jax.Array]:
        return env.meta_parameters[level].kl_regularizer_betas.get(i)

    return get_kl_regularizer_beta_fn


def put_kl_regularizer_beta(i: int, level: int):
    def put_kl_regularizer_beta_fn(env: GodState, kl_regularizer_beta: Parameter[jax.Array]) -> GodState:
        return env.transform(["meta_parameters", level, "kl_regularizer_betas", i], lambda _: kl_regularizer_beta)

    return put_kl_regularizer_beta_fn


def get_opt_state(i: int, level: int):
    def get_opt_state_fn(env: GodState) -> State[optax.OptState]:
        return env.learning_states[level].opt_states.get(i)

    return get_opt_state_fn


def put_opt_state(i: int, level: int):
    def put_opt_state_fn(env: GodState, opt_state: State[optax.OptState]) -> GodState:
        return env.transform(["learning_states", level, "opt_states", i], lambda _: opt_state)

    return put_opt_state_fn


def get_forward_mode_jacobian(i: int, level: int):
    def get_forward_mode_jacobian_fn(env: GodState) -> State[JACOBIAN]:
        return env.learning_states[level].influence_tensors.get(i)

    return get_forward_mode_jacobian_fn


def put_forward_mode_jacobian(i: int, level: int):
    def put_forward_mode_jacobian_fn(env: GodState, jacobian: State[JACOBIAN]) -> GodState:
        return env.transform(["learning_states", level, "influence_tensors", i], lambda _: jacobian)

    return put_forward_mode_jacobian_fn


def get_uoro_state(i: int, level: int):
    def get_uoro_state_fn(env: GodState) -> UOROState:
        return env.learning_states[level].uoros.get(i)

    return get_uoro_state_fn


def put_uoro_state(i: int, level: int):
    def put_uoro_state_fn(env: GodState, uoro_state: UOROState) -> GodState:
        return env.transform(["learning_states", level, "uoros", i], lambda _: uoro_state)

    return put_uoro_state_fn


@dataclass(frozen=True)
class Lens:
    get: Callable[[GodState], jax.Array]
    put: Callable[[GodState, jax.Array], GodState]


def make_lens(model_states: slice, learning_states: slice, params: slice) -> Lens:
    is_leaf = lambda x: x is None or isinstance(x, (Parameter, State))

    def predicate(x):
        if isinstance(x, Parameter):
            return x.is_learnable
        if isinstance(x, State):
            return params.stop in x.is_stateful
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


def put_logs(level: int):
    def put_logs_fn(env: GodState, logs: Logs) -> GodState:
        return env.transform(
            ["level_meta", level, "log"],
            lambda old: State(
                value=old.value.set(
                    **{k: v for k, v in logs.serialize().items() if v is not None},
                ),
                is_stateful=old.is_stateful,
            ),
        )

    return put_logs_fn


def get_logs(level: int):
    def get_logs_fn(env: GodState) -> Logs:
        return env.level_meta[level].log.value

    return get_logs_fn


def create_meta_interfaces(config: GodConfig, i: int) -> tuple[list[dict[str, GodInterface[GodState]]], int]:
    i = max(i, 0) + len(config.hyperparameters)

    default_interface: GodInterface[GodState] = default_god_interface()
    meta_interfaces: list[dict[str, GodInterface[GodState]]] = [{} for _ in range(len(config.levels))]

    # 1. nodes first
    # Compute shared param ids once
    node_ids: dict[str, int] = {}
    for name, node in config.nodes.items():
        i += 1
        node_ids[name] = i

    # Do parameter sharing base model across all levels.
    for level in range(len(config.levels)):
        for name, node in config.nodes.items():
            i += 1
            ni = node_ids[name]
            match node:
                case NNLayer():
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_mlp_param=get_mlp_param(ni, 0),
                        put_mlp_param=put_mlp_param(ni, 0),
                    )
                case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_vanilla_rnn_state=get_vanilla_rnn_state(i, level),
                        put_vanilla_rnn_state=put_vanilla_rnn_state(i, level),
                        get_vanilla_rnn_param=get_vanilla_rnn_param(ni, 0),
                        put_vanilla_rnn_param=put_vanilla_rnn_param(ni, 0),
                        get_time_constant=get_time_constant(
                            [*config.hyperparameters].index(time_constant),
                            config.hyperparameters[time_constant].level,
                        ),
                    )
                case GRULayer(n, use_bias, use_random_init, time_constant):
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_gru_state=get_gru_state(i, level),
                        put_gru_state=put_gru_state(i, level),
                        get_gru_param=get_gru_param(ni, 0),
                        put_gru_param=put_gru_param(ni, 0),
                        get_time_constant=get_time_constant(
                            [*config.hyperparameters].index(time_constant),
                            config.hyperparameters[time_constant].level,
                        ),
                    )
                case LSTMLayer(n, use_bias, use_random_init, time_constant):
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_lstm_state=get_lstm_state(i, level),
                        put_lstm_state=put_lstm_state(i, level),
                        get_lstm_param=get_lstm_param(ni, 0),
                        put_lstm_param=put_lstm_param(ni, 0),
                        get_time_constant=get_time_constant(
                            [*config.hyperparameters].index(time_constant),
                            config.hyperparameters[time_constant].level,
                        ),
                    )
                case Scan():
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_autoregressive_predictions=get_autoregressive_predictions(i, level),
                        put_autoregressive_predictions=put_autoregressive_predictions(i, level),
                    )
                case ReparameterizeLayer():
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                    )
                case _:
                    interface = default_interface

            meta_interfaces[level][name] = interface

    # 2. optimizers
    for level, meta_config in enumerate(config.levels):
        for name, assignment in meta_config.learner.optimizer.items():
            i += 1
            match assignment.optimizer:
                case SGDConfig() | SGDNormalizedConfig() | AdamConfig() | ExponentiatedGradientConfig() as opt:
                    interface = copy.replace(
                        default_interface,
                        put_logs=put_logs(level),
                        get_logs=get_logs(level),
                        take_prng=prng_factory(i, level),
                        put_prng=put_prng(i, level),
                        get_opt_state=get_opt_state(i, level),
                        put_opt_state=put_opt_state(i, level),
                        get_learning_rate=get_learning_rate(
                            [*config.hyperparameters].index(opt.learning_rate),
                            config.hyperparameters[opt.learning_rate].level,
                        ),
                        put_learning_rate=put_learning_rate(
                            [*config.hyperparameters].index(opt.learning_rate),
                            config.hyperparameters[opt.learning_rate].level,
                        ),
                        get_weight_decay=get_weight_decay(
                            [*config.hyperparameters].index(opt.weight_decay),
                            config.hyperparameters[opt.weight_decay].level,
                        ),
                        put_weight_decay=put_weight_decay(
                            [*config.hyperparameters].index(opt.weight_decay),
                            config.hyperparameters[opt.weight_decay].level,
                        ),
                        get_momentum=get_momentum(
                            [*config.hyperparameters].index(opt.momentum),
                            config.hyperparameters[opt.momentum].level,
                        ),
                        put_momentum=put_momentum(
                            [*config.hyperparameters].index(opt.momentum),
                            config.hyperparameters[opt.momentum].level,
                        ),
                    )
                case _:
                    interface = default_interface

            meta_interfaces[level][name] = interface

    # 3. all hyperparameters
    for name, hp in config.hyperparameters.items():
        idx = [*config.hyperparameters].index(name)
        match hp.kind:
            case "learning_rate":
                interface = copy.replace(
                    default_interface,
                    put_logs=put_logs(hp.level),
                    get_logs=get_logs(hp.level),
                    take_prng=prng_factory(idx, hp.level),
                    put_prng=put_prng(idx, hp.level),
                    get_learning_rate=get_learning_rate(idx, hp.level),
                    put_learning_rate=put_learning_rate(idx, hp.level),
                )
            case "weight_decay":
                interface = copy.replace(
                    default_interface,
                    put_logs=put_logs(hp.level),
                    get_logs=get_logs(hp.level),
                    take_prng=prng_factory(idx, hp.level),
                    put_prng=put_prng(idx, hp.level),
                    get_weight_decay=get_weight_decay(idx, hp.level),
                    put_weight_decay=put_weight_decay(idx, hp.level),
                )
            case "momentum":
                interface = copy.replace(
                    default_interface,
                    put_logs=put_logs(hp.level),
                    get_logs=get_logs(hp.level),
                    take_prng=prng_factory(idx, hp.level),
                    put_prng=put_prng(idx, hp.level),
                    get_momentum=get_momentum(idx, hp.level),
                    put_momentum=put_momentum(idx, hp.level),
                )
            case "time_constant":
                interface = copy.replace(
                    default_interface,
                    put_logs=put_logs(hp.level),
                    get_logs=get_logs(hp.level),
                    take_prng=prng_factory(idx, hp.level),
                    put_prng=put_prng(idx, hp.level),
                    get_time_constant=get_time_constant(idx, hp.level),
                    put_time_constant=put_time_constant(idx, hp.level),
                )
            case "kl_regularizer_beta":
                interface = copy.replace(
                    default_interface,
                    put_logs=put_logs(hp.level),
                    get_logs=get_logs(hp.level),
                    take_prng=prng_factory(idx, hp.level),
                    put_prng=put_prng(idx, hp.level),
                    get_kl_regularizer_beta=get_kl_regularizer_beta(idx, hp.level),
                    put_kl_regularizer_beta=put_kl_regularizer_beta(idx, hp.level),
                )
        meta_interfaces[hp.level][name] = interface

    return meta_interfaces, i


def create_learn_interfaces(
    config: GodConfig, i: int
) -> tuple[list[tuple[GodInterface[GodState], GodInterface[GodState]]], int]:

    def learner_to_interface(learner: GradientMethod, i: int, level: int, is_val: bool) -> GodInterface[GodState]:
        # for level 0, the val learner should actually use the non val's learner
        if is_val:
            state_lens = make_lens(slice(level, level + 1), slice(level, level), slice(level, level))
            param_lens1 = make_lens(slice(0, level), slice(0, level), slice(0, level))
            param_lens2 = make_lens(slice(level, level), slice(level, level), slice(level, level + 1))

            def get_param(env: GodState) -> jax.Array:
                return to_vector((param_lens1.get(env), param_lens2.get(env))).vector

            def put_param(env: GodState, vector: jax.Array) -> GodState:
                new_param1, new_param2 = to_vector((param_lens1.get(env), param_lens2.get(env))).to_param(vector)
                env = param_lens1.put(env, new_param1)
                env = param_lens2.put(env, new_param2)
                return env

            param_lens = Lens(get=get_param, put=put_param)
        else:
            state_lens = make_lens(slice(0, level), slice(0, level), slice(0, level))
            param_lens = make_lens(slice(level, level), slice(level, level), slice(level, level + 1))

        match learner:
            case RTRLConfig() | RTRLFiniteHvpConfig():
                return copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                    get_tick=get_tick(i, level),
                    put_tick=put_tick(i, level),
                    advance_tick=advance_tick(i, level),
                    get_state=state_lens.get,
                    put_state=state_lens.put,
                    get_param=param_lens.get,
                    put_param=param_lens.put,
                    get_forward_mode_jacobian=get_forward_mode_jacobian(i, level),
                    put_forward_mode_jacobian=put_forward_mode_jacobian(i, level),
                )
            case RFLOConfig(time_constant):
                return copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                    get_tick=get_tick(i, level),
                    put_tick=put_tick(i, level),
                    advance_tick=advance_tick(i, level),
                    get_state=state_lens.get,
                    put_state=state_lens.put,
                    get_param=param_lens.get,
                    put_param=param_lens.put,
                    get_forward_mode_jacobian=get_forward_mode_jacobian(i, level),
                    put_forward_mode_jacobian=put_forward_mode_jacobian(i, level),
                    get_time_constant=get_time_constant(
                        [*config.hyperparameters].index(time_constant),
                        config.hyperparameters[time_constant].level,
                    ),
                )
            case UOROConfig() | UOROFiniteDiffConfig():
                return copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                    get_tick=get_tick(i, level),
                    put_tick=put_tick(i, level),
                    advance_tick=advance_tick(i, level),
                    get_state=state_lens.get,
                    put_state=state_lens.put,
                    get_param=param_lens.get,
                    put_param=param_lens.put,
                    get_uoro_state=get_uoro_state(i, level),
                    put_uoro_state=put_uoro_state(i, level),
                )
            case _:
                return copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                    get_tick=get_tick(i, level),
                    put_tick=put_tick(i, level),
                    advance_tick=advance_tick(i, level),
                    get_state=state_lens.get,
                    put_state=state_lens.put,
                    get_param=param_lens.get,
                    put_param=param_lens.put,
                )

    i = max(i, 0) + len(config.hyperparameters)
    meta_interfaces: list[tuple[GodInterface[GodState], GodInterface[GodState]]] = []
    for level, meta_config in enumerate(config.levels):
        i += 2
        model_interface = learner_to_interface(meta_config.learner.model_learner.method, i, level, True)
        optimizer_interface = learner_to_interface(meta_config.learner.optimizer_learner.method, i + 1, level, False)
        meta_interfaces.append((model_interface, optimizer_interface))

    return meta_interfaces, i


def create_task_interfaces(config: GodConfig, i: int) -> tuple[list[GodInterface[GodState]], int]:
    i = max(i, 0) + len(config.hyperparameters)
    meta_interfaces: list[GodInterface[GodState]] = []
    for level, meta_config in enumerate(config.levels):
        match meta_config.objective_fn:
            case ELBOObjective(beta, likelihood):
                interface = copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                    get_kl_regularizer_beta=get_kl_regularizer_beta(
                        [*config.hyperparameters].index(beta),
                        config.hyperparameters[beta].level,
                    ),
                )
            case _:
                interface = copy.replace(
                    default_god_interface(),
                    put_logs=put_logs(level),
                    get_logs=get_logs(level),
                    take_prng=prng_factory(i, level),
                    put_prng=put_prng(i, level),
                )
        meta_interfaces.append(interface)

    return meta_interfaces, i
