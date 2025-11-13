import copy
from itertools import islice
import jax
from jaxtyping import PyTree
import equinox as eqx

from meta_learn_lib.config import GodConfig, RFLOConfig
from meta_learn_lib.env import GodState, Hyperparameter, Parameter
from meta_learn_lib.interface import (
    GeneralInterface,
    InferenceInterface,
    LearnInterface,
    get_default_general_interface,
    get_default_inference_interface,
    get_default_learn_interface,
)
from meta_learn_lib.lib_types import batched
from meta_learn_lib.util import to_vector
from meta_learn_lib.util_lib import get_optimizer, get_updater


def get_inference_prng(env: GodState, i: int) -> tuple[jax.Array, GodState]:
    """Assumes you will be operating in vmapped mode so no need to deal with as batched mode"""
    prng, new_prng = jax.random.split(env.prng[i].b)
    return prng, env.transform(["prng", i], lambda _: batched(new_prng))


def get_learning_prng(env: GodState, i: int) -> tuple[jax.Array, GodState]:
    prng, new_prng = jax.random.split(env.prng_learning[i])
    return prng, env.transform(["prng_learning", i], lambda _: new_prng)


def filter_hyperparam(pytree: PyTree) -> jax.Array:
    learnable, static = eqx.partition(
        pytree,
        lambda x: x.learnable if isinstance(x, Hyperparameter) else True,
        is_leaf=lambda x: x is None or isinstance(x, Hyperparameter),
    )
    return to_vector(learnable).vector


def obtain_param(env: GodState, i: int) -> jax.Array:
    return filter_hyperparam(env.parameters[i])


def vector_to_param(env: GodState, param: jax.Array, i: int) -> Parameter:
    learnable, static = eqx.partition(
        env.parameters[i],
        lambda x: x.learnable if isinstance(x, Hyperparameter) else True,
        is_leaf=lambda x: x is None or isinstance(x, Hyperparameter),
    )
    _learnable = to_vector(learnable).to_param(param)
    return eqx.combine(_learnable, static, is_leaf=lambda x: x is None or isinstance(x, Hyperparameter))


def place_param(env: GodState, param: jax.Array, i: int) -> GodState:
    new_param = vector_to_param(env, param, i)
    return env.transform(["parameters", i], lambda _: new_param)


def create_learn_interfaces(config: GodConfig) -> dict[int, LearnInterface[GodState]]:
    default_interpreter: LearnInterface[GodState] = get_default_learn_interface()
    interpreters: dict[int, LearnInterface[GodState]] = {}

    def get_state_pytree(env: GodState, i) -> PyTree:
        if i == 0 or config.treat_inference_state_as_online:
            inference_part = (
                dict(islice(env.inference_states.items(), i + 1))
                if not config.ignore_validation_inference_recurrence
                else {0: env.inference_states[0]}
            )
        else:
            inference_part = {}

        state = (
            inference_part,
            dict(islice(env.learning_states.items(), i)),
            dict(islice(env.parameters.items(), i)),
        )
        learnable, static = eqx.partition(
            state,
            lambda x: x.learnable if isinstance(x, Hyperparameter) else True,
            is_leaf=lambda x: x is None or isinstance(x, Hyperparameter),
        )
        return learnable

    def get_state(env: GodState, i) -> jax.Array:
        return to_vector(get_state_pytree(env, i)).vector

    def put_state(env: GodState, state: jax.Array, i) -> GodState:
        if i == 0 or config.treat_inference_state_as_online:
            inference_part = (
                dict(islice(env.inference_states.items(), i + 1))
                if not config.ignore_validation_inference_recurrence
                else {0: env.inference_states[0]}
            )
        else:
            inference_part = {}

        temp = (
            inference_part,
            dict(islice(env.learning_states.items(), i)),
            dict(islice(env.parameters.items(), i)),
        )
        learnable, static = eqx.partition(
            temp,
            lambda x: x.learnable if isinstance(x, Hyperparameter) else True,
            is_leaf=lambda x: x is None or isinstance(x, Hyperparameter),
        )
        _learnable = to_vector(learnable).to_param(state)
        _inference_states, _learning_states, _params = eqx.combine(
            _learnable, static, is_leaf=lambda x: x is None or isinstance(x, Hyperparameter)
        )

        inference_states = env.inference_states.update(_inference_states)
        learning_states = env.learning_states.update(_learning_states)
        params = env.parameters.update(_params)

        return env.set(inference_states=inference_states, learning_states=learning_states, parameters=params)

    for j, (_, learn_config) in enumerate(sorted(config.learners.items())):
        interpreter = copy.replace(
            default_interpreter,
            get_state_pytree=lambda env, i=j: get_state_pytree(env, i),
            get_state=lambda env, i=j: get_state(env, i),
            put_state=lambda env, state, i=j: put_state(env, state, i),
            get_param=lambda env, i=j: obtain_param(env, i),
            put_param=lambda env, param, i=j: place_param(env, param, i),
            get_sgd_param=lambda env, i=j: env.parameters[i + 1].learning_parameter.learning_rate,
            get_optimizer=lambda env, i=j, _lc=learn_config: get_optimizer(
                _lc.optimizer, lambda p: vector_to_param(env, p, i)
            )(env.parameters[i + 1].learning_parameter),
            get_updater=get_updater(learn_config),
            get_opt_state=lambda env, i=j: env.learning_states[i].opt_state,
            put_opt_state=lambda env, opt_state, i=j: env.transform(
                ["learning_states", i, "opt_state"], lambda _: opt_state
            ),
            get_rflo_timeconstant=lambda env, i=j: env.parameters[i + 1].learning_parameter.rflo_timeconstant,
            get_influence_tensor=lambda env, i=j: env.learning_states[i].influence_tensor,
            put_influence_tensor=lambda env, influence_tensor, i=j: env.transform(
                ["learning_states", i, "influence_tensor"], lambda _: influence_tensor
            ),
            get_influence_tensor_squared=lambda env, i=j: env.learning_states[i].influence_tensor_squared,
            put_influence_tensor_squared=lambda env, influence_tensor_squared, i=j: env.transform(
                ["learning_states", i, "influence_tensor_squared"], lambda _: influence_tensor_squared
            ),
            get_uoro=lambda env, i=j: env.learning_states[i].uoro,
            put_uoro=lambda env, uoro, i=j: env.transform(["learning_states", i, "uoro"], lambda _: uoro),
            learn_config=learn_config,
            put_logs=lambda env, logs, i=j: env.transform(
                ["general", i, "logs"],
                lambda old: old.set(
                    **{k: getattr(logs, k) for k in logs._pclass_fields if getattr(logs, k) is not None}
                ),
            ),
            put_special_logs=lambda env, special_logs, i=j: env.transform(
                ["general", i, "special_logs"],
                lambda old: old.set(
                    **{
                        k: getattr(special_logs, k)
                        for k in special_logs._pclass_fields
                        if getattr(special_logs, k) is not None
                    }
                ),
            ),
            get_prng=lambda env, i=j: get_learning_prng(env, i),
            get_rflo_t=lambda env, i=j: env.learning_states[i].rflo_t,
            put_rflo_t=lambda env, t, i=j: env.transform(["learning_states", i, "rflo_t"], lambda _: t),
            get_rtrl_t=lambda env, i=j: env.learning_states[i].rtrl_t,
            put_rtrl_t=lambda env, t, i=j: env.transform(["learning_states", i, "rtrl_t"], lambda _: t),
        )
        interpreters[j] = interpreter

    return interpreters


def create_validation_learn_interfaces(
    config: GodConfig, learn_interfaces: dict[int, LearnInterface[GodState]]
) -> dict[int, LearnInterface[GodState]]:
    default_interpreter: LearnInterface[GodState] = get_default_learn_interface()
    interpreters: dict[int, LearnInterface[GodState]] = {}

    for j, _ in enumerate(islice(config.learners.items(), 1, None), 1):

        def get_state_pytree(env: GodState, i) -> PyTree:
            return env.inference_states[i]

        def get_state(env: GodState, i) -> jax.Array:
            return to_vector(get_state_pytree(env, i)).vector

        def put_state(env: GodState, state: jax.Array, i) -> GodState:
            _inference_states = to_vector(env.inference_states[i]).to_param(state)
            return env.transform(["inference_states", i], lambda _: _inference_states)

        interpreter = copy.replace(
            default_interpreter,
            get_state_pytree=lambda env, i=j: get_state_pytree(env, i),
            get_state=lambda env, i=j: get_state(env, i),
            put_state=lambda env, state, i=j: put_state(env, state, i),
            get_param=lambda env, i=j: learn_interfaces[i].get_state(env),
            put_param=lambda env, param, i=j: learn_interfaces[i].put_state(env, param),
            get_rflo_timeconstant=lambda env: env.parameters[1].learning_parameter.rflo_timeconstant,
            get_influence_tensor=lambda env, i=j: env.validation_learning_states[i].influence_tensor,
            put_influence_tensor=lambda env, influence_tensor, i=j: env.transform(
                ["validation_learning_states", i, "influence_tensor"], lambda _: influence_tensor
            ),
            get_influence_tensor_squared=lambda env, i=j: env.validation_learning_states[i].influence_tensor_squared,
            put_influence_tensor_squared=lambda env, influence_tensor_squared, i=j: env.transform(
                ["validation_learning_states", i, "influence_tensor_squared"], lambda _: influence_tensor_squared
            ),
            get_uoro=lambda env, i=j: env.validation_learning_states[i].uoro,
            put_uoro=lambda env, uoro, i=j: env.transform(["validation_learning_states", i, "uoro"], lambda _: uoro),
            learn_config=config.learners[0],
            put_logs=lambda env, logs, i=j: env.transform(
                ["general", i, "logs"],
                lambda old: old.set(
                    **{k: getattr(logs, k) for k in logs._pclass_fields if getattr(logs, k) is not None}
                ),
            ),
            put_special_logs=lambda env, special_logs, i=j: env.transform(
                ["general", i, "special_logs"],
                lambda old: old.set(
                    **{
                        k: getattr(special_logs, k)
                        for k in special_logs._pclass_fields
                        if getattr(special_logs, k) is not None
                    }
                ),
            ),
            get_prng=lambda env, i=j: get_learning_prng(env, i),
            get_rflo_t=lambda env, i=j: env.validation_learning_states[i].rflo_t,
            put_rflo_t=lambda env, t, i=j: env.transform(["validation_learning_states", i, "rflo_t"], lambda _: t),
            get_rtrl_t=lambda env, i=j: env.validation_learning_states[i].rtrl_t,
            put_rtrl_t=lambda env, t, i=j: env.transform(["validation_learning_states", i, "rtrl_t"], lambda _: t),
        )
        interpreters[j] = interpreter

    return interpreters


def create_transition_interfaces(config: GodConfig) -> dict[int, dict[int, InferenceInterface[GodState]]]:
    default_interpreter: InferenceInterface[GodState] = get_default_inference_interface()
    interpreters: dict[int, dict[int, InferenceInterface[GodState]]] = {}
    match config.learners[min(config.learners.keys())].learner:
        case RFLOConfig(_time_constant, use_reverse_mode):
            time_constant = _time_constant
        case _:
            time_constant = 1.0

    def get_state(env: GodState) -> jax.Array:
        return to_vector(env.inference_states[0]).vector

    for j, _ in enumerate(sorted(config.data.items())):
        _interpreter = copy.replace(
            default_interpreter,
            get_state=lambda env: get_state(env),
            get_readout_param=lambda env: env.parameters[0].readout_fn,
            get_prng=lambda env, i=j: get_inference_prng(env, i),
            _get_prng=lambda env, i=j: env.prng[i],
            get_rflo_timeconstant=lambda env: time_constant,
        )
        for k, _ in sorted(config.transition_function.items()):
            interpreter = copy.replace(
                _interpreter,
                get_rnn_state=lambda env, i=j, l=k: env.inference_states[i][l].rnn,
                put_rnn_state=lambda env, rnn_state, i=j, l=k: env.transform(
                    ["inference_states", i, l, "rnn"], lambda _: rnn_state
                ),
                get_rnn_param=lambda env, l=k: env.parameters[0].transition_parameter.param[l].rnn,
                get_lstm_state=lambda env, i=j, l=k: env.inference_states[i][l].lstm,
                put_lstm_state=lambda env, lstm_state, i=j, l=k: env.transform(
                    ["inference_states", i, l, "lstm"], lambda _: lstm_state
                ),
                get_lstm_param=lambda env, l=k: env.parameters[0].transition_parameter.param[l].lstm,
                get_gru_param=lambda env, l=k: env.parameters[0].transition_parameter.param[l].gru,
            )
            interpreters.setdefault(j, {})[k] = interpreter

    return interpreters


def create_general_interfaces(config: GodConfig) -> dict[int, GeneralInterface[GodState]]:
    default_interpreter: GeneralInterface[GodState] = get_default_general_interface()
    interpreters: dict[int, GeneralInterface[GodState]] = {}

    for j, _ in enumerate(sorted(config.data.items())):
        interpreter = copy.replace(
            default_interpreter,
            get_current_virtual_minibatch=lambda env, i=j: env.general[i].current_virtual_minibatch,
            put_current_virtual_minibatch=lambda env, value, i=j: env.transform(
                ["general", i, "current_virtual_minibatch"], lambda _: value
            ),
            put_logs=lambda env, logs, i=j: env.transform(
                ["general", i, "logs"],
                lambda old: old.set(
                    **{k: getattr(logs, k) for k in logs._pclass_fields if getattr(logs, k) is not None}
                ),
            ),
        )
        interpreters[j] = interpreter

    return interpreters
