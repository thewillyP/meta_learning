import copy
from itertools import islice
import jax
from jaxtyping import PyTree

from lib.config import GodConfig, RFLOConfig
from lib.env import GodState
from lib.interface import (
    GeneralInterface,
    InferenceInterface,
    LearnInterface,
    get_default_general_interface,
    get_default_inference_interface,
    get_default_learn_interface,
)
from lib.lib_types import batched
from lib.util import to_vector
from lib.util_lib import get_optimizer


def get_inference_prng(env: GodState, i: int) -> tuple[jax.Array, GodState]:
    """Assumes you will be operating in vmapped mode so no need to deal with as batched mode"""
    prng, new_prng = jax.random.split(env.prng[i].b)
    return prng, copy.replace(env, prng=env.prng | {i: batched(new_prng)})


def get_learning_prng(env: GodState, i: int) -> tuple[jax.Array, GodState]:
    prng, new_prng = jax.random.split(env.prng_learning[i])
    return prng, copy.replace(env, prng_learning=env.prng_learning | {i: new_prng})


def create_learn_interfaces(config: GodConfig) -> dict[int, LearnInterface[GodState]]:
    default_interpreter: LearnInterface[GodState] = get_default_learn_interface()
    interpreters: dict[int, LearnInterface[GodState]] = {}

    for j, (_, learn_config) in enumerate(sorted(config.learners.items())):

        def get_state_pytree(env: GodState, i) -> PyTree:
            return (
                dict(islice(env.inference_states.items(), i + 1))
                if not config.ignore_validation_inference_recurrence
                else env.inference_states[0],
                dict(islice(env.learning_states.items(), i)),
                dict(islice(env.parameters.items(), i)),
            )

        def get_state(env: GodState, i) -> jax.Array:
            return to_vector(get_state_pytree(env, i)).vector

        def put_state(env: GodState, state: jax.Array, i) -> GodState:
            _inference_states, _learning_states, _params = to_vector(
                (
                    dict(islice(env.inference_states.items(), i + 1))
                    if not config.ignore_validation_inference_recurrence
                    else env.inference_states[0],
                    dict(islice(env.learning_states.items(), i)),
                    dict(islice(env.parameters.items(), i)),
                )
            ).to_param(state)
            if not config.ignore_validation_inference_recurrence:
                inference_states = dict(islice(env.inference_states.items(), i + 1, None)) | _inference_states
            else:
                inference_states = dict(islice(env.inference_states.items(), 1, None)) | _inference_states
            learning_states = dict(islice(env.learning_states.items(), i, None)) | _learning_states
            params = dict(islice(env.parameters.items(), i, None)) | _params
            return copy.replace(
                env,
                inference_states=inference_states,
                learning_states=learning_states,
                parameters=params,
            )

        interpreter = copy.replace(
            default_interpreter,
            get_state_pytree=lambda env, i=j: get_state_pytree(env, i),
            get_state=lambda env, i=j: get_state(env, i),
            put_state=lambda env, state, i=j: put_state(env, state, i),
            get_param=lambda env, i=j: to_vector(env.parameters[i]).vector,
            put_param=lambda env, param, i=j: copy.replace(
                env,
                parameters=env.parameters | {i: to_vector(env.parameters[i]).to_param(param)},
            ),
            get_sgd_param=lambda env, i=j: env.parameters[i + 1].learning_parameter.learning_rate,
            get_optimizer=lambda env, i=j, _lc=learn_config: get_optimizer(_lc)(env.parameters[i + 1]),
            get_opt_state=lambda env, i=j: env.learning_states[i].opt_state,
            put_opt_state=lambda env, opt_state, i=j: copy.replace(
                env,
                learning_states=env.learning_states
                | {
                    i: copy.replace(
                        env.learning_states[i],
                        opt_state=opt_state,
                    )
                },
            ),
            get_rflo_timeconstant=lambda env, i=j: env.parameters[i + 1].learning_parameter.rflo_timeconstant,
            get_influence_tensor=lambda env, i=j: env.learning_states[i].influence_tensor,
            put_influence_tensor=lambda env, influence_tensor, i=j: copy.replace(
                env,
                learning_states=env.learning_states
                | {
                    i: copy.replace(
                        env.learning_states[i],
                        influence_tensor=influence_tensor,
                    )
                },
            ),
            get_uoro=lambda env, i=j: env.learning_states[i].uoro,
            put_uoro=lambda env, uoro, i=j: copy.replace(
                env,
                learning_states=env.learning_states
                | {
                    i: copy.replace(
                        env.learning_states[i],
                        uoro=uoro,
                    )
                },
            ),
            learn_config=learn_config,
            put_logs=lambda env, logs, i=j: copy.replace(
                env,
                general=env.general
                | {
                    i: copy.replace(
                        env.general[i],
                        logs=logs,
                    )
                },
            ),
            put_special_logs=lambda env, special_logs, i=j: copy.replace(
                env,
                general=env.general
                | {
                    i: copy.replace(
                        env.general[i],
                        special_logs=special_logs,
                    )
                },
            ),
            get_prng=lambda env, i=j: get_learning_prng(env, i),
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
            inference_states = env.inference_states | {i: _inference_states}
            return copy.replace(
                env,
                inference_states=inference_states,
            )

        interpreter = copy.replace(
            default_interpreter,
            get_state_pytree=lambda env, i=j: get_state_pytree(env, i),
            get_state=lambda env, i=j: get_state(env, i),
            put_state=lambda env, state, i=j: put_state(env, state, i),
            get_param=lambda env, i=j: learn_interfaces[i].get_state(env),
            put_param=lambda env, param, i=j: learn_interfaces[i].put_state(env, param),
            get_rflo_timeconstant=lambda env: env.parameters[1].learning_parameter.rflo_timeconstant,
            get_influence_tensor=lambda env, i=j: env.validation_learning_states[i].influence_tensor,
            put_influence_tensor=lambda env, influence_tensor, i=j: copy.replace(
                env,
                validation_learning_states=env.validation_learning_states
                | {
                    i: copy.replace(
                        env.validation_learning_states[i],
                        influence_tensor=influence_tensor,
                    )
                },
            ),
            get_uoro=lambda env, i=j: env.validation_learning_states[i].uoro,
            put_uoro=lambda env, uoro, i=j: copy.replace(
                env,
                validation_learning_states=env.validation_learning_states
                | {
                    i: copy.replace(
                        env.validation_learning_states[i],
                        uoro=uoro,
                    )
                },
            ),
            learn_config=config.learners[0],
            put_logs=lambda env, logs, i=j: copy.replace(
                env,
                general=env.general
                | {
                    i: copy.replace(
                        env.general[i],
                        logs=logs,
                    )
                },
            ),
            put_special_logs=lambda env, special_logs, i=j: copy.replace(
                env,
                general=env.general
                | {
                    i: copy.replace(
                        env.general[i],
                        special_logs=special_logs,
                    )
                },
            ),
            get_prng=lambda env, i=j: get_learning_prng(env, i),
        )
        interpreters[j] = interpreter

    return interpreters


def create_transition_interfaces(config: GodConfig) -> dict[int, dict[int, InferenceInterface[GodState]]]:
    default_interpreter: InferenceInterface[GodState] = get_default_inference_interface()
    interpreters: dict[int, dict[int, InferenceInterface[GodState]]] = {}
    match config.learners[min(config.learners.keys())].learner:
        case RFLOConfig(_time_constant):
            time_constant = _time_constant
        case _:
            time_constant = 1.0

    for j, _ in enumerate(sorted(config.data.items())):
        _interpreter = copy.replace(
            default_interpreter,
            get_readout_param=lambda env: env.parameters[0].readout_fn,
            get_prng=lambda env, i=j: get_inference_prng(env, i),
            _get_prng=lambda env, i=j: env.prng[i],
            get_rflo_timeconstant=lambda env: time_constant,
        )
        for k, _ in sorted(config.transition_function.items()):
            interpreter = copy.replace(
                _interpreter,
                get_rnn_state=lambda env, i=j, l=k: env.inference_states[i][l].rnn,
                put_rnn_state=lambda env, rnn_state, i=j, l=k: copy.replace(
                    env,
                    inference_states=env.inference_states
                    | {
                        i: env.inference_states[i]
                        | {
                            l: copy.replace(
                                env.inference_states[i][l],
                                rnn=rnn_state,
                            )
                        }
                    },
                ),
                get_rnn_param=lambda env, l=k: env.parameters[0].transition_parameter[l].rnn,
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
            put_current_virtual_minibatch=lambda env, value, i=j: copy.replace(
                env,
                general=env.general
                | {
                    i: copy.replace(
                        env.general[i],
                        current_virtual_minibatch=value,
                    )
                },
            ),
        )
        interpreters[j] = interpreter

    return interpreters
