import copy
from itertools import islice
import jax

from lib.config import GodConfig
from lib.env import GodState
from lib.interface import (
    InferenceInterface,
    LearnInterface,
    get_default_inference_interface,
    get_default_learn_interface,
)
from lib.util import to_vector
from lib.util_lib import get_optimizer


def get_prng(env: GodState) -> tuple[jax.Array, GodState]:
    prng, new_prng = jax.random.split(env.prng)
    return prng, copy.replace(env, prng=new_prng)


def create_learn_interfaces(config: GodConfig) -> dict[int, LearnInterface[GodState]]:
    default_interpreter: LearnInterface[GodState] = get_default_learn_interface()
    interpreters: dict[int, LearnInterface[GodState]] = {}

    for j, (_, learn_config) in enumerate(sorted(config.learners.items())):

        def get_state(env: GodState, i) -> jax.Array:
            return to_vector(
                (
                    dict(islice(env.inference_states.items(), i + 1)),
                    dict(islice(env.learning_states.items(), i)),
                    dict(islice(env.parameters.items(), i)),
                )
            ).vector

        def put_state(env: GodState, state: jax.Array, i) -> GodState:
            _inference_states, _learning_states, _params = to_vector(
                (
                    dict(islice(env.inference_states.items(), i + 1)),
                    dict(islice(env.learning_states.items(), i)),
                    dict(islice(env.parameters.items(), i)),
                )
            ).to_param(state)
            inference_states = dict(islice(env.inference_states.items(), i + 1, None)) | _inference_states
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
            get_state=lambda env, i=j: get_state(env, i),
            put_state=lambda env, state, i=j: put_state(env, state, i),
            get_param=lambda env, i=j: to_vector(env.parameters[i]).vector,
            put_param=lambda env, param, i=j: copy.replace(
                env,
                parameters=env.parameters | {i: to_vector(env.parameters[i]).to_param(param)},
            ),
            get_sgd_param=lambda env, i=j: env.parameters[i + 1].learning_parameter.learning_rate,
            get_optimizer=lambda env, i=j: get_optimizer(learn_config)(env.parameters[i + 1]),
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
            get_prng=lambda env: get_prng(env),
        )
        interpreters[j] = interpreter

    return interpreters


def create_transition_interfaces(config: GodConfig) -> dict[int, dict[int, InferenceInterface]]:
    default_interpreter: InferenceInterface[GodState] = get_default_inference_interface()
    interpreters: dict[int, dict[int, InferenceInterface[GodState]]] = {}

    for j, _ in enumerate(sorted(config.data.items())):
        _interpreter = copy.replace(
            default_interpreter,
            get_readout_param=lambda env: env.parameters[0].readout_fn,
            get_prng=lambda env: get_prng(env),
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
