from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
import equinox as eqx
import toolz
import matfree
import matfree.decomp
import matfree.eig

from meta_learn_lib.config import *
from meta_learn_lib.interface import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.util import filter_cond, jacobian_matrix_product, to_vector


def rtrl[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[GRADIENT, STAT]],
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

            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            t = learn_interface.get_tick(env)
            mu = config.damping
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def state_fn(state: jax.Array) -> tuple[jax.Array, None]:
                _env = learn_interface.put_state(env, state)
                _env, _ = transition(_env, data)
                state = learn_interface.get_state(_env)
                return state, None

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, data)
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
            credit_gr, readout_stat = readout_gr(env, data)
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
    transition: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: UOROConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        std = config.std
        state_shape = learn_interface.get_state(env).shape
        match config.distribution:
            case "uniform":
                distribution = lambda key: jax.random.uniform(key, state_shape, minval=-std, maxval=std)
            case "normal":
                distribution = lambda key: jax.random.normal(key, state_shape) * std

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
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
                _env, _ = transition(_env, data)
                state = learn_interface.get_state(_env)
                return state

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, data)
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

            credit_gr, readout_stat = readout_gr(env, data)
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
    transition: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, STAT]],
    readout_gr: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[GRADIENT, STAT]],
    learn_interface: GodInterface[ENV],
    config: RFLOConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def step(arr: ENV, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            env = eqx.combine(arr, static)
            s = learn_interface.get_state(env)
            p = learn_interface.get_param(env)
            mu = config.damping
            alpha = learn_interface.get_time_constant(env).value
            influence_tensor_s = learn_interface.get_forward_mode_jacobian(env)

            def param_fn(param: jax.Array) -> tuple[jax.Array, tuple[ENV, STAT]]:
                _env = learn_interface.put_param(env, param)
                _env, stat = transition(_env, data)
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

            credit_gr, readout_stat = readout_gr(env, data)
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
    transition: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, STAT]],
    readout: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[LOSS, STAT]],
    learn_interface: GodInterface[ENV],
    config: BPTTConfig,
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        def loss_fn(param: jax.Array, ds: tuple[TR_DATA, VL_DATA]) -> tuple[LOSS, tuple[ENV, STAT]]:
            env = learn_interface.put_param(env_init, param)
            arr_init, static = eqx.partition(env, eqx.is_array)

            def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[STAT, LOSS]]:
                _env = eqx.combine(arr, static)

                t = learn_interface.get_tick(_env)
                s = filter_cond(
                    t % config.truncate_at == 0,
                    lambda _: jax.lax.stop_gradient(learn_interface.get_state(_env)),
                    lambda _: learn_interface.get_state(_env),
                    None,
                )
                _env = learn_interface.put_state(_env, s)

                _env, trans_stat = transition(_env, data)
                loss, readout_stat = readout(_env, data)
                arr, _ = eqx.partition(_env, eqx.is_array)
                return arr, (trans_stat | readout_stat, loss)

            arr, (stats, losses) = jax.lax.scan(inference_fn, arr_init, ds)
            env = eqx.combine(arr, static)
            return jnp.sum(losses), (env, stats)

        param = learn_interface.get_param(env)
        grad, (env, stats) = eqx.filter_grad(loss_fn, has_aux=True)(param, ds_init)
        return env, GRADIENT(grad), stats

    return gradient_fn


def identity[ENV, TR_DATA, VL_DATA](
    transition: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, STAT]],
    readout: Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[LOSS, STAT]],
    learn_interface: GodInterface[ENV],
) -> Callable[[ENV, tuple[TR_DATA, VL_DATA]], tuple[ENV, GRADIENT, STAT]]:

    def gradient_fn(env_init: ENV, ds_init: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, GRADIENT, STAT]:
        arr_init, static = eqx.partition(env_init, eqx.is_array)

        def inference_fn(arr, data: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[STAT, LOSS]]:
            env = eqx.combine(arr, static)
            env, trans_stat = transition(env, data)
            loss, readout_stat = readout(env, data)
            arr, _ = eqx.partition(env, eqx.is_array)
            return arr, (trans_stat | readout_stat, loss)

        arr, (stats, losses) = jax.lax.scan(inference_fn, arr_init, ds_init)
        env = eqx.combine(arr, static)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return env, GRADIENT(grad), stats

    return gradient_fn


# # add inference to the the validation lista and the size will work out
# def create_meta_learner[ENV, DATA](
#     config: GodConfig,
#     transition_fns: list[Callable[[ENV, DATA], ENV]],
#     readout_fns: list[Callable[[ENV, tuple[DATA, jax.Array]], tuple[tuple[STAT, ...], LOSS]]],
#     resets: list[Callable[[ENV], ENV]],
#     test_reset: Callable[[ENV], ENV],
#     learn_interfaces: list[LearnInterface[ENV]],
#     validation_learn_interfaces: list[LearnInterface[ENV]],
#     general_interfaces: list[GeneralInterface[ENV]],
#     virtual_minibatches: list[int],
#     last_unpadded_lengths: list[int],
# ):
#     """from here on out the types stop making sense because I have to rely on dynamic typing to get the algorithm to work. just make sure the data is in the correct shape and everything should work out thats the only assumption I need to make"""

#     gr_fns = []
#     for (
#         learn_interface,
#         general_interface,
#         data_config,
#         virtual_minibatch,
#         last_unpadded_length,
#         transition_fn,
#         statistic_fn,
#     ) in zip(
#         [learn_interfaces[0]] + validation_learn_interfaces,
#         general_interfaces,
#         config.data.values(),
#         virtual_minibatches,
#         last_unpadded_lengths,
#         transition_fns,
#         readout_fns,
#     ):
#         readout_gr = take_jacobian(statistic_fn, learn_interface)

#         match config.learners[0].learner:
#             case RTRLConfig() as rtrl_config:

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     readout_gr=readout_gr,
#                     learn_interface=learn_interface,
#                     rtrl_config=rtrl_config,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return rtrl(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         readout_gr,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                         rtrl_config,
#                     )(env, tr_data)

#             case RTRLHessianDecompConfig() as rtrl_config:

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     readout_gr=readout_gr,
#                     learn_interface=learn_interface,
#                     rtrl_config=rtrl_config,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return rtrl_hessian_decomp(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         readout_gr,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                         rtrl_config,
#                     )(env, tr_data)

#             case RTRLFiniteHvpConfig() as rtrl_config:

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     readout_gr=readout_gr,
#                     learn_interface=learn_interface,
#                     rtrl_config=rtrl_config,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return rtrl_finite_hvp(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         readout_gr,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                         rtrl_config,
#                     )(env, tr_data)

#             case BPTTConfig():

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     learn_interface=learn_interface,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return bptt(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                     )(env, tr_data)

#             case IdentityConfig():

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     learn_interface=learn_interface,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return identity(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                     )(env, tr_data)
#             case RFLOConfig() as rflo_config:

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     readout_gr=readout_gr,
#                     learn_interface=learn_interface,
#                     rflo_config=rflo_config,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return rflo(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         readout_gr,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                         rflo_config,
#                     )(env, tr_data)
#             case UOROConfig(std):

#                 def _learner(
#                     env: ENV,
#                     data: tuple[traverse[DATA], jax.Array],
#                     transition_fn=transition_fn,
#                     statistic_fn=statistic_fn,
#                     readout_gr=readout_gr,
#                     learn_interface=learn_interface,
#                     std=std,
#                 ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
#                     tr_data, mask = data
#                     return uoro(
#                         lambda e, d: (transition_fn(e, d), ()),
#                         statistic_fn,
#                         readout_gr,
#                         learn_interface,
#                         lambda x: x,
#                         lambda x: (x, mask),
#                         std,
#                     )(env, tr_data)

#         learner = average_gradients(
#             gr_fn=_learner,
#             num_times_to_avg_in_timeseries=data_config.num_times_to_avg_in_timeseries,
#             general_interface=general_interface,
#             virtual_minibatches=virtual_minibatch,
#             last_unpadded_length=last_unpadded_length,
#             get_traverse=lambda data: data[0],
#             put_traverse=lambda tr, data: (tr, data[1]),
#         )

#         gr_fns.append(learner)

#     readouts = []
#     for transition, statistics in zip(transition_fns, readout_fns):

#         def make_readout(transition=transition, statistics=statistics):
#             def readout(env: ENV, ds: tuple[traverse[DATA], jax.Array]) -> tuple[tuple[STAT, ...], LOSS]:
#                 _ds, mask = ds
#                 arr, static = eqx.partition(env, eqx.is_array)

#                 def step(_arr, data: DATA) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
#                     __env = eqx.combine(_arr, static)
#                     __env = transition(__env, data)
#                     read_stat, loss = statistics(__env, (data, mask))
#                     _arr, _ = eqx.partition(__env, eqx.is_array)
#                     return _arr, (read_stat, loss)

#                 _, (stats, losses) = jax.lax.scan(step, arr, _ds.d)
#                 return stats, jnp.sum(losses)

#             return readout

#         readouts.append(make_readout())

#     readout_grs = [lambda env, data, g=gr_fn: g(env, data)[1:] for gr_fn in gr_fns[1:]]

#     learner0 = optimization(
#         lambda env, data: gr_fns[0](resets[0](env), data), learn_interfaces[0], general_interfaces[0]
#     )
#     for learn_config, vl_readout_gr, vl_readout, vl_reset, learn_interface, general_interface in zip(
#         toolz.drop(1, config.learners.values()),
#         readout_grs,
#         readouts[1:],
#         resets[1:],
#         learn_interfaces[1:],
#         general_interfaces[1:],
#     ):
#         learner0 = lambda env, data, r=vl_reset, l=learner0: l(r(env), data)

#         match learn_config.learner:
#             case RTRLConfig() as rtrl_config:
#                 _learner = rtrl(
#                     learner0,
#                     vl_readout,
#                     vl_readout_gr,
#                     learn_interface,
#                     lambda x: x[0],
#                     lambda x: x[1],
#                     rtrl_config,
#                 )
#             case RTRLHessianDecompConfig() as rtrl_config:
#                 _learner = rtrl_hessian_decomp(
#                     learner0,
#                     vl_readout,
#                     vl_readout_gr,
#                     learn_interface,
#                     lambda x: x[0],
#                     lambda x: x[1],
#                     rtrl_config,
#                 )
#             case RTRLFiniteHvpConfig() as rtrl_config:
#                 _learner = rtrl_finite_hvp(
#                     learner0,
#                     vl_readout,
#                     vl_readout_gr,
#                     learn_interface,
#                     lambda x: x[0],
#                     lambda x: x[1],
#                     rtrl_config,
#                 )
#             case BPTTConfig():
#                 _learner = bptt(learner0, vl_readout, learn_interface, lambda x: x[0], lambda x: x[1])
#             case IdentityConfig():
#                 _learner = identity(learner0, vl_readout, learn_interface, lambda x: x[0], lambda x: x[1])
#             case RFLOConfig() as rflo_config:
#                 _learner = rflo(
#                     learner0,
#                     vl_readout,
#                     vl_readout_gr,
#                     learn_interface,
#                     lambda x: x[0],
#                     lambda x: x[1],
#                     rflo_config,
#                 )
#             case UOROConfig(std):
#                 _learner = uoro(
#                     learner0,
#                     vl_readout,
#                     vl_readout_gr,
#                     learn_interface,
#                     lambda x: x[0],
#                     lambda x: x[1],
#                     std,
#                 )

#         learner0 = optimization(_learner, learn_interface, general_interface)

#     def final_learner(env: ENV, data: traverse) -> tuple[ENV, tuple[STAT, ...], traverse[LOSS]]:
#         arr, static = eqx.partition(env, eqx.is_array)

#         def step(_arr, d) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
#             tr_data, vl_data = d
#             _env = eqx.combine(_arr, static)
#             _env, stats = learner0(_env, tr_data)
#             temp_env = general_interfaces[0].put_current_virtual_minibatch(_env, jnp.nan)
#             readout_stat, loss = readouts[0](test_reset(temp_env), vl_data)
#             _arr, _ = eqx.partition(_env, eqx.is_array)
#             return _arr, (stats + readout_stat, loss)

#         _env, (stats, losses) = jax.lax.scan(step, arr, data.d)
#         env = eqx.combine(_env, static)
#         return env, stats, traverse(losses)

#     return final_learner
