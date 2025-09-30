from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
import equinox as eqx
import toolz

from meta_learn_lib.config import (
    BPTTConfig,
    GodConfig,
    IdentityConfig,
    RFLOConfig,
    RTRLConfig,
    RTRLFiniteHvpConfig,
    RTRLHessianDecompConfig,
    UOROConfig,
)
from meta_learn_lib.env import Logs
from meta_learn_lib.interface import GeneralInterface, LearnInterface
from meta_learn_lib.lib_types import GRADIENT, JACOBIAN, LOSS, STAT, batched, traverse
from meta_learn_lib.util import filter_cond, jacobian_matrix_product, to_vector


def do_optimizer[ENV](env: ENV, gr: GRADIENT, learn_interface: LearnInterface[ENV]) -> ENV:
    param = learn_interface.get_param(env)
    optimizer = learn_interface.get_optimizer(env)
    opt_state = learn_interface.get_opt_state(env)
    updates, new_opt_state = optimizer.update(gr, opt_state, param)
    new_param = optax.apply_updates(param, updates)
    env = learn_interface.put_opt_state(env, new_opt_state)
    env = learn_interface.put_param(env, new_param)

    return env


def optimization[ENV, DATA](
    gr_fn: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]],
    learn_interface: LearnInterface[ENV],
    general_interface: GeneralInterface[ENV],
) -> Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...]]]:
    def f(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...]]:
        env, tr_stat, gr = gr_fn(env, data)
        env = general_interface.put_logs(env, Logs(gradient=gr))
        env = do_optimizer(env, gr, learn_interface)
        return env, tr_stat

    return f


def average_gradients[ENV, DATA, TR_DATA](
    gr_fn: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]],
    num_times_to_avg_in_timeseries: int,
    general_interface: GeneralInterface[ENV],
    virtual_minibatches: int,
    last_unpadded_length: int,
    get_traverse: Callable[[DATA], traverse[TR_DATA]],
    put_traverse: Callable[[traverse[TR_DATA], DATA], DATA],
) -> Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def f(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        # counters for tracking when loss should be padded
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)

        # compute weights for averaging gradients due to potential padding
        traverse_data = get_traverse(data)
        N: int = jax.tree_util.tree_leaves(traverse_data)[0].shape[0]
        chunk_size = N // num_times_to_avg_in_timeseries
        last_chunk_valid = last_unpadded_length - (num_times_to_avg_in_timeseries - 1) * chunk_size
        _weights = jnp.array([chunk_size] * (num_times_to_avg_in_timeseries - 1) + [last_chunk_valid]).astype(float)
        weights = filter_cond(
            current_virtual_minibatch % virtual_minibatches == 0,
            lambda _: _weights,
            lambda _: jnp.ones(num_times_to_avg_in_timeseries),
            None,
        )

        # main code
        arr, static = eqx.partition(env, eqx.is_array)
        _data = jax.tree.map(lambda x: jnp.stack(jnp.split(x, num_times_to_avg_in_timeseries)), traverse_data)

        def step(e: ENV, _d: traverse[TR_DATA]) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            d = put_traverse(_d, data)
            _env = eqx.combine(e, static)
            _env, stat, gr = gr_fn(_env, d)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (gr, stat)

        _env, (grads, _stats) = jax.lax.scan(step, arr, _data)
        env = eqx.combine(_env, static)
        stats = jax.tree.map(lambda x: jnp.average(x, axis=0, weights=weights), _stats)
        return env, stats, GRADIENT(jnp.average(grads, axis=0, weights=weights))

    return f


def rtrl[ENV, TR_DATA, VL_DATA, DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...]]],
    readout: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], GRADIENT]],
    learn_interface: LearnInterface[ENV],
    get_tr: Callable[[DATA], TR_DATA],
    get_vl: Callable[[DATA], VL_DATA],
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, ds: traverse[DATA]) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(_arr: ENV, data: DATA) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            _env = eqx.combine(_arr, static)
            s__p = learn_interface.get_state(_env), learn_interface.get_param(_env)
            s__p_vector = to_vector(s__p)

            def state_fn(state__param: jax.Array) -> tuple[jax.Array, tuple[ENV, tuple[STAT, ...]]]:
                state, param = s__p_vector.to_param(state__param)
                __env = learn_interface.put_state(_env, state)
                __env = learn_interface.put_param(__env, param)
                __env, stat = transition(__env, get_tr(data))
                state = learn_interface.get_state(__env)
                return state, (__env, stat)

            influence_tensor = learn_interface.get_influence_tensor(_env)
            state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
            _, new_influence_tensor, (_env, tr_stat) = jacobian_matrix_product(
                state_fn, s__p_vector.vector, state_jacobian
            )
            new_influence_tensor = JACOBIAN(new_influence_tensor)
            vl_stat, credit_gr = readout_gr(_env, get_vl(data))
            credit_assignment = GRADIENT(credit_gr @ new_influence_tensor)
            output_gr = take_gradient(readout, learn_interface)(_env, get_vl(data))[1]
            grad = credit_assignment + output_gr

            _env = learn_interface.put_influence_tensor(_env, new_influence_tensor)
            _arr, _ = eqx.partition(_env, eqx.is_array)
            return _arr, (grad, tr_stat + vl_stat)

        _env, (grads, stats) = jax.lax.scan(step, arr, ds.d)
        env = eqx.combine(_env, static)
        return env, stats, GRADIENT(jnp.sum(grads, axis=0))

    return gradient_fn


def rtrl_hessian_decomp[ENV, TR_DATA, VL_DATA, DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...]]],
    readout: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], GRADIENT]],
    learn_interface: LearnInterface[ENV],
    get_tr: Callable[[DATA], TR_DATA],
    get_vl: Callable[[DATA], VL_DATA],
    epsilon: float,
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, ds: traverse[DATA]) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(_arr: ENV, data: DATA) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            _env = eqx.combine(_arr, static)

            s = to_vector(learn_interface.get_state(_env))
            p = to_vector(learn_interface.get_param(_env))

            def state_fn(state: jax.Array) -> tuple[jax.Array, tuple[ENV, tuple[STAT, ...]]]:
                state = s.to_param(state)
                __env = learn_interface.put_state(_env, state)
                __env, stat = transition(__env, get_tr(data))
                state = learn_interface.get_state(__env)
                return state, (__env, stat)

            def param_fn(param: jax.Array) -> jax.Array:
                param = p.to_param(param)
                __env = learn_interface.put_param(_env, param)
                __env, stat = transition(__env, get_tr(data))
                state = learn_interface.get_state(__env)
                return state

            influence_tensor = learn_interface.get_influence_tensor(_env)
            hessian, (_env, tr_stat) = eqx.filter_jacrev(state_fn, has_aux=True)(s.vector)

            # recompose hessian with normalized eigenvalues
            eigenvalues, eigenvectors = jnp.linalg.eigh(hessian, symmetrize_input=True)
            spectral_radius = jnp.max(jnp.abs(eigenvalues))  # abs handles complex correctly
            normalized_eigenvalues = eigenvalues / (spectral_radius + epsilon)
            reformed_hessian = (eigenvectors * normalized_eigenvalues) @ eigenvectors.T

            # threshold = 1e-6
            # diff_matrix = jnp.abs(hessian - jnp.transpose(hessian))
            # nontrivial_differences = jnp.sum(diff_matrix > threshold)

            dhdp = eqx.filter_jacfwd(param_fn, has_aux=False)(p.vector)
            new_influence_tensor = reformed_hessian @ influence_tensor + dhdp

            new_influence_tensor = JACOBIAN(new_influence_tensor)
            vl_stat, credit_gr = readout_gr(_env, get_vl(data))
            credit_assignment = GRADIENT(credit_gr @ new_influence_tensor)
            output_gr = take_gradient(readout, learn_interface)(_env, get_vl(data))[1]
            grad = credit_assignment + output_gr

            _env = learn_interface.put_influence_tensor(_env, new_influence_tensor)

            # # logging
            # hessian_max = jnp.max(jnp.abs(hessian))
            # nonzero_mask = jnp.abs(hessian) > 0
            # hessian_min = jnp.min(jnp.where(nonzero_mask, jnp.abs(hessian), jnp.inf))
            # nonzero_eigenvalue_mask = jnp.abs(eigenvalues) > 1e-6
            # smallest_nonzero_eigenvalue = jnp.min(jnp.where(nonzero_eigenvalue_mask, jnp.abs(eigenvalues), jnp.inf))

            # symmetry_error = jnp.max(jnp.abs(hessian - hessian.T))
            # eigenvalue_inf_count = jnp.sum(jnp.isinf(eigenvalues))
            # eigenvalue_nan_count: jax.Array = jnp.sum(jnp.isnan(eigenvalues))
            # zero_eigenvalue_count = jnp.sum(jnp.abs(eigenvalues) < 1e-6)
            # # immediate_influence_contains_nans = jnp.any(~jnp.isfinite(dhdp))
            # stuff = jnp.stack(
            #     [
            #         hessian_max,
            #         hessian_min,
            #         smallest_nonzero_eigenvalue,
            #         symmetry_error,
            #         eigenvalue_inf_count,
            #         eigenvalue_nan_count,
            #         zero_eigenvalue_count,
            #     ]
            # )
            # _env = learn_interface.put_logs(
            #     _env,
            #     Logs(
            #         immediate_influence_contains_nans=stuff,
            #     ),
            # )

            # infl = filter_cond(
            #     nontrivial_differences > 0, lambda _: influence_tensor, lambda _: new_influence_tensor, None
            # )
            # _env = learn_interface.put_influence_tensor(_env, infl)

            # grad = filter_cond(nontrivial_differences > 0, lambda _: jnp.zeros_like(grad), lambda _: grad, None)

            _arr, _ = eqx.partition(_env, eqx.is_array)
            return _arr, (grad, tr_stat + vl_stat)

        _env, (grads, stats) = jax.lax.scan(step, arr, ds.d)
        env = eqx.combine(_env, static)
        return env, stats, GRADIENT(jnp.sum(grads, axis=0))

    return gradient_fn


def rtrl_finite_hvp[ENV, TR_DATA, VL_DATA, DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...]]],
    readout: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    readout_gr: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], GRADIENT]],
    learn_interface: LearnInterface[ENV],
    get_tr: Callable[[DATA], TR_DATA],
    get_vl: Callable[[DATA], VL_DATA],
    epsilon: float,
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, ds: traverse[DATA]) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(_arr: ENV, data: DATA) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            _env = eqx.combine(_arr, static)

            s = to_vector(learn_interface.get_state(_env))
            p = to_vector(learn_interface.get_param(_env))

            def state_fn(state: jax.Array) -> jax.Array:
                state = s.to_param(state)
                __env = learn_interface.put_state(_env, state)
                __env, stat = transition(__env, get_tr(data))
                state = learn_interface.get_state(__env)
                return state

            def param_fn(param: jax.Array) -> jax.Array:
                param = p.to_param(param)
                __env = learn_interface.put_param(_env, param)
                __env, stat = transition(__env, get_tr(data))
                state = learn_interface.get_state(__env)
                __arr, _ = eqx.partition(__env, eqx.is_array)
                return state, (__arr, stat)

            influence_tensor = learn_interface.get_influence_tensor(_env)

            def finite_hvp(v):
                return (state_fn(s.vector + epsilon * v) - state_fn(s.vector - epsilon * v)) / (2 * epsilon)

            hmp = eqx.filter_vmap(finite_hvp, in_axes=1, out_axes=1)(influence_tensor)

            dhdp, (_arr, tr_stat) = eqx.filter_jacfwd(param_fn, has_aux=True)(p.vector)
            _env = eqx.combine(_arr, static)
            new_influence_tensor = hmp + dhdp
            new_influence_tensor = JACOBIAN(new_influence_tensor)

            vl_stat, credit_gr = readout_gr(_env, get_vl(data))
            credit_assignment = GRADIENT(credit_gr @ new_influence_tensor)
            output_gr = take_gradient(readout, learn_interface)(_env, get_vl(data))[1]
            grad = credit_assignment + output_gr

            _env = learn_interface.put_influence_tensor(_env, new_influence_tensor)

            _arr, _ = eqx.partition(_env, eqx.is_array)
            return _arr, (grad, tr_stat + vl_stat)

        _env, (grads, stats) = jax.lax.scan(step, arr, ds.d)
        env = eqx.combine(_env, static)
        return env, stats, GRADIENT(jnp.sum(grads, axis=0))

    return gradient_fn


# tuple[TR_DATA, jax.Array] is for first level
def bptt[ENV, TR_DATA, VL_DATA, DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...]]],
    readout: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
    get_tr: Callable[[DATA], TR_DATA],
    get_vl: Callable[[DATA], VL_DATA],
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, ds: traverse[DATA]) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        def loss_fn(param: jax.Array, _ds: traverse[DATA]) -> tuple[LOSS, tuple[ENV, tuple[STAT, ...]]]:
            _env = learn_interface.put_param(env, param)
            arr, static = eqx.partition(_env, eqx.is_array)

            def inference_fn(_arr, data: DATA) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
                tr_data = get_tr(data)
                vl_data = get_vl(data)
                __env = eqx.combine(_arr, static)
                __env, trans_stat = transition(__env, tr_data)
                read_stat, loss = readout(__env, vl_data)
                _arr, _ = eqx.partition(__env, eqx.is_array)
                return _arr, (trans_stat + read_stat, loss)

            __env, (stats, losses) = jax.lax.scan(inference_fn, arr, _ds.d)
            _env = eqx.combine(__env, static)
            return jnp.sum(losses), (_env, stats)

        param = learn_interface.get_param(env)
        grad, (env, stats) = eqx.filter_grad(loss_fn, has_aux=True)(param, ds)
        return env, stats, GRADIENT(grad)

    return gradient_fn


def identity[ENV, TR_DATA, VL_DATA, DATA](
    transition: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...]]],
    readout: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
    get_tr: Callable[[DATA], TR_DATA],
    get_vl: Callable[[DATA], VL_DATA],
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, ds: traverse[DATA]) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        arr, static = eqx.partition(env, eqx.is_array)

        def inference_fn(_arr, data: DATA) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
            tr_data = get_tr(data)
            vl_data = get_vl(data)
            _env = eqx.combine(_arr, static)
            _env, trans_stat = transition(_env, tr_data)
            read_stat, loss = readout(_env, vl_data)
            _arr, _ = eqx.partition(_env, eqx.is_array)
            return _arr, (trans_stat + read_stat, loss)

        _env, (stats, losses) = jax.lax.scan(inference_fn, arr, ds.d)
        env = eqx.combine(_env, static)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return env, stats, GRADIENT(grad)

    return gradient_fn


def take_jacobian[ENV, DATA, OUT: jax.Array, AUX](
    transition: Callable[[ENV, DATA], tuple[AUX, OUT]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[AUX, JACOBIAN]]:
    def jacobian_fn(env: ENV, data: DATA) -> tuple[AUX, JACOBIAN]:
        s = learn_interface.get_state(env)

        def fn(state: jax.Array, data: DATA) -> tuple[OUT, AUX]:
            _env = learn_interface.put_state(env, state)
            aux, out = transition(_env, data)
            return out, aux

        jacobian, aux = eqx.filter_grad(fn, has_aux=True)(s, data)
        return aux, JACOBIAN(jacobian)

    return jacobian_fn


def take_gradient[ENV, DATA, OUT: jax.Array, AUX](
    transition: Callable[[ENV, DATA], tuple[AUX, OUT]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[AUX, GRADIENT]]:
    def gradient_fn(env: ENV, data: DATA) -> tuple[AUX, GRADIENT]:
        p = learn_interface.get_param(env)

        def fn(param: jax.Array, data: DATA) -> tuple[OUT, AUX]:
            _env = learn_interface.put_param(env, param)
            aux, out = transition(_env, data)
            return out, aux

        gradient, aux = eqx.filter_grad(fn, has_aux=True)(p, data)
        return aux, GRADIENT(gradient)

    return gradient_fn


# add inference to the the validation lista and the size will work out
def create_meta_learner[ENV, DATA](
    config: GodConfig,
    transition_fns: list[Callable[[ENV, DATA], ENV]],
    readout_fns: list[Callable[[ENV, tuple[DATA, jax.Array]], tuple[tuple[STAT, ...], LOSS]]],
    resets: list[Callable[[ENV], ENV]],
    test_reset: Callable[[ENV], ENV],
    learn_interfaces: list[LearnInterface[ENV]],
    validation_learn_interfaces: list[LearnInterface[ENV]],
    general_interfaces: list[GeneralInterface[ENV]],
    virtual_minibatches: list[int],
    last_unpadded_lengths: list[int],
):
    """from here on out the types stop making sense because I have to rely on dynamic typing to get the algorithm to work. just make sure the data is in the correct shape and everything should work out thats the only assumption I need to make"""

    gr_fns = []
    for (
        learn_interface,
        general_interface,
        data_config,
        virtual_minibatch,
        last_unpadded_length,
        transition_fn,
        statistic_fn,
    ) in zip(
        [learn_interfaces[0]] + validation_learn_interfaces,
        general_interfaces,
        config.data.values(),
        virtual_minibatches,
        last_unpadded_lengths,
        transition_fns,
        readout_fns,
    ):
        readout_gr = take_jacobian(statistic_fn, learn_interface)

        match config.learners[0].learner:
            case RTRLConfig():

                def _learner(
                    env: ENV,
                    data: tuple[traverse[DATA], jax.Array],
                    transition_fn=transition_fn,
                    statistic_fn=statistic_fn,
                    readout_gr=readout_gr,
                    learn_interface=learn_interface,
                ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
                    tr_data, mask = data
                    return rtrl(
                        lambda e, d: (transition_fn(e, d), ()),
                        statistic_fn,
                        readout_gr,
                        learn_interface,
                        lambda x: x,
                        lambda x: (x, mask),
                    )(env, tr_data)

            case RTRLHessianDecompConfig(epsilon):

                def _learner(
                    env: ENV,
                    data: tuple[traverse[DATA], jax.Array],
                    transition_fn=transition_fn,
                    statistic_fn=statistic_fn,
                    readout_gr=readout_gr,
                    learn_interface=learn_interface,
                    epsilon=epsilon,
                ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
                    tr_data, mask = data
                    return rtrl_hessian_decomp(
                        lambda e, d: (transition_fn(e, d), ()),
                        statistic_fn,
                        readout_gr,
                        learn_interface,
                        lambda x: x,
                        lambda x: (x, mask),
                        epsilon,
                    )(env, tr_data)

            case RTRLFiniteHvpConfig(epsilon):

                def _learner(
                    env: ENV,
                    data: tuple[traverse[DATA], jax.Array],
                    transition_fn=transition_fn,
                    statistic_fn=statistic_fn,
                    readout_gr=readout_gr,
                    learn_interface=learn_interface,
                    epsilon=epsilon,
                ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
                    tr_data, mask = data
                    return rtrl_finite_hvp(
                        lambda e, d: (transition_fn(e, d), ()),
                        statistic_fn,
                        readout_gr,
                        learn_interface,
                        lambda x: x,
                        lambda x: (x, mask),
                        epsilon,
                    )(env, tr_data)

            case BPTTConfig():

                def _learner(
                    env: ENV,
                    data: tuple[traverse[DATA], jax.Array],
                    transition_fn=transition_fn,
                    statistic_fn=statistic_fn,
                    learn_interface=learn_interface,
                ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
                    tr_data, mask = data
                    return bptt(
                        lambda e, d: (transition_fn(e, d), ()),
                        statistic_fn,
                        learn_interface,
                        lambda x: x,
                        lambda x: (x, mask),
                    )(env, tr_data)

            case IdentityConfig():

                def _learner(
                    env: ENV,
                    data: tuple[traverse[DATA], jax.Array],
                    transition_fn=transition_fn,
                    statistic_fn=statistic_fn,
                    learn_interface=learn_interface,
                ) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
                    tr_data, mask = data
                    return identity(
                        lambda e, d: (transition_fn(e, d), ()),
                        statistic_fn,
                        learn_interface,
                        lambda x: x,
                        lambda x: (x, mask),
                    )(env, tr_data)
            case RFLOConfig():
                ...
            case UOROConfig():
                ...

        learner = average_gradients(
            gr_fn=_learner,
            num_times_to_avg_in_timeseries=data_config.num_times_to_avg_in_timeseries,
            general_interface=general_interface,
            virtual_minibatches=virtual_minibatch,
            last_unpadded_length=last_unpadded_length,
            get_traverse=lambda data: data[0],
            put_traverse=lambda tr, data: (tr, data[1]),
        )

        gr_fns.append(learner)

    readouts = []
    for transition, statistics in zip(transition_fns, readout_fns):

        def make_readout(transition=transition, statistics=statistics):
            def readout(env: ENV, ds: tuple[traverse[DATA], jax.Array]) -> tuple[tuple[STAT, ...], LOSS]:
                _ds, mask = ds
                arr, static = eqx.partition(env, eqx.is_array)

                def step(_arr, data: DATA) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
                    __env = eqx.combine(_arr, static)
                    __env = transition(__env, data)
                    read_stat, loss = statistics(__env, (data, mask))
                    _arr, _ = eqx.partition(__env, eqx.is_array)
                    return _arr, (read_stat, loss)

                _, (stats, losses) = jax.lax.scan(step, arr, _ds.d)
                return stats, jnp.sum(losses)

            return readout

        readouts.append(make_readout())

    readout_grs = [lambda env, data, g=gr_fn: g(env, data)[1:] for gr_fn in gr_fns[1:]]

    learner0 = optimization(
        lambda env, data: gr_fns[0](resets[0](env), data), learn_interfaces[0], general_interfaces[0]
    )
    for learn_config, vl_readout_gr, vl_readout, vl_reset, learn_interface, general_interface in zip(
        toolz.drop(1, config.learners.values()),
        readout_grs,
        readouts[1:],
        resets[1:],
        learn_interfaces[1:],
        general_interfaces[1:],
    ):
        learner0 = lambda env, data, r=vl_reset, l=learner0: l(r(env), data)

        match learn_config.learner:
            case RTRLConfig():
                _learner = rtrl(learner0, vl_readout, vl_readout_gr, learn_interface, lambda x: x[0], lambda x: x[1])
            case RTRLHessianDecompConfig(epsilon):
                _learner = rtrl_hessian_decomp(
                    learner0, vl_readout, vl_readout_gr, learn_interface, lambda x: x[0], lambda x: x[1], epsilon
                )
            case RTRLFiniteHvpConfig(epsilon):
                _learner = rtrl_finite_hvp(
                    learner0, vl_readout, vl_readout_gr, learn_interface, lambda x: x[0], lambda x: x[1], epsilon
                )
            case BPTTConfig():
                _learner = bptt(learner0, vl_readout, learn_interface, lambda x: x[0], lambda x: x[1])
            case IdentityConfig():
                _learner = identity(learner0, vl_readout, learn_interface, lambda x: x[0], lambda x: x[1])
            case RFLOConfig():
                ...
            case UOROConfig():
                ...

        learner0 = optimization(_learner, learn_interface, general_interface)

    def final_learner(env: ENV, data: traverse) -> tuple[ENV, tuple[STAT, ...], traverse[LOSS]]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(_arr, d) -> tuple[PyTree, tuple[tuple[STAT, ...], LOSS]]:
            tr_data, vl_data = d
            _env = eqx.combine(_arr, static)
            _env, stats = learner0(_env, tr_data)
            temp_env = general_interfaces[0].put_current_virtual_minibatch(_env, jnp.nan)
            readout_stat, loss = readouts[0](test_reset(temp_env), vl_data)
            _arr, _ = eqx.partition(_env, eqx.is_array)
            return _arr, (stats + readout_stat, loss)

        _env, (stats, losses) = jax.lax.scan(step, arr, data.d)
        env = eqx.combine(_env, static)
        return env, stats, traverse(losses)

    return final_learner
