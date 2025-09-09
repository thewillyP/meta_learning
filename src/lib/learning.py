"""
1. initial seed of inference and readout
2. first make learner takes the same learn interface for both recurrent and readout
3. second takes the other stuff
4. scan over config.learners
5. after every make learner, compose with optimizer step. then pass into next loop
6.

learning function still takes two different functions, one for recurrent and one for readout
but for inference I can reuse the same function just mask out one input. will contain repetition but whatever
This separation is necessary for efficient rtrl for oho where I dont want to do a backward pass twice
"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import toolz

from lib.config import BPTTConfig, GodConfig, IdentityConfig, RFLOConfig, RTRLConfig, UOROConfig
from lib.interface import GeneralInterface, LearnInterface
from lib.lib_types import GRADIENT, JACOBIAN, LOSS, STAT, batched, traverse
from lib.util import filter_cond, to_vector


def do_optimizer[ENV](env: ENV, gr: GRADIENT, learn_interface: LearnInterface[ENV]) -> ENV:
    param = learn_interface.get_param(env)
    optimizer = learn_interface.get_optimizer(env)
    opt_state = learn_interface.get_opt_state(env)
    updates, new_opt_state = optimizer.update(gr, opt_state, param)
    new_param = optax.apply_updates(param, updates)
    env = learn_interface.put_opt_state(env, new_opt_state)
    env = learn_interface.put_param(env, new_param)
    return env


def optimization[ENV, TR_DATA, VL_DATA](
    gr_fn: Callable[[ENV, TR_DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]],
    validation: Callable[[ENV, VL_DATA], tuple[tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, traverse[tuple[TR_DATA, VL_DATA]]], tuple[ENV, traverse[tuple[STAT, ...]], traverse[LOSS]]]:
    def f(env: ENV, data: traverse[tuple[TR_DATA, VL_DATA]]) -> tuple[ENV, traverse[tuple[STAT, ...]], traverse[LOSS]]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(e: ENV, d: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[LOSS, tuple[STAT, ...]]]:
            _env = eqx.combine(e, static)
            tr_data, vl_data = d
            _env, tr_stat, gr = gr_fn(_env, tr_data)
            _env = do_optimizer(_env, gr, learn_interface)
            x = validation(_env, vl_data)
            print(x)
            vl_stats, loss = x
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (loss, tr_stat + vl_stats)

        _env, (losses, total_stats) = jax.lax.scan(step, arr, data.d)
        env = eqx.combine(_env, static)
        return env, traverse(total_stats), traverse(losses)

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
        env = general_interface.put_current_avg_in_timeseries(env, 0)
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)

        # compute weights for averaging gradients due to potential padding
        traverse_data = get_traverse(data)
        N: int = jax.tree_util.tree_leaves(traverse_data)[0].shape[0]
        chunk_size = N // num_times_to_avg_in_timeseries
        last_chunk_valid = last_unpadded_length - (num_times_to_avg_in_timeseries - 1) * chunk_size
        _weights = jnp.array([chunk_size] * (num_times_to_avg_in_timeseries - 1) + [last_chunk_valid]).astype(float)
        weights = filter_cond(
            current_virtual_minibatch > 0 and current_virtual_minibatch % virtual_minibatches == 0,
            lambda _: _weights,
            lambda _: jnp.ones(num_times_to_avg_in_timeseries),
            None,
        )

        # main code
        arr, static = eqx.partition(env, eqx.is_array)
        print(traverse_data)
        _data = jax.tree.map(lambda x: jnp.stack(jnp.split(x, num_times_to_avg_in_timeseries)), traverse_data)
        print(_data)

        def step(e: ENV, _d: traverse[TR_DATA]) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            d = put_traverse(_d, data)
            print("afafasdfdsafaf", d)
            _env = eqx.combine(e, static)
            current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(_env)
            _env, stat, gr = gr_fn(_env, d)
            _env = general_interface.put_current_avg_in_timeseries(_env, current_avg_in_timeseries + 1)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (gr, stat)

        _env, (grads, _stats) = jax.lax.scan(step, arr, _data)
        env = eqx.combine(_env, static)
        stats = jax.tree.map(lambda x: jnp.average(x, axis=0, weights=weights), _stats)
        return env, stats, GRADIENT(jnp.average(grads, axis=0, weights=weights))

    return f


def bptt[ENV, DATA](
    transition: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        def loss_fn(param: jax.Array, data: DATA) -> tuple[LOSS, tuple[ENV, STAT]]:
            _env = learn_interface.put_param(env, param)
            _env, stat, loss = transition(_env, data)
            return loss, (_env, stat)

        param = learn_interface.get_param(env)
        grad, (env, stat) = eqx.filter_grad(loss_fn, has_aux=True)(param, data)
        return env, stat, GRADIENT(grad)

    return gradient_fn


def identity[ENV, DATA](
    transition: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        _env, stat, loss = transition(env, data)
        param = learn_interface.get_param(env)
        grad = jnp.zeros_like(param)
        return _env, stat, GRADIENT(grad)

    return gradient_fn


def take_jacobian[ENV, DATA, OUT: jax.Array](
    transition: Callable[[ENV, DATA], OUT],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], JACOBIAN]:
    def jacobian_fn(env: ENV, data: DATA) -> JACOBIAN:
        s__p = learn_interface.get_state(env), learn_interface.get_param(env)
        s__p_vector = to_vector(s__p)

        def fn(state__param: jax.Array, data: DATA) -> OUT:
            state, param = s__p_vector.to_param(state__param)
            _env = learn_interface.put_state(env, state)
            _env = learn_interface.put_param(env, param)
            out = transition(_env, data)
            return out

        state__param = s__p_vector.vector
        jacobian = eqx.filter_grad(fn)(state__param, data)
        return JACOBIAN(jacobian)

    return jacobian_fn


# add inference to the the validation lista and the size will work out
def create_meta_learner[ENV, DATA](
    config: GodConfig,
    statistics_fns: list[Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], LOSS]]],
    resets: list[Callable[[ENV], ENV]],
    test_reset: Callable[[ENV], ENV],
    learn_interfaces: list[LearnInterface[ENV]],
    validation_learn_interfaces: list[LearnInterface[ENV]],
    general_interfaces: list[GeneralInterface[ENV]],
    virtual_minibatches: list[int],
    last_unpadded_lengths: list[int],
):
    """from here on out the types stop making sense because I have to rely on dynamic typing to get the algorithm to work. just make sure the data is in the correct shape and everything should work out thats the only assumption I need to make"""
    _readout_gr = take_jacobian(lambda env, data: statistics_fns[0](env, data)[2], learn_interfaces[0])
    readout_gr = lambda env, data: GRADIENT(_readout_gr(env, data))

    gr_fns = []
    for learn_interface, general_interface, data_config, virtual_minibatch, last_unpadded_length, transition in zip(
        [learn_interfaces[0]] + validation_learn_interfaces,
        general_interfaces,
        config.data.values(),
        virtual_minibatches,
        last_unpadded_lengths,
        statistics_fns,
    ):
        match config.learners[0].learner:
            case RTRLConfig():
                ...
            case BPTTConfig():
                _learner = bptt(transition, learn_interface)

            case IdentityConfig():
                _learner = identity(transition, learn_interface)
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
            get_traverse=lambda data: traverse(jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data[0].b.d)),
            put_traverse=lambda tr, data: (
                batched(traverse(jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), tr.d))),
                data[1],
            ),
        )

        gr_fns.append(learner)

    readouts = statistics_fns[1:] + [lambda env, data: statistics_fns[0](test_reset(env), data)]
    readouts = [lambda env, data, r=readout: r(env, data)[1:] for readout in readouts]
    __learner0 = optimization(lambda env, data: gr_fns[0](resets[0](env), data), readouts[0], learn_interfaces[0])

    def _learner0(env: ENV, data: traverse[tuple[DATA, DATA]]) -> tuple[ENV, tuple[STAT, ...], LOSS]:
        env, stats, losses = __learner0(env, data)
        loss = jnp.mean(losses.d)
        return env, stats.d, LOSS(loss)

    learner0 = _learner0

    readout_grs = [lambda env, data, g=gr_fn: g(env, data)[2] for gr_fn in gr_fns[1:]]

    for learn_config, vl_readout_gr, vl_readout, vl_reset, learn_interface in zip(
        toolz.drop(1, config.learners.values()), readout_grs, readouts[1:], resets[1:], learn_interfaces[1:]
    ):
        match learn_config.learner:
            case RTRLConfig():
                ...
            case BPTTConfig():
                _learner = bptt(learner0, learn_interface)
            case IdentityConfig():
                _learner = identity(learner0, learn_interface)
            case RFLOConfig():
                ...
            case UOROConfig():
                ...

        learner0__ = optimization(
            lambda env, data, r=vl_reset: _learner(r(env), data),
            vl_readout,
            learn_interface,
        )

        def learner0_(
            env: ENV, data: traverse[tuple[DATA, DATA]], learner0__=learner0__
        ) -> tuple[ENV, tuple[STAT, ...], LOSS]:
            env, stats, losses = learner0__(env, data)
            loss = jnp.mean(losses.d)
            return env, stats.d, LOSS(loss)

        learner0 = learner0_

    return learner0__
