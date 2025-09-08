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

from lib.config import BPTTConfig, GodConfig, IdentityConfig, RFLOConfig, RTRLConfig, UOROConfig
from lib.interface import GeneralInterface, LearnInterface
from lib.lib_types import GRADIENT, JACOBIAN, LOSS, STAT, traverse
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
    validation: Callable[[ENV, VL_DATA], tuple[LOSS, STAT]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, traverse[tuple[TR_DATA, VL_DATA]]], tuple[ENV, traverse[tuple[STAT, ...]], traverse[LOSS]]]:
    def f(env: ENV, data: traverse[tuple[TR_DATA, VL_DATA]]) -> tuple[ENV, traverse[tuple[STAT, ...]], traverse[LOSS]]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(e: ENV, d: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[LOSS, tuple[STAT, ...], STAT]]:
            _env = eqx.combine(e, static)
            tr_data, vl_data = d
            _env, tr_stat, gr = gr_fn(_env, tr_data)
            _env = do_optimizer(_env, gr, learn_interface)
            loss, vl_stat = validation(_env, vl_data)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (loss, tr_stat, vl_stat)

        _env, (losses, tr_stats, vl_stats) = jax.lax.scan(step, arr, data.d)
        total_stats = tr_stats + (vl_stats,)
        env = eqx.combine(_env, static)
        return env, traverse(total_stats), traverse(losses)

    return f


def average_gradients[ENV, DATA, TRV_DATA](
    gr_fn: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]],
    num_times_to_avg_in_timeseries: int,
    general_interface: GeneralInterface[ENV],
    virtual_minibatches: int,
    last_unpadded_length: int,
    get_traverse: Callable[[DATA], traverse[TRV_DATA]],
    put_traverse: Callable[[DATA, traverse[TRV_DATA]], DATA],
) -> Callable[[ENV, DATA], tuple[ENV, GRADIENT, STAT]]:
    def f(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        # counters for tracking when loss should be padded
        env = general_interface.put_current_avg_in_timeseries(env, 0)
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)

        # compute weights for averaging gradients due to potential padding
        traverse_data = get_traverse(data)
        N: int = jax.tree_util.tree_leaves(traverse_data)[0].shape[0]
        chunk_size = N // num_times_to_avg_in_timeseries
        last_chunk_valid = last_unpadded_length - (num_times_to_avg_in_timeseries - 1) * chunk_size
        _weights = jnp.array([chunk_size] * (num_times_to_avg_in_timeseries - 1) + [last_chunk_valid])
        weights = filter_cond(
            current_virtual_minibatch > 0 and current_virtual_minibatch % virtual_minibatches == 0,
            lambda _: _weights,
            lambda _: jnp.ones(num_times_to_avg_in_timeseries),
            None,
        )

        # main code
        arr, static = eqx.partition(env, eqx.is_array)
        traverse_datas = jax.tree.map(lambda x: jnp.array_split(x, num_times_to_avg_in_timeseries), traverse_data)

        def step(e: ENV, _d: traverse[TRV_DATA]) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            d = put_traverse(data, _d)
            _env = eqx.combine(e, static)
            current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(_env)
            _env, stat, gr = gr_fn(_env, d)
            _env = general_interface.put_current_avg_in_timeseries(_env, current_avg_in_timeseries + 1)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (gr, stat)

        _env, (grads, _stats) = jax.lax.scan(step, arr, traverse_datas)
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
    transition: Callable[[ENV, DATA], tuple[ENV, LOSS, STAT]],
    vl_statistics_fns: list[Callable[[ENV, tuple[DATA, jax.Array]], tuple[ENV, LOSS, STAT]]],
    resets: list[Callable[[ENV], ENV]],
    learn_interfaces: list[LearnInterface[ENV]],
    validation_learn_interfaces: list[LearnInterface[ENV]],
) -> None:
    readout_gr = take_jacobian(transition, learn_interfaces[0])

    vl_gradients = []
    for learn_config, data_config

    for learn_config, vl_model, reset, learn_interface in zip(
        config.learners.values(), vl_statistics_fns, resets, learn_interfaces
    ):
        match learn_config.learner:
            case RTRLConfig():
                ...
            case BPTTConfig():
                learner = bptt(transition, learn_interface)
                learner = average_gradients(
            case IdentityConfig():
                ...
            case RFLOConfig():
                ...
            case UOROConfig():
                ...

        # def with_reset(
        #     env: ENV, data: DATA
        # ) -> tuple[ENV, GRADIENT, STAT]:
        #     env = resets[1](resets[0](env))
        #     return _learner(env, data)


# class OfflineLearning:
#     def createLearner[ENV, DATA](
#         self,
#         activationStep: Controller[Data, Interpreter, Env, REC_STATE],
#         readoutStep: Controller[Data, Interpreter, Env, Pred],
#         lossFunction: LossFn[Pred, Data],
#         _: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
#     ) -> Library[Traversable[Data], Interpreter, Env, Traversable[Pred]]:
#         rnnStep = lambda data: activationStep(data).then(readoutStep(data))
#         rnn2Loss = lambda data: rnnStep(data).fmap(lambda p: lossFunction(p, data))
#         rnnWithLoss = accumulateM(rnn2Loss, add, LOSS(jnp.array(0.0)))

#         @do()
#         def rnnWithGradient(data: Traversable[Data]) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
#             interpreter = yield from ask(PX[Interpreter]())
#             return doGradient(rnnWithLoss(data), interpreter.getRecurrentParam, interpreter.putRecurrentParam)

#         return Library(
#             model=traverseM(rnnStep),
#             modelLossFn=rnnWithLoss,
#             modelGradient=rnnWithGradient,
#         )
