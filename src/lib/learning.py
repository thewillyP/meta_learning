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

from lib.interface import GeneralInterface, LearnInterface
from lib.lib_types import GRADIENT, LOSS, STAT, traverse
from lib.util import filter_cond


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
    gr_fn: Callable[[ENV, TR_DATA], tuple[ENV, GRADIENT, STAT]],
    validation: Callable[[ENV, VL_DATA], tuple[LOSS, STAT]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, traverse[tuple[TR_DATA, VL_DATA]]], tuple[ENV, LOSS, traverse[STAT], traverse[STAT]]]:
    def f(env: ENV, data: traverse[tuple[TR_DATA, VL_DATA]]) -> tuple[ENV, LOSS]:
        arr, static = eqx.partition(env, eqx.is_array)

        def step(e: ENV, d: tuple[TR_DATA, VL_DATA]) -> tuple[ENV, tuple[LOSS, traverse[STAT], traverse[STAT]]]:
            _env = eqx.combine(e, static)
            tr_data, vl_data = d
            _env, gr, tr_stat = gr_fn(_env, tr_data)
            _env = do_optimizer(_env, gr, learn_interface)
            loss, vl_stat = validation(_env, vl_data)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (loss, traverse(tr_stat), traverse(vl_stat))

        _env, (losses, tr_stats, vl_stats) = jax.lax.scan(step, arr, data.d)
        env = eqx.combine(_env, static)
        return env, LOSS(jnp.mean(losses)), tr_stats, vl_stats

    return f


def average_gradients[ENV, DATA](
    gr_fn: Callable[[ENV, traverse[DATA]], tuple[ENV, GRADIENT, STAT]],
    num_times_to_avg_in_timeseries: int,
    general_interface: GeneralInterface[ENV],
    virtual_minibatches: int,
    last_unpadded_length: int,
) -> Callable[[ENV, traverse[DATA]], tuple[ENV, GRADIENT, STAT]]:
    def f(env: ENV, data: traverse[DATA]) -> tuple[ENV, GRADIENT, STAT]:
        # counters for tracking when loss should be padded
        env = general_interface.put_current_avg_in_timeseries(env, 0)
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)

        # compute weights for averaging gradients due to potential padding
        N: int = jax.tree_util.tree_leaves(data)[0].shape[0]
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
        _data = jax.tree.map(lambda x: jnp.array_split(x, num_times_to_avg_in_timeseries), data.d)

        def step(e: ENV, d: DATA) -> tuple[ENV, tuple[GRADIENT, STAT]]:
            _env = eqx.combine(e, static)
            current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(_env)
            _env, gr, stat = gr_fn(_env, d)
            _env = general_interface.put_current_avg_in_timeseries(_env, current_avg_in_timeseries + 1)
            e, _ = eqx.partition(_env, eqx.is_array)
            return e, (gr, stat)

        _env, (grads, stats) = jax.lax.scan(step, arr, _data)
        env = eqx.combine(_env, static)
        return (
            env,
            GRADIENT(jnp.average(grads, axis=0, weights=weights)),
            STAT(jnp.average(stats, axis=0, weights=weights)),
        )

    return f


def bptt[ENV, DATA](
    transition: Callable[[ENV, DATA], tuple[ENV, LOSS, STAT]], learn_interface: LearnInterface[ENV]
) -> Callable[[ENV, DATA], tuple[ENV, GRADIENT, STAT]]:
    def gradient_fn(env: ENV, data: DATA) -> tuple[ENV, GRADIENT, STAT]:
        def loss_fn(param: jax.Array, data: DATA) -> tuple[LOSS, tuple[ENV, STAT]]:
            _env = learn_interface.put_param(env, param)
            _env, loss, stat = transition(_env, data)
            return loss, (_env, stat)

        param = learn_interface.get_param(env)
        grad, (env, stat) = eqx.filter_grad(loss_fn, has_aux=True)(param, data)
        return env, GRADIENT(grad), stat

    return gradient_fn


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
