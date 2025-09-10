from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import toolz

from lib.config import BPTTConfig, GodConfig, IdentityConfig, RFLOConfig, RTRLConfig, UOROConfig
from lib.interface import GeneralInterface, LearnInterface
from lib.lib_types import GRADIENT, JACOBIAN, LOSS, STAT, batched, traverse
from lib.util import filter_cond, jacobian_matrix_product, to_vector


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
        _data = jax.tree.map(lambda x: jnp.stack(jnp.split(x, num_times_to_avg_in_timeseries)), traverse_data)

        def step(e: ENV, _d: traverse[TR_DATA]) -> tuple[ENV, tuple[GRADIENT, tuple[STAT, ...]]]:
            d = put_traverse(_d, data)
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


def rtrl[ENV, DATA](
    transition: Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], LOSS]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[ENV, tuple[STAT, ...], GRADIENT]]:
    def gradient_fn(env: ENV, data: DATA) -> tuple[ENV, tuple[STAT, ...], GRADIENT]:
        s__p = learn_interface.get_state(env), learn_interface.get_param(env)
        s__p_vector = to_vector(s__p)

        def state_fn(state__param: jax.Array) -> tuple[tuple[jax.Array, LOSS], tuple[ENV, tuple[STAT, ...]]]:
            state, param = s__p_vector.to_param(state__param)
            _env = learn_interface.put_state(env, state)
            _env = learn_interface.put_param(_env, param)
            _env, stat, loss = transition(_env, data)
            state = learn_interface.get_state(_env)
            return (state, loss), (_env, stat)

        influence_tensor = learn_interface.get_influence_tensor(env)
        state_jacobian = jnp.vstack([influence_tensor, jnp.eye(influence_tensor.shape[1])])
        (
            _,
            (new_influence_tensor, grad),
            (stat, env),
        ) = jacobian_matrix_product(state_fn, s__p_vector.vector, state_jacobian)
        env = learn_interface.put_influence_tensor(env, JACOBIAN(new_influence_tensor))
        return env, stat, GRADIENT(grad)

    return gradient_fn


# class PastFacingLearn(ABC):
#     @abstractmethod
#     def creditAssignment[Interpreter, Env](
#         self,
#         recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]],
#         activationStep: Agent[Interpreter, Env, REC_STATE],
#     ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]: ...

#     class _CreateModel_Can[Env](PutRecurrentState[Env], PutRecurrentParam[Env], Protocol): ...

#     @staticmethod
#     def createRnnForward[Interpreter: _CreateModel_Can[Env], Env](activationStep: Agent[Interpreter, Env, REC_STATE]):
#         @do()
#         def _creatingModel():
#             interpreter = yield from ask(PX[Interpreter]())
#             env = yield from get(PX[Env]())

#             def agentFn(actv: REC_STATE, param: REC_PARAM) -> tuple[REC_STATE, Env]:
#                 return (
#                     interpreter.putRecurrentState(actv)
#                     .then(interpreter.putRecurrentParam(param))
#                     .then(activationStep)
#                     .func(interpreter, env)
#                 )

#             return pure(agentFn, PX[tuple[Interpreter, Env]]())

#         return _creatingModel()

#     class _PastFacingLearner_Can[Env](
#         GetRecurrentState[Env],
#         PutRecurrentState[Env],
#         GetRecurrentParam[Env],
#         PutRecurrentParam[Env],
#         Protocol,
#     ): ...

#     def createLearner[Data, Interpreter: _PastFacingLearner_Can[Env], Env, Pred](
#         self,
#         activationStep: Controller[Data, Interpreter, Env, REC_STATE],
#         readoutStep: Controller[Data, Interpreter, Env, Pred],
#         lossFunction: LossFn[Pred, Data],
#         lossGradientWrtActiv: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
#     ) -> Library[IdentityF[Data], Interpreter, Env, IdentityF[Pred]]:
#         def immediateLoss(data: Data):
#             return readoutStep(data).fmap(lambda p: lossFunction(p, data))

#         @do()
#         def rnnGradient(data: Data) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
#             interpreter = yield from ask(PX[Interpreter]())

#             grad_rec = yield from self.creditAssignment(lossGradientWrtActiv(data), activationStep(data))
#             grad_readout = yield from doGradient(
#                 immediateLoss(data),
#                 interpreter.getRecurrentParam,
#                 interpreter.putRecurrentParam,
#             )

#             return pure(grad_rec + grad_readout, PX[tuple[Interpreter, Env]]())

#         return Library[IdentityF[Data], Interpreter, Env, IdentityF[Pred]](
#             model=lambda data: activationStep(data.value).then(readoutStep(data.value)),
#             modelLossFn=lambda data: activationStep(data.value).then(immediateLoss(data.value)),
#             modelGradient=lambda data: rnnGradient(data.value),
#         )


# class InfluenceTensorLearner(PastFacingLearn, ABC):
#     class _InfluenceTensorLearner_Can[Env](
#         GetInfluenceTensor[Env],
#         PutInfluenceTensor[Env],
#         PastFacingLearn._PastFacingLearner_Can[Env],
#         PutLogs[Env],
#         GetGlobalLogConfig[Env],
#         Protocol,
#     ): ...

#     @abstractmethod
#     def getInfluenceTensor[Interpreter, Env](
#         self, activationStep: Agent[Interpreter, Env, REC_STATE]
#     ) -> Agent[Interpreter, Env, Jacobian[REC_PARAM]]: ...

#     def creditAssignment[Interpreter: _InfluenceTensorLearner_Can[Env], Env](
#         self,
#         recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]],
#         activationStep: Agent[Interpreter, Env, REC_STATE],
#     ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]:
#         @do()
#         def _creditAssignment() -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
#             interpreter = yield from ask(PX[Interpreter]())
#             infT = yield from self.getInfluenceTensor(activationStep)
#             stop_influence = yield from interpreter.getGlobalLogConfig.fmap(lambda x: x.stop_influence)
#             log_influence = yield from interpreter.getGlobalLogConfig.fmap(lambda x: x.log_influence)
#             if not stop_influence:
#                 _ = yield from interpreter.putInfluenceTensor(JACOBIAN(infT.value))

#             influenceTensor = yield from interpreter.getInfluenceTensor
#             signal = yield from recurrentError
#             recurrentGradient = Gradient[REC_PARAM](signal.value @ influenceTensor)

#             if log_influence:
#                 _ = yield from interpreter.putLogs(Logs(influenceTensor=influenceTensor))
#             return pure(recurrentGradient, PX[tuple[Interpreter, Env]]())

#         return _creditAssignment()


# class RTRL(InfluenceTensorLearner):
#     class _UpdateInfluence_Can[Env](
#         GetInfluenceTensor[Env],
#         GetRecurrentState[Env],
#         PutRecurrentState[Env],
#         GetRecurrentParam[Env],
#         PutRecurrentParam[Env],
#         PutLogs[Env],
#         GetLogConfig[Env],
#         GetGlobalLogConfig[Env],
#         GetPRNG[Env],
#         Protocol,
#     ): ...

#     def __init__(self, use_fwd: bool):
#         self.immediateJacFn = eqx.filter_jacfwd if use_fwd else eqx.filter_jacrev

#     @do()
#     def getInfluenceTensor[Interpreter: _UpdateInfluence_Can[Env], Env](
#         self, activationStep: Agent[Interpreter, Env, REC_STATE]
#     ) -> G[Agent[Interpreter, Env, Jacobian[REC_PARAM]]]:
#         interpreter = yield from ask(PX[Interpreter]())
#         rnnForward = yield from self.createRnnForward(activationStep)
#         influenceTensor = yield from interpreter.getInfluenceTensor
#         actv0 = yield from interpreter.getRecurrentState
#         param0 = yield from interpreter.getRecurrentParam

#         wrtActvFn = lambda a: rnnForward(a, param0)[0]
#         immediateJacobian__InfluenceTensor_product: Array = jacobian_matrix_product(wrtActvFn, actv0, influenceTensor)
#         immediateInfluence: Array
#         env: Env
#         immediateInfluence, env = self.immediateJacFn(lambda p: rnnForward(actv0, p), has_aux=True)(param0)
#         newInfluenceTensor = Jacobian[REC_PARAM](immediateJacobian__InfluenceTensor_product + immediateInfluence)

#         log_condition = yield from interpreter.getLogConfig.fmap(lambda x: x.log_special)
#         lanczos_iterations = yield from interpreter.getLogConfig.fmap(lambda x: x.lanczos_iterations)
#         log_expensive = yield from interpreter.getLogConfig.fmap(lambda x: x.log_expensive)
#         log_influence = yield from interpreter.getGlobalLogConfig.fmap(lambda x: x.log_influence)
#         subkey = yield from interpreter.updatePRNG()

#         if log_condition:
#             v0: Array = jnp.array(jax.random.normal(subkey, actv0.shape))
#             tridag = matfree.decomp.tridiag_sym(lanczos_iterations, custom_vjp=False)
#             get_eig = matfree.eig.eigh_partial(tridag)
#             fn = lambda v: jvp(lambda a: wrtActvFn(a), actv0, v)
#             eigvals, _ = get_eig(fn, v0)
#             _ = yield from interpreter.putLogs(Logs(jac_eigenvalue=jnp.max(eigvals)))

#         if log_expensive:
#             _ = yield from interpreter.putLogs(Logs(hessian=eqx.filter_jacrev(wrtActvFn)(actv0)))

#         _ = yield from put(env)
#         if log_influence:
#             _ = yield from interpreter.putLogs(Logs(immediateInfluenceTensor=immediateInfluence))
#         return pure(newInfluenceTensor, PX[tuple[Interpreter, Env]]())


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


def take_jacobian[ENV, DATA, OUT: jax.Array, AUX](
    transition: Callable[[ENV, DATA], tuple[OUT, AUX]],
    learn_interface: LearnInterface[ENV],
) -> Callable[[ENV, DATA], tuple[JACOBIAN, AUX]]:
    def jacobian_fn(env: ENV, data: DATA) -> tuple[JACOBIAN, AUX]:
        s__p = learn_interface.get_state(env), learn_interface.get_param(env)
        s__p_vector = to_vector(s__p)

        def fn(state__param: jax.Array, data: DATA) -> OUT:
            state, param = s__p_vector.to_param(state__param)
            _env = learn_interface.put_state(env, state)
            _env = learn_interface.put_param(_env, param)
            out, aux = transition(_env, data)
            return out, aux

        state__param = s__p_vector.vector
        jacobian, aux = eqx.filter_grad(fn, has_aux=True)(state__param, data)
        return JACOBIAN(jacobian), aux

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
    _readout_gr = take_jacobian(lambda env, data: (statistics_fns[0](env, data)[2], None), learn_interfaces[0])
    readout_gr = lambda env, data: GRADIENT(_readout_gr(env, data)[0])

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
