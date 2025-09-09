from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from lib.config import DelayAddOnlineConfig, FashionMnistConfig, GodConfig, MnistConfig
from lib.interface import ClassificationInterface, GeneralInterface
from lib.lib_types import LOSS, STAT
from lib.util import accuracy_hard, filter_cond, get_loss_fn


def make_statistics_fns[ENV, DATA, OUT](
    config: GodConfig,
    inferences: dict[int, Callable[[ENV, DATA], tuple[ENV, OUT]]],
    data_interface: ClassificationInterface[DATA],
    get_out: Callable[[OUT], jax.Array],
    general_interfaces: dict[int, GeneralInterface[ENV]],
    virtual_minibatches: dict[int, int],
    last_unpadded_lengths: dict[int, int],
) -> dict[int, Callable[[ENV, tuple[DATA, jax.Array]], tuple[ENV, tuple[STAT, ...], LOSS]]]:
    model_loss_fns = {}
    for i, (general_interface, virtual_minibatch, last_unpadded_length, data_config) in enumerate(
        zip(
            general_interfaces.values(),
            virtual_minibatches.values(),
            last_unpadded_lengths.values(),
            config.data.values(),
        )
    ):
        loss_fn = make_loss_fn(
            config,
            general_interface,
            virtual_minibatch,
            last_unpadded_length,
            data_config.num_times_to_avg_in_timeseries,
        )

        statistics_fn = make_statistics_fn(
            config,
            general_interface,
            virtual_minibatch,
            last_unpadded_length,
            data_config.num_times_to_avg_in_timeseries,
        )

        def model_loss_fn(env: ENV, ds: tuple[DATA, jax.Array], j=i) -> tuple[ENV, tuple[STAT, ...], LOSS]:
            data, mask = ds
            env, _preds = inferences[j](env, data)
            preds = get_out(_preds)
            targets = data_interface.get_target(data)
            loss = loss_fn(env, preds, targets, mask)
            stat = statistics_fn(env, preds, targets, mask)
            return env, (stat,), loss

        model_loss_fns[i] = model_loss_fn

    return model_loss_fns


def make_loss_fn[ENV](
    config: GodConfig,
    general_interface: GeneralInterface[ENV],
    virtual_minibatch: int,
    last_unpadded_length: int,
    num_times_to_avg_in_timeseries: int,
) -> Callable[[ENV, jax.Array, jax.Array, jax.Array], LOSS]:
    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            _loss_fn = get_loss_fn(config.loss_fn)

        case MnistConfig() | FashionMnistConfig():
            __loss_fn = get_loss_fn(config.loss_fn)

            def loss_sequence_length(p: jax.Array, t: jax.Array) -> jax.Array:
                sequence_length = p.shape[0]

                def new_loss_fn(pred: jax.Array, target: jax.Array) -> jax.Array:
                    label, idx = target
                    return jax.lax.cond(
                        idx == sequence_length - 1,
                        lambda p: __loss_fn(p[None, :], jnp.array([label])).sum(),
                        lambda _: 0.0,
                        pred,
                    )

                return eqx.filter_vmap(new_loss_fn)(p, t)

            _loss_fn = eqx.filter_vmap(loss_sequence_length)

    def loss_fn(env: ENV, pred: jax.Array, target: jax.Array, batch_mask: jax.Array) -> LOSS:
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)
        current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(env)
        loss = _loss_fn(pred, target)
        sequence_masked_loss = filter_cond(
            current_virtual_minibatch % virtual_minibatch == 0
            and current_avg_in_timeseries == num_times_to_avg_in_timeseries - 1,
            lambda l: l * (jnp.arange(l.shape[1]) < last_unpadded_length),
            lambda l: l,
            loss,
        )

        mask_shape = (batch_mask.shape[0],) + (1,) * (sequence_masked_loss.ndim - 1)
        mask_expanded = jnp.reshape(batch_mask, mask_shape)
        valid_loss = jnp.where(mask_expanded, sequence_masked_loss, 0.0)

        return LOSS(jnp.sum(valid_loss) / jnp.sum(mask_expanded))

    return loss_fn


# 1. account for padding 2. account for sequence filtering 3. take the mean()


def make_statistics_fn[ENV](
    config: GodConfig,
    general_interface: GeneralInterface[ENV],
    virtual_minibatch: int,
    last_unpadded_length: int,
    num_times_to_avg_in_timeseries: int,
) -> Callable[[ENV, jax.Array, jax.Array, jax.Array], STAT]:
    match config.dataset:
        case DelayAddOnlineConfig():
            _statistics_fn = lambda _, __: jnp.array(0.0)

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
            seq_len = 784 // n_in - 1
            _statistics_fn = lambda pred, target: accuracy_hard(pred, target[..., 0]) * (target[..., 1] == seq_len)
            # premptively multiply by series length so averaging cancels out the factor

    def statistics_fn(env: ENV, pred: jax.Array, target: jax.Array, batch_mask: jax.Array) -> STAT:
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)
        current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(env)
        statistics = _statistics_fn(pred, target)
        sequence_masked_statistics = filter_cond(
            current_virtual_minibatch % virtual_minibatch == 0
            and current_avg_in_timeseries == num_times_to_avg_in_timeseries - 1,
            lambda s: s * (jnp.arange(s.shape[1]) < last_unpadded_length),
            lambda s: s,
            statistics,
        )

        mask_shape = (batch_mask.shape[0],) + (1,) * (sequence_masked_statistics.ndim - 1)
        mask_expanded = jnp.reshape(batch_mask, mask_shape)
        valid_statistics = jnp.where(mask_expanded, sequence_masked_statistics, 0.0)

        return jnp.sum(valid_statistics) / jnp.sum(mask_expanded)

    return statistics_fn
