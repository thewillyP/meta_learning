from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from meta_learn_lib.config import CIFAR10Config, DelayAddOnlineConfig, FashionMnistConfig, GodConfig, MnistConfig
from meta_learn_lib.interface import ClassificationInterface, GeneralInterface
from meta_learn_lib.lib_types import LOSS, STAT
from meta_learn_lib.util import accuracy_hard, filter_cond, get_loss_fn


def make_statistics_fns[ENV, DATA, OUT](
    config: GodConfig,
    readouts: dict[int, Callable[[ENV, DATA], OUT]],
    data_interface: ClassificationInterface[DATA],
    get_out: Callable[[OUT], jax.Array],
    general_interfaces: dict[int, GeneralInterface[ENV]],
    virtual_minibatches: dict[int, int],
    last_unpadded_lengths: dict[int, int],
    get_logs: Callable[[ENV], tuple[jax.Array, ...]],
) -> dict[int, Callable[[ENV, tuple[DATA, jax.Array]], tuple[tuple[STAT, ...], LOSS]]]:
    model_loss_fns = {}
    for i, (general_interface, virtual_minibatch, last_unpadded_length) in enumerate(
        zip(
            general_interfaces.values(),
            virtual_minibatches.values(),
            last_unpadded_lengths.values(),
        )
    ):
        loss_fn = make_loss_fn(
            config,
            general_interface,
            virtual_minibatch,
            last_unpadded_length,
        )

        statistics_fn = make_statistics_fn(
            config,
            general_interface,
            virtual_minibatch,
            last_unpadded_length,
            get_logs,
        )

        def model_loss_fn(
            env: ENV, ds: tuple[DATA, jax.Array], j=i, loss_fn=loss_fn, statistics_fn=statistics_fn
        ) -> tuple[tuple[STAT, ...], LOSS]:
            data, mask = ds
            preds = get_out(readouts[j](env, data))
            targets = data_interface.get_target(data)
            seq_num = data_interface.get_sequence(data)
            loss = loss_fn(env, preds, targets, seq_num, mask)
            stat = statistics_fn(env, preds, targets, seq_num, mask, loss)
            return (stat,), loss

        model_loss_fns[i] = model_loss_fn

    return model_loss_fns


def make_loss_fn[ENV](
    config: GodConfig,
    general_interface: GeneralInterface[ENV],
    virtual_minibatch: int,
    last_unpadded_length: int,
) -> Callable[[ENV, jax.Array, jax.Array, jax.Array, jax.Array], LOSS]:
    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            _loss_fn = get_loss_fn(config.loss_fn)

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
            seq_len = 784 // n_in - 1
            __loss_fn = get_loss_fn(config.loss_fn)

            def loss_sequence_length(pred: jax.Array, target: jax.Array) -> jax.Array:
                label, idx = target
                return jax.lax.cond(
                    idx == seq_len,
                    lambda p: __loss_fn(p[None, :], jnp.array([label])).sum(),
                    lambda _: 0.0,
                    pred,
                )

            _loss_fn = eqx.filter_vmap(loss_sequence_length)

        case CIFAR10Config(n_in):
            seq_len = 3072 // n_in - 1
            __loss_fn = get_loss_fn(config.loss_fn)

            def loss_sequence_length(pred: jax.Array, target: jax.Array) -> jax.Array:
                label, idx = target
                return jax.lax.cond(
                    idx == seq_len,
                    lambda p: __loss_fn(p[None, :], jnp.array([label])).sum(),
                    lambda _: 0.0,
                    pred,
                )

            _loss_fn = eqx.filter_vmap(loss_sequence_length)

    def loss_fn(env: ENV, pred: jax.Array, target: jax.Array, seq_num: jax.Array, batch_mask: jax.Array) -> LOSS:
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)
        loss = _loss_fn(pred, target)
        sequence_masked_loss = filter_cond(
            current_virtual_minibatch % virtual_minibatch == 0,
            lambda l: l * (seq_num < last_unpadded_length),
            lambda l: l,
            loss,
        )

        mask_shape = (batch_mask.shape[0],) + (1,) * (sequence_masked_loss.ndim - 1)
        mask_expanded = jnp.reshape(batch_mask, mask_shape)
        valid_loss = jnp.where(mask_expanded, sequence_masked_loss, 0.0)

        return LOSS(jnp.sum(valid_loss) / jnp.sum(mask_expanded))

    return loss_fn


def make_statistics_fn[ENV](
    config: GodConfig,
    general_interface: GeneralInterface[ENV],
    virtual_minibatch: int,
    last_unpadded_length: int,
    get_logs: Callable[[ENV], tuple[jax.Array, ...]],
) -> Callable[[ENV, jax.Array, jax.Array, jax.Array], STAT]:
    match config.dataset:
        case DelayAddOnlineConfig():
            _statistics_fn = lambda _, __: jnp.array([0.0])

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
            seq_len = 784 // n_in - 1
            _statistics_fn = lambda pred, target: accuracy_hard(pred, target[..., 0]) * (target[..., 1] == seq_len)
            # premptively multiply by series length so averaging cancels out the factor

        case CIFAR10Config(n_in):
            seq_len = 3072 // n_in - 1
            _statistics_fn = lambda pred, target: accuracy_hard(pred, target[..., 0]) * (target[..., 1] == seq_len)

    def statistics_fn(
        env: ENV, pred: jax.Array, target: jax.Array, seq_num: jax.Array, batch_mask: jax.Array, loss: jax.Array
    ) -> STAT:
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)
        statistics = _statistics_fn(pred, target)
        sequence_masked_statistics = filter_cond(
            current_virtual_minibatch % virtual_minibatch == 0,
            lambda s: s * (seq_num < last_unpadded_length),
            lambda s: s,
            statistics,
        )

        mask_shape = (batch_mask.shape[0],) + (1,) * (sequence_masked_statistics.ndim - 1)
        mask_expanded = jnp.reshape(batch_mask, mask_shape)
        valid_statistics = jnp.where(mask_expanded, sequence_masked_statistics, 0.0)

        statistic = jnp.sum(valid_statistics) / jnp.sum(mask_expanded)
        logs = get_logs(env)
        return (
            loss,
            statistic,
        ) + logs

    return statistics_fn
