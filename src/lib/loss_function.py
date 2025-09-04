from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from lib.config import DelayAddOnlineConfig, FashionMnistConfig, GodConfig, MnistConfig
from lib.interface import GeneralInterface
from lib.lib_types import LOSS
from lib.util import filter_cond, get_loss_fn


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
            _loss_fn = get_loss_fn(config.loss_fn)

            def loss_sequence_length(p: jax.Array, t: jax.Array) -> jax.Array:
                sequence_length = p.shape[0]

                def new_loss_fn(pred: jax.Array, target: jax.Array) -> jax.Array:
                    label, idx = target
                    return jax.lax.cond(idx == sequence_length - 1, lambda p: _loss_fn(p, label), lambda _: 0.0, pred)

                return eqx.filter_vmap(new_loss_fn)(p, t)

            _loss_fn = eqx.filter_vmap(loss_sequence_length)

    def loss_fn(env: ENV, pred: jax.Array, target: jax.Array, batch_mask: jax.Array) -> LOSS:
        current_virtual_minibatch = general_interface.get_current_virtual_minibatch(env)
        current_avg_in_timeseries = general_interface.get_current_avg_in_timeseries(env)
        loss = _loss_fn(pred, target)
        sequence_masked_loss = filter_cond(
            current_virtual_minibatch % virtual_minibatch == 0
            and current_avg_in_timeseries == num_times_to_avg_in_timeseries - 1,
            lambda l: l * (jnp.arange(l.shape[1]) < (l.shape[1] - last_unpadded_length)),
            lambda l: l,
            loss,
        )

        valid_loss = sequence_masked_loss[batch_mask]
        return LOSS(jnp.mean(valid_loss))

    return loss_fn


# 1. account for padding 2. account for sequence filtering 3. take the mean()
