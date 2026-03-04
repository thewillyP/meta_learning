from typing import Callable
import jax
import jax.numpy as jnp
import optax

from meta_learn_lib.config import *
from meta_learn_lib.env import Outputs
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import LOSS, STAT
from meta_learn_lib.util import accuracy_hard, accuracy_soft


def log_p_z(prior: ELBOObjective.Prior, z: jax.Array) -> jax.Array:
    match prior:
        case ELBOObjective.GaussianPrior(mu, log_sigma):
            return -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2 * log_sigma + (z - mu) ** 2 / jnp.exp(2 * log_sigma))


def kl(posterior: ELBOObjective.Posterior, prior: ELBOObjective.Prior, outputs: Outputs) -> jax.Array:
    match (posterior, prior):
        case (ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(prior_mu, prior_log_sigma)):
            mu = outputs.mu
            log_sigma = outputs.log_sigma
            return 0.5 * jnp.sum(
                2 * (prior_log_sigma - log_sigma)
                + (jnp.exp(2 * log_sigma) + (mu - prior_mu) ** 2) / jnp.exp(2 * prior_log_sigma)
                - 1
            )
        case _:
            return outputs.log_q_z - log_p_z(prior, outputs.z)


def create_loss_fn[ENV](
    objective_fn: ObjectiveFn,
    label_mask_value: float,
    unlabeled_mask_value: float,
    task_interface: GodInterface[ENV],
) -> Callable[[ENV, Outputs, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]]:

    match objective_fn:
        case CrossEntropyObjective(mode):
            match mode:
                case "cross_entropy_with_integer_labels":
                    _loss_fn = lambda logits, y: optax.losses.softmax_cross_entropy_with_integer_labels(logits, y)
                case "cross_entropy":
                    _loss_fn = lambda logits, y: optax.safe_softmax_cross_entropy(logits, y)

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                raw_loss = _loss_fn(pred, target)
                loss = LOSS(jnp.sum(jnp.where(mask, raw_loss, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0))
                hard_acc = jnp.sum(jnp.where(mask, accuracy_hard(pred, target), 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)
                soft_acc = jnp.sum(jnp.where(mask, accuracy_soft(pred, target), 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)
                return loss, {
                    "loss": loss,
                    "accuracy_hard": jax.lax.stop_gradient(hard_acc),
                    "accuracy_soft": jax.lax.stop_gradient(soft_acc),
                }

        case RegressionObjective():

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                raw_loss = optax.losses.squared_error(pred, target)
                loss = LOSS(jnp.sum(jnp.where(mask, raw_loss, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0))
                return loss, {"loss": loss}

        case BernoulliObjective():

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                raw_loss = optax.losses.sigmoid_binary_cross_entropy(pred, target)
                loss = LOSS(jnp.sum(jnp.where(mask, raw_loss, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0))
                return loss, {"loss": loss}

        case ELBOObjective(beta_hp, likelihood, posterior, prior):
            inner_loss_fn = create_loss_fn(likelihood, unlabeled_mask_value, unlabeled_mask_value, task_interface)

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                x, _ = data
                recon_loss, stats = inner_loss_fn(env, outputs, (x, x))

                beta = task_interface.get_kl_regularizer_beta(env).value
                kl_value = kl(posterior, prior, outputs)

                loss = LOSS(recon_loss + beta * kl_value)
                stats["recon_loss"] = jax.lax.stop_gradient(recon_loss)
                stats["kl"] = jax.lax.stop_gradient(kl_value)
                stats["elbo_loss"] = jax.lax.stop_gradient(loss)
                return loss, stats

    return loss_fn


def create_readout_loss_fns[ENV](
    config: GodConfig,
    task_interfaces: list[GodInterface[ENV]],
    readouts: list[Callable[[ENV, tuple[jax.Array, jax.Array]], Outputs]],
) -> list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]]]:

    def compose(loss_fn, readout):
        def fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
            outputs = readout(env, data)
            return loss_fn(env, outputs, data)

        return fn

    return [
        compose(
            create_loss_fn(
                meta_config.objective_fn,
                config.label_mask_value,
                config.unlabeled_mask_value,
                task_interface,
            ),
            readout,
        )
        for meta_config, task_interface, readout in zip(config.levels, task_interfaces, readouts)
    ]
