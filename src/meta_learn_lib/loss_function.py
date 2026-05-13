import jax
import jax.numpy as jnp
import optax
from typing import Callable

from meta_learn_lib.config import *
from meta_learn_lib.env import Logs, Outputs
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import LOSS, NamedStat, STAT, S_ID
from meta_learn_lib.constants import TASK
from meta_learn_lib.util import accuracy, hyperparameter_reparametrization, scalar


def log_p_z(prior: ELBOObjective.Prior, z: jax.Array) -> jax.Array:
    match prior:
        case ELBOObjective.GaussianPrior(mu, log_var):
            return -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + log_var + (z - mu) ** 2 / jnp.exp(log_var), axis=-1)


def kl(posterior: ELBOObjective.Posterior, prior: ELBOObjective.Prior, outputs: Outputs, mask: jax.Array) -> jax.Array:
    # mask is element-wise (shape matches target); KL is per-example. Collapse trailing
    # feature axes to get a per-example mask, then weight the per-example KL by it.
    # Substitute safe values before exp(log_var) so masked-out positions (which can carry
    # huge log_var from -1e10-padded inputs through an un-normalized encoder) don't produce
    # inf, which would survive `where` forward but propagate NaN through `0 * inf` in the gradient.
    example_mask = jnp.any(mask, axis=tuple(range(2, mask.ndim)))
    match (posterior, prior):
        case (ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(prior_mu, prior_log_var)):
            safe = example_mask[..., None]
            mu = jnp.where(safe, outputs.mu, 0.0)
            log_var = jnp.where(safe, outputs.log_var, 0.0)
            kl_per_example = -0.5 * jnp.sum(
                1 + log_var - prior_log_var - (jnp.exp(log_var) + (mu - prior_mu) ** 2) / jnp.exp(prior_log_var),
                axis=-1,
            )
            return jnp.sum(jnp.where(example_mask, kl_per_example, 0.0)) / jnp.maximum(jnp.sum(example_mask), 1.0)
        case _:
            kl_per_example = outputs.log_q_z - log_p_z(prior, outputs.z)
            return jnp.sum(jnp.where(example_mask, kl_per_example, 0.0)) / jnp.maximum(jnp.sum(example_mask), 1.0)


def nan_if_masked(value: jax.Array, has_label: jax.Array) -> jax.Array:
    return jnp.where(has_label, value, jnp.nan)


def get_env_stats[ENV](
    config: GodConfig,
    env: ENV,
    interfaces: dict[S_ID, GodInterface[ENV]],
    level: int,
) -> STAT:
    stat: STAT = {}
    task_interface = interfaces[(TASK, level)]

    for hp_name, hp_config in config.hyperparameters.items():
        if hp_config.level != level:
            continue
        if (hp_name, hp_config.level) not in interfaces:
            continue
        iface = interfaces[(hp_name, hp_config.level)]
        match hp_config.kind:
            case "learning_rate":
                raw = iface.learning_rate.get(env)
            case "weight_decay":
                raw = iface.weight_decay.get(env)
            case "momentum":
                raw = iface.momentum.get(env)
            case "time_constant":
                raw = iface.time_constant.get(env)
            case "kl_regularizer_beta":
                raw = iface.kl_regularizer_beta.get(env)
        forward, _ = hyperparameter_reparametrization(hp_config.hyperparameter_parametrization)
        values = forward(raw)
        for i in range(hp_config.count):
            stat[f"level{level}/{hp_config.kind}/{hp_name}/{i}"] = scalar(jax.lax.stop_gradient(values[i]))

    logs: Logs = task_interface.logs.get(env)
    if logs.gradient is not None:
        stat[f"level{level}/gradient_norm"] = scalar(jax.lax.stop_gradient(jnp.linalg.norm(logs.gradient)))
    if logs.largest_eigenvalue is not None:
        stat[f"level{level}/largest_eigenvalue"] = scalar(jax.lax.stop_gradient(logs.largest_eigenvalue))
    if logs.hessian_contains_nans is not None:
        stat[f"level{level}/hessian_contains_nans"] = scalar(jax.lax.stop_gradient(logs.hessian_contains_nans))
    if logs.largest_jac_eigenvalue is not None:
        stat[f"level{level}/largest_jac_eigenvalue"] = scalar(jax.lax.stop_gradient(logs.largest_jac_eigenvalue))
    if logs.influence_tensor_norm is not None:
        stat[f"level{level}/influence_tensor_norm"] = scalar(jax.lax.stop_gradient(logs.influence_tensor_norm))

    return stat


def create_loss_fn[ENV](
    objective_fn: ObjectiveFn,
    label_mask_value: float,
    unlabeled_mask_value: float,
    task_interface: GodInterface[ENV],
) -> Callable[[ENV, Outputs, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]]:

    def reduce(masked: jax.Array, mask: jax.Array, reduction: Reduction) -> jax.Array:
        feature_dims = tuple(range(2, masked.ndim))
        match reduction:
            case "sum":
                per_example = jnp.sum(masked, axis=feature_dims)
                example_mask = jnp.any(mask, axis=feature_dims)
                return jnp.sum(jnp.where(example_mask, per_example, 0.0)) / jnp.maximum(jnp.sum(example_mask), 1.0)
            case "mean":
                return jnp.sum(masked) / jnp.maximum(jnp.sum(mask), 1.0)

    match objective_fn:
        case CrossEntropyObjective(mode):
            match mode:
                case "cross_entropy_with_integer_labels":
                    _loss_fn = lambda logits, y: optax.losses.softmax_cross_entropy_with_integer_labels(
                        logits, jnp.asarray(y, dtype=int)
                    )
                    _to_onehot = lambda target, num_classes: jax.nn.one_hot(target, num_classes)
                case "cross_entropy":
                    _loss_fn = lambda logits, y: optax.safe_softmax_cross_entropy(logits, y)
                    _to_onehot = lambda target, num_classes: target

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                has_label = jnp.any(mask)
                raw_loss = _loss_fn(pred, target)
                loss = LOSS(jnp.sum(jnp.where(mask, raw_loss, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0))
                target_onehot = _to_onehot(target, pred.shape[-1])
                acc = jnp.sum(jnp.where(mask, accuracy(pred, target_onehot), 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)
                return loss, {
                    "loss": scalar(nan_if_masked(loss, has_label)),
                    "accuracy": scalar(jax.lax.stop_gradient(nan_if_masked(acc, has_label))),
                }

        case RegressionObjective(reduction):

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                has_label = jnp.any(mask)
                raw_loss = optax.losses.squared_error(pred, target)
                loss = LOSS(reduce(jnp.where(mask, raw_loss, 0.0), mask, reduction))
                return loss, {"loss": scalar(nan_if_masked(loss, has_label))}

        case BernoulliObjective(reduction):

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                has_label = jnp.any(mask)
                raw_loss = optax.losses.sigmoid_binary_cross_entropy(pred, target)
                loss = LOSS(reduce(jnp.where(mask, raw_loss, 0.0), mask, reduction))
                return loss, {"loss": scalar(nan_if_masked(loss, has_label))}

        case ELBOObjective(beta_hp, likelihood, posterior, prior):
            inner_loss_fn = create_loss_fn(likelihood, label_mask_value, label_mask_value, task_interface)

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                x, target = data
                # Mask source = y (target); x may be padded with a safe value the encoder tolerates
                # (e.g. 0.0). Synthesize a target where padded slots carry label_mask_value so the
                # inner loss's `target != label_mask_value` mask still works.
                example_mask = jnp.any(target != label_mask_value, axis=tuple(range(2, target.ndim)))
                pixel_mask = example_mask.reshape(example_mask.shape + (1,) * (x.ndim - example_mask.ndim))
                masked_x = jnp.where(pixel_mask, x, label_mask_value)
                recon_loss, stats = inner_loss_fn(env, outputs, (x, masked_x))

                beta = task_interface.kl_regularizer_beta.get(env)[0]
                kl_value = kl(posterior, prior, outputs, example_mask)

                loss = LOSS(recon_loss + beta * kl_value)
                safe_mask = example_mask[..., None]
                num_valid = jnp.maximum(jnp.sum(example_mask) * outputs.mu.shape[-1], 1.0)
                mu_mean = jnp.sum(jnp.where(safe_mask, outputs.mu, 0.0)) / num_valid
                lv_mean = jnp.sum(jnp.where(safe_mask, outputs.log_var, 0.0)) / num_valid
                mu_std = jnp.sqrt(jnp.sum(jnp.where(safe_mask, (outputs.mu - mu_mean) ** 2, 0.0)) / num_valid)
                lv_std = jnp.sqrt(jnp.sum(jnp.where(safe_mask, (outputs.log_var - lv_mean) ** 2, 0.0)) / num_valid)
                stats["kl"] = scalar(jax.lax.stop_gradient(kl_value))
                stats["elbo_loss"] = scalar(jax.lax.stop_gradient(loss))
                stats["mu_mean"] = scalar(jax.lax.stop_gradient(mu_mean))
                stats["mu_std"] = scalar(jax.lax.stop_gradient(mu_std))
                stats["log_var_mean"] = scalar(jax.lax.stop_gradient(lv_mean))
                stats["log_var_std"] = scalar(jax.lax.stop_gradient(lv_std))
                return loss, stats

        case NoopObjective():

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                return LOSS(jnp.zeros(())), {}

    return loss_fn


def create_readout_loss_fns[ENV](
    config: GodConfig,
    interfaces: dict[S_ID, GodInterface[ENV]],
    readouts: list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, Outputs]]],
) -> list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]]:

    def compose(
        loss_fn: Callable[[ENV, Outputs, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]],
        readout: Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, Outputs]],
        level: int,
        collect_predictions: bool,
    ) -> Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]:
        def fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, LOSS, STAT]:
            env, outputs = readout(env, data)
            loss, stat = loss_fn(env, outputs, data)
            stat = {f"level{level}/{k}": v for k, v in stat.items()}
            stat |= get_env_stats(config, env, interfaces, level)
            if collect_predictions:
                pred = jax.lax.stop_gradient(outputs.prediction)
                stat[f"level{level}/prediction"] = NamedStat(
                    pred,
                    ("batch", "minibatch") + ("feature",) * (pred.ndim - 2),
                )
                _, y_data = data
                stat[f"level{level}/label"] = NamedStat(
                    jax.lax.stop_gradient(y_data),
                    ("batch", "minibatch") + ("feature",) * (y_data.ndim - 2),
                )
            return env, loss, stat

        return fn

    return [
        compose(
            create_loss_fn(
                meta_config.objective_fn,
                config.label_mask_value,
                config.unlabeled_mask_value,
                interfaces[(TASK, level)],
            ),
            readout,
            level,
            meta_config.collect_predictions,
        )
        for level, (meta_config, readout) in enumerate(zip(config.levels, readouts))
    ]
