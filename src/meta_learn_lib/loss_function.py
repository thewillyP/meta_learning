import jax
import jax.numpy as jnp
import optax
from typing import Callable

from meta_learn_lib.config import *
from meta_learn_lib.env import Logs, Outputs
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import LOSS, STAT
from meta_learn_lib.util import accuracy, hyperparameter_reparametrization


def log_p_z(prior: ELBOObjective.Prior, z: jax.Array) -> jax.Array:
    match prior:
        case ELBOObjective.GaussianPrior(mu, log_var):
            return -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + log_var + (z - mu) ** 2 / jnp.exp(log_var), axis=-1)


def kl(posterior: ELBOObjective.Posterior, prior: ELBOObjective.Prior, outputs: Outputs, mask: jax.Array) -> jax.Array:
    match (posterior, prior):
        case (ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(prior_mu, prior_log_var)):
            mu = outputs.mu
            log_var = outputs.log_var
            kl_per_example = -0.5 * jnp.sum(
                1 + log_var - prior_log_var - (jnp.exp(log_var) + (mu - prior_mu) ** 2) / jnp.exp(prior_log_var),
                axis=-1,
            )
            return jnp.sum(jnp.where(mask, kl_per_example, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)
        case _:
            kl_per_example = outputs.log_q_z - log_p_z(prior, outputs.z)
            return jnp.sum(jnp.where(mask, kl_per_example, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)


def nan_if_masked(value: jax.Array, has_label: jax.Array) -> jax.Array:
    return jnp.where(has_label, value, jnp.nan)


def get_env_stats[ENV](
    config: GodConfig,
    env: ENV,
    meta_interfaces: dict[str, GodInterface[ENV]],
    task_interface: GodInterface[ENV],
    level: int,
) -> STAT:
    stat: STAT = {}

    def get_hp_value(interface: GodInterface[ENV], kind: HyperparameterConfig.Kind) -> jax.Array:
        match kind:
            case "learning_rate":
                p = interface.get_learning_rate(env)
            case "weight_decay":
                p = interface.get_weight_decay(env)
            case "momentum":
                p = interface.get_momentum(env)
            case "time_constant":
                p = interface.get_time_constant(env)
            case "kl_regularizer_beta":
                p = interface.get_kl_regularizer_beta(env)
        return p.value

    for hp_name, hp_config in config.hyperparameters.items():
        if hp_config.level != level:
            continue
        iface = meta_interfaces.get(hp_name)
        if iface is None:
            continue
        raw = get_hp_value(iface, hp_config.kind)
        forward, _ = hyperparameter_reparametrization(hp_config.hyperparameter_parametrization)
        values = forward(raw)
        for i in range(hp_config.count):
            stat[f"level{level}/{hp_config.kind}/{hp_name}/{i}"] = jax.lax.stop_gradient(values[i])

    logs: Logs = task_interface.get_logs(env)
    if logs.gradient is not None:
        stat[f"level{level}/gradient_norm"] = jax.lax.stop_gradient(jnp.linalg.norm(logs.gradient))
    if logs.largest_eigenvalue is not None:
        stat[f"level{level}/largest_eigenvalue"] = jax.lax.stop_gradient(logs.largest_eigenvalue)
    if logs.hessian_contains_nans is not None:
        stat[f"level{level}/hessian_contains_nans"] = jax.lax.stop_gradient(logs.hessian_contains_nans)
    if logs.largest_jac_eigenvalue is not None:
        stat[f"level{level}/largest_jac_eigenvalue"] = jax.lax.stop_gradient(logs.largest_jac_eigenvalue)
    if logs.influence_tensor_norm is not None:
        stat[f"level{level}/influence_tensor_norm"] = jax.lax.stop_gradient(logs.influence_tensor_norm)

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
                    "loss": nan_if_masked(loss, has_label),
                    "accuracy": jax.lax.stop_gradient(nan_if_masked(acc, has_label)),
                }

        case RegressionObjective(reduction):

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                has_label = jnp.any(mask)
                raw_loss = optax.losses.squared_error(pred, target)
                loss = LOSS(reduce(jnp.where(mask, raw_loss, 0.0), mask, reduction))
                return loss, {"loss": nan_if_masked(loss, has_label)}

        case BernoulliObjective(reduction):

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                _, target = data
                pred = outputs.prediction
                mask = target != label_mask_value
                has_label = jnp.any(mask)
                raw_loss = optax.losses.sigmoid_binary_cross_entropy(pred, target)
                loss = LOSS(reduce(jnp.where(mask, raw_loss, 0.0), mask, reduction))
                return loss, {"loss": nan_if_masked(loss, has_label)}

        case ELBOObjective(beta_hp, likelihood, posterior, prior):
            inner_loss_fn = create_loss_fn(likelihood, unlabeled_mask_value, unlabeled_mask_value, task_interface)

            def loss_fn(env: ENV, outputs: Outputs, data: tuple[jax.Array, jax.Array]) -> tuple[LOSS, STAT]:
                x, target = data
                recon_loss, stats = inner_loss_fn(env, outputs, (x, x))

                beta = task_interface.get_kl_regularizer_beta(env).value
                mask = target != label_mask_value
                kl_value = kl(posterior, prior, outputs, mask)

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
    meta_interfaces: list[dict[str, GodInterface[ENV]]],
) -> list[Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]]:

    def compose(
        loss_fn: Callable[[ENV, Outputs, tuple[jax.Array, jax.Array]], tuple[LOSS, STAT]],
        readout: Callable[[ENV, tuple[jax.Array, jax.Array]], Outputs],
        level: int,
        interfaces: dict[str, GodInterface[ENV]],
        task_interface: GodInterface[ENV],
    ) -> Callable[[ENV, tuple[jax.Array, jax.Array]], tuple[ENV, LOSS, STAT]]:
        def fn(env: ENV, data: tuple[jax.Array, jax.Array]) -> tuple[ENV, LOSS, STAT]:
            outputs = readout(env, data)
            loss, stat = loss_fn(env, outputs, data)
            stat = {f"level{level}/{k}": v for k, v in stat.items()}
            stat |= get_env_stats(config, env, interfaces, task_interface, level)
            return env, loss, stat

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
            level,
            interfaces,
            task_interface,
        )
        for level, (meta_config, task_interface, readout, interfaces) in enumerate(
            zip(config.levels, task_interfaces, readouts, meta_interfaces)
        )
    ]
