from typing import Callable

import jax
import jax.numpy as jnp
import optax
from meta_learn_lib.config import AdamConfig, LearnConfig, SGDClipConfig, SGDConfig, SGDNormalizedConfig
from meta_learn_lib.env import Parameter
from meta_learn_lib.util import hyperparameter_reparametrization


def get_optimizer(
    learn_config: LearnConfig, format_to_param: Callable[[jax.Array], Parameter]
) -> Callable[[Parameter], optax.GradientTransformation]:
    # forward, _ = hyperparameter_reparametrization(learn_config.hyperparameter_parametrization)

    match learn_config.optimizer:
        case SGDConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.chain(
                optax.add_decayed_weights(lr_forward(pr.learning_parameter.weight_decay.value)),
                optax.sgd(wd_forward(pr.learning_parameter.learning_rate.value), momentum=momentum),
            )
        case SGDNormalizedConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.add_decayed_weights(lr_forward(pr.learning_parameter.weight_decay.value)),
                optax.sgd(wd_forward(pr.learning_parameter.learning_rate.value), momentum=momentum),
            )
        case SGDClipConfig(learning_rate, weight_decay, momentum, threshold, sharpness):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)

            def update_fn(updates, state, _):
                grads_flat, _ = jax.flatten_util.ravel_pytree(updates)
                grad_norm = jnp.linalg.norm(grads_flat)
                clipped_norm = grad_norm - jax.nn.softplus(sharpness * (grad_norm - threshold)) / sharpness
                scale = clipped_norm / (grad_norm + 1e-6)
                updates_scaled = jax.tree.map(lambda g: g * scale, updates)
                return updates_scaled, state

            return lambda pr: optax.chain(
                optax.GradientTransformation(lambda _: (), update_fn),
                optax.add_decayed_weights(lr_forward(pr.learning_parameter.weight_decay.value)),
                optax.sgd(wd_forward(pr.learning_parameter.learning_rate.value), momentum=momentum),
            )
        case AdamConfig(learning_rate, weight_decay):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.adamw(
                lr_forward(pr.learning_parameter.learning_rate.value),
                weight_decay=wd_forward(pr.learning_parameter.weight_decay.value),
            )
