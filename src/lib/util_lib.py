from typing import Callable

import jax
import jax.numpy as jnp
import optax
from lib.config import AdamConfig, LearnConfig, SGDClipConfig, SGDConfig, SGDNormalizedConfig
from lib.env import Parameter
from lib.util import hyperparameter_reparametrization


def get_optimizer(learn_config: LearnConfig) -> Callable[[Parameter], optax.GradientTransformation]:
    forward, _ = hyperparameter_reparametrization(learn_config.hyperparameter_parametrization)

    match learn_config.optimizer:
        case SGDConfig(learning_rate):
            return lambda pr: optax.sgd(forward(pr.learning_parameter.learning_rate))
        case SGDNormalizedConfig(learning_rate):
            return lambda pr: optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.sgd(forward(pr.learning_parameter.learning_rate)),
            )
        case SGDClipConfig(learning_rate, threshold, sharpness):

            def update_fn(updates, state, _):
                grads_flat, _ = jax.flatten_util.ravel_pytree(updates)
                grad_norm = jnp.linalg.norm(grads_flat)
                clipped_norm = grad_norm - jax.nn.softplus(sharpness * (grad_norm - threshold)) / sharpness
                scale = clipped_norm / (grad_norm + 1e-6)
                updates_scaled = jax.tree.map(lambda g: g * scale, updates)
                return updates_scaled, state

            return lambda pr: optax.chain(
                optax.GradientTransformation(lambda _: (), update_fn),
                optax.sgd(forward(pr.learning_parameter.learning_rate)),
            )
        case AdamConfig(learning_rate):
            return lambda pr: optax.adam(forward(pr.learning_parameter.learning_rate))
