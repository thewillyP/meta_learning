from typing import Callable

import jax
import jax.numpy as jnp
import optax
from pyrsistent.typing import PMap
from meta_learn_lib.config import (
    AdamConfig,
    ExponentiatedGradientConfig,
    LearnConfig,
    Optimizer,
    RecurrenceConfig,
    SGDConfig,
    SGDNormalizedConfig,
)
import equinox as eqx
from meta_learn_lib.env import InferenceParameter, LearningParameter, Parameter, TransitionParameter
from meta_learn_lib.util import filter_cond, hyperparameter_reparametrization, to_vector


def get_optimizer(
    optimizer: Optimizer, format_to_param: Callable[[jax.Array], Parameter]
) -> Callable[[LearningParameter], optax.GradientTransformation]:
    def add_clipping(threshold: float, sharpness: float) -> optax.GradientTransformation:
        def update_fn(updates, state, _):
            grads_flat, _ = jax.flatten_util.ravel_pytree(updates)
            grad_norm = jnp.linalg.norm(grads_flat)
            clipped_norm = grad_norm - jax.nn.softplus(sharpness * (grad_norm - threshold)) / sharpness
            scale = clipped_norm / (grad_norm + 1e-6)
            updates_scaled = jax.tree.map(lambda g: g * scale, updates)
            return updates_scaled, state

        clipping = optax.GradientTransformation(lambda _: (), update_fn)
        return clipping

    match optimizer:
        case SGDConfig(learning_rate, weight_decay, momentum, add_clip):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.chain(
                add_clipping(add_clip.threshold, add_clip.sharpness) if add_clip is not None else optax.identity(),
                optax.add_decayed_weights(wd_forward(pr.weight_decay.value)),
                optax.sgd(lr_forward(pr.learning_rate.value), momentum=momentum),
            )
        case SGDNormalizedConfig(learning_rate, weight_decay, momentum):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.chain(
                optax.normalize_by_update_norm(scale_factor=1.0),
                optax.add_decayed_weights(wd_forward(pr.weight_decay.value)),
                optax.sgd(lr_forward(pr.learning_rate.value), momentum=momentum),
            )
        case AdamConfig(learning_rate, weight_decay, add_clip):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)
            return lambda pr: optax.chain(
                add_clipping(add_clip.threshold, add_clip.sharpness) if add_clip is not None else optax.identity(),
                optax.adamw(
                    lr_forward(pr.learning_rate.value),
                    weight_decay=wd_forward(pr.weight_decay.value),
                ),
            )

        case ExponentiatedGradientConfig(learning_rate, weight_decay, momentum, add_clip):
            lr_forward, _ = hyperparameter_reparametrization(learning_rate.hyperparameter_parametrization)
            wd_forward, _ = hyperparameter_reparametrization(weight_decay.hyperparameter_parametrization)

            def update_rule(pr: LearningParameter) -> optax.GradientTransformation:
                def init_fn(params):
                    return (jnp.zeros_like(params), jnp.zeros_like(params), 1)

                def update_fn(grad, prev_m__prev_v__count, param):
                    prev_m, prev_v, count = prev_m__prev_v__count
                    # g = filter_cond(count == 0, lambda g: g, lambda g: (1 - momentum) * g, grad)
                    new_m = (
                        momentum * prev_m + (1 - momentum) * grad * jnp.sign(param) + wd_forward(pr.weight_decay.value)
                    )
                    new_m_unbiased = new_m / (1 - momentum**count)
                    new_v = 0.999 * prev_v + 0.001 * ((grad * jnp.sign(param)) ** 2)
                    new_v_unbiased = new_v / (1 - 0.999**count)
                    power = -lr_forward(pr.learning_rate.value) * new_m_unbiased / (jnp.sqrt(new_v_unbiased) + 1e-8)

                    return jnp.exp(power), (new_m, new_v, count + 1)

                return optax.chain(
                    add_clipping(add_clip.threshold, add_clip.sharpness) if add_clip is not None else optax.identity(),
                    optax.GradientTransformation(init_fn, update_fn),
                )

            return update_rule

        case RecurrenceConfig(_recurrent_optimizer, _readout_optimizer):
            recurrent_optimizer = get_optimizer(_recurrent_optimizer, format_to_param)
            readout_optimizer = get_optimizer(_readout_optimizer, format_to_param)

            def update_rule(pr: LearningParameter) -> optax.GradientTransformation:
                def init_fn(params):
                    param_pytree = format_to_param(params)
                    rec_param_tree, out_param_tree = eqx.partition(
                        param_pytree, Parameter(transition_parameter=True, readout_fn=False, learning_parameter=False)
                    )

                    opt_state1 = recurrent_optimizer(pr.multiple_parameters[0]).init(to_vector(rec_param_tree).vector)
                    opt_state2 = readout_optimizer(pr.multiple_parameters[1]).init(to_vector(out_param_tree).vector)
                    return (opt_state1, opt_state2)

                def update_fn(grad, state1__state2, param):
                    state1, state2 = state1__state2
                    grad_pytree = format_to_param(grad)
                    param_pytree = format_to_param(param)
                    rec_tree, out_tree = eqx.partition(
                        grad_pytree, Parameter(transition_parameter=True, readout_fn=False, learning_parameter=False)
                    )
                    rec_param_tree, out_param_tree = eqx.partition(
                        param_pytree, Parameter(transition_parameter=True, readout_fn=False, learning_parameter=False)
                    )
                    rec_vec = to_vector(rec_tree)
                    out_vec = to_vector(out_tree)
                    rec_update, new_state1 = recurrent_optimizer(pr.multiple_parameters[0]).update(
                        rec_vec.vector, state1, to_vector(rec_param_tree).vector
                    )
                    out_update, new_state2 = readout_optimizer(pr.multiple_parameters[1]).update(
                        out_vec.vector, state2, to_vector(out_param_tree).vector
                    )
                    new_rec_tree = rec_vec.to_param(rec_update)
                    new_out_tree = out_vec.to_param(out_update)
                    combined_update = eqx.combine(new_rec_tree, new_out_tree)
                    return to_vector(combined_update).vector, (new_state1, new_state2)

                return optax.GradientTransformation(init_fn, update_fn)

            return update_rule


def get_updater(learn_config: LearnConfig) -> Callable[[optax.Params, optax.Updates], optax.Params]:
    match learn_config.optimizer:
        case ExponentiatedGradientConfig():
            return lambda p, u: p * u
        case _:
            return optax.apply_updates
