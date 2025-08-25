from typing import Callable, Literal, Optional
import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field


from lib.lib_types import *


class Logs(eqx.Module):
    gradient: Optional[jax.Array] = eqx.field(default=None)


class SpecialLogs(eqx.Module):
    influence_tensor: Optional[jax.Array] = eqx.field(default=None)
    immediate_influence_tensor: Optional[jax.Array] = eqx.field(default=None)
    largest_jac_eigenvalue: Optional[jax.Array] = eqx.field(default=None)
    jacobian: Optional[jax.Array] = eqx.field(default=None)


class CustomSequential(eqx.Module):
    model: eqx.nn.Sequential

    def __init__(
        self, layer_defs: list[tuple[int, bool, Callable[[jax.Array], jax.Array]]], input_size: int, key: PRNG
    ):
        layers = []
        in_size = input_size
        layer_keys = jax.random.split(key, len(layer_defs))

        for (out_size, use_bias, activation), k in zip(layer_defs, layer_keys):
            layers.append(eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=k))
            layers.append(eqx.nn.Lambda(activation))
            in_size = out_size

        self.model = eqx.nn.Sequential(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(x)


class RNNState(eqx.Module):
    activation: jax.Array
    n_h: int = eqx.field(static=True)
    n_in: int = eqx.field(static=True)
    activation_fn: Literal["tanh", "relu", "sigmoid", "identity", "softmax"] = eqx.field(static=True)


class RNN(eqx.Module):
    w_rec: jax.Array
    b_rec: Optional[jax.Array]


class UOROState(eqx.Module):
    A: jax.Array
    B: jax.Array


class General(eqx.Module):
    current_virtual_minibatch: int
    logs: Optional[Logs] = eqx.field(default=None)
    special_logs: Optional[SpecialLogs] = eqx.field(default=None)


class InferenceState(eqx.Module):
    rnn: Optional[RNNState] = eqx.field(default=None)


class LearningState(eqx.Module):
    influence_tensor: Optional[JACOBIAN] = eqx.field(default=None)
    uoro: Optional[UOROState] = eqx.field(default=None)
    opt_state: Optional[optax.OptState] = eqx.field(default=None)


class InferenceParameter(eqx.Module):
    rnn: Optional[RNN] = eqx.field(default=None)


class LearningParameter(eqx.Module):
    learning_rate: Optional[jax.Array] = eqx.field(default=None)
    rflo_timeconstant: Optional[float] = eqx.field(static=True, default=None)


class Parameter(eqx.Module):
    transition_parameter: Optional[dict[int, InferenceParameter]] = eqx.field(default=None)
    readout_fn: Optional[CustomSequential] = eqx.field(default=None)
    learning_parameter: Optional[LearningParameter] = eqx.field(default=None)


class GodState(eqx.Module):
    learning_states: dict[int, LearningState]
    inference_states: dict[int, dict[int, InferenceState]]
    validation_learning_states: dict[int, LearningState]
    parameters: dict[int, Parameter]
    general: dict[int, General]
    prng: dict[int, batched[PRNG]]
    prng_learning: dict[int, PRNG]
    start_epoch: int


class GodState(PClass):
    learning_states: dict[int, LearningState] = field()
    inference_states: dict[int, dict[int, InferenceState]] = field()
    validation_learning_states: dict[int, LearningState] = field()
    parameters: dict[int, Parameter] = field()
    general: dict[int, General] = field()
    prng: dict[int, batched[PRNG]] = field()
    prng_learning: dict[int, PRNG] = field()
    start_epoch: int = field()


# Register as JAX pytree
def _godstate_tree_flatten(obj: GodState):
    """Flatten GodState for JAX pytree operations"""
    leaves = [
        obj.learning_states,
        obj.inference_states,
        obj.validation_learning_states,
        obj.parameters,
        obj.general,
        obj.prng,
        obj.prng_learning,
        obj.start_epoch,
    ]
    keys = [
        "learning_states",
        "inference_states",
        "validation_learning_states",
        "parameters",
        "general",
        "prng",
        "prng_learning",
        "start_epoch",
    ]
    return leaves, keys


def _godstate_tree_unflatten(keys, leaves):
    """Unflatten to reconstruct GodState"""
    return GodState(**dict(zip(keys, leaves)))


# Register the pytree
jax.tree_util.register_pytree_node(GodState, _godstate_tree_flatten, _godstate_tree_unflatten)
