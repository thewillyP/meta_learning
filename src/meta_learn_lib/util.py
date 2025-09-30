from typing import Callable
import jax
import jax.flatten_util
import jax.numpy as jnp
import math
import equinox as eqx
import equinox.internal as eqxi
import optax
import jax.lax as lax
from meta_learn_lib.lib_types import LOSS, PRNG, FractionalList


def jvp(f, primal, tangent):
    return jax.jvp(f, (primal,), (tangent,), has_aux=True)


def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return eqx.filter_vmap(wrapper, in_axes=(None, 1), out_axes=(None, 1, None))(primal, matrix)


def create_fractional_list(percentages: list[float]) -> FractionalList | None:
    """Create a FractionalList from a list of percentages.

    Args:
        percentages (list[float]): A list of percentages that sum to 1.0.

    Returns:
        FractionalList | None: A FractionalList if the input is valid, otherwise None.
    """
    if not percentages or abs(sum(percentages) - 1.0) > 1e-6:
        return None
    return FractionalList(percentages)


def subset_n(n: int, percentages: FractionalList) -> list[int]:
    """Given a number n and a list of percentages, return a list of integers
    that represent the subset sizes of n according to the percentages.

    Args:
        n (int): The total number to be divided into subsets.
        percentages (list[float]): A list of percentages that sum to 1.0.

    Returns:
        list[int]: A list of integers representing the subset sizes.
    """
    subset_sizes = [int(n * p) for p in percentages]
    total_assigned = sum(subset_sizes)
    remainder = n - total_assigned

    for i in range(remainder):
        subset_sizes[i % len(subset_sizes)] += 1

    return subset_sizes


def reshape_timeseries(arr: jax.Array, target_time_dim: int) -> tuple[jax.Array, int]:
    num_examples, time_series_length, *rest = arr.shape

    num_virtual_minibatches = math.ceil(time_series_length / target_time_dim)
    pad_length = (-time_series_length) % num_virtual_minibatches
    new_time_dim = (time_series_length + pad_length) // num_virtual_minibatches

    if pad_length > 0:
        pad_width = [(0, 0), (0, pad_length)] + [(0, 0)] * len(rest)
        arr_padded = jnp.pad(arr, pad_width, constant_values=0)
    else:
        arr_padded = arr

    new_shape = (num_examples, num_virtual_minibatches, new_time_dim) + tuple(rest)

    # Length of non-padded portion in last minibatch
    last_minibatch_length = new_time_dim - pad_length

    return arr_padded.reshape(new_shape), last_minibatch_length


def get_activation_fn(s: str) -> Callable[[jax.Array], jax.Array]:
    match s:
        case "tanh":
            return jax.nn.tanh
        case "relu":
            return jax.nn.relu
        case "sigmoid":
            return jax.nn.sigmoid
        case "identity":
            return lambda x: x
        case "softmax":
            return jax.nn.softmax
        case _:
            raise ValueError("Invalid activation function")


def get_loss_fn(s: str) -> Callable[[jax.Array, jax.Array], LOSS]:
    match s:
        case "cross_entropy":
            return lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))
        case "cross_entropy_with_integer_labels":
            return lambda a, b: LOSS(optax.losses.softmax_cross_entropy_with_integer_labels(a, b))
        case "mse":
            return lambda a, b: LOSS(optax.losses.squared_error(a, b))
        case _:
            raise ValueError("Invalid loss function")


def accuracy_hard(preds: jax.Array, labels: jax.Array) -> jax.Array:
    pred_classes = jnp.argmax(preds, axis=-1)
    return pred_classes == labels


def accuracy_soft(preds: jax.Array, labels: jax.Array) -> jax.Array:
    pred_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(labels, axis=-1)
    return pred_classes == true_classes


def accuracy_with_sequence_filter(
    preds: jnp.ndarray,  # [B, N, C]
    labels: jnp.ndarray,  # [B, N, 2]
    n: int,  # sequence number to filter
) -> jnp.ndarray:  # [B,]
    def _accuracy_seq_filter_single(
        preds: jnp.ndarray,  # [N, C]
        labels: jnp.ndarray,  # [N, 2]
        n: int,  # sequence number to filter
    ) -> float:
        class_indices = labels[:, 0].astype(jnp.int32)
        sequence_numbers = labels[:, 1].astype(jnp.int32)
        pred_classes = jnp.argmax(preds, axis=-1)
        correct = pred_classes == class_indices
        # Mask: 1 where sequence == n, else 0
        mask = (sequence_numbers == n).astype(jnp.float32)
        correct_masked: jax.Array = correct.astype(jnp.float32) * mask
        total = jnp.sum(mask)
        correct_total = jnp.sum(correct_masked)
        return jax.lax.cond(total > 0, lambda: correct_total / total, lambda: 0.0)

    # Vectorized over batch dimension: preds [B, N, C], labels [B, N, 2]
    batched_accuracy_with_sequence_filter = eqx.filter_vmap(
        _accuracy_seq_filter_single,
        in_axes=(0, 0, None),  # vmap over batch; n is shared
    )

    return batched_accuracy_with_sequence_filter(preds, labels, n)


class Vector[T](eqx.Module):
    vector: jax.Array
    to_param: Callable[[jax.Array], T] = eqx.field(static=True)


def to_vector[T](tree: T) -> Vector[T]:
    """Convert a pytree to a Vector, which contains a flattened array and non-parameter parts."""
    params, nonparams = eqx.partition(tree, eqx.is_inexact_array)
    vector, to_param = jax.flatten_util.ravel_pytree(params)
    return Vector(vector=vector, to_param=lambda a: eqx.combine(to_param(a), nonparams))


def hyperparameter_reparametrization(
    reparametrization: str,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    match reparametrization:
        case "identity":
            reparam_fn = lambda lr: lr
            reparam_inverse = lambda lr: lr
        case "softplus":
            reparam_fn = jax.nn.softplus
            reparam_inverse = lambda y: jnp.log(jnp.expm1(y))
        case "relu":
            reparam_fn = lambda x: jnp.maximum(x, 0.0)
            reparam_inverse = lambda y: y  # Note: not strictly correct, as relu is not invertible
        case _:
            raise ValueError("Invalid hyperparameter reparametrization")

    return reparam_fn, reparam_inverse


def infinite_keys(key: PRNG):
    """Generate infinite stream of JAX keys."""
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def filter_cond(pred, true_fun, false_fun, *operands):
    """Taken from https://github.com/patrick-kidger/equinox/issues/709"""
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)
