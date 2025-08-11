from typing import Callable
import jax
import jax.numpy as jnp
import math
from lib.lib_types import FractionalList


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
