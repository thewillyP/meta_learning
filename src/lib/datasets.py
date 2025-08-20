import jax
import jax.numpy as jnp
from typing import Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset

from lib.lib_types import PRNG


class IteratorWrapper[T](IterableDataset[T]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator

    def __iter__(self):
        for item in self.iterator:
            yield item


def create_multi_epoch_dataloader[T](
    iter: Iterator[T], num_minibatches_in_epoch: int
) -> DataLoader[tuple[jax.Array, jax.Array]]:
    """Returns a PyTorch DataLoader that sequentially loads virtual minibatches for specific examples"""
    dataset = IteratorWrapper[T](iter)
    return DataLoader(dataset, batch_size=num_minibatches_in_epoch, collate_fn=jax_collate_fn(jnp.stack))


class VirtualMinibatchDataset(Dataset[tuple[jax.Array, jax.Array]]):
    def __init__(self, X_data: jax.Array, Y_data: jax.Array, example_indices):
        self.X_selected = X_data[example_indices]  # shape: (batch_size, num_virtual_minibatches, ...)
        self.Y_selected = Y_data[example_indices]  # shape: (batch_size, num_virtual_minibatches, ...)
        self.num_virtual_batches = self.X_selected.shape[1]

    def __len__(self):
        return self.num_virtual_batches

    def __getitem__(self, idx):
        return self.X_selected[:, idx], self.Y_selected[:, idx]


def create_example_indices_generator(num_examples: int, batch_size: int, rng_key: PRNG):
    """Generator that yields random batches of example indices (exhaustible)"""
    indices = jnp.arange(num_examples)
    rng_key, subkey = jax.random.split(rng_key)
    shuffled_indices = jax.random.permutation(subkey, indices)

    for start_idx in range(0, num_examples, batch_size):
        end_idx = min(start_idx + batch_size, num_examples)
        yield shuffled_indices[start_idx:end_idx]


def jax_collate_fn(f):
    def _collate(batch):
        return jax.tree.map(lambda *xs: f(xs), *batch)

    return _collate


def create_sequential_virtual_dataloader(
    X_data: jax.Array, Y_data: jax.Array, example_indices, num_virtual_minibatches: int
) -> DataLoader[tuple[jax.Array, jax.Array]]:
    """Returns a PyTorch DataLoader that sequentially loads virtual minibatches for specific examples"""
    dataset = VirtualMinibatchDataset(X_data, Y_data, example_indices)
    return DataLoader(
        dataset,
        batch_size=num_virtual_minibatches,
        shuffle=False,
        collate_fn=jax_collate_fn(lambda x: jnp.stack(x).swapaxes(0, 1)),
    )


def standard_dataloader(
    X_data: jax.Array,
    Y_data: jax.Array,
    num_examples: int,
    batch_size: int,
    num_virtual_minibatches: int,
    rng_key: PRNG,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    for indices in create_example_indices_generator(num_examples, batch_size, rng_key):
        for batch in create_sequential_virtual_dataloader(X_data, Y_data, indices, num_virtual_minibatches):
            yield batch


def generate_add_task_dataset(N: int, t_1: int, t_2: int, tau_task: int, rng_key: PRNG):
    """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
    N = N // tau_task

    x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float32)

    y = 0.5 + 0.5 * jnp.roll(x, t_1) - 0.25 * jnp.roll(x, t_2)

    X = jnp.asarray([x, 1 - x]).T
    Y = jnp.asarray([y, 1 - y]).T

    # Temporally stretch according to the desire timescale of change.
    X = jnp.tile(X, tau_task).reshape((1, tau_task * N, 2))  # outer=1 bc only 1 example
    Y = jnp.tile(Y, tau_task).reshape((1, tau_task * N, 2))

    return X, Y


# Transforms
def flatten_and_cast(pic, x_pixels):
    """Convert image to flat (y, x_pixels) JAX array."""
    arr = jnp.array(pic, dtype=jnp.float32) / 255.0  # normalize
    flat = arr.ravel()  # flatten in scanline order (row-major)
    total_pixels = flat.shape[0]

    if total_pixels % x_pixels != 0:
        raise ValueError(f"Cannot reshape array of size {total_pixels} into shape (-1, {x_pixels}).")

    return flat.reshape(-1, x_pixels)


def target_transform(label, sequence_length):
    """Convert scalar label to (784, 2) JAX array: [class_label, sequence_number]."""
    labels = jnp.zeros((sequence_length, 2), dtype=jnp.int32)
    labels = labels.at[:, 0].set(label)  # Repeat class label
    labels = labels.at[:, 1].set(jnp.arange(sequence_length))  # Sequence numbers
    return labels
