import jax
import jax.numpy as jnp
from typing import Callable, Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision
from toolz import mapcat
import math

from lib.config import *
from lib.lib_types import PRNG, FractionalList
from lib.util import infinite_keys, reshape_timeseries, subset_n


class PyTreeDataset(Dataset):
    def __init__(self, pytree_data):
        self.data = pytree_data
        leaves = jax.tree.leaves(pytree_data)
        if not leaves:
            raise ValueError("PyTree has no leaves!")
        self.n_samples = len(leaves[0])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return jax.tree.map(lambda x: x[idx], self.data)


def create_dataloader(config: GodConfig, percentages: FractionalList, prng: PRNG, test_prng: PRNG):
    dataloader_prng, dataset_gen_prng = jax.random.split(prng, 2)

    # trying to get it into consistent shape: [num virtual minibatches, batch, time series, features...] for all datasets
    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            data_size = dict(zip(config.data.keys(), subset_n(n, percentages)))
            dataset_te = generate_add_task_dataset(nTest, t1, t2, tau_task, test_prng)
            dataloader_te = DataLoader(
                PyTreeDataset(dataset_te), batch_size=config.test_batch_size, collate_fn=jax_collate_fn
            )
            n_in_shape = dataset_te[0].shape[1:]

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, (i, data_config) in enumerate(sorted(config.data.items())):
                data_prng, dataset_gen_prng = jax.random.split(dataset_gen_prng, 2)
                X_vl, Y_vl = generate_add_task_dataset(data_size[i], t1, t2, tau_task, data_prng)
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]

                def get_dataloader(rng: PRNG, X_vl=X_vl, Y_vl=Y_vl, data_config=data_config):
                    return standard_dataloader(
                        X_vl,
                        Y_vl,
                        X_vl.shape[0],
                        1,
                        rng,
                    )

                datasets[idx] = get_dataloader

                if idx == 0:
                    total_tr_vb = virtual_minibatches[0] * math.ceil(X_vl.shape[0] / 1) * 1
                # 1. check when to reset after consume concrete example
                # 2. check when is last padded minibatch

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
            n_in_shape = (n_in,)

            match config.dataset:
                case MnistConfig():
                    dataset_factory = torchvision.datasets.MNIST
                case FashionMnistConfig():
                    dataset_factory = torchvision.datasets.FashionMNIST

            dataset = dataset_factory(root=f"{config.data_root_dir}/data", train=True, download=True)
            dataset_te = dataset_factory(root=f"{config.data_root_dir}/data", train=False, download=True)
            xs = jax.vmap(flatten_and_cast, in_axes=(0, None))(dataset.data.numpy(), n_in)
            ys = jax.vmap(target_transform, in_axes=(0, None))(dataset.targets.numpy(), xs.shape[1])
            xs_te = jax.vmap(flatten_and_cast, in_axes=(0, None))(dataset_te.data.numpy(), n_in)
            ys_te = jax.vmap(target_transform, in_axes=(0, None))(dataset_te.targets.numpy(), xs_te.shape[1])
            dataset_te = (xs_te, ys_te)
            dataloader_te = DataLoader(
                PyTreeDataset(dataset_te), batch_size=config.test_batch_size, collate_fn=jax_collate_fn
            )

            perm = jax.random.permutation(dataset_gen_prng, len(xs))
            split_indices = jnp.cumsum(jnp.array(subset_n(len(xs), percentages)))[:-1]
            val_indices = jnp.split(perm, split_indices)

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, ((i, data_config), val_idx) in enumerate(zip(sorted(config.data.items()), val_indices)):
                X_vl = xs[val_idx]
                Y_vl = ys[val_idx]
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]

                def get_dataloader(rng: PRNG, X_vl=X_vl, Y_vl=Y_vl, data_config=data_config):
                    return standard_dataloader(
                        X_vl,
                        Y_vl,
                        X_vl.shape[0],
                        data_config.num_examples_in_minibatch,
                        rng,
                    )

                datasets[idx] = get_dataloader

                if idx == 0:
                    total_tr_vb = (
                        virtual_minibatches[0]
                        * math.ceil(X_vl.shape[0] / data_config.num_examples_in_minibatch)
                        * data_config.num_examples_in_minibatch
                    )

    subkeys = jax.random.split(dataloader_prng, len(datasets))
    train_loader = datasets[0](subkeys[0])
    vl_loaders = [mapcat(datasets[i], infinite_keys(subkeys[i])) for i in range(1, len(datasets))]

    # create hierarchical dataloader for meta learning
    virtual_minibatches_per_turn = [l.num_virtual_minibatches_per_turn for l in config.learners.values()]
    for num_vb, vl_loader in zip(virtual_minibatches_per_turn[:-1], vl_loaders):
        train_loader = create_multi_epoch_dataloader(zip(train_loader, vl_loader), num_vb)

    dataloader = create_multi_epoch_dataloader(train_loader, virtual_minibatches_per_turn[-1])
    return dataloader, dataloader_te, virtual_minibatches, n_in_shape, last_unpadded_length, total_tr_vb


class IteratorWrapper[T](IterableDataset[T]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator

    def __iter__(self):
        for item in self.iterator:
            yield item


def create_multi_epoch_dataloader[T](iter: Iterator[T], num_minibatches_in_epoch: int) -> DataLoader[T]:
    """Returns a PyTorch DataLoader that sequentially loads virtual minibatches for specific examples"""
    dataset = IteratorWrapper[T](iter)
    return DataLoader(dataset, batch_size=num_minibatches_in_epoch, collate_fn=jax_collate_fn)


class VirtualMinibatchDataset(Dataset[tuple[jax.Array, jax.Array, jax.Array]]):
    def __init__(self, X_data: jax.Array, Y_data: jax.Array, example_indices, batch_mask: jax.Array):
        self.X_selected = X_data[example_indices]  # (batch_size, num_virtual_minibatches, ...)
        self.Y_selected = Y_data[example_indices]  # (batch_size, num_virtual_minibatches, ...)
        self.batch_mask = batch_mask  # (batch_size,)
        self.num_virtual_batches = self.X_selected.shape[1]

    def __len__(self):
        return self.num_virtual_batches

    def __getitem__(self, idx):
        X = self.X_selected[:, idx].swapaxes(0, 1)
        Y = self.Y_selected[:, idx].swapaxes(0, 1)
        mask = self.batch_mask
        return X, Y, mask


def create_example_indices_generator(num_examples: int, batch_size: int, rng_key: PRNG):
    """Generator that yields random batches of example indices with padding and masks"""
    indices = jnp.arange(num_examples)
    rng_key, subkey = jax.random.split(rng_key)
    shuffled_indices = jax.random.permutation(subkey, indices)

    for start_idx in range(0, num_examples, batch_size):
        end_idx = min(start_idx + batch_size, num_examples)
        batch_indices = shuffled_indices[start_idx:end_idx]
        actual_batch_size = len(batch_indices)
        padding_size = batch_size - actual_batch_size

        padded_indices = jnp.concatenate([batch_indices, jnp.repeat(batch_indices[-1], padding_size)])

        batch_mask = jnp.concatenate([jnp.ones(actual_batch_size, dtype=bool), jnp.zeros(padding_size, dtype=bool)])

        yield padded_indices, batch_mask


def jax_collate_fn(batch):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *batch)


def jax_squeeze_batch(batch):
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=0) if x.shape[0] == 1 else x, batch)


def create_sequential_virtual_dataloader(
    X_data: jax.Array, Y_data: jax.Array, example_indices, batch_mask: jax.Array
) -> DataLoader[tuple[jax.Array, jax.Array, jax.Array]]:
    """Returns a PyTorch DataLoader that sequentially loads virtual minibatches for specific examples"""
    dataset = VirtualMinibatchDataset(X_data, Y_data, example_indices, batch_mask)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: jax_squeeze_batch(jax_collate_fn(b)),
    )


def standard_dataloader(
    X_data: jax.Array,
    Y_data: jax.Array,
    num_examples: int,
    batch_size: int,
    rng_key: PRNG,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array]]:
    for indices, batch_mask in create_example_indices_generator(num_examples, batch_size, rng_key):
        for batch in create_sequential_virtual_dataloader(X_data, Y_data, indices, batch_mask):
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
