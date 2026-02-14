from itertools import islice
import jax
import jax.numpy as jnp
from typing import Callable, Iterable, Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import torch
import torchvision
from toolz import mapcat
import math
import numpy as np
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms.transforms import Lambda

from meta_learn_lib.config import *
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import PRNG, FractionalList
from meta_learn_lib.util import infinite_keys, reshape_timeseries, subset_n


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: Lambda, target_transform: Lambda):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[idx]
        return self.transform(x), self.target_transform(y)


def make_timeseries_transforms(n_consume: int, ignore_value: float) -> Lambda:
    def reshape_to_timeseries(arr: np.ndarray, pad_value: float) -> np.ndarray:
        length: int = arr.shape[0]
        num_vb = math.ceil(length / n_consume)
        pad_length = (-length) % num_vb
        new_time_dim = (length + pad_length) // num_vb
        if pad_length > 0:
            pad_width = [(0, pad_length)] + [(0, 0)] * (arr.ndim - 1)
            arr = np.pad(arr, pad_width, constant_values=pad_value)
        return arr.reshape(num_vb, new_time_dim, *arr.shape[1:])

    transform = torchvision.transforms.Lambda(lambda x: reshape_to_timeseries(x, ignore_value))
    return transform


def take_datasets(
    seed: PRNG,
    remaining: list[Dataset],
    n: int,
    batch_indices: list[int],
    n_consume: int,
    x_mask: float,
    y_mask: float,
) -> tuple[list[Dataset], list[Dataset]]:
    x_transform = make_timeseries_transforms(n_consume, x_mask)
    y_transform = make_timeseries_transforms(n_consume, y_mask)
    keys = jax.random.split(seed, len(batch_indices))

    def make_dataset(idx: int, key: PRNG) -> tuple[Dataset, Dataset]:
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        take_n = min(n, len(remaining[idx]))
        taken, leftover = random_split(remaining[idx], [take_n, len(remaining[idx]) - take_n], generator=generator)
        return TransformedDataset(taken, x_transform, y_transform), leftover

    datasets_out, new_remaining = zip(*map(make_dataset, batch_indices, keys))
    full_remaining = remaining.copy()
    for idx, leftover in zip(batch_indices, new_remaining):
        full_remaining[idx] = leftover
    return list(datasets_out), full_remaining


def image_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int,
    width: int,
    channel: int,
    patch_h: int,
    patch_w: int,
    augment: Lambda,
    y_mask: float,
    label_last_only: bool,
) -> tuple[Lambda, Lambda]:
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f"image ({height}, {width}) not divisible by patch ({patch_h}, {patch_w})")
    seq_len = (height // patch_h) * (width // patch_w)
    x_transform = torchvision.transforms.Compose(
        [
            augment,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
            torchvision.transforms.Lambda(
                lambda x: (
                    x.numpy()
                    .reshape(x.shape[0], height // patch_h, patch_h, width // patch_w, patch_w)
                    .transpose(1, 3, 0, 2, 4)
                    .reshape(seq_len, channel, patch_h, patch_w)
                )
            ),
        ]
    )

    def make_targets(y):
        y_val = np.array(y)
        fill = y_val if not label_last_only else np.array(y_mask, dtype=y_val.dtype)
        arr = np.full((seq_len,), fill)
        arr[-1] = y_val
        return arr

    y_transform = torchvision.transforms.Lambda(make_targets)
    return x_transform, y_transform


def dataset_sources(task: Task, root_dir: str, is_test: bool, y_mask: float) -> list[Dataset]:
    match task:
        case MNISTTaskFamily(patch_h, patch_w, label_last_only, add_spurious_pixel_to_train, domain):

            def domain_to_dataset(domain: MNISTTaskFamily.Domain) -> Dataset:
                match domain:
                    case "mnist":
                        factory = torchvision.datasets.MNIST
                        mean, std = MNIST_MEAN, MNIST_STD
                    case "fashion_mnist":
                        factory = torchvision.datasets.FashionMNIST
                        mean, std = FASHION_MNIST_MEAN, FASHION_MNIST_STD
                x_transform, y_transform = image_transforms(
                    mean=mean,
                    std=std,
                    height=MNIST_HEIGHT,
                    width=MNIST_WIDTH,
                    channel=MNIST_CHANNEL,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    augment=torchvision.transforms.Lambda(lambda x: x),
                    y_mask=y_mask,
                    label_last_only=label_last_only,
                )
                return factory(
                    root=f"{root_dir}/data",
                    train=not is_test,
                    download=True,
                    transform=x_transform,
                    target_transform=y_transform,
                )

            return [domain_to_dataset(d) for d in domain]

        case CIFAR10TaskFamily(patch_h, patch_w, label_last_only) | CIFAR100TaskFamily(
            patch_h, patch_w, label_last_only
        ):
            match task:
                case CIFAR10TaskFamily():
                    factory = torchvision.datasets.CIFAR10
                    mean, std = CIFAR10_MEAN, CIFAR10_STD
                case CIFAR100TaskFamily():
                    factory = torchvision.datasets.CIFAR100
                    mean, std = CIFAR100_MEAN, CIFAR100_STD
            augment = (
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomCrop(32, padding=4),
                        torchvision.transforms.RandomHorizontalFlip(),
                    ]
                )
                if not is_test
                else torchvision.transforms.Lambda(lambda x: x)
            )
            x_transform, y_transform = image_transforms(
                mean=mean,
                std=std,
                height=CIFAR_HEIGHT,
                width=CIFAR_WIDTH,
                channel=CIFAR_CHANNEL,
                patch_h=patch_h,
                patch_w=patch_w,
                augment=augment,
                y_mask=y_mask,
                label_last_only=label_last_only,
            )
            return [
                factory(
                    root=f"{root_dir}/data",
                    train=not is_test,
                    download=True,
                    transform=x_transform,
                    target_transform=y_transform,
                ),
            ]
        case DelayAddTaskFamily(t1_lb, t1_ub, t2_lb, t2_ub, tau_task_lb, tau_task_ub, t_train, n_train, t_test, n_test):
            ...


def create_dataloader(config: GodConfig, prng: PRNG, test_prng: PRNG):
    dataloader_prng, dataset_gen_prng = jax.random.split(prng, 2)

    def make_dataloader_fn(X, Y, batch_size):
        def get_dataloader(rng: PRNG):
            return standard_dataloader(X, Y, X.shape[0], batch_size, rng)

        return get_dataloader

    # trying to get it into consistent shape: [num virtual minibatches, batch, time series, features...] for all datasets
    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            data_size = dict(zip(config.data.keys(), subset_n(n, percentages)))
            dataset_te = generate_add_task_dataset(nTest, t1, t2, tau_task, test_prng)
            # Convert test dataset to same format as training datasets
            X_te, _ = reshape_timeseries(dataset_te[0], dataset_te[0].shape[1])
            Y_te, _ = reshape_timeseries(dataset_te[1], dataset_te[1].shape[1])

            n_in_shape = dataset_te[0].shape[2:]

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            last_unpadded_lengths: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, (i, data_config) in enumerate(sorted(config.data.items())):
                data_prng, dataset_gen_prng = jax.random.split(dataset_gen_prng, 2)
                X_vl, Y_vl = generate_add_task_dataset(data_size[i], t1, t2, tau_task, data_prng)
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]
                last_unpadded_lengths[idx] = last_unpadded_length

                datasets[idx] = make_dataloader_fn(X_vl, Y_vl, 1)

                if idx == 0:
                    total_tr_vb = virtual_minibatches[0] * math.ceil(X_vl.shape[0] / 1)
                # 1. check when to reset after consume concrete example
                # 2. check when is last padded minibatch

            # Add test dataset info
            virtual_minibatches[len(datasets)] = 1
            last_unpadded_lengths[len(datasets)] = 0
            datasets[len(datasets)] = make_dataloader_fn(X_te, Y_te, 1)

        case MnistConfig(n_in, add_spurious_pixel_to_train) | FashionMnistConfig(n_in, add_spurious_pixel_to_train):
            n_in_shape = (n_in,)

            match config.dataset:
                case MnistConfig():
                    dataset_factory = torchvision.datasets.MNIST
                case FashionMnistConfig():
                    dataset_factory = torchvision.datasets.FashionMNIST

            transform = torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32))

            dataset = dataset_factory(
                root=f"{config.data_root_dir}/data",
                train=True,
                download=True,
                transform=transform,
            )
            dataset_te = dataset_factory(
                root=f"{config.data_root_dir}/data",
                train=False,
                download=True,
                transform=transform,
            )

            generator1 = torch.Generator().manual_seed(
                jax.random.randint(dataset_gen_prng, shape=(), minval=0, maxval=2**31 - 1).item()
            )
            torch_datasets = random_split(dataset, subset_n(len(dataset), percentages), generator=generator1)
            if add_spurious_pixel_to_train:
                torch_datasets[0] = SpuriousMNISTDataset(torch_datasets[0])

            xs_te = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(dataset_te.data.numpy(), n_in, True)
            ys_te = jax.vmap(target_transform, in_axes=(0, None))(dataset_te.targets.numpy(), xs_te.shape[1])

            # Convert test dataset to same format as training datasets
            X_te, _ = reshape_timeseries(xs_te, xs_te.shape[1])
            Y_te, _ = reshape_timeseries(ys_te, ys_te.shape[1])

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            last_unpadded_lengths: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, ((i, data_config), torch_ds) in enumerate(zip(sorted(config.data.items()), torch_datasets)):
                subset_data = torch.stack([torch_ds[i][0] for i in range(len(torch_ds))])
                subset_targets = torch.tensor([torch_ds[i][1] for i in range(len(torch_ds))])

                X_vl = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(subset_data.numpy(), n_in, True)
                Y_vl = jax.vmap(target_transform, in_axes=(0, None))(subset_targets.numpy(), X_vl.shape[1])

                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]
                last_unpadded_lengths[idx] = last_unpadded_length

                datasets[idx] = make_dataloader_fn(X_vl, Y_vl, data_config.num_examples_in_minibatch)

                if idx == 0:
                    total_tr_vb = virtual_minibatches[0] * math.ceil(
                        X_vl.shape[0] / data_config.num_examples_in_minibatch
                    )

            # Add test dataset info
            virtual_minibatches[len(datasets)] = 1
            last_unpadded_lengths[len(datasets)] = 0
            datasets[len(datasets)] = make_dataloader_fn(X_te, Y_te, config.data[0].num_examples_in_minibatch)

        case CIFAR10Config(n_in):
            n_in_shape = (n_in,)

            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_test = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            dataset = torchvision.datasets.CIFAR10(
                root=f"{config.data_root_dir}/data", train=True, download=True, transform=transform_train
            )
            dataset_te = torchvision.datasets.CIFAR10(
                root=f"{config.data_root_dir}/data", train=False, download=True, transform=transform_test
            )

            # Collect transformed training data into numpy array
            xs_list = []
            ys_list = []
            for img_tensor, label in dataset:
                # img_tensor is (3, 32, 32) after transforms
                xs_list.append(img_tensor.numpy())
                ys_list.append(label)

            xs_te_list = []
            ys_te_list = []
            for img_tensor, label in dataset_te:
                xs_te_list.append(img_tensor.numpy())
                ys_te_list.append(label)

            transformed_data = np.array(xs_list)  # (50000, 3, 32, 32)
            transformed_targets = np.array(ys_list)  # (50000,)
            transformed_data_te = np.array(xs_te_list)
            transformed_targets_te = np.array(ys_te_list)

            xs = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(transformed_data, n_in, False)
            ys = jax.vmap(target_transform, in_axes=(0, None))(transformed_targets, xs.shape[1])
            xs_te = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(transformed_data_te, n_in, False)
            ys_te = jax.vmap(target_transform, in_axes=(0, None))(transformed_targets_te, xs_te.shape[1])

            # Convert test dataset to same format as training datasets
            X_te, _ = reshape_timeseries(xs_te, xs_te.shape[1])
            Y_te, _ = reshape_timeseries(ys_te, ys_te.shape[1])

            perm = jax.random.permutation(dataset_gen_prng, len(xs))
            split_indices = jnp.cumsum(jnp.array(subset_n(len(xs), percentages)))[:-1]
            val_indices = jnp.split(perm, split_indices)

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            last_unpadded_lengths: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, ((i, data_config), val_idx) in enumerate(zip(sorted(config.data.items()), val_indices)):
                X_vl = xs[val_idx]
                Y_vl = ys[val_idx]
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]
                last_unpadded_lengths[idx] = last_unpadded_length

                datasets[idx] = make_dataloader_fn(X_vl, Y_vl, data_config.num_examples_in_minibatch)

                if idx == 0:
                    total_tr_vb = virtual_minibatches[0] * math.ceil(
                        X_vl.shape[0] / data_config.num_examples_in_minibatch
                    )

            # Add test dataset info
            virtual_minibatches[len(datasets)] = 1
            last_unpadded_lengths[len(datasets)] = 0
            datasets[len(datasets)] = make_dataloader_fn(X_te, Y_te, config.data[0].num_examples_in_minibatch)

        case CIFAR100Config(n_in):
            n_in_shape = (n_in,)

            transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_test = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            dataset = torchvision.datasets.CIFAR100(
                root=f"{config.data_root_dir}/data", train=True, download=True, transform=transform_train
            )
            dataset_te = torchvision.datasets.CIFAR100(
                root=f"{config.data_root_dir}/data", train=False, download=True, transform=transform_test
            )

            # Collect transformed training data into numpy array
            xs_list = []
            ys_list = []
            for img_tensor, label in dataset:
                # img_tensor is (3, 32, 32) after transforms
                xs_list.append(img_tensor.numpy())
                ys_list.append(label)

            xs_te_list = []
            ys_te_list = []
            for img_tensor, label in dataset_te:
                xs_te_list.append(img_tensor.numpy())
                ys_te_list.append(label)

            transformed_data = np.array(xs_list)  # (50000, 3, 32, 32)
            transformed_targets = np.array(ys_list)  # (50000,)
            transformed_data_te = np.array(xs_te_list)
            transformed_targets_te = np.array(ys_te_list)

            xs = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(transformed_data, n_in, False)
            ys = jax.vmap(target_transform, in_axes=(0, None))(transformed_targets, xs.shape[1])
            xs_te = jax.vmap(flatten_and_cast, in_axes=(0, None, None))(transformed_data_te, n_in, False)
            ys_te = jax.vmap(target_transform, in_axes=(0, None))(transformed_targets_te, xs_te.shape[1])

            # Convert test dataset to same format as training datasets
            X_te, _ = reshape_timeseries(xs_te, xs_te.shape[1])
            Y_te, _ = reshape_timeseries(ys_te, ys_te.shape[1])

            perm = jax.random.permutation(dataset_gen_prng, len(xs))
            split_indices = jnp.cumsum(jnp.array(subset_n(len(xs), percentages)))[:-1]
            val_indices = jnp.split(perm, split_indices)

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            last_unpadded_lengths: dict[int, int] = {}
            total_tr_vb: int = 0
            for idx, ((i, data_config), val_idx) in enumerate(zip(sorted(config.data.items()), val_indices)):
                X_vl = xs[val_idx]
                Y_vl = ys[val_idx]
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]
                last_unpadded_lengths[idx] = last_unpadded_length

                datasets[idx] = make_dataloader_fn(X_vl, Y_vl, data_config.num_examples_in_minibatch)

                if idx == 0:
                    total_tr_vb = virtual_minibatches[0] * math.ceil(
                        X_vl.shape[0] / data_config.num_examples_in_minibatch
                    )

            # Add test dataset info
            virtual_minibatches[len(datasets)] = 1
            last_unpadded_lengths[len(datasets)] = 0
            datasets[len(datasets)] = make_dataloader_fn(X_te, Y_te, config.data[0].num_examples_in_minibatch)

    subkeys = jax.random.split(dataloader_prng, len(datasets))
    loaders = [mapcat(datasets[i], infinite_keys(subkeys[i])) for i in range(len(datasets))]

    # create hierarchical dataloader for meta learning
    virtual_minibatches_per_turn = [l.num_virtual_minibatches_per_turn for l in config.learners.values()]
    train_loader = loaders[0]
    for num_vb, vl_loader in zip(virtual_minibatches_per_turn, loaders[1:]):
        train_loader = create_multi_epoch_dataloader(zip(train_loader, vl_loader), num_vb)

    test_loader = datasets[len(datasets) - 1](subkeys[-1])
    return train_loader, test_loader, virtual_minibatches, n_in_shape, last_unpadded_lengths, total_tr_vb


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


class VirtualMinibatchDataset(Dataset[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]):
    def __init__(self, X_data: jax.Array, Y_data: jax.Array, example_indices, batch_mask: jax.Array):
        self.X_selected = X_data[example_indices]  # (batch_size, num_virtual_minibatches, ...)
        self.Y_selected = Y_data[example_indices]  # (batch_size, num_virtual_minibatches, ...)
        self.batch_mask = batch_mask  # (batch_size,)
        self.num_virtual_batches = self.X_selected.shape[1]

        # Create sequence numbers: replicated across batch, incremental across sequence
        batch_size = self.X_selected.shape[0]
        self.n_consume = jnp.broadcast_to(jnp.arange(self.X_selected.shape[2]), (batch_size, self.X_selected.shape[2]))

    def __len__(self):
        return self.num_virtual_batches

    def __getitem__(self, idx):
        X = self.X_selected[:, idx].swapaxes(0, 1)
        Y = self.Y_selected[:, idx].swapaxes(0, 1)
        sequence_nums = self.n_consume.swapaxes(0, 1)
        mask = self.batch_mask
        return X, Y, sequence_nums, mask


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
def flatten_and_cast(pic, x_pixels, normalize: bool):
    """Convert image to flat (y, x_pixels) JAX array."""
    arr = jnp.array(pic, dtype=jnp.float32)
    arr = arr / 255.0 if normalize else arr
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


def add_spurious_features(image, label, k):
    correlated_image = image.clone()
    pixel_location = (label * 10) % 28
    correlated_image[pixel_location : pixel_location + k, pixel_location : pixel_location + k] = 255.0
    return correlated_image


class SpuriousMNISTDataset(Dataset):
    def __init__(self, dataset, k=1):
        self.dataset = dataset
        self.k = k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        correlated_image = add_spurious_features(image, label, self.k)
        return correlated_image, label
