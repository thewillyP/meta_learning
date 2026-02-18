import itertools
import jax
import jax.numpy as jnp
from typing import Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import torch
import torchvision
from toolz import mapcat, concat
import math
import numpy as np
from torchvision.transforms.transforms import Lambda
from PIL import Image

from meta_learn_lib.config import *
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.util import infinite_keys


class SpuriousMNISTDataset(Dataset):
    def __init__(self, dataset: Dataset, k: int = 1):
        self.dataset = dataset
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        image, label = self.dataset[idx]
        image = np.array(image)
        pixel_location = (label * 10) % image.shape[0]
        h_end = min(pixel_location + self.k, image.shape[0])
        w_end = min(pixel_location + self.k, image.shape[1])
        image[pixel_location:h_end, pixel_location:w_end] = 255
        return Image.fromarray(image), label


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


class IteratorDataset(IterableDataset):
    def __init__(self, iterator: Iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator


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


def numpy_collate_fn(batch):
    return jax.tree.map(lambda x: np.asarray(x), batch)


def jax_collate_fn(batch):
    return jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *batch)


def generate_add_task_dataset(N: int, t_1: int, t_2: int, tau_task: int, rng_key: PRNG):
    """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
    N = N // tau_task

    x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float32)

    y = 0.5 + 0.5 * jnp.roll(x, t_1) - 0.25 * jnp.roll(x, t_2)

    X = jnp.asarray([x, 1 - x]).T
    Y = jnp.asarray([y, 1 - y]).T

    # Temporally stretch according to the desire timescale of change.
    X = jnp.tile(X, tau_task).reshape(tau_task * N, 2)
    Y = jnp.tile(Y, tau_task).reshape(tau_task * N, 2)

    return X, Y


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
    n_consume: int,
    x_mask: float,
    y_mask: float,
) -> tuple[list[Dataset], list[Dataset]]:
    x_transform = make_timeseries_transforms(n_consume, x_mask)
    y_transform = make_timeseries_transforms(n_consume, y_mask)
    keys = jax.random.split(seed, len(remaining))

    def make_dataset(idx: int, key: PRNG) -> tuple[Dataset, Dataset]:
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        take_n = min(n, len(remaining[idx]))
        if take_n == 0:
            raise ValueError(
                f"Task {idx}: no examples remaining (requested {n}, available {len(remaining[idx])}). "
                f"Earlier levels likely consumed all data from this source."
            )
        taken, leftover = random_split(remaining[idx], [take_n, len(remaining[idx]) - take_n], generator=generator)
        taken = TransformedDataset(taken, x_transform, y_transform)
        xs, ys = jax_collate_fn(numpy_collate_fn([taken[i] for i in range(len(taken))]))
        return PyTreeDataset((xs, ys)), leftover

    datasets_out, new_remaining = zip(*map(make_dataset, range(len(remaining)), keys))
    return list(datasets_out), list(new_remaining)


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
                    x.reshape(x.shape[0], height // patch_h, patch_h, width // patch_w, patch_w)
                    .permute(1, 3, 0, 2, 4)
                    .reshape(seq_len, channel, patch_h, patch_w)
                )
            ),
        ]
    )

    def make_targets(y):
        y_val = torch.tensor(y)
        fill = y_val if not label_last_only else torch.tensor(y_mask, dtype=y_val.dtype)
        arr = torch.full((seq_len,), fill)
        arr[-1] = y_val
        return arr

    y_transform = torchvision.transforms.Lambda(make_targets)
    return x_transform, y_transform


def dataset_sources(
    task: Task,
    root_dir: str,
    is_test: bool,
    y_mask: float,
    num_tasks: int,
    seed: PRNG,
) -> list[Dataset]:

    def split_dataset(ds: Dataset, count: int, key: PRNG) -> list[Dataset]:
        if count == 0:
            return []
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        sizes = [len(ds) // count] * count
        sizes[-1] += len(ds) - sum(sizes)
        return list(random_split(ds, sizes, generator=generator))

    match task:
        case MNISTTaskFamily(patch_h, patch_w, label_last_only, add_spurious_pixel_to_train, domain, normalize):
            domain_list = sorted(domain)
            seed_assign, seed_split = jax.random.split(seed)
            assignments = jax.random.randint(seed_assign, shape=(num_tasks,), minval=0, maxval=len(domain_list))
            split_keys = jax.random.split(seed_split, len(domain_list))

            def make_domain_datasets(d: MNISTTaskFamily.Domain) -> Dataset:
                match d:
                    case "mnist":
                        factory = torchvision.datasets.MNIST
                        mean, std = (MNIST_MEAN, MNIST_STD) if normalize else ((0.0,), (1.0,))
                    case "fashion_mnist":
                        factory = torchvision.datasets.FashionMNIST
                        mean, std = (FASHION_MNIST_MEAN, FASHION_MNIST_STD) if normalize else ((0.0,), (1.0,))

                ds = factory(
                    root=f"{root_dir}/data",
                    train=not is_test,
                    download=True,
                )

                if add_spurious_pixel_to_train and not is_test:
                    ds = SpuriousMNISTDataset(ds)

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

                return TransformedDataset(ds, x_transform, y_transform)

            counts = [int((assignments == i).sum()) for i in range(len(domain_list))]
            return [
                ds
                for d, c, k in zip(domain_list, counts, split_keys)
                for ds in split_dataset(make_domain_datasets(d), c, k)
            ]

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
            ds = factory(
                root=f"{root_dir}/data",
                train=not is_test,
                download=True,
                transform=x_transform,
                target_transform=y_transform,
            )
            return split_dataset(ds, num_tasks, seed)
        case DelayAddTaskFamily(t1_lb, t1_ub, t2_lb, t2_ub, tau_task_lb, tau_task_ub, t_train, n_train, t_test, n_test):
            keys = jax.random.split(seed, num_tasks)
            t = t_test if is_test else t_train
            n = n_test if is_test else n_train

            def make_task(key: PRNG) -> Dataset:
                k1, k2, k3, k4 = jax.random.split(key, 4)
                t1 = jax.random.randint(k1, shape=(), minval=t1_lb, maxval=t1_ub + 1).item()
                t2 = jax.random.randint(k2, shape=(), minval=t2_lb, maxval=t2_ub + 1).item()
                tau_task = jax.random.randint(k3, shape=(), minval=tau_task_lb, maxval=tau_task_ub + 1).item()
                example_keys = jax.random.split(k4, n)
                X, Y = jax.vmap(lambda k: generate_add_task_dataset(t, t1, t2, tau_task, k))(example_keys)
                return torch.utils.data.TensorDataset(torch.from_numpy(np.array(X)), torch.from_numpy(np.array(Y)))

            return [make_task(k) for k in keys]


def task_iterator(
    dataset: Dataset,
    batch: int,
    x_mask: float,
    y_mask: float,
    key: PRNG,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        generator=generator,
        collate_fn=jax_collate_fn,
        drop_last=False,
    )
    for X_batch, Y_batch in loader:
        if X_batch.shape[0] < batch:
            pad_size = batch - X_batch.shape[0]
            X_pad = jnp.full((pad_size, *X_batch.shape[1:]), x_mask)
            Y_pad = jnp.full((pad_size, *Y_batch.shape[1:]), y_mask)
            X_batch = jnp.concatenate([X_batch, X_pad])
            Y_batch = jnp.concatenate([Y_batch, Y_pad])
        for vb in range(X_batch.shape[1]):
            x = X_batch[:, vb].swapaxes(0, 1)
            y = Y_batch[:, vb].swapaxes(0, 1)
            yield x, y


def batch_iterator(iters, batch_size):
    for batch_iters in itertools.batched(iters, batch_size):
        for batch in zip(*batch_iters):
            yield jax.tree.map(lambda *xs: jnp.stack(xs), *batch)


def validate_dataloader_config(config: GodConfig) -> list[str]:
    errors = []

    if len(config.levels) == 0:
        errors.append("At least one level is required to create a dataloader")
        return errors

    running_divisor = 1
    for i, level in enumerate(config.levels):
        running_divisor *= level.meta_opt.batch
        if config.num_tasks % running_divisor != 0:
            errors.append(
                f"Level {i}: num_tasks ({config.num_tasks}) not divisible by cumulative batch product ({running_divisor})"
            )
            continue
        chunk_size = config.num_tasks // running_divisor
        tasks_per_stream = level.dataset_validation.task_batch_size
        if chunk_size % tasks_per_stream != 0:
            errors.append(
                f"Level {i}: chunk size ({chunk_size}) must be divisible by task_batch_size ({tasks_per_stream})"
            )

    return errors


def create_data_sources(config: GodConfig, prng: PRNG) -> tuple[list[list[Dataset]], list[tuple[int, ...]]]:
    k1, k2, prng = jax.random.split(prng, 3)

    def get_sources(c: list[MetaConfig], k: PRNG) -> list[tuple[MetaConfig, PRNG]]:
        keys = jax.random.split(k, len(c))
        return [
            (level, jax.random.key(level.test_seed) if level.dataset_validation.is_test else key)
            for level, key in zip(c, keys)
        ]

    # 1. Create dataset sources
    unique_task_pairs = {
        (level.dataset_source, level.dataset_validation.is_test): key for level, key in get_sources(config.levels, k1)
    }

    remaining = {
        (task, is_test): dataset_sources(
            task=task,
            root_dir=config.data_root_dir,
            is_test=is_test,
            y_mask=config.label_mask_value,
            num_tasks=config.num_tasks,
            seed=key,
        )
        for (task, is_test), key in unique_task_pairs.items()
    }

    # 2. Take from sources for each level
    level_datasets: list[list[Dataset]] = []
    for level, tk in get_sources(config.levels, k2):
        key_pair = (level.dataset_source, level.dataset_validation.is_test)
        taken, remaining[key_pair] = take_datasets(
            seed=tk,
            remaining=remaining[key_pair],
            n=level.dataset_validation.num_examples_total,
            n_consume=level.dataset_validation.num_steps_in_timeseries,
            x_mask=config.unlabeled_mask_value,
            y_mask=config.label_mask_value,
        )
        level_datasets.append(taken)

    # 3. Extract feature dimension shapes per level
    feature_shapes = [datasets[0][0][0].shape[2:] for datasets in level_datasets]
    return level_datasets, feature_shapes


def create_dataloader(
    config: GodConfig,
    data_sources: list[list[Dataset]],
    prng: PRNG,
    task_distribution_prng: PRNG,
) -> DataLoader:
    k1, prng = jax.random.split(prng, 2)

    # 3. Build nested loaders top-down
    global_perm = jax.random.permutation(task_distribution_prng, config.num_tasks)

    def make_task_loader(
        task_indices: jax.Array,
        level: MetaConfig,
        datasets: list[Dataset],
        key: PRNG,
    ) -> Iterator:
        tasks_per_stream = level.dataset_validation.task_batch_size
        task_keys = jax.random.split(key, len(task_indices))
        task_iters = [
            task_iterator(
                datasets[idx],
                level.dataset_validation.num_examples_in_minibatch,
                config.unlabeled_mask_value,
                config.label_mask_value,
                tkey,
            )
            for idx, tkey in zip(task_indices.tolist(), task_keys)
        ]
        # might want to switch vb and multitask axes.

        streams = [
            map(
                lambda batch: jax.tree.map(lambda x: x.swapaxes(0, 1), batch),
                batch_iterator(list(chunk), tasks_per_stream),
            )
            for chunk in itertools.batched(task_iters, tasks_per_stream)
        ]
        return concat(streams)

    def make_level_loader(
        meta_config: MetaConfig,
        datasets: list[Dataset],
        xss: list[tuple[MetaConfig, list[Dataset]]],
        task_indices: jax.Array,
        key: PRNG,
    ) -> Iterator:
        batch = meta_config.meta_opt.batch
        num_steps = meta_config.meta_opt.num_steps
        is_test = meta_config.dataset_validation.is_test
        child_key, val_key = jax.random.split(key)
        val_key = jax.random.key(meta_config.test_seed) if is_test else val_key

        chunks = jnp.split(task_indices, batch)
        child_keys = jax.random.split(child_key, batch)
        val_keys_per_child = [infinite_keys(vk) for vk in jax.random.split(val_key, batch)]
        partition = zip(chunks, child_keys, val_keys_per_child)

        if len(xss) == 0:
            f = lambda c, ck: make_task_loader(c, meta_config, datasets, ck)
        else:
            x, *xs = xss
            mc, ds = x
            f = lambda c, ck: make_level_loader(mc, ds, xs, c, ck)

        children = [
            zip(f(chunk, ckey), mapcat(lambda k, c=chunk: make_task_loader(c, meta_config, datasets, k), vks))
            for chunk, ckey, vks in partition
        ]
        train_loader = batch_iterator(children, len(children))

        return DataLoader(
            IteratorDataset(train_loader),
            batch_size=num_steps,
            collate_fn=jax_collate_fn,
        )

    xs = list(reversed(list(zip(config.levels, data_sources))))
    return make_level_loader(config.levels[-1], data_sources[-1], xs[1:], global_perm, k1)
