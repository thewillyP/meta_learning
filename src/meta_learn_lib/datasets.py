from functools import partial
import itertools
import jax
import jax.numpy as jnp
from typing import Callable, Iterator, NamedTuple
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision
from toolz import mapcat
import math
import numpy as np
from torchvision.transforms.transforms import Lambda
from PIL import Image

from meta_learn_lib.config import *
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.util import infinite_keys


type EpochTransform = Callable[[jax.Array, PRNG], jax.Array]


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


def get_seq_len(task: Task, is_test: bool) -> int:
    match task:
        case MNISTTaskFamily(ph, pw, _, _, _, _):
            return (MNIST_HEIGHT // ph) * (MNIST_WIDTH // pw)
        case CIFAR10TaskFamily(ph, pw, _):
            return (CIFAR_HEIGHT // ph) * (CIFAR_WIDTH // pw)
        case CIFAR100TaskFamily(ph, pw, _):
            return (CIFAR_HEIGHT // ph) * (CIFAR_WIDTH // pw)
        case DelayAddTaskFamily(_, _, _, _, _, _, t_train, _, t_test, _):
            return t_test if is_test else t_train
        case GaussianNoiseTaskFamily(_, _):
            return 1


def numpy_collate_fn(batch):
    return jax.tree.map(lambda x: np.asarray(x), batch)


def jax_collate_fn(batch):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *batch)


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


def make_jax_timeseries_reshape(n_consume: int, pad_value: float) -> Callable[[jax.Array], jax.Array]:
    """Returns a JAX function: (seq_len, features...) -> (num_vb, time, features...)"""

    def reshape(arr: jax.Array) -> jax.Array:
        length = arr.shape[0]
        num_vb = math.ceil(length / n_consume)
        pad_length = (-length) % num_vb
        if pad_length > 0:
            pad_width = [(0, pad_length)] + [(0, 0)] * (arr.ndim - 1)
            arr = jnp.pad(arr, pad_width, constant_values=arr.dtype.type(pad_value))
        return arr.reshape(num_vb, -1, *arr.shape[1:])

    return reshape


class PrematerializedTask(NamedTuple):
    xs: jax.Array
    ys: jax.Array
    x_epoch: EpochTransform
    y_epoch: EpochTransform


@partial(jax.jit, static_argnums=(2,))
def jax_random_crop(key: PRNG, img: jax.Array, padding: int) -> jax.Array:
    """img: (C, H, W) -> (C, H, W)"""
    h, w = img.shape[1], img.shape[2]
    padded = jnp.pad(img, ((0, 0), (padding, padding), (padding, padding)))
    k1, k2 = jax.random.split(key)
    top = jax.random.randint(k1, (), 0, 2 * padding + 1)
    left = jax.random.randint(k2, (), 0, 2 * padding + 1)
    return jax.lax.dynamic_slice(padded, (0, top, left), (img.shape[0], h, w))


@jax.jit
def jax_random_hflip(key: PRNG, img: jax.Array) -> jax.Array:
    """img: (C, H, W) -> (C, H, W)"""
    return jax.lax.cond(jax.random.bernoulli(key), lambda: jnp.flip(img, axis=-1), lambda: img)


def identity_epoch_transform(x: jax.Array, key: PRNG) -> jax.Array:
    return x


def make_jax_augment(task: Task, augment: bool) -> EpochTransform:
    match (task, augment):
        case (CIFAR10TaskFamily() | CIFAR100TaskFamily(), True):

            def augment_fn(x: jax.Array, key: PRNG) -> jax.Array:
                k1, k2 = jax.random.split(key)
                x = jax_random_crop(k1, x, padding=4)
                x = jax_random_hflip(k2, x)
                return x

            return augment_fn
        case _:
            return identity_epoch_transform


def make_patch_reshape(
    height: int, width: int, channel: int, patch_h: int, patch_w: int
) -> Callable[[jax.Array], jax.Array]:
    """Returns a JAX function: (C, H, W) -> (seq_len, C, patch_h, patch_w)"""
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f"image ({height}, {width}) not divisible by patch ({patch_h}, {patch_w})")
    seq_len = (height // patch_h) * (width // patch_w)

    def reshape(x: jax.Array) -> jax.Array:
        return (
            x.reshape(channel, height // patch_h, patch_h, width // patch_w, patch_w)
            .transpose(1, 3, 0, 2, 4)
            .reshape(seq_len, channel, patch_h, patch_w)
        )

    return reshape


def image_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int,
    width: int,
    channel: int,
    patch_h: int,
    patch_w: int,
    y_mask: float,
    label_last_only: bool,
    pixel_transform: MNISTTaskFamily.PixelTransform,
) -> tuple[Lambda, Lambda, Callable[[jax.Array], jax.Array]]:
    """Returns (x_pre, y_pre, patch_reshape).

    x_pre: torchvision transform, PIL -> (C, H, W) tensor (prematerialized once)
    y_pre: torchvision transform, int -> (seq_len,) tensor (prematerialized once)
    patch_reshape: JAX function, (C, H, W) -> (seq_len, C, patch_h, patch_w) (applied per-epoch)
    """
    seq_len = (height // patch_h) * (width // patch_w)

    match pixel_transform:
        case "normalize":
            tv_transform = torchvision.transforms.Normalize(mean, std)
        case "binarize":
            tv_transform = torchvision.transforms.Lambda(lambda x: (x > 0.5).float())
        case "raw":
            tv_transform = torchvision.transforms.Lambda(lambda x: x)

    x_pre = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            tv_transform,
        ]
    )

    def make_targets(y):
        y_val = torch.tensor(y)
        fill = y_val if not label_last_only else torch.tensor(y_mask, dtype=y_val.dtype)
        arr = torch.full((seq_len,), fill)
        arr[-1] = y_val
        return arr

    y_pre = torchvision.transforms.Lambda(make_targets)
    patch_reshape_fn = make_patch_reshape(height, width, channel, patch_h, patch_w)

    return x_pre, y_pre, patch_reshape_fn


type DatasetWithReshape = tuple[Dataset, Callable[[jax.Array], jax.Array]]


def dataset_sources(
    task: Task,
    root_dir: str,
    is_test: bool,
    y_mask: float,
    num_tasks: int,
    seed: PRNG,
) -> list[DatasetWithReshape]:

    def split_dataset(ds: Dataset, count: int, key: PRNG) -> list[Dataset]:
        if count == 0:
            return []
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        sizes = [len(ds) // count] * count
        sizes[-1] += len(ds) - sum(sizes)
        return list(random_split(ds, sizes, generator=generator))

    match task:
        case MNISTTaskFamily(
            patch_h,
            patch_w,
            label_last_only,
            add_spurious_pixel_to_train,
            domain,
            pixel_transform,
        ):
            domain_list = sorted(domain)
            seed_assign, seed_split = jax.random.split(seed)
            assignments = jax.random.randint(seed_assign, shape=(num_tasks,), minval=0, maxval=len(domain_list))
            split_keys = jax.random.split(seed_split, len(domain_list))

            def make_domain_dataset(d: MNISTTaskFamily.Domain) -> DatasetWithReshape:
                match d:
                    case "mnist":
                        factory = torchvision.datasets.MNIST
                        mean, std = MNIST_MEAN, MNIST_STD
                    case "fashion_mnist":
                        factory = torchvision.datasets.FashionMNIST
                        mean, std = FASHION_MNIST_MEAN, FASHION_MNIST_STD

                x_pre, y_pre, patch_reshape_fn = image_transforms(
                    mean=mean,
                    std=std,
                    height=MNIST_HEIGHT,
                    width=MNIST_WIDTH,
                    channel=MNIST_CHANNEL,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    y_mask=y_mask,
                    label_last_only=label_last_only,
                    pixel_transform=pixel_transform,
                )

                if add_spurious_pixel_to_train and not is_test:
                    ds = factory(root=f"{root_dir}/data", train=not is_test, download=True)
                    ds = SpuriousMNISTDataset(ds)
                    ds = TransformedDataset(ds, x_pre, y_pre)
                else:
                    ds = factory(
                        root=f"{root_dir}/data",
                        train=not is_test,
                        download=True,
                        transform=x_pre,
                        target_transform=y_pre,
                    )

                return ds, patch_reshape_fn

            counts = [int((assignments == i).sum()) for i in range(len(domain_list))]
            return [
                (split, pr)
                for d, c, k in zip(domain_list, counts, split_keys)
                for ds, pr in [make_domain_dataset(d)]
                for split in split_dataset(ds, c, k)
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

            x_pre, y_pre, patch_reshape_fn = image_transforms(
                mean=mean,
                std=std,
                height=CIFAR_HEIGHT,
                width=CIFAR_WIDTH,
                channel=CIFAR_CHANNEL,
                patch_h=patch_h,
                patch_w=patch_w,
                y_mask=y_mask,
                label_last_only=label_last_only,
                pixel_transform="normalize",
            )
            ds = factory(
                root=f"{root_dir}/data", train=not is_test, download=True, transform=x_pre, target_transform=y_pre
            )
            return [(split, patch_reshape_fn) for split in split_dataset(ds, num_tasks, seed)]

        case DelayAddTaskFamily(t1_lb, t1_ub, t2_lb, t2_ub, tau_task_lb, tau_task_ub, t_train, n_train, t_test, n_test):
            keys = jax.random.split(seed, num_tasks)
            t = t_test if is_test else t_train
            n = n_test if is_test else n_train

            def make_task(key: PRNG) -> DatasetWithReshape:
                k1, k2, k3, k4 = jax.random.split(key, 4)
                t1 = jax.random.randint(k1, shape=(), minval=t1_lb, maxval=t1_ub + 1).item()
                t2 = jax.random.randint(k2, shape=(), minval=t2_lb, maxval=t2_ub + 1).item()
                tau_task = jax.random.randint(k3, shape=(), minval=tau_task_lb, maxval=tau_task_ub + 1).item()
                example_keys = jax.random.split(k4, n)
                X, Y = jax.vmap(lambda k: generate_add_task_dataset(t, t1, t2, tau_task, k))(example_keys)
                return PyTreeDataset((X, Y)), lambda x: x

            return [make_task(k) for k in keys]

        case GaussianNoiseTaskFamily(shape, n):
            keys = jax.random.split(seed, num_tasks)

            def make_noise_task(key: PRNG) -> DatasetWithReshape:
                xs = jax.random.normal(key, (n, 1, *shape))
                return PyTreeDataset((xs, xs)), lambda x: x

            return [make_noise_task(k) for k in keys]


def take_datasets(
    seed: PRNG,
    remaining: list[DatasetWithReshape],
    n: int,
    n_consume: int,
    x_mask: float,
    y_mask: float,
    augment_fn: EpochTransform,
) -> tuple[list[PrematerializedTask], list[DatasetWithReshape]]:
    ts_x_reshape = make_jax_timeseries_reshape(n_consume, x_mask)
    ts_y_reshape = make_jax_timeseries_reshape(n_consume, y_mask)
    keys = jax.random.split(seed, len(remaining))

    def make_dataset(idx: int, key: PRNG) -> tuple[PrematerializedTask, DatasetWithReshape]:
        ds, xr = remaining[idx]
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        take_n = min(n, len(ds))
        if take_n == 0:
            raise ValueError(
                f"Task {idx}: no examples remaining (requested {n}, available {len(ds)}). "
                f"Earlier levels likely consumed all data from this source."
            )
        taken, leftover = random_split(ds, [take_n, len(ds) - take_n], generator=generator)
        xs, ys = jax_collate_fn(numpy_collate_fn([taken[i] for i in range(len(taken))]))

        def x_epoch(x: jax.Array, key: PRNG) -> jax.Array:
            x = augment_fn(x, key)
            x = xr(x)
            return ts_x_reshape(x)

        def y_epoch(y: jax.Array, key: PRNG) -> jax.Array:
            return ts_y_reshape(y)

        return PrematerializedTask(xs, ys, x_epoch, y_epoch), (leftover, xr)

    datasets_out, new_remaining = zip(*map(make_dataset, range(len(remaining)), keys))
    return list(datasets_out), list(new_remaining)


def task_iterator(
    task: PrematerializedTask,
    batch: int,
    x_mask: float,
    y_mask: float,
    key: PRNG,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    shuffle_key, k1, k2 = jax.random.split(key, 3)
    xs = jax.vmap(task.x_epoch)(task.xs, jax.random.split(k1, task.xs.shape[0]))
    ys = jax.vmap(task.y_epoch)(task.ys, jax.random.split(k2, task.ys.shape[0]))

    perm = jax.random.permutation(shuffle_key, xs.shape[0])
    xs, ys = xs[perm], ys[perm]

    for i in range(math.ceil(xs.shape[0] / batch)):
        X_batch = xs[i * batch : (i + 1) * batch]
        Y_batch = ys[i * batch : (i + 1) * batch]

        if X_batch.shape[0] < batch:
            pad_size = batch - X_batch.shape[0]
            X_batch = jnp.concatenate([X_batch, jnp.full((pad_size, *X_batch.shape[1:]), x_mask, dtype=X_batch.dtype)])
            Y_batch = jnp.concatenate([Y_batch, jnp.full((pad_size, *Y_batch.shape[1:]), y_mask, dtype=Y_batch.dtype)])

        for vb in range(X_batch.shape[1]):
            x = X_batch[:, vb].swapaxes(0, 1)
            y = Y_batch[:, vb].swapaxes(0, 1)
            yield x, y


def batch_iterator(iters, batch_size, axis):
    for batch_iters in itertools.batched(iters, batch_size):
        for batch in zip(*batch_iters):
            yield jax.tree.map(lambda *xs: jnp.stack(xs, axis=axis), *batch)


def stack_batches(stream: Iterator, batch_size: int) -> Iterator:
    for group in itertools.batched(stream, batch_size):
        yield jax.tree.map(lambda *xs: jnp.stack(xs), *group)


def validate_dataloader_config(config: GodConfig) -> list[str]:
    errors = []

    if len(config.levels) == 0:
        errors.append("At least one level is required to create a dataloader")
        return errors

    running_divisor = 1
    for i, level in enumerate(config.levels):
        running_divisor *= level.nested.batch
        if config.num_tasks % running_divisor != 0:
            errors.append(
                f"Level {i}: num_tasks ({config.num_tasks}) not divisible by cumulative batch product ({running_divisor})"
            )
            continue
        chunk_size = config.num_tasks // running_divisor
        tasks_per_stream = level.validation.batch
        if chunk_size % tasks_per_stream != 0:
            errors.append(
                f"Level {i}: chunk size ({chunk_size}) must be divisible by validation.batch ({tasks_per_stream})"
            )

    return errors


def create_data_sources(
    config: GodConfig, prng: PRNG
) -> tuple[list[list[PrematerializedTask]], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    k1, k2, prng = jax.random.split(prng, 3)

    def get_sources(c: list[MetaConfig], k: PRNG) -> list[tuple[MetaConfig, PRNG]]:
        keys = jax.random.split(k, len(c))
        return [
            (level, jax.random.key(level.test_seed) if level.dataset.is_test else key) for level, key in zip(c, keys)
        ]

    # 1. Create dataset sources
    unique_task_pairs = {
        (level.dataset_source, level.dataset.is_test): key for level, key in get_sources(config.levels, k1)
    }

    remaining: dict[tuple, list[DatasetWithReshape]] = {
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
    level_datasets: list[list[PrematerializedTask]] = []
    for level, tk in get_sources(config.levels, k2):
        key_pair = (level.dataset_source, level.dataset.is_test)
        taken, remaining[key_pair] = take_datasets(
            seed=tk,
            remaining=remaining[key_pair],
            n=level.dataset.num_examples_total,
            n_consume=level.validation.num_steps,
            x_mask=config.unlabeled_mask_value,
            y_mask=config.label_mask_value,
            augment_fn=make_jax_augment(level.dataset_source, level.dataset.augment),
        )
        level_datasets.append(taken)

    # 3. Extract feature dimension shapes per level
    dummy_key = jax.random.key(0)
    shapes = [
        (
            datasets[0].x_epoch(datasets[0].xs[0], dummy_key).shape[2:],
            datasets[0].y_epoch(datasets[0].ys[0], dummy_key).shape[2:],
        )
        for datasets in level_datasets
    ]
    return level_datasets, shapes


def create_dataloader(
    config: GodConfig,
    data_sources: list[list[PrematerializedTask]],
    prng: PRNG,
    task_distribution_prng: PRNG,
) -> Iterator:
    k1, prng = jax.random.split(prng, 2)

    global_perm = jax.random.permutation(task_distribution_prng, config.num_tasks)

    def make_task_loader(
        task_indices: jax.Array,
        level: MetaConfig,
        datasets: list[PrematerializedTask],
        key: PRNG,
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        tasks_per_stream = level.validation.batch
        task_keys = jax.random.split(key, len(task_indices))
        task_iters = [
            task_iterator(
                datasets[idx],
                level.dataset.num_examples_in_minibatch,
                config.unlabeled_mask_value,
                config.label_mask_value,
                tkey,
            )
            for idx, tkey in zip(task_indices.tolist(), task_keys)
        ]
        return batch_iterator(task_iters, tasks_per_stream, axis=1)

    def make_nil_loader() -> Iterator:
        while True:
            yield None

    def make_level_loader(
        levels: list[tuple[MetaConfig, list[PrematerializedTask]]],
        task_indices: jax.Array,
        key: PRNG,
    ) -> Iterator:
        if len(levels) == 0:
            return make_nil_loader()

        (meta_config, datasets), *rest = levels

        batch = meta_config.nested.batch
        num_steps = meta_config.nested.num_steps
        is_test = meta_config.dataset.is_test
        child_key, val_key = jax.random.split(key)
        val_key = jax.random.key(meta_config.test_seed) if is_test else val_key

        chunks = jnp.split(task_indices, batch)
        child_keys = jax.random.split(child_key, batch)
        val_keys_per_child = [infinite_keys(vk) for vk in jax.random.split(val_key, batch)]

        def f_val(c: jax.Array, k: PRNG) -> Iterator[tuple[jax.Array, jax.Array]]:
            return make_task_loader(c, meta_config, datasets, k)

        def nest_validation(
            val_stream: Iterator[tuple[jax.Array, jax.Array]],
            lower_levels: list[tuple[MetaConfig, list[PrematerializedTask]]],
        ) -> Iterator:
            for lower_meta, _ in reversed(lower_levels):
                val_stream = stack_batches(val_stream, lower_meta.nested.batch)
            return val_stream

        children = [
            zip(
                make_level_loader(rest, chunk, ckey),
                map(lambda v: (v, v), nest_validation(mapcat(lambda k, c=chunk: f_val(c, k), vks), rest)),
            )
            for chunk, ckey, vks in zip(chunks, child_keys, val_keys_per_child)
        ]
        train_loader = batch_iterator(children, len(children), axis=0)

        return stack_batches(train_loader, num_steps)

    levels_with_data = list(reversed(list(zip(config.levels, data_sources))))
    return make_level_loader(levels_with_data, global_perm, k1)
