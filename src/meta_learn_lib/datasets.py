from functools import partial
import itertools
import jax
import jax.numpy as jnp
from typing import Callable, Iterator, NamedTuple
from torch.utils.data import Dataset, random_split
import torch
import torchvision
from toolz import mapcat
import math
import numpy as np
from torchvision.transforms.transforms import Lambda
from jaxtyping import PyTree
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
        case MNISTTaskFamily(ph, pw, _, _, _) | FashionMNISTTaskFamily(ph, pw, _, _, _):
            return (MNIST_HEIGHT // ph) * (MNIST_WIDTH // pw)
        case CIFAR10TaskFamily(ph, pw, _):
            return (CIFAR_HEIGHT // ph) * (CIFAR_WIDTH // pw)
        case CIFAR100TaskFamily(ph, pw, _):
            return (CIFAR_HEIGHT // ph) * (CIFAR_WIDTH // pw)
        case DelayAddTaskFamily(_, _, _, _, _, _, t_train, _, t_test, _):
            return t_test if is_test else t_train
        case GaussianNoiseTaskFamily(_, _):
            return 1
        case GridTaskFamily(_, _, _, _, _):
            return 1
        case MNISTSequenceTaskFamily(_, _):
            return 1
        case SOSTaskFamily(grid_size, _, _, _, ph, pw, _, _):
            return (grid_size // ph) * (grid_size // pw)


def get_pixel_mean_std(task: Task) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Return per-channel (mean, std) used at dataset load time. None means raw / non-image."""
    match task:
        case MNISTTaskFamily(pixel_transform="normalize"):
            return MNIST_MEAN, MNIST_STD
        case FashionMNISTTaskFamily(pixel_transform="normalize"):
            return FASHION_MNIST_MEAN, FASHION_MNIST_STD
        case MNISTSequenceTaskFamily(pixel_transform="normalize"):
            return MNIST_MEAN, MNIST_STD
        case CIFAR10TaskFamily(_, _, _):
            return CIFAR10_MEAN, CIFAR10_STD
        case CIFAR100TaskFamily(_, _, _):
            return CIFAR100_MEAN, CIFAR100_STD
        case _:
            return None


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


def make_image_preprocessor(mean: tuple[float, ...], std: tuple[float, ...], pixel_transform: PixelTransform) -> Lambda:
    """Returns a torchvision transform applied to an already-tensor (C, H, W) image.
    Callers are responsible for getting the input into tensor form first (e.g. ToTensor for PIL)."""
    match pixel_transform:
        case "normalize":
            return torchvision.transforms.Normalize(mean, std)
        case "binarize":
            return torchvision.transforms.Lambda(lambda x: (x > 0.5).float())
        case "raw":
            return torchvision.transforms.Lambda(lambda x: x)


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
    pixel_transform: PixelTransform,
) -> tuple[Lambda, Lambda, Callable[[jax.Array], jax.Array]]:
    """Returns (x_pre, y_pre, patch_reshape).

    x_pre: torchvision transform, PIL -> (C, H, W) tensor (prematerialized once)
    y_pre: torchvision transform, int -> (seq_len,) tensor (prematerialized once)
    patch_reshape: JAX function, (C, H, W) -> (seq_len, C, patch_h, patch_w) (applied per-epoch)
    """
    seq_len = (height // patch_h) * (width // patch_w)

    x_pre = make_image_preprocessor(mean, std, pixel_transform)

    def make_targets(y):
        y_val = torch.as_tensor(y)
        if label_last_only:
            arr = torch.full((seq_len, *y_val.shape), float(y_mask), dtype=y_val.dtype)
        else:
            arr = y_val.unsqueeze(0).expand(seq_len, *y_val.shape).clone()
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
        case (
            MNISTTaskFamily(patch_h, patch_w, label_last_only, add_spurious_pixel_to_train, pixel_transform)
            | FashionMNISTTaskFamily(patch_h, patch_w, label_last_only, add_spurious_pixel_to_train, pixel_transform)
        ) as t:
            match t:
                case MNISTTaskFamily():
                    factory = torchvision.datasets.MNIST
                    mean, std = MNIST_MEAN, MNIST_STD
                case FashionMNISTTaskFamily():
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
            pil_x_pre = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), x_pre])

            if add_spurious_pixel_to_train and not is_test:
                ds = factory(root=f"{root_dir}/data", train=not is_test, download=True)
                ds = SpuriousMNISTDataset(ds)
                ds = TransformedDataset(ds, pil_x_pre, y_pre)
            else:
                ds = factory(
                    root=f"{root_dir}/data",
                    train=not is_test,
                    download=True,
                    transform=pil_x_pre,
                    target_transform=y_pre,
                )

            return [(split, patch_reshape_fn) for split in split_dataset(ds, num_tasks, seed)]

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
            pil_x_pre = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), x_pre])
            ds = factory(
                root=f"{root_dir}/data", train=not is_test, download=True, transform=pil_x_pre, target_transform=y_pre
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

        case GridTaskFamily(dim, min_value, max_value, n_per_axis, _, mode):
            keys = jax.random.split(seed, num_tasks)

            def make_grid_task(key: PRNG) -> DatasetWithReshape:
                n = n_per_axis**dim
                probs = jnp.linspace(min_value, max_value, n_per_axis)
                match mode:
                    case "uniform":
                        axis_values = probs
                    case "quantile":
                        axis_values = jax.scipy.stats.norm.ppf(probs)
                axes = [axis_values] * dim
                grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(n, dim)
                xs = grid[:, None, :]
                return PyTreeDataset((xs, xs)), lambda x: x

            return [make_grid_task(k) for k in keys]

        case MNISTSequenceTaskFamily(time_series_length, pixel_transform):
            x_pre = make_image_preprocessor(MNIST_MEAN, MNIST_STD, pixel_transform)
            pil_x_pre = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), x_pre])
            base = torchvision.datasets.MNIST(
                root=f"{root_dir}/data", train=not is_test, download=True, transform=pil_x_pre
            )
            splits = split_dataset(base, num_tasks, seed)
            keys = jax.random.split(seed, num_tasks)

            def make_seq_task(split: Dataset, key: PRNG) -> DatasetWithReshape:
                images, labels = jax_collate_fn(numpy_collate_fn([split[i] for i in range(len(split))]))
                n_seq = len(split) // time_series_length
                perm = jax.random.permutation(key, len(split))[: n_seq * time_series_length]
                image_seqs = images[perm].reshape(n_seq, time_series_length, MNIST_CHANNEL, MNIST_HEIGHT, MNIST_WIDTH)
                label_seqs = labels[perm].reshape(n_seq, time_series_length)
                xs = image_seqs[:, None, ...]
                ys = label_seqs[:, None, :]
                return PyTreeDataset((xs, ys)), lambda x: x

            return [make_seq_task(s, k) for s, k in zip(splits, keys)]

        case SOSTaskFamily(grid_size, sigma_x, sigma_y, n, patch_h, patch_w, region, region_mode):
            keys = jax.random.split(seed, num_tasks)
            x_min, x_max, y_min, y_max = region

            x_pre, y_pre, patch_reshape_fn = image_transforms(
                mean=(0.0,),  # unused for "raw"
                std=(1.0,),
                height=grid_size,
                width=grid_size,
                channel=1,
                patch_h=patch_h,
                patch_w=patch_w,
                y_mask=y_mask,
                label_last_only=False,
                pixel_transform="raw",
            )

            def in_region(cx: jax.Array, cy: jax.Array) -> jax.Array:
                return (x_min <= cx) & (cx <= x_max) & (y_min <= cy) & (cy <= y_max)

            def make_sos_task(key: PRNG) -> DatasetWithReshape:
                def sample_n_acceptable(k: PRNG, want: int) -> jax.Array:
                    pool: list[np.ndarray] = []
                    accepted = 0
                    while accepted < want:
                        k, k_batch = jax.random.split(k)
                        candidates = jax.random.uniform(k_batch, (want * 2, 2), minval=0.0, maxval=float(grid_size))
                        match region_mode:
                            case "full":
                                mask = jnp.ones((candidates.shape[0],), dtype=bool)
                            case "exclude_region":
                                mask = ~in_region(candidates[:, 0], candidates[:, 1])
                            case "only_region":
                                mask = in_region(candidates[:, 0], candidates[:, 1])
                        kept = np.asarray(candidates[np.asarray(mask)])
                        pool.append(kept)
                        accepted += kept.shape[0]
                    return jnp.asarray(np.concatenate(pool, axis=0)[:want])

                centers = sample_n_acceptable(key, n)
                cxs, cys = centers[:, 0], centers[:, 1]

                xv, yv = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing="xy")

                def render(cx: jax.Array, cy: jax.Array) -> jax.Array:
                    return 0.5 * jnp.exp(-((xv - cx) ** 2) / (4 * sigma_x**2)) + 0.5 * jnp.exp(
                        -((yv - cy) ** 2) / (4 * sigma_y**2)
                    )

                # (n, 1, H, W) images and (n, 2) labels, both as torch tensors so they flow through
                # the same TransformedDataset(x_pre, y_pre) pipeline as MNIST/CIFAR.
                images_np = np.array(jax.vmap(render)(cxs, cys)[:, None, :, :])
                labels_np = np.array(jnp.stack([cxs, cys], axis=-1))
                raw_ds = PyTreeDataset((torch.from_numpy(images_np), torch.from_numpy(labels_np)))
                return TransformedDataset(raw_ds, x_pre, y_pre), patch_reshape_fn

            return [make_sos_task(k) for k in keys]


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


def regroup_leading(arr: jax.Array, outer: int, inner: int) -> jax.Array:
    """Take arr of shape (outer*inner, A0, A1, *rest). Split the leading axis into
    (outer, inner), move `inner` from axis 1 to axis 3, then flatten (outer, A0)
    into a single leading axis. Returns shape (outer*A0, A1, inner, *rest).

    Used to convert a "minibatch * group" leading axis into a "yields * group"
    leading axis where each yield carries `inner` as a feature dim.
    """
    arr = arr.reshape(outer, inner, *arr.shape[1:])
    perm = (0, 2, 3, 1) + tuple(range(4, arr.ndim))
    arr = arr.transpose(perm)
    return arr.reshape(outer * arr.shape[1], *arr.shape[2:])


def task_epoch_tensor(
    task: PrematerializedTask,
    batch: int,
    x_mask: float,
    y_mask: float,
    key: PRNG,
) -> tuple[jax.Array, jax.Array]:
    """Eager equivalent of task_iterator's per-epoch yield stream.

    Returns (xs, ys) each of shape (Y_task, time, batch, *features), where
    Y_task = num_mb * num_vb. Slicing arr[i] gives the i-th yield equivalent.
    """
    shuffle_key, k1, k2 = jax.random.split(key, 3)
    xs = jax.vmap(task.x_epoch)(task.xs, jax.random.split(k1, task.xs.shape[0]))
    ys = jax.vmap(task.y_epoch)(task.ys, jax.random.split(k2, task.ys.shape[0]))

    perm = jax.random.permutation(shuffle_key, xs.shape[0])
    xs, ys = xs[perm], ys[perm]

    N = xs.shape[0]
    num_mb = math.ceil(N / batch)
    pad_size = num_mb * batch - N
    if pad_size > 0:
        x_pad = jnp.full((pad_size, *xs.shape[1:]), x_mask, dtype=xs.dtype)
        y_pad = jnp.full((pad_size, *ys.shape[1:]), y_mask, dtype=ys.dtype)
        xs = jnp.concatenate([xs, x_pad])
        ys = jnp.concatenate([ys, y_pad])

    return regroup_leading(xs, num_mb, batch), regroup_leading(ys, num_mb, batch)


def task_loader_eager(
    task_indices: jax.Array,
    level: MetaConfig,
    datasets: list[PrematerializedTask],
    key: PRNG,
    x_mask: float,
    y_mask: float,
) -> tuple[jax.Array, jax.Array]:
    """Eager equivalent of make_task_loader = batch_iterator(task_iters, tasks_per_stream, axis=1).

    Returns (xs, ys) each of shape (Y_loader, time, tasks_per_stream, batch_per_task, *features).
    Y_loader = (chunk_size / tasks_per_stream) * Y_task.
    """
    tasks_per_stream = level.validation.batch
    batch_per_task = level.dataset.num_examples_in_minibatch
    chunk_size = len(task_indices)
    num_groups = chunk_size // tasks_per_stream
    task_keys = jax.random.split(key, chunk_size)

    per_task_x, per_task_y = zip(
        *[
            task_epoch_tensor(datasets[idx], batch_per_task, x_mask, y_mask, tkey)
            for idx, tkey in zip(task_indices.tolist(), task_keys)
        ]
    )

    return (
        regroup_leading(jnp.stack(per_task_x, axis=0), num_groups, tasks_per_stream),
        regroup_leading(jnp.stack(per_task_y, axis=0), num_groups, tasks_per_stream),
    )


def rechunk_pytrees[T: PyTree](iterator: Iterator[T], chunk_size: int) -> Iterator[T]:
    """Rechunk a stream of pytrees (whose leaves are tensors with a leading axis)
    into a stream of pytrees where every leaf's leading axis is exactly `chunk_size`.
    Leftover state lives in the frame across yields — every item is visited before
    reshuffling (mirrors original lazy mapcat continuation)."""
    buffer: list[PyTree] = []
    buffered = 0
    for pytree in iterator:
        buffer.append(pytree)
        buffered += jax.tree.leaves(pytree)[0].shape[0]
        while buffered >= chunk_size:
            combined = (
                buffer[0] if len(buffer) == 1 else jax.tree.map(lambda *arrs: jnp.concatenate(arrs, axis=0), *buffer)
            )
            yield jax.tree.map(lambda x: x[:chunk_size], combined)
            buffered -= chunk_size
            buffer = [jax.tree.map(lambda x: x[chunk_size:], combined)] if buffered > 0 else []


def nest_val_eager[T: PyTree](val: T, lower_levels: list[tuple[MetaConfig, list[PrematerializedTask]]]) -> T:
    """Eager equivalent of nest_validation: add one leading-axis split (Y, ...) -> (Y // B, B, ...)
    per lower level. Caller invariant: every leaf's leading axis is divisible by every lower B."""
    for lower_meta, _ in reversed(lower_levels):
        b = lower_meta.nested.batch
        val = jax.tree.map(lambda x: x.reshape(x.shape[0] // b, b, *x.shape[1:]), val)
    return val


def level_eager_gen(
    levels: list[tuple[MetaConfig, list[PrematerializedTask]]],
    task_indices: jax.Array,
    key: PRNG,
    K: int,
    x_mask: float,
    y_mask: float,
) -> Iterator:
    """Eager recursive equivalent of make_level_loader. Returns an iterator yielding
    one (K, *yield_shape) batch per next(). Setup runs once; state (val streams,
    leftover partial chunks, child generators) lives in the underlying iterators.
    Composed via map/zip — no manual yield loop."""
    if len(levels) == 0:
        return itertools.repeat(None)

    (meta, datasets), *rest = levels
    batch = meta.nested.batch
    num_steps = meta.nested.num_steps
    is_test = meta.dataset.is_test

    chunks = jnp.split(task_indices, batch)
    child_key, val_key = jax.random.split(key)
    val_key = jax.random.key(meta.test_seed) if is_test else val_key
    val_keys = jax.random.split(val_key, batch)
    child_keys = jax.random.split(child_key, batch)
    lower_consumption = math.prod(lower_meta.nested.batch for lower_meta, _ in rest)

    K_child = K * num_steps
    raw_val_per_chunk = K_child * lower_consumption

    def make_val_stream(chunk: jax.Array, vk: PRNG) -> Iterator[tuple[jax.Array, jax.Array]]:
        return map(lambda k: task_loader_eager(chunk, meta, datasets, k, x_mask, y_mask), infinite_keys(vk))

    def batch_step(chunk_pairs: tuple) -> PyTree:
        chunks_pytree = [(c, ((val := nest_val_eager(v, rest)), val)) for c, v in chunk_pairs]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *chunks_pytree)
        return jax.tree.map(lambda arr: arr.reshape(K, num_steps, *arr.shape[1:]), stacked)

    chunk_streams = [
        zip(
            level_eager_gen(rest, chunk, ck, K_child, x_mask, y_mask),
            rechunk_pytrees(make_val_stream(chunk, vk), raw_val_per_chunk),
        )
        for chunk, ck, vk in zip(chunks, child_keys, val_keys)
    ]

    return map(batch_step, zip(*chunk_streams))


def create_dataloader(
    config: GodConfig,
    data_sources: list[list[PrematerializedTask]],
    prng: PRNG,
    task_distribution_prng: PRNG,
) -> Iterator:
    k1, prng = jax.random.split(prng, 2)
    global_perm = jax.random.permutation(task_distribution_prng, config.num_tasks)
    levels_with_data = list(reversed(list(zip(config.levels, data_sources))))

    innermost_level = levels_with_data[-1][0]
    seq_len = get_seq_len(innermost_level.dataset_source, innermost_level.dataset.is_test)
    num_vb = math.ceil(seq_len / innermost_level.validation.num_steps)
    num_mb = math.ceil(innermost_level.dataset.num_examples_total / innermost_level.dataset.num_examples_in_minibatch)
    consumption = math.prod(meta.nested.num_steps for meta, _ in levels_with_data)
    full_epoch_yields = (num_mb * num_vb) // consumption
    K = config.dataloader_chunk_size if config.dataloader_chunk_size is not None else full_epoch_yields

    return mapcat(
        lambda batch: (jax.tree.map(lambda leaf: leaf[i], batch) for i in range(K)),
        level_eager_gen(
            levels_with_data,
            global_perm,
            k1,
            K,
            config.unlabeled_mask_value,
            config.label_mask_value,
        ),
    )
