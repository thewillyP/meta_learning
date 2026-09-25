from meta_learn_lib.constants import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    CIFAR_CHANNEL,
    CIFAR_HEIGHT,
    CIFAR_WIDTH,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    MNIST_CHANNEL,
    MNIST_HEIGHT,
    MNIST_MEAN,
    MNIST_STD,
    MNIST_WIDTH,
)
from meta_learn_lib.data_source.source import (
    Augmentation,
    CIFAR100TaskFamily,
    CIFAR10TaskFamily,
    Cifar,
    DelayAddTaskFamily,
    FashionMNISTTaskFamily,
    GaussianNoiseTaskFamily,
    GridTaskFamily,
    HorizontalFlip,
    Vision,
    MNISTSequenceTaskFamily,
    MNISTTaskFamily,
    Mnist,
    NTMCopyTaskFamily,
    RandomCrop,
    SOSTaskFamily,
    Task,
)
from meta_learn_lib.lib_types import PRNG, PixelTransform

import math
from typing import Callable, Literal, overload
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np
from PIL import Image
from plum import dispatch
import torch
from torch.utils.data import Dataset, Subset, random_split
import torchvision
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms.v2 import Compose, Lambda, Normalize, ToDtype, ToImage, Transform

type Sequencer = Callable[[np.ndarray], np.ndarray]
type Supply = tuple[Dataset, Sequencer]


class SpuriousMNISTDataset(Dataset):
    def __init__(self, dataset: Dataset, k: int):
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


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Transform, target_transform: Transform):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[idx]
        return self.transform(x), self.target_transform(y)


class PyTreeDataset(Dataset):
    def __init__(self, pytree_data: PyTree):
        self.data = jax.tree.map(np.asarray, pytree_data)
        leaves = jax.tree.leaves(pytree_data)
        if not leaves:
            raise ValueError("PyTree has no leaves!")
        self.n_samples = len(leaves[0])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> PyTree:
        return jax.tree.map(lambda x: x[idx], self.data)


def numpy_collate_fn(batch: list) -> PyTree:
    return jax.tree.map(lambda x: np.asarray(x), batch)


def jax_collate_fn(batch: list) -> PyTree:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *batch)


def generate_add_task_dataset(N: int, t_1: int, t_2: int, tau_task: int, rng_key: PRNG) -> tuple[jax.Array, jax.Array]:
    N = N // tau_task

    x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float32)

    y = 0.5 + 0.5 * jnp.roll(x, t_1) - 0.25 * jnp.roll(x, t_2)

    X = jnp.asarray([x, 1 - x]).T
    Y = jnp.asarray([y, 1 - y]).T

    X = jnp.tile(X, tau_task).reshape(tau_task * N, 2)
    Y = jnp.tile(Y, tau_task).reshape(tau_task * N, 2)

    return X, Y


@overload
def augmenter(a: RandomCrop) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        _, height, width = img.shape
        padded = np.pad(img, ((0, 0), (a.padding, a.padding), (a.padding, a.padding)))
        top, left = rng.integers(0, 2 * a.padding + 1, size=2)
        return padded[:, top : top + height, left : left + width]

    return crop


@overload
def augmenter(a: HorizontalFlip) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    return lambda img, rng: img[..., ::-1] if rng.random() < 0.5 else img


@overload
def augmenter(a: Augmentation) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    raise NotImplementedError


@dispatch
def augmenter(a: Augmentation) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    raise NotImplementedError


def make_patch_reshape(height: int, width: int, channel: int, patch_h: int, patch_w: int) -> Sequencer:
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f"image ({height}, {width}) not divisible by patch ({patch_h}, {patch_w})")
    seq_len = (height // patch_h) * (width // patch_w)

    def reshape(x: np.ndarray) -> np.ndarray:
        return (
            x.reshape(channel, height // patch_h, patch_h, width // patch_w, patch_w)
            .transpose(1, 3, 0, 2, 4)
            .reshape(seq_len, channel, patch_h, patch_w)
        )

    return reshape


def make_image_preprocessor(
    mean: tuple[float, ...], std: tuple[float, ...], pixel_transform: PixelTransform
) -> Transform:
    match pixel_transform:
        case "normalize":
            return Normalize(mean, std)
        case "binarize":
            return Lambda(lambda x: (x > 0.5).float())
        case "raw":
            return Lambda(lambda x: x)


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
) -> tuple[Transform, Transform, Sequencer]:
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

    y_pre = Lambda(make_targets)
    patch_reshape_fn = make_patch_reshape(height, width, channel, patch_h, patch_w)

    return x_pre, y_pre, patch_reshape_fn


def split_dataset(ds: Dataset, count: int, key: PRNG) -> list[Dataset]:
    if count == 0:
        return []
    generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
    sizes = [len(ds) // count] * count
    sizes[-1] += len(ds) - sum(sizes)
    return list(random_split(ds, sizes, generator=generator))


@overload
def factory(t: MNISTTaskFamily) -> type[MNIST]:
    return torchvision.datasets.MNIST


@overload
def factory(t: FashionMNISTTaskFamily) -> type[MNIST]:
    return torchvision.datasets.FashionMNIST


@overload
def factory(t: Mnist) -> type[MNIST]:
    raise NotImplementedError


@overload
def factory(t: CIFAR10TaskFamily) -> type[CIFAR10]:
    return torchvision.datasets.CIFAR10


@overload
def factory(t: CIFAR100TaskFamily) -> type[CIFAR10]:
    return torchvision.datasets.CIFAR100


@overload
def factory(t: Cifar) -> type[CIFAR10]:
    raise NotImplementedError


@overload
def factory(t: Vision) -> type[MNIST] | type[CIFAR10]:
    raise NotImplementedError


@dispatch
def factory(t: Vision) -> type[MNIST] | type[CIFAR10]:
    raise NotImplementedError


@overload
def normalization(t: MNISTTaskFamily) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return MNIST_MEAN, MNIST_STD


@overload
def normalization(t: FashionMNISTTaskFamily) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return FASHION_MNIST_MEAN, FASHION_MNIST_STD


@overload
def normalization(t: CIFAR10TaskFamily) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return CIFAR10_MEAN, CIFAR10_STD


@overload
def normalization(t: CIFAR100TaskFamily) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return CIFAR100_MEAN, CIFAR100_STD


@overload
def normalization(t: Vision) -> tuple[tuple[float, ...], tuple[float, ...]]:
    raise NotImplementedError


@dispatch
def normalization(t: Vision) -> tuple[tuple[float, ...], tuple[float, ...]]:
    raise NotImplementedError


@overload
def dataset_sources(
    t: Mnist, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    mean, std = normalization(t)
    x_pre, y_pre, patch_reshape_fn = image_transforms(
        mean=mean,
        std=std,
        height=MNIST_HEIGHT,
        width=MNIST_WIDTH,
        channel=MNIST_CHANNEL,
        patch_h=t.patch_h,
        patch_w=t.patch_w,
        y_mask=y_mask,
        label_last_only=t.label_last_only,
        pixel_transform=t.pixel_transform,
    )
    pil_x_pre = Compose([ToImage(), ToDtype(torch.float32, scale=True), x_pre])

    if t.add_spurious_pixel_to_train and not is_test:
        ds = factory(t)(root=f"{root_dir}/data", train=not is_test, download=True)
        ds = SpuriousMNISTDataset(ds, 1)
        ds = TransformedDataset(ds, pil_x_pre, y_pre)
    else:
        ds = factory(t)(
            root=f"{root_dir}/data",
            train=not is_test,
            download=True,
            transform=pil_x_pre,
            target_transform=y_pre,
        )

    return [(split, patch_reshape_fn) for split in split_dataset(ds, num_tasks, seed)]


@overload
def dataset_sources(
    t: Cifar, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    mean, std = normalization(t)
    x_pre, y_pre, patch_reshape_fn = image_transforms(
        mean=mean,
        std=std,
        height=CIFAR_HEIGHT,
        width=CIFAR_WIDTH,
        channel=CIFAR_CHANNEL,
        patch_h=t.patch_h,
        patch_w=t.patch_w,
        y_mask=y_mask,
        label_last_only=t.label_last_only,
        pixel_transform="normalize",
    )
    pil_x_pre = Compose([ToImage(), ToDtype(torch.float32, scale=True), x_pre])
    ds = factory(t)(
        root=f"{root_dir}/data", train=not is_test, download=True, transform=pil_x_pre, target_transform=y_pre
    )
    return [(split, patch_reshape_fn) for split in split_dataset(ds, num_tasks, seed)]


@overload
def dataset_sources(
    t: DelayAddTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    keys = jax.random.split(seed, num_tasks)
    length = t.t_test if is_test else t.t_train
    n = t.n_test if is_test else t.n_train

    def make_task(key: PRNG) -> Supply:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        t1 = jax.random.randint(k1, shape=(), minval=t.t1_lb, maxval=t.t1_ub + 1).item()
        t2 = jax.random.randint(k2, shape=(), minval=t.t2_lb, maxval=t.t2_ub + 1).item()
        tau_task = jax.random.randint(k3, shape=(), minval=t.tau_task_lb, maxval=t.tau_task_ub + 1).item()
        example_keys = jax.random.split(k4, n)
        X, Y = jax.vmap(lambda k: generate_add_task_dataset(length, t1, t2, tau_task, k))(example_keys)
        return PyTreeDataset((X, Y)), lambda x: x

    return [make_task(PRNG(k)) for k in keys]


@overload
def dataset_sources(
    t: GaussianNoiseTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    keys = jax.random.split(seed, num_tasks)

    def make_noise_task(key: PRNG) -> Supply:
        xs = jax.random.normal(key, (t.n, 1, *t.shape))
        return PyTreeDataset((xs, xs)), lambda x: x

    return [make_noise_task(PRNG(k)) for k in keys]


@overload
def dataset_sources(
    t: GridTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    keys = jax.random.split(seed, num_tasks)

    def make_grid_task(key: PRNG) -> Supply:
        n = t.n_per_axis**t.dim
        probs = jnp.linspace(t.min_value, t.max_value, t.n_per_axis)
        match t.mode:
            case "uniform":
                axis_values = probs
            case "quantile":
                axis_values = jax.scipy.stats.norm.ppf(probs)
        axes = [axis_values] * t.dim
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(n, t.dim)
        xs = grid[:, None, :]
        return PyTreeDataset((xs, xs)), lambda x: x

    return [make_grid_task(PRNG(k)) for k in keys]


@overload
def dataset_sources(
    t: MNISTSequenceTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    x_pre = make_image_preprocessor(MNIST_MEAN, MNIST_STD, t.pixel_transform)
    pil_x_pre = Compose([ToImage(), ToDtype(torch.float32, scale=True), x_pre])
    base = torchvision.datasets.MNIST(root=f"{root_dir}/data", train=not is_test, download=True, transform=pil_x_pre)
    splits = split_dataset(base, num_tasks, seed)
    keys = jax.random.split(seed, num_tasks)

    def make_seq_task(split: Dataset, key: PRNG) -> Supply:
        images, labels = jax_collate_fn(numpy_collate_fn([split[i] for i in range(len(split))]))
        n_seq = len(split) // t.time_series_length
        perm = jax.random.permutation(key, len(split))[: n_seq * t.time_series_length]
        image_seqs = images[perm].reshape(n_seq, t.time_series_length, MNIST_CHANNEL, MNIST_HEIGHT, MNIST_WIDTH)
        label_seqs = labels[perm].reshape(n_seq, t.time_series_length)
        xs = image_seqs[:, None, ...]
        ys = label_seqs[:, None, :]
        return PyTreeDataset((xs, ys)), lambda x: x

    return [make_seq_task(s, PRNG(k)) for s, k in zip(splits, keys)]


@overload
def dataset_sources(
    t: SOSTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    keys = jax.random.split(seed, num_tasks)
    x_min, x_max, y_min, y_max = t.region

    x_pre, y_pre, patch_reshape_fn = image_transforms(
        mean=(0.0,),
        std=(1.0,),
        height=t.grid_size,
        width=t.grid_size,
        channel=1,
        patch_h=t.patch_h,
        patch_w=t.patch_w,
        y_mask=y_mask,
        label_last_only=False,
        pixel_transform="raw",
    )

    def in_region(cx: jax.Array, cy: jax.Array) -> jax.Array:
        return (x_min <= cx) & (cx <= x_max) & (y_min <= cy) & (cy <= y_max)

    def make_sos_task(key: PRNG) -> Supply:
        def sample_n_acceptable(
            k: PRNG,
            want: int,
            rejection_mode: Literal["full", "exclude_region", "only_region"],
        ) -> jax.Array:
            pool: list[np.ndarray] = []
            accepted = 0
            while accepted < want:
                k, k_batch = jax.random.split(k)
                candidates = jax.random.uniform(k_batch, (want * 2, 2), minval=0.0, maxval=float(t.grid_size))
                match rejection_mode:
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

        match t.region_mode:
            case "grid":
                n_per_axis = int(round(math.sqrt(t.n)))
                assert n_per_axis * n_per_axis == t.n, f"SOS grid mode requires n to be a perfect square, got n={t.n}"
                cx_vals = jnp.linspace(x_min, x_max, n_per_axis)
                cy_vals = jnp.linspace(y_min, y_max, n_per_axis)
                cx_grid, cy_grid = jnp.meshgrid(cx_vals, cy_vals, indexing="ij")
                centers = jnp.stack([cx_grid.reshape(-1), cy_grid.reshape(-1)], axis=-1)
            case non_grid:
                centers = sample_n_acceptable(key, t.n, non_grid)
        cxs, cys = centers[:, 0], centers[:, 1]

        xv, yv = jnp.meshgrid(jnp.arange(t.grid_size), jnp.arange(t.grid_size), indexing="xy")

        def render(cx: jax.Array, cy: jax.Array) -> jax.Array:
            return 0.5 * jnp.exp(-((xv - cx) ** 2) / (4 * t.sigma_x**2)) + 0.5 * jnp.exp(
                -((yv - cy) ** 2) / (4 * t.sigma_y**2)
            )

        images_np = np.array(jax.vmap(render)(cxs, cys)[:, None, :, :])
        labels_np = np.array(jnp.stack([cxs, cys], axis=-1))
        raw_ds = PyTreeDataset((torch.from_numpy(images_np), torch.from_numpy(labels_np)))
        return TransformedDataset(raw_ds, x_pre, y_pre), patch_reshape_fn

    return [make_sos_task(PRNG(k)) for k in keys]


@overload
def dataset_sources(
    t: NTMCopyTaskFamily, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    keys = jax.random.split(seed, num_tasks)
    V = t.bits_per_vector
    T_max = t.max_seq_len
    total = 2 * T_max + 1
    n = t.n_test if is_test else t.n_train

    def make_copy_task(key: PRNG) -> Supply:
        example_keys = jax.random.split(key, n)

        def gen_one(k: PRNG) -> tuple[jax.Array, jax.Array]:
            k_len, k_vec = jax.random.split(k)
            T = jax.random.randint(k_len, (), t.min_seq_len, t.max_seq_len + 1)
            vec_pool = jax.random.bernoulli(k_vec, 0.5, (T_max, V)).astype(jnp.float32)
            t_idx = jnp.arange(total)
            in_input_phase = t_idx < T
            is_eos = t_idx == T
            in_output_phase = (t_idx > T) & (t_idx < 2 * T + 1)
            input_idx = jnp.clip(t_idx, 0, T_max - 1)
            output_idx = jnp.clip(t_idx - T - 1, 0, T_max - 1)
            X_input_bits = jnp.where(in_input_phase[:, None], vec_pool[input_idx], 0.0)
            X_eos_bits = jnp.where(is_eos[:, None], jnp.ones((1, V)), 0.0)
            X_extra_channel = is_eos.astype(jnp.float32)[:, None]
            X = jnp.concatenate([X_input_bits + X_eos_bits, X_extra_channel], axis=1)
            Y = jnp.where(in_output_phase[:, None], vec_pool[output_idx], y_mask)
            return X, Y

        Xs, Ys = jax.vmap(gen_one)(PRNG(example_keys))
        return PyTreeDataset((Xs, Ys)), lambda x: x

    return [make_copy_task(PRNG(k)) for k in keys]


@overload
def dataset_sources(
    t: Task, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    raise NotImplementedError


@dispatch
def dataset_sources(
    t: Task, root_dir: str, is_test: bool, y_mask: float, num_tasks: int, seed: PRNG
) -> list[tuple[Dataset, Callable[[np.ndarray], np.ndarray]]]:
    raise NotImplementedError


def take_datasets(
    seed: PRNG, remaining: list[Supply], n: int, shuffle: bool
) -> tuple[list[Subset[tuple]], list[Supply]]:
    keys = jax.random.split(seed, len(remaining))

    def make_dataset(idx: int, key: PRNG) -> tuple[Subset[tuple], Supply]:
        ds, sequence = remaining[idx]
        generator = torch.Generator().manual_seed(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1).item())
        take_n = min(n, len(ds))
        if take_n == 0:
            raise ValueError(
                f"Task {idx}: no examples remaining (requested {n}, available {len(ds)}). "
                f"Earlier levels likely consumed all data from this source."
            )
        if shuffle:
            taken, leftover = random_split(ds, [take_n, len(ds) - take_n], generator=generator)
        else:
            taken = Subset(ds, list(range(take_n)))
            leftover = Subset(ds, list(range(take_n, len(ds))))
        return taken, (leftover, sequence)

    taken, new_remaining = zip(*map(make_dataset, range(len(remaining)), keys))
    return list(taken), list(new_remaining)
