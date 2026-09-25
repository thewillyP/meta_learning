from meta_learn_lib.lib_types import PixelTransform

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Task: ...


@dataclass(frozen=True)
class Vision(Task): ...


@dataclass(frozen=True)
class Mnist(Vision):
    patch_h: int
    patch_w: int
    label_last_only: bool
    add_spurious_pixel_to_train: bool
    pixel_transform: PixelTransform


@dataclass(frozen=True)
class MNISTTaskFamily(Mnist): ...


@dataclass(frozen=True)
class FashionMNISTTaskFamily(Mnist): ...


@dataclass(frozen=True)
class Cifar(Vision):
    patch_h: int
    patch_w: int
    label_last_only: bool


@dataclass(frozen=True)
class CIFAR10TaskFamily(Cifar): ...


@dataclass(frozen=True)
class CIFAR100TaskFamily(Cifar): ...


@dataclass(frozen=True)
class DelayAddTaskFamily(Task):
    t1_lb: int
    t1_ub: int
    t2_lb: int
    t2_ub: int
    tau_task_lb: int
    tau_task_ub: int
    t_train: int
    n_train: int
    t_test: int
    n_test: int


@dataclass(frozen=True)
class GaussianNoiseTaskFamily(Task):
    shape: tuple[int, ...]
    n: int


@dataclass(frozen=True)
class GridTaskFamily(Task):
    dim: int
    min_value: float
    max_value: float
    n_per_axis: int
    tag: int
    mode: Literal["uniform", "quantile"]


@dataclass(frozen=True)
class MNISTSequenceTaskFamily(Task):
    time_series_length: int
    pixel_transform: PixelTransform


@dataclass(frozen=True)
class SOSTaskFamily(Task):
    grid_size: int
    sigma_x: float
    sigma_y: float
    n: int
    patch_h: int
    patch_w: int
    region: tuple[float, float, float, float]
    region_mode: Literal["full", "exclude_region", "only_region", "grid"]
    tag: int


@dataclass(frozen=True)
class NTMCopyTaskFamily(Task):
    min_seq_len: int
    max_seq_len: int
    bits_per_vector: int
    n_train: int
    n_test: int


@dataclass(frozen=True)
class Augmentation: ...


@dataclass(frozen=True)
class RandomCrop(Augmentation):
    padding: int


@dataclass(frozen=True)
class HorizontalFlip(Augmentation): ...


@dataclass(frozen=True)
class Pool:
    task: Task
    split: Literal["train", "test"]


@dataclass(frozen=True)
class Fresh: ...


@dataclass(frozen=True)
class Fixed:
    seed: int


type Seeding = Fresh | Fixed


@dataclass(frozen=True)
class Draw:
    pool: Pool
    take: int
    shuffle: bool
    augment: tuple[Augmentation, ...]
    seeding: Seeding


@dataclass(frozen=True)
class DataConfig:
    root_dir: str
    num_tasks: int
    label_mask_value: float
    unlabeled_mask_value: float
    workers: int
