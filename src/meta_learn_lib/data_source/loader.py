from meta_learn_lib.construct.data import Axis, Leaf, Pair, Plan, Window, size, windowed
from meta_learn_lib.data_source.source import (
    Chunked,
    DataConfig,
    Draw,
    EveryPass,
    EveryTick,
    Examples,
    Fixed,
    Fresh,
    Independent,
    Level,
    Padded,
    Pool,
    Seeding,
    Shared,
    Sharing,
    Sources,
    Straddle,
    Tasks,
)
from meta_learn_lib.data_source.tasks import (
    PrematerializedTask,
    Supply,
    augmentation,
    dataset_sources,
    take_datasets,
)
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.utility.util import fold_in

from collections.abc import Sequence
import math
from typing import overload
import grain
from grain.experimental import ZipMapDataset
import jax
from jaxtyping import PyTree
import numpy as np
from plum import dispatch

type Taken = list[PrematerializedTask] | tuple[Taken, Taken]


class Extent(grain.MapDataset):
    def __init__(self, parent: grain.MapDataset, length: int):
        super().__init__(parent)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        return self._parent[index]


def seeded(seed: int, seeding: Seeding) -> int:
    match seeding:
        case Fresh():
            return seed
        case Fixed(fixed):
            return fixed


def materialize(
    plan: Plan, sources: Sources, seed: int, pools: dict[Pool, list[Supply]], config: DataConfig
) -> tuple[Taken, dict[Pool, list[Supply]]]:
    def take(
        plan: Plan, sources: Sources, seed: int, pools: dict[Pool, list[Supply]]
    ) -> tuple[Taken, dict[Pool, list[Supply]]]:
        match plan, sources:
            case Leaf(), Draw() as draw:
                k_pool, k_take = jax.random.split(jax.random.key(seeded(seed, draw.seeding)))
                if draw.pool in pools:
                    supply = pools[draw.pool]
                else:
                    supply = dataset_sources(
                        draw.pool.task,
                        config.root_dir,
                        draw.pool.split == "test",
                        config.label_mask_value,
                        config.num_tasks,
                        PRNG(k_pool),
                    )
                datasets, leftover = take_datasets(PRNG(k_take), supply, draw.take, draw.shuffle)
                return datasets, {**pools, draw.pool: leftover}
            case Pair(_, left, right), Level(train, val, _):
                first, after_train = take(left, train, fold_in(seed, 0), pools)
                second, after_val = take(right, val, fold_in(seed, 1), after_train)
                return (first, second), after_val
            case _:
                raise ValueError(f"sources {sources} do not mirror plan {plan}")

    return take(plan, sources, seed, pools)


def features(data: Taken) -> PyTree:
    match data:
        case tuple((first, second)):
            return (features(first), features(second))
        case list():
            example, *_ = data
            return (example.sequence(example.xs[0]).shape[1:], example.ys[0].shape[1:])


def ticks(
    axes: tuple[Axis, ...],
    draw: Draw,
    datasets: list[PrematerializedTask],
    indices: np.ndarray,
    seed: int,
    config: DataConfig,
) -> grain.MapDataset:
    batch = [i for i, a in enumerate(axes) if not windowed(a)]
    if len(batch) != len(draw.over):
        raise ValueError(
            f"the term has batch axes {tuple(axes[i] for i in batch)} but the source declares roles {draw.over}"
        )
    task_axes = [i for i, role in zip(batch, draw.over) if role == Tasks()]
    example_axes = [i for i, role in zip(batch, draw.over) if role == Examples()]
    match [i for i, a in enumerate(axes) if windowed(a)]:
        case [*outer, innermost]:
            inner, window = [innermost], size(axes[innermost])
        case _:
            outer, inner, window = [], [], 1
    per_tick = math.prod(size(axes[i]) for i in task_axes)
    per_task = math.prod(size(axes[i]) for i in example_axes)
    grouped = math.prod(size(axes[i]) for i in outer)
    if len(indices) % per_tick != 0:
        raise ValueError(f"{len(indices)} tasks cannot be dealt {per_tick} per tick for {draw}")
    groups = len(indices) // per_tick
    dealing, drawing = fold_in(seed, 0), fold_in(seed, 1)
    augment = augmentation(draw.augment)

    def pad_to(a: np.ndarray, n: int, mask: float) -> np.ndarray:
        return np.pad(a, [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1), constant_values=mask)

    def windows_of(a: np.ndarray, mask: float) -> np.ndarray:
        count = math.ceil(len(a) / window)
        return pad_to(a, count * window, mask).reshape(count, window, *a.shape[1:])

    def minibatches(i: int) -> grain.MapDataset:
        task = datasets[i]
        ordering, augmenting = fold_in(drawing, i, 0), fold_in(drawing, i, 1)

        def augmented(index: int, xy: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            x, y = xy
            return augment(x, fold_in(augmenting, index)), y

        def framed(xy: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            x, y = xy
            return windows_of(task.sequence(x), config.unlabeled_mask_value), windows_of(y, config.label_mask_value)

        def padded(items: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
            xs, ys = zip(*items)
            return (
                pad_to(np.stack(xs), per_task, config.unlabeled_mask_value),
                pad_to(np.stack(ys), per_task, config.label_mask_value),
            )

        examples = grain.MapDataset.range(len(task.xs)).map(lambda j: (task.xs[j], task.ys[j]))
        if draw.shuffle:
            examples = examples.shuffle(ordering)
        examples = examples.map_with_index(augmented).map(framed)
        match draw.boundary:
            case Straddle():
                return examples.repeat().batch(per_task, drop_remainder=True)
            case Padded():
                return examples.batch(per_task, batch_fn=padded).repeat()

    streams = {int(i): minibatches(int(i)) for i in indices}
    example = datasets[int(indices[0])]
    num_vb = math.ceil(len(example.sequence(example.xs[0])) / window)
    per_pass = math.ceil(len(example.xs) / per_task) * num_vb
    if (groups * per_pass) % grouped != 0:
        raise ValueError(f"{groups * per_pass} ticks per pass are not divisible by the {grouped} per yield of {draw}")

    def dealt(cycle: int, g: int) -> list[int]:
        tasks = np.sort(indices)
        order = np.random.default_rng(fold_in(dealing, cycle)).permutation(tasks) if draw.shuffle else tasks
        return order.reshape(groups, per_tick)[g].tolist()

    def tick(j: int) -> tuple[np.ndarray, np.ndarray]:
        match draw.regroup:
            case EveryPass():
                p, r = divmod(j, groups * per_pass)
                g, t = divmod(r, per_pass)
                group, position = dealt(p, g), p * per_pass + t
            case EveryTick():
                group, position = dealt(*divmod(j, groups)), j
        m, v = divmod(position, num_vb)
        xs, ys = zip(*[streams[i][m] for i in group])
        return np.stack(xs)[:, :, v], np.stack(ys)[:, :, v]

    order = outer + task_axes + example_axes + inner
    sizes = tuple(size(axes[i]) for i in order)

    def shaped(block: np.ndarray) -> np.ndarray:
        return np.moveaxis(block.reshape(sizes + block.shape[4:]), list(range(len(order))), order)

    return (
        grain.MapDataset.range(groups * per_pass)
        .map_with_index(lambda j, _: tick(j))
        .batch(grouped)
        .map(lambda block: jax.tree.map(shaped, block))
    )


@overload
def share(r: Chunked, n: int, indices: np.ndarray, seed: int) -> list[tuple[np.ndarray, int]]:
    if len(indices) % n != 0:
        raise ValueError(f"{len(indices)} tasks cannot be chunked across {n} learners")
    return [(chunk, fold_in(seed, i)) for i, chunk in enumerate(indices.reshape(n, len(indices) // n))]


@overload
def share(r: Shared, n: int, indices: np.ndarray, seed: int) -> list[tuple[np.ndarray, int]]:
    return [(indices, seed)] * n


@overload
def share(r: Independent, n: int, indices: np.ndarray, seed: int) -> list[tuple[np.ndarray, int]]:
    return [(indices, fold_in(seed, i)) for i in range(n)]


@overload
def share(r: Sharing, n: int, indices: np.ndarray, seed: int) -> list[tuple[np.ndarray, int]]:
    raise NotImplementedError


@dispatch
def share(r: Sharing, n: int, indices: np.ndarray, seed: int) -> list[tuple[np.ndarray, int]]:
    raise NotImplementedError


def stream(
    plan: Plan, sources: Sources, data: Taken, seed: int, passes: int | None, config: DataConfig
) -> grain.MapDataset:
    def go(plan: Plan, sources: Sources, data: Taken, indices: np.ndarray, seed: int) -> grain.MapDataset:
        match plan, sources, data:
            case Leaf(axes), Draw() as draw, list() as datasets:
                return ticks(axes, draw, datasets, indices, seeded(seed, draw.seeding), config)
            case Pair(axes, left, right), Level(train, val, sharing), tuple((first, second)):

                def learner(indices_: np.ndarray, seed_: int) -> grain.MapDataset:
                    trainee = go(left, train, first, indices_, fold_in(seed_, 0))
                    validation = go(right, val, second, indices_, fold_in(seed_, 1))
                    return ZipMapDataset([trainee, Extent(validation, len(trainee))])

                def lift(above: tuple[Axis, ...], indices_: np.ndarray, seed_: int) -> grain.MapDataset:
                    match above:
                        case (Window(k), *rest):
                            inner = lift(tuple(rest), indices_, seed_)
                            if len(inner) % k != 0:
                                raise ValueError(
                                    f"{len(inner)} yields below are not divisible by the {k} per step of Scan"
                                )
                            return inner.batch(k)
                        case (axis, *rest):
                            copies = share(sharing, size(axis), indices_, seed_)
                            return ZipMapDataset([lift(tuple(rest), i, s) for i, s in copies]).map(
                                lambda items: jax.tree.map(lambda *xs: np.stack(xs), *items)
                            )
                        case _:
                            return learner(indices_, seed_)

                return lift(axes, indices, seed)
            case _:
                raise ValueError(f"sources {sources} do not mirror plan {plan}")

    tasks = np.random.default_rng(fold_in(seed, 2)).permutation(config.num_tasks)
    return go(plan, sources, data, tasks, seed).repeat(passes)
