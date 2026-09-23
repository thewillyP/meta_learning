from meta_learn_lib.construct.data import Axis, Batch, Leaf, Pair, Plan, Window
from meta_learn_lib.construct.term import Examples, Same, Tasks
from meta_learn_lib.data_source.source import (
    DataConfig,
    Draw,
    EveryMinibatch,
    EveryPass,
    Fixed,
    Fresh,
    Level,
    Padded,
    Pool,
    Seeding,
    Sources,
    Straddle,
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

from collections.abc import Callable
import math
import grain
from grain.experimental import ZipMapDataset
import jax
from jaxtyping import PyTree
import numpy as np

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


def stacked(copies: list[grain.MapDataset]) -> grain.MapDataset:
    return ZipMapDataset(copies).map(lambda items: jax.tree.map(lambda *xs: np.stack(xs), *items))


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
            case Pair(_, left, right), Level(train, val):
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
    per_tick, per_task, window, windows = 1, 1, 1, 0
    for axis in axes:
        match axis:
            case Batch(n, Tasks()):
                per_tick *= n
            case Batch(n, Examples()):
                per_task *= n
            case Batch(_, Same()):
                pass
            case Window(n):
                window, windows = n, windows + 1
    if len(indices) % per_tick != 0:
        raise ValueError(f"{len(indices)} tasks cannot be dealt {per_tick} per tick for {draw}")
    groups = len(indices) // per_tick
    tasks = np.sort(indices)
    example = datasets[int(tasks[0])]
    n_examples = len(example.xs)
    per_pass = math.ceil(n_examples / per_task) * per_task
    num_vb = math.ceil(len(example.sequence(example.xs[0])) / window)
    unit = (window,) if windows > 0 else ()
    dealing, drawing = fold_in(seed, 0), fold_in(seed, 1)
    augment = augmentation(draw.augment)

    def pad_to(a: np.ndarray, n: int, mask: float) -> np.ndarray:
        return a if len(a) == n else np.concatenate([a, np.full((n - len(a), *a.shape[1:]), mask, a.dtype)])

    def examples(i: int) -> grain.MapDataset:
        task = datasets[i]
        ordering, augmenting = fold_in(drawing, i, 0), fold_in(drawing, i, 1)

        def example(index: int, j: int) -> tuple[np.ndarray, np.ndarray]:
            x = task.sequence(augment(task.xs[j], fold_in(augmenting, index)))
            return (
                pad_to(x, num_vb * window, config.unlabeled_mask_value),
                pad_to(task.ys[j], num_vb * window, config.label_mask_value),
            )

        drawn = grain.MapDataset.range(len(task.xs))
        if draw.shuffle:
            drawn = drawn.shuffle(ordering)
        return drawn.map_with_index(example)

    streams = {int(i): examples(int(i)) for i in tasks}
    masked = jax.tree.map(
        np.full_like, streams[int(tasks[0])][0], (config.unlabeled_mask_value, config.label_mask_value)
    )

    def dealt(cycle: int, g: int, t: int) -> int:
        order = np.random.default_rng(fold_in(dealing, cycle)).permutation(tasks) if draw.shuffle else tasks
        return int(order[g * per_tick + t])

    def slot(t: int) -> grain.MapDataset:
        def at(u: int, _: int) -> PyTree:
            match draw.regroup:
                case EveryPass():
                    cycle, r = divmod(u, groups * per_pass)
                    g, e = divmod(r, per_pass)
                    task, position = dealt(cycle, g, t), cycle * per_pass + e
                case EveryMinibatch():
                    cycle, r = divmod(u, groups * per_task)
                    task, position = dealt(cycle, r // per_task, t), u
            match draw.boundary:
                case Straddle():
                    return streams[task][position]
                case Padded():
                    return streams[task][position] if position % per_pass < n_examples else masked

        return grain.MapDataset.range(groups * per_pass).map_with_index(at)

    def cut(below: grain.MapDataset, depth: int) -> grain.MapDataset:
        def piece(j: int, _: int) -> PyTree:
            def windowed(a: np.ndarray) -> np.ndarray:
                sequenced = np.moveaxis(a, depth, 0)
                v = j % num_vb
                return sequenced[v * window : (v + 1) * window].reshape(*unit, *sequenced.shape[1:])

            return jax.tree.map(windowed, below[j // num_vb])

        return grain.MapDataset.range(len(below) * num_vb).map_with_index(piece)

    def build(above: tuple[Axis, ...], pick: Callable[[int], grain.MapDataset], scans: int) -> grain.MapDataset:
        match above:
            case (Window(k), *rest):
                below = build(tuple(rest), pick, scans - 1)
                if scans == 1:
                    return cut(below, len(rest))
                if len(below) % k != 0:
                    raise ValueError(f"{len(below)} ticks per pass are not divisible by the {k} per step of {draw}")
                return below.batch(k)
            case (Batch(n, Tasks()), *rest):

                def task_slot(s: int) -> Callable[[int], grain.MapDataset]:
                    return lambda t: pick(t * n + s)

                return stacked([build(tuple(rest), task_slot(s), scans) for s in range(n)])
            case (Batch(n, Examples()), *rest):

                def example_slot(s: int) -> Callable[[int], grain.MapDataset]:
                    return lambda t: pick(t)[s::n]

                return stacked([build(tuple(rest), example_slot(s), scans) for s in range(n)])
            case (Batch(n, Same()), *rest):
                return stacked([build(tuple(rest), pick, scans)] * n)
            case _:
                return pick(0) if windows > 0 else cut(pick(0), 0)

    return build(axes, slot, windows)


def stream(
    plan: Plan, sources: Sources, data: Taken, seed: int, passes: int | None, config: DataConfig
) -> grain.MapDataset:
    def go(plan: Plan, sources: Sources, data: Taken, indices: np.ndarray, seed: int) -> grain.MapDataset:
        match plan, sources, data:
            case Leaf(axes), Draw() as draw, list() as datasets:
                return ticks(axes, draw, datasets, indices, seeded(seed, draw.seeding), config)
            case Pair(axes, left, right), Level(train, val), tuple((first, second)):

                def learner(indices_: np.ndarray, seed_: int) -> grain.MapDataset:
                    trainee = go(left, train, first, indices_, fold_in(seed_, 0))
                    validation = go(right, val, second, indices_, fold_in(seed_, 1))
                    return ZipMapDataset([trainee, Extent(validation, len(trainee))])

                def lift(above: tuple[Axis, ...], indices_: np.ndarray, seed_: int) -> grain.MapDataset:
                    match above:
                        case (Window(k), *rest):
                            below = lift(tuple(rest), indices_, seed_)
                            if len(below) % k != 0:
                                raise ValueError(
                                    f"{len(below)} yields below are not divisible by the {k} per step of Scan"
                                )
                            return below.batch(k)
                        case (Batch(n, Tasks()), *rest):
                            if len(indices_) % n != 0:
                                raise ValueError(f"{len(indices_)} tasks cannot be chunked across {n} learners")
                            chunks = indices_.reshape(n, len(indices_) // n)
                            return stacked(
                                [lift(tuple(rest), chunk, fold_in(seed_, i)) for i, chunk in enumerate(chunks)]
                            )
                        case (Batch(n, Examples()), *rest):
                            return stacked([lift(tuple(rest), indices_, fold_in(seed_, i)) for i in range(n)])
                        case (Batch(n, Same()), *rest):
                            return stacked([lift(tuple(rest), indices_, seed_)] * n)
                        case _:
                            return learner(indices_, seed_)

                return lift(axes, indices, seed)
            case _:
                raise ValueError(f"sources {sources} do not mirror plan {plan}")

    tasks = np.random.default_rng(fold_in(seed, 2)).permutation(config.num_tasks)
    return go(plan, sources, data, tasks, seed).repeat(passes)
