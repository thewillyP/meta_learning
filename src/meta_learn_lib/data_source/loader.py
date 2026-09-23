from meta_learn_lib.construct.data import Axis, Batch, Leaf, Pair, Plan, Window
from meta_learn_lib.construct.term import Examples, Same, Tasks
from meta_learn_lib.data_source.source import (
    Augmentation,
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
    augmenter,
    dataset_sources,
    take_datasets,
)
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.utility.util import fold_in

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
import math
from typing import NamedTuple
import grain
from grain.experimental import ZipMapDataset
from grain.transforms import DatasetSelectionMap
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


class SequenceWindow(NamedTuple):
    size: int
    batch_axes_before: int


class SingleStep(NamedTuple): ...


class StepBundle(NamedTuple):
    size: int
    batch_axes_before: int


class Layout(NamedTuple):
    batches: tuple[Batch, ...]
    innermost: SequenceWindow | SingleStep
    bundles: tuple[StepBundle, ...]
    tasks_per_tick: int
    examples_per_task: int


def layout(axes: tuple[Axis, ...]) -> Layout:
    batches: list[Batch] = []
    scans: list[tuple[int, int]] = []
    tasks, examples = 1, 1
    for axis in axes:
        match axis:
            case Batch(n, Tasks()):
                batches.append(axis)
                tasks *= n
            case Batch(n, Examples()):
                batches.append(axis)
                examples *= n
            case Batch():
                batches.append(axis)
            case Window(size):
                scans.append((size, len(batches)))
    match scans:
        case [*outer, (size, before)]:
            innermost, bundles = SequenceWindow(size, before), [StepBundle(k, b) for k, b in outer]
        case _:
            innermost, bundles = SingleStep(), []
    return Layout(tuple(batches), innermost, tuple(bundles), tasks, examples)


def drawn(task: PrematerializedTask, draw: Draw, seed: int, length: int, config: DataConfig) -> grain.MapDataset:
    def padded(a: np.ndarray, mask: float) -> np.ndarray:
        return a if len(a) == length else np.concatenate([a, np.full((length - len(a), *a.shape[1:]), mask, a.dtype)])

    def augmented(examples: grain.MapDataset, augmentation: Augmentation) -> grain.MapDataset:
        apply = augmenter(augmentation)

        def on_x(xy: tuple[np.ndarray, np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
            x, y = xy
            return apply(x, rng), y

        return examples.random_map(on_x)

    def framed(xy: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x, y = xy
        return padded(task.sequence(x), config.unlabeled_mask_value), padded(y, config.label_mask_value)

    order = grain.MapDataset.range(len(task.xs)).seed(seed)
    if draw.shuffle:
        order = order.shuffle()
    examples = order.map(lambda j: (task.xs[j], task.ys[j]))
    return reduce(augmented, draw.augment, examples).map(framed)


@dataclass(frozen=True)
class Schedule(DatasetSelectionMap):
    draw: Draw
    per_tick: int
    per_task: int
    groups: int
    per_pass: int
    n_examples: int
    seed: int
    slot: int

    def __len__(self) -> int:
        return self.groups * self.per_pass

    def __getitem__(self, u: int) -> tuple[int, int]:
        match self.draw.regroup:
            case EveryPass():
                cycle, r = divmod(u, self.groups * self.per_pass)
                g, e = divmod(r, self.per_pass)
                consumed = cycle * self.per_pass + e
            case EveryMinibatch():
                cycle, r = divmod(u, self.groups * self.per_task)
                g, e = divmod(r, self.per_task)
                consumed = cycle * self.per_task + e
        count = self.groups * self.per_tick
        order = (
            np.random.default_rng(fold_in(self.seed, cycle)).permutation(count) if self.draw.shuffle else range(count)
        )
        task = int(order[g * self.per_tick + self.slot])
        match self.draw.boundary:
            case Straddle():
                return task, consumed
            case Padded():
                passes, within = divmod(consumed, self.per_pass)
                return (task, passes * self.n_examples + within) if within < self.n_examples else (count, 0)


def scheduled(
    draw: Draw,
    datasets: list[PrematerializedTask],
    indices: np.ndarray,
    seed: int,
    shape: Layout,
    length: int,
    config: DataConfig,
) -> Callable[[int], grain.MapDataset]:
    if len(indices) % shape.tasks_per_tick != 0:
        raise ValueError(f"{len(indices)} tasks cannot be dealt {shape.tasks_per_tick} per tick for {draw}")
    tasks = np.sort(indices)
    n_examples = len(datasets[int(tasks[0])].xs)
    per_pass = math.ceil(n_examples / shape.examples_per_task) * shape.examples_per_task
    dealing, drawing = fold_in(seed, 0), fold_in(seed, 1)
    streams = [drawn(datasets[int(i)], draw, fold_in(drawing, int(i)), length, config) for i in tasks]
    first, *_ = streams
    masks = grain.MapDataset.source(
        [jax.tree.map(np.full_like, first[0], (config.unlabeled_mask_value, config.label_mask_value))]
    )

    def slot(t: int) -> grain.MapDataset:
        groups = len(tasks) // shape.tasks_per_tick
        schedule = Schedule(
            draw, shape.tasks_per_tick, shape.examples_per_task, groups, per_pass, n_examples, dealing, t
        )
        return grain.MapDataset.select_from_datasets([*streams, masks], schedule)

    return slot


def assembled(shape: Layout, slot: Callable[[int], grain.MapDataset], length: int) -> grain.MapDataset:
    sequence_axis = len(shape.batches)

    def grouped(above: tuple[Batch, ...], t: int) -> grain.MapDataset:
        match above:
            case (Batch(n, Tasks()), *rest):
                return stacked([grouped(tuple(rest), t * n + s) for s in range(n)])
            case (Batch(n, Examples()), *rest):
                return grouped(tuple(rest), t).batch(n)
            case (Batch(n, Same()), *rest):
                return stacked([grouped(tuple(rest), t)] * n)
            case _:
                return slot(t)

    def stepped(examples: grain.MapDataset) -> grain.MapDataset:
        def step(j: int, _: int) -> PyTree:
            return jax.tree.map(lambda a: np.take(a, j % length, axis=sequence_axis), examples[j // length])

        return grain.MapDataset.range(len(examples) * length).map_with_index(step)

    def windowed(examples: grain.MapDataset, window: SequenceWindow) -> grain.MapDataset:
        per_example = length // window.size

        def window_of(j: int, _: int) -> PyTree:
            def sliced(a: np.ndarray) -> np.ndarray:
                v = j % per_example
                steps = np.moveaxis(a, sequence_axis, 0)[v * window.size : (v + 1) * window.size]
                return np.moveaxis(steps, 0, window.batch_axes_before)

            return jax.tree.map(sliced, examples[j // per_example])

        return grain.MapDataset.range(len(examples) * per_example).map_with_index(window_of)

    def bundled(ticks: grain.MapDataset, bundle: StepBundle) -> grain.MapDataset:
        if len(ticks) % bundle.size != 0:
            raise ValueError(f"{len(ticks)} ticks per pass are not divisible by the {bundle.size} per step of Scan")
        return ticks.batch(bundle.size).map(
            lambda tick: jax.tree.map(lambda a: np.moveaxis(a, 0, bundle.batch_axes_before), tick)
        )

    examples = grouped(shape.batches, 0)
    match shape.innermost:
        case SequenceWindow() as window:
            ticks = windowed(examples, window)
        case SingleStep():
            ticks = stepped(examples)
    return reduce(bundled, reversed(shape.bundles), ticks)


def ticks(
    axes: tuple[Axis, ...],
    draw: Draw,
    datasets: list[PrematerializedTask],
    indices: np.ndarray,
    seed: int,
    config: DataConfig,
) -> grain.MapDataset:
    shape = layout(axes)
    match shape.innermost:
        case SequenceWindow(size, _):
            window = size
        case SingleStep():
            window = 1
    example = datasets[int(indices[0])]
    length = math.ceil(len(example.sequence(example.xs[0])) / window) * window
    return assembled(shape, scheduled(draw, datasets, indices, seed, shape, length, config), length)


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
