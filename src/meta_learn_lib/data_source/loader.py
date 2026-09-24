from meta_learn_lib.construct.data import Batch, Leaf, Pair, Plan, Steps, Window
from meta_learn_lib.construct.term import Examples, Over, Same, Tasks
from meta_learn_lib.data_source.source import Augmentation, DataConfig, Draw, Fixed, Fresh, Level, Pool, Sources
from meta_learn_lib.data_source.tasks import Sequencer, Supply, augmenter, dataset_sources, take_datasets
from meta_learn_lib.lib_types import PRNG

from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
import itertools
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np


@dataclass(frozen=True)
class Table:
    xs: np.ndarray
    ys: np.ndarray
    sequence: Sequencer
    draw: Draw


type Taken = Table | tuple[Taken, Taken]


def seeded(key: PRNG, draw: Draw) -> PRNG:
    match draw.seeding:
        case Fresh():
            return key
        case Fixed(seed):
            return PRNG(jax.random.key(seed))


def materialize(
    plan: Plan, sources: Sources, key: PRNG, pools: dict[Pool, list[Supply]], config: DataConfig
) -> tuple[Taken, dict[Pool, list[Supply]]]:
    def go(
        plan: Plan, sources: Sources, key: PRNG, pools: dict[Pool, list[Supply]]
    ) -> tuple[Taken, dict[Pool, list[Supply]]]:
        match plan, sources:
            case Pair(left, right), Level(train, val):
                k_train, k_val = jax.random.split(key)
                first, after_train = go(left, train, PRNG(k_train), pools)
                second, after_val = go(right, val, PRNG(k_val), after_train)
                return (first, second), after_val
            case Steps() | Window() | Batch(), Draw() as draw:
                k_pool, k_take = jax.random.split(seeded(key, draw))
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
                tasks, leftover = take_datasets(PRNG(k_take), supply, draw.take, draw.shuffle)
                first, *_ = tasks
                table = Table(np.stack([t.xs for t in tasks]), np.stack([t.ys for t in tasks]), first.sequence, draw)
                return table, {**pools, draw.pool: leftover}
            case _:
                raise ValueError(f"sources {sources} do not mirror plan {plan}")

    return go(plan, sources, key, pools)


def features(taken: Taken) -> PyTree:
    match taken:
        case (first, second):
            return (features(first), features(second))
        case Table(xs, ys, sequence, _):
            return (sequence(jnp.asarray(xs[0, 0])).shape[1:], ys[0, 0].shape[1:])


def steps(xs: jax.Array) -> jax.Array:
    k, n, t, *rest = xs.shape
    return xs.reshape(k * n * t, *rest)


def chunks(n: int, ticks: jax.Array) -> jax.Array:
    if len(ticks) % n != 0:
        raise ValueError(f"{len(ticks)} ticks per pass are not divisible by a window of {n}")
    return ticks.reshape(len(ticks) // n, n, *ticks.shape[1:])


def lanes(parts: list[jax.Array]) -> jax.Array:
    return jnp.stack(parts, axis=1)


def split(over: Over, n: int, xs: jax.Array) -> list[jax.Array]:
    match over:
        case Tasks():
            return [xs[i::n] for i in range(n)]
        case Examples():
            return [xs[:, i::n] for i in range(n)]
        case Same():
            return [xs] * n


def epoch(plan: Leaf, xs: jax.Array) -> jax.Array:
    match plan:
        case Steps():
            return steps(xs)
        case Window(n, below):
            return chunks(n, epoch(below, xs))
        case Batch(n, over, below):
            return lanes([epoch(below, part) for part in split(over, n, xs)])


def lanes_of(plan: Leaf) -> tuple[int, int]:
    match plan:
        case Steps():
            return 1, 1
        case Window(_, below):
            return lanes_of(below)
        case Batch(n, over, below):
            tasks, examples = lanes_of(below)
            match over:
                case Tasks():
                    return n * tasks, examples
                case Examples():
                    return tasks, n * examples
                case Same():
                    return tasks, examples


def innermost(plan: Leaf) -> int | None:
    match plan:
        case Steps():
            return None
        case Window(n, below):
            match innermost(below):
                case None:
                    return n
                case inner:
                    return inner
        case Batch(_, _, below):
            return innermost(below)


def padded(plan: Leaf, xs: jax.Array, mask: float) -> jax.Array:
    tasks, examples = lanes_of(plan)
    k, n, t, *rest = xs.shape
    if k % tasks != 0:
        raise ValueError(f"{k} tasks cannot be dealt into {tasks} lanes for {plan}")
    match innermost(plan):
        case None:
            missing = 0
        case window:
            missing = -t % window
    return jnp.pad(xs, [(0, 0), (0, -n % examples), (0, missing), *[(0, 0) for _ in rest]], constant_values=mask)


def drawn(table: Table, tasks: jax.Array, key: PRNG) -> tuple[jax.Array, jax.Array]:
    def shuffled(x: jax.Array, y: jax.Array, k_task: jax.Array) -> tuple[jax.Array, jax.Array]:
        order = jax.random.permutation(k_task, len(x)) if table.draw.shuffle else jnp.arange(len(x))
        return x[order], y[order]

    def augmented(x: jax.Array, k_example: jax.Array) -> jax.Array:
        def apply(img: jax.Array, step: tuple[Augmentation, jax.Array]) -> jax.Array:
            augmentation, k_step = step
            return augmenter(augmentation)(img, PRNG(k_step))

        applications = zip(table.draw.augment, jax.random.split(k_example, len(table.draw.augment)))
        return table.sequence(reduce(apply, applications, x))

    k_order, k_augment = jax.random.split(key)
    xs, ys = jnp.asarray(table.xs)[tasks], jnp.asarray(table.ys)[tasks]
    xs, ys = jax.vmap(shuffled)(xs, ys, jax.random.split(k_order, len(tasks))[tasks])
    return jax.vmap(jax.vmap(augmented))(xs, jax.random.split(k_augment, xs.shape[:2])[tasks]), ys


def ticks(
    plan: Leaf, table: Table, tasks: jax.Array, key: PRNG, config: DataConfig
) -> Iterator[tuple[jax.Array, jax.Array]]:
    for e in itertools.count():
        xs, ys = drawn(table, tasks, PRNG(jax.random.fold_in(key, e)))
        yield from zip(
            epoch(plan, padded(plan, xs, config.unlabeled_mask_value)),
            epoch(plan, padded(plan, ys, config.label_mask_value)),
        )


def stream(plan: Plan, taken: Taken, key: PRNG, config: DataConfig) -> Iterator[PyTree]:
    k_tasks, k_leaves = jax.random.split(key)
    tasks = jax.random.permutation(k_tasks, config.num_tasks)

    def go(plan: Plan, taken: Taken, key: PRNG) -> Iterator[PyTree]:
        match plan, taken:
            case Pair(left, right), (first, second):
                k_left, k_right = jax.random.split(key)
                return zip(go(left, first, PRNG(k_left)), go(right, second, PRNG(k_right)))
            case (Steps() | Window() | Batch()) as leaf, Table() as table:
                return ticks(leaf, table, tasks, seeded(key, table.draw), config)
            case _:
                raise ValueError(f"data {taken} does not mirror plan {plan}")

    return go(plan, taken, PRNG(k_leaves))
