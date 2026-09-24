from meta_learn_lib.construct.data import Batch, Every, Leaf, Plan, Steps, Window
from meta_learn_lib.construct.term import Examples, Same, Tasks
from meta_learn_lib.data_source.source import Augmentation, DataConfig, Draw, Fixed, Fresh, Pool, Sources
from meta_learn_lib.data_source.tasks import Sequencer, Supply, Table, augmenter, dataset_sources, take_datasets
from meta_learn_lib.lib_types import PRNG

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import partial, reduce
import itertools
import math
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

type Taken = Table | tuple[Taken, Taken]


@dataclass(frozen=True)
class Factors:
    time: int
    tasks: int
    examples: int
    every: int


def seeded(key: PRNG, draw: Draw) -> PRNG:
    match draw.seeding:
        case Fresh():
            return key
        case Fixed(seed):
            return PRNG(jax.random.key(seed))


def materialize(sources: Sources, key: PRNG, config: DataConfig) -> Taken:
    draws, treedef = jax.tree.flatten(sources)
    pools: dict[Pool, list[Supply]] = {}
    tables: list[Table] = []
    for draw, k in zip(draws, jax.random.split(key, len(draws))):
        k_pool, k_take = jax.random.split(seeded(PRNG(k), draw))
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
        table, leftover = take_datasets(PRNG(k_take), supply, draw.take, draw.shuffle)
        pools = {**pools, draw.pool: leftover}
        tables = [*tables, table]
    return jax.tree.unflatten(treedef, tables)


def features(taken: Taken) -> PyTree:
    def shapes(table: Table) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return table.sequence(jnp.asarray(table.xs[0, 0])).shape[1:], table.ys[0, 0].shape[1:]

    return jax.tree.map(shapes, taken)


def factors(leaf: Leaf) -> Factors:
    match leaf:
        case Steps():
            return Factors(1, 1, 1, 1)
        case Window(n, below):
            f = factors(below)
            return Factors(n * f.time, f.tasks, f.examples, f.every)
        case Every(n, below):
            f = factors(below)
            return Factors(f.time, f.tasks, f.examples, n * f.every)
        case Batch(n, over, below):
            f = factors(below)
            match over:
                case Tasks():
                    return Factors(f.time, n * f.tasks, f.examples, f.every)
                case Examples():
                    return Factors(f.time, f.tasks, n * f.examples, f.every)
                case Same():
                    return f


def divided(leaf: Window | Batch, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    k, n, t = shape
    match leaf:
        case Window(m, _):
            return (k, n, t // m)
        case Batch(m, over, _):
            match over:
                case Tasks():
                    return (k // m, n, t)
                case Examples():
                    return (k, n // m, t)
                case Same():
                    return shape


def ticks_per_epoch(leaf: Leaf, shape: tuple[int, int, int]) -> Fraction:
    match leaf:
        case Steps():
            return Fraction(math.prod(shape))
        case Every(m, below):
            return ticks_per_epoch(below, shape) / m
        case Window(_, below) | Batch(_, _, below):
            return ticks_per_epoch(below, divided(leaf, shape))


def epochs_per_block(leaf: Leaf, shape: tuple[int, int, int]) -> int:
    match leaf:
        case Steps():
            return 1
        case Every(_, below):
            return math.lcm(epochs_per_block(below, shape), ticks_per_epoch(leaf, shape).denominator)
        case Window(_, below) | Batch(_, _, below):
            return epochs_per_block(below, divided(leaf, shape))


def padded(leaf: Leaf, xs: jax.Array, mask: float) -> jax.Array:
    f = factors(leaf)
    l, k, n, t, *rest = xs.shape
    if k % f.tasks != 0:
        raise ValueError(f"{k} tasks cannot be dealt into {f.tasks} lanes for {leaf}")
    return jnp.pad(
        xs, [(0, 0), (0, 0), (0, -n % f.examples), (0, -t % f.time), *[(0, 0) for _ in rest]], constant_values=mask
    )


def dealt(leaf: Window | Batch, xs: jax.Array) -> jax.Array:
    l, k, n, t, *rest = xs.shape
    match leaf:
        case Window(m, below):
            p = factors(below).time
            parts = jnp.moveaxis(xs.reshape(l, k, n, t // (m * p), m, p, *rest), 4, 0)
            return parts.reshape(m, l, k, n, t // m, *rest)
        case Batch(m, over, _):
            match over:
                case Tasks():
                    return jnp.moveaxis(xs.reshape(l, k // m, m, n, t, *rest), 2, 0)
                case Examples():
                    return jnp.moveaxis(xs.reshape(l, k, n // m, m, t, *rest), 3, 0)
                case Same():
                    return jnp.broadcast_to(xs, (m, *xs.shape))


def epoch(leaf: Leaf, xs: jax.Array) -> jax.Array:
    match leaf:
        case Steps():
            l, k, n, t, *rest = xs.shape
            return xs.reshape(l * k * n * t, *rest)
        case Every(m, below):
            ticks = epoch(below, xs)
            if len(ticks) % m != 0:
                raise ValueError(f"{len(ticks)} ticks are not divisible by {m} for {leaf}")
            return ticks.reshape(len(ticks) // m, m, *ticks.shape[1:])
        case Window(_, below) | Batch(_, _, below):
            return jax.vmap(lambda part: epoch(below, part), out_axes=1)(dealt(leaf, xs))


def drawn(
    sequence: Sequencer, draw: Draw, xs: jax.Array, ys: jax.Array, tasks: jax.Array, key: PRNG
) -> tuple[jax.Array, jax.Array]:
    def shuffled(x: jax.Array, y: jax.Array, k_task: jax.Array) -> tuple[jax.Array, jax.Array]:
        order = jax.random.permutation(k_task, len(x)) if draw.shuffle else jnp.arange(len(x))
        return x[order], y[order]

    def augmented(x: jax.Array, k_example: jax.Array) -> jax.Array:
        def apply(img: jax.Array, step: tuple[Augmentation, jax.Array]) -> jax.Array:
            augmentation, k_step = step
            return augmenter(augmentation)(img, PRNG(k_step))

        applications = zip(draw.augment, jax.random.split(k_example, len(draw.augment)))
        return sequence(reduce(apply, applications, x))

    k_order, k_augment = jax.random.split(key)
    xs, ys = xs[tasks], ys[tasks]
    xs, ys = jax.vmap(shuffled)(xs, ys, jax.random.split(k_order, len(tasks))[tasks])
    return jax.vmap(jax.vmap(augmented))(xs, jax.random.split(k_augment, xs.shape[:2])[tasks]), ys


def block(
    leaf: Leaf,
    draw: Draw,
    sequence: Sequencer,
    count: int,
    config: DataConfig,
    xs: jax.Array,
    ys: jax.Array,
    tasks: jax.Array,
    key: PRNG,
) -> tuple[jax.Array, jax.Array]:
    drawn_x, drawn_y = zip(
        *[drawn(sequence, draw, xs, ys, tasks, PRNG(jax.random.fold_in(key, l))) for l in range(count)]
    )
    return (
        epoch(leaf, padded(leaf, jnp.stack(drawn_x), config.unlabeled_mask_value)),
        epoch(leaf, padded(leaf, jnp.stack(drawn_y), config.label_mask_value)),
    )


def ticks(
    leaf: Leaf, draw: Draw, table: Table, tasks: jax.Array, key: PRNG, config: DataConfig
) -> Iterator[tuple[jax.Array, jax.Array]]:
    xs, ys = jnp.asarray(table.xs), jnp.asarray(table.ys)
    k, n, *_ = table.xs.shape
    t = len(table.sequence(xs[0, 0]))
    f = factors(leaf)
    count = epochs_per_block(leaf, (k, n + -n % f.examples, t + -t % f.time))
    build = jax.jit(partial(block, leaf, draw, table.sequence, count, config))
    for b in itertools.count():
        ex, ey = build(xs, ys, tasks, PRNG(jax.random.fold_in(key, b)))
        yield from zip(ex, ey)


def stream(plan: Plan, sources: Sources, taken: Taken, key: PRNG, config: DataConfig) -> Iterator[PyTree]:
    treedef = jax.tree.structure(plan)
    if jax.tree.structure(sources) != treedef or jax.tree.structure(taken) != treedef:
        raise ValueError(f"sources {sources} and data {taken} must mirror the plan {plan}")
    k_tasks, k_leaves = jax.random.split(key)
    tasks = jax.random.permutation(k_tasks, config.num_tasks)
    keys = jax.tree.unflatten(treedef, list(jax.random.split(k_leaves, treedef.num_leaves)))
    lanes = jax.tree.map(
        lambda leaf, draw, table, k: ticks(leaf, draw, table, tasks, seeded(PRNG(k), draw), config),
        plan,
        sources,
        taken,
        keys,
    )
    for parts in zip(*jax.tree.leaves(lanes)):
        yield jax.tree.unflatten(treedef, parts)
