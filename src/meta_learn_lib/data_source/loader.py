from meta_learn_lib.construct.data import Batch, Chunk, Leaf, Plan, Steps, draw_of
from meta_learn_lib.construct.term import Examples, Same, Tasks
from meta_learn_lib.data_source.source import DataConfig, Draw, Fixed, Fresh, Pool
from meta_learn_lib.data_source.tasks import (
    Example,
    Sequencer,
    Source,
    Supply,
    dataset_sources,
    examples,
    take_datasets,
)
from meta_learn_lib.lib_types import PRNG

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import grain
from grain.experimental import FlatMapIterDataset, FlatMapTransform, ZipIterDataset
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np
from torch.utils.data import Subset

type Lane = list[grain.MapDataset] | grain.MapDataset


@dataclass(frozen=True)
class Feed:
    leaf: Leaf
    tasks: list[Subset[tuple]]
    sequence: Sequencer


type Taken = Feed | tuple[Taken, Taken]


def seeded(key: PRNG, draw: Draw) -> PRNG:
    match draw.seeding:
        case Fresh():
            return key
        case Fixed(seed):
            return PRNG(jax.random.key(seed))


def materialize(plan: Plan, key: PRNG, config: DataConfig) -> Taken:
    leaves, treedef = jax.tree.flatten(plan)
    pools: dict[Pool, list[Supply]] = {}
    feeds: list[Feed] = []
    for leaf, k in zip(leaves, jax.random.split(key, len(leaves))):
        draw = draw_of(leaf)
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
        tasks, leftover = take_datasets(PRNG(k_take), supply, draw.take, draw.shuffle)
        first, *_ = supply
        _, sequence = first
        pools = {**pools, draw.pool: leftover}
        feeds = [*feeds, Feed(leaf, tasks, sequence)]
    return jax.tree.unflatten(treedef, feeds)


def features(taken: Taken) -> PyTree:
    def shapes(feed: Feed) -> tuple[tuple[int, ...], tuple[int, ...]]:
        x, y = Source(feed.tasks[0])[0]
        return feed.sequence(np.asarray(x)).shape[1:], np.asarray(y).shape[1:]

    return jax.tree.map(shapes, taken)


def stacked(parts: Sequence[Example]) -> Example:
    xs, ys = zip(*parts)
    return np.stack(xs), np.stack(ys)


def abreast(parts: Sequence[Example]) -> Example:
    xs, ys = zip(*parts)
    return np.stack(xs, axis=1), np.stack(ys, axis=1)


class Timesteps(FlatMapTransform):
    def flat_map(self, element: Example) -> list[Example]:
        x, y = element
        return [(x[t], y[t]) for t in range(len(x))]


def sequences(lane: Lane) -> grain.MapDataset:
    match lane:
        case list():
            return grain.MapDataset.concatenate(lane).repeat()
        case _:
            return lane


def build(leaf: Leaf, lane: Lane) -> grain.IterDataset:
    match leaf:
        case Steps():
            reading = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
            return FlatMapIterDataset(sequences(lane).to_iter_dataset(reading), Timesteps())
        case Chunk(n, below):
            return build(below, lane).batch(n, batch_fn=stacked)
        case Batch(n, Examples(), Steps() as below):
            return build(below, sequences(lane).batch(n, batch_fn=abreast))
        case Batch(n, over, below):
            match over:
                case Tasks():
                    parts = [lane[i::n] for i in range(n)]
                case Examples():
                    parts = [sequences(lane)[i::n] for i in range(n)]
                case Same():
                    parts = [lane] * n
            return ZipIterDataset([build(below, part) for part in parts]).map(stacked)


def lanes(leaf: Leaf) -> int:
    match leaf:
        case Steps():
            return 1
        case Chunk(_, below):
            return lanes(below)
        case Batch(n, over, below):
            match over:
                case Tasks():
                    return n * lanes(below)
                case Examples() | Same():
                    return lanes(below)


def ticks(feed: Feed, order: list[int], key: PRNG) -> grain.IterDataset:
    if len(order) % lanes(feed.leaf) != 0:
        raise ValueError(f"{len(order)} tasks cannot be dealt into {lanes(feed.leaf)} lanes for {feed.leaf}")
    draw = draw_of(feed.leaf)
    seeds = jax.random.randint(seeded(key, draw), (len(feed.tasks),), 0, 2**31 - 1).tolist()
    return build(feed.leaf, [examples(feed.tasks[i], draw, feed.sequence, seeds[i]) for i in order])


def stream(taken: Taken, key: PRNG, config: DataConfig) -> Iterator[PyTree]:
    feeds, treedef = jax.tree.flatten(taken)
    k_tasks, k_leaves = jax.random.split(key)
    order = jax.random.permutation(k_tasks, config.num_tasks).tolist()
    leaves = [ticks(feed, order, PRNG(k)) for feed, k in zip(feeds, jax.random.split(k_leaves, len(feeds)))]
    for parts in ZipIterDataset(leaves):
        yield jax.tree.unflatten(treedef, jax.tree.map(jnp.asarray, parts))
