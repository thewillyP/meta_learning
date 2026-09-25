from meta_learn_lib.construct.data import Batch, Every, Leaf, Plan, Steps, Window, draw_of
from meta_learn_lib.construct.term import Examples, Same, Tasks
from meta_learn_lib.data_source.source import Augmentation, DataConfig, Draw, Fixed, Fresh, Pool
from meta_learn_lib.data_source.tasks import Sequencer, Supply, augmenter, dataset_sources, take_datasets
from meta_learn_lib.lib_types import PRNG

from absl import flags
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import grain
from grain.experimental import ZipIterDataset, ZipMapDataset
import numpy as np
from torch.utils.data import Subset

type Example = tuple[np.ndarray, np.ndarray]
type Masks = tuple[float, float]


@dataclass(frozen=True)
class Feed:
    leaf: Leaf
    tasks: list[Subset[tuple]]
    sequence: Sequencer


class Source:
    def __init__(self, records: Subset[tuple]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple:
        match self.records[index]:
            case (x, y):
                return x, y
            case other:
                raise ValueError(f"{self.records} has no example at {index}: {other}")


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


def augmented(ds: grain.MapDataset, augmentation: Augmentation) -> grain.MapDataset:
    augment = augmenter(augmentation)

    def apply(xy: tuple, rng: np.random.Generator) -> tuple:
        x, y = xy
        return augment(np.asarray(x), rng), y

    return ds.random_map(apply)


def examples(task: Subset[tuple], draw: Draw, sequence: Sequencer, seed: int) -> grain.MapDataset:
    def sequenced(xy: tuple) -> Example:
        x, y = xy
        return np.asarray(sequence(np.asarray(x))), np.asarray(y)

    ds = grain.MapDataset.source(Source(task)).seed(seed)
    if draw.shuffle:
        ds = ds.shuffle()
    return reduce(augmented, draw.augment, ds).map(sequenced)


def at(ds: grain.MapDataset, i: int) -> Example:
    match ds[i]:
        case (x, y):
            return x, y
        case other:
            raise ValueError(f"{ds} has no example at {i}: {other}")


def masked(like: Example, count: int, masks: Masks) -> grain.MapDataset:
    mask_x, mask_y = masks
    x, y = like
    return grain.MapDataset.source([(np.full_like(x, mask_x), np.full_like(y, mask_y))] * count)


def to_multiple(task: grain.MapDataset, m: int, masks: Masks) -> grain.MapDataset:
    return grain.MapDataset.concatenate([task, masked(at(task, 0), -len(task) % m, masks)])


def windows(parent: grain.MapDataset, w: int, axis: int, fan_out: int, masks: Masks) -> grain.MapDataset:
    mask_x, mask_y = masks

    def window(a: np.ndarray, i: int, mask: float) -> np.ndarray:
        pad = [(0, 0)] * a.ndim
        pad[axis] = (0, fan_out * w - a.shape[axis])
        return np.moveaxis(np.take(np.pad(a, pad, constant_values=mask), range(i * w, (i + 1) * w), axis=axis), axis, 0)

    def cut(i: int, _) -> Example:
        x, y = at(parent, i // fan_out)
        return window(x, i % fan_out, mask_x), window(y, i % fan_out, mask_y)

    return grain.MapDataset.range(len(parent) * fan_out).map_with_index(cut)


def step(xy: Example) -> Example:
    x, y = xy
    return x[0], y[0]


def grouped(stream: grain.MapDataset, e: int) -> grain.MapDataset:
    match e:
        case 1:
            return stream.map(lambda xy: stacked([xy]))
        case _:
            return stream.batch(e, batch_fn=stacked)


def depth(leaf: Leaf) -> int:
    match leaf:
        case Steps():
            return 0
        case Window(_, below) | Every(_, below):
            return depth(below)
        case Batch(_, _, below):
            return 1 + depth(below)


def windowed(leaf: Leaf) -> bool:
    match leaf:
        case Steps():
            return False
        case Window():
            return True
        case Every(_, below) | Batch(_, _, below):
            return windowed(below)


def time_extent(leaf: Leaf) -> int:
    match leaf:
        case Steps():
            return 1
        case Window(w, below):
            return w * time_extent(below)
        case Every(_, below) | Batch(_, _, below):
            return time_extent(below)


def hoisted(leaf: Leaf) -> tuple[Leaf, list[int], list[int]]:
    match leaf:
        case Steps():
            return leaf, [], []
        case Every(e, below):
            inner, everys, positions = hoisted(below)
            return inner, [e, *everys], [0, *[p + 1 for p in positions]]
        case Window(w, below):
            inner, everys, positions = hoisted(below)
            return Window(w, inner), everys, [p + 1 for p in positions]
        case Batch(m, over, below):
            inner, everys, positions = hoisted(below)
            return Batch(m, over, inner), everys, [p + 1 for p in positions]


def ticks(feed: Feed, order: list[int], key: PRNG, config: DataConfig) -> grain.MapDataset:
    draw = draw_of(feed.leaf)
    masks = (config.unlabeled_mask_value, config.label_mask_value)
    seeds = jax.random.randint(seeded(key, draw), (len(feed.tasks),), 0, 2**31 - 1).tolist()
    tasks = [examples(feed.tasks[i], draw, feed.sequence, seeds[i]) for i in order]
    x, _ = at(tasks[0], 0)
    time = x.shape[0]
    inner, everys, positions = hoisted(feed.leaf)
    padded_time = -(-time // time_extent(inner)) * time_extent(inner)

    def epoch(leaf: Leaf, tasks: list[grain.MapDataset]) -> grain.MapDataset:
        match leaf:
            case Steps():
                return grain.MapDataset.concatenate(tasks)
            case Window(w, below):
                if windowed(below):
                    return epoch(below, tasks).batch(w, batch_fn=stacked)
                return windows(epoch(below, tasks), w, depth(below), padded_time // w, masks)
            case Every():
                raise ValueError(f"{leaf} is hoisted above the epoch")
            case Batch(m, over, below):
                match over:
                    case Tasks():
                        if len(tasks) % m != 0:
                            raise ValueError(f"{len(tasks)} tasks cannot be dealt into {m} lanes for {leaf}")
                        lanes = [epoch(below, tasks[i::m]) for i in range(m)]
                    case Examples():
                        padded = [to_multiple(task, m, masks) for task in tasks]
                        match below:
                            case Steps():
                                return epoch(below, padded).batch(m, batch_fn=stacked)
                            case _:
                                lanes = [epoch(below, [task[i::m] for task in padded]) for i in range(m)]
                    case Same():
                        return epoch(below, tasks).map(lambda xy: stacked([xy] * m))
                return ZipMapDataset(lanes).map(stacked)

    def reordered(xy: Example) -> Example:
        x, y = xy
        return np.moveaxis(x, range(len(positions)), positions), np.moveaxis(y, range(len(positions)), positions)

    units = epoch(inner, tasks)
    if not windowed(inner):
        units = windows(units, 1, depth(inner), time, masks).map(step)
    stream = reduce(grouped, reversed(everys), units.repeat())
    if positions == list(range(len(positions))):
        return stream
    return stream.map(reordered)


def stream(taken: Taken, key: PRNG, config: DataConfig) -> Iterator[PyTree]:
    feeds, treedef = jax.tree.flatten(taken)
    k_tasks, k_leaves = jax.random.split(key)
    order = jax.random.permutation(k_tasks, config.num_tasks).tolist()
    lanes = [ticks(feed, order, PRNG(k), config) for feed, k in zip(feeds, jax.random.split(k_leaves, len(feeds)))]
    ds = ZipIterDataset(
        [lane.to_iter_dataset(grain.ReadOptions(num_threads=1, prefetch_buffer_size=64)) for lane in lanes]
    )
    if config.workers > 0:
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=config.workers))
    for parts in ds:
        yield jax.tree.unflatten(treedef, jax.tree.map(jnp.asarray, parts))
