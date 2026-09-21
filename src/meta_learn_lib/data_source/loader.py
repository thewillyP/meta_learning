from meta_learn_lib.construct.data import Axis, Leaf, Pair, Plan, size, windowed
from meta_learn_lib.data_source.source import (
    Chunked,
    DataConfig,
    Draw,
    Examples,
    Fixed,
    Fresh,
    Independent,
    Level,
    Pool,
    Seeding,
    Shared,
    Sharing,
    Sources,
    Tasks,
)
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.data_source.tasks import (
    PrematerializedTask,
    augmentation,
    dataset_sources,
    rechunk_pytrees,
    regroup_leading,
    take_datasets,
    task_epoch_tensor,
)

from collections.abc import Iterator
from dataclasses import dataclass
import itertools
import math
from typing import Callable, overload
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from plum import dispatch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Placement:
    leaf: Leaf
    draw: Draw
    index: int
    chunk: int
    outer: tuple[Axis, ...]


def ticks(axes: tuple[Axis, ...]) -> int:
    return math.prod(size(a) for a in axes if windowed(a))


def batched(axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
    return tuple(a for a in axes if not windowed(a))


def window(leaf: Leaf) -> int:
    inner = [size(a) for a in leaf.axes if windowed(a)]
    return inner[-1] if inner else 1


def count(plan: Plan) -> int:
    match plan:
        case Leaf():
            return 1
        case Pair(_, left, right):
            return count(left) + count(right)


@overload
def narrow(r: Chunked, chunk: int, n: int) -> int:
    if chunk % n != 0:
        raise ValueError(f"{chunk} tasks cannot be chunked across {n} replicas")
    return chunk // n


@overload
def narrow(r: Shared, chunk: int, n: int) -> int:
    return chunk


@overload
def narrow(r: Independent, chunk: int, n: int) -> int:
    return chunk


@overload
def narrow(r: Sharing, chunk: int, n: int) -> int:
    raise NotImplementedError


@dispatch
def narrow(r: Sharing, chunk: int, n: int) -> int:
    raise NotImplementedError


@overload
def replicate(r: Chunked, n: int, indices: jax.Array, key: PRNG) -> list[tuple[jax.Array, PRNG]]:
    return [(c, PRNG(jax.random.fold_in(key, i))) for i, c in enumerate(jnp.split(indices, n))]


@overload
def replicate(r: Shared, n: int, indices: jax.Array, key: PRNG) -> list[tuple[jax.Array, PRNG]]:
    return [(indices, key)]


@overload
def replicate(r: Independent, n: int, indices: jax.Array, key: PRNG) -> list[tuple[jax.Array, PRNG]]:
    return [(indices, PRNG(jax.random.fold_in(key, i))) for i in range(n)]


@overload
def replicate(r: Sharing, n: int, indices: jax.Array, key: PRNG) -> list[tuple[jax.Array, PRNG]]:
    raise NotImplementedError


@dispatch
def replicate(r: Sharing, n: int, indices: jax.Array, key: PRNG) -> list[tuple[jax.Array, PRNG]]:
    raise NotImplementedError


def stacked(shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    return map(
        lambda items: jax.tree.map(lambda *xs: jnp.stack(xs).reshape(shape + xs[0].shape), *items), zip(*streams)
    )


def broadcast(shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    (only,) = streams
    return map(lambda item: jax.tree.map(lambda x: jnp.broadcast_to(x, shape + x.shape), item), only)


@overload
def gather(r: Chunked, shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    return stacked(shape, streams)


@overload
def gather(r: Shared, shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    return broadcast(shape, streams)


@overload
def gather(r: Independent, shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    return stacked(shape, streams)


@overload
def gather(r: Sharing, shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    raise NotImplementedError


@dispatch
def gather(r: Sharing, shape: tuple[int, ...], streams: list[Iterator[PyTree]]) -> Iterator[PyTree]:
    raise NotImplementedError


def placements(plan: Plan, sources: Sources, index: int, chunk: int, outer: tuple[Axis, ...]) -> list[Placement]:
    match plan, sources:
        case Leaf() as leaf, Draw() as draw:
            return [Placement(leaf, draw, index, chunk, outer)]
        case Pair(axes, left, right), Level(train, val, replicas):
            below = (*outer, *axes)
            sub = narrow(replicas, chunk, math.prod(size(a) for a in batched(axes)))
            return placements(left, train, index, sub, below) + placements(right, val, index + count(left), sub, below)
        case _:
            raise ValueError(f"sources {sources} do not mirror plan {plan}")


def positions(placed: Placement) -> tuple[list[int], list[int]]:
    axes = batched(placed.leaf.axes)
    if len(axes) != len(placed.draw.over):
        raise ValueError(
            f"leaf {placed.index}: the term has batch axes {axes} but the source declares roles {placed.draw.over}"
        )
    tasks: list[int] = []
    examples: list[int] = []
    for i, role in enumerate(placed.draw.over):
        match role:
            case Tasks():
                tasks.append(i)
            case Examples():
                examples.append(i)
    return tasks, examples


def counts(placed: Placement) -> tuple[int, int]:
    axes = batched(placed.leaf.axes)
    tasks, examples = positions(placed)
    return math.prod(size(axes[i]) for i in tasks), math.prod(size(axes[i]) for i in examples)


def validate(config: DataConfig, plan: Plan) -> list[str]:
    errors: list[str] = []
    try:
        placed = placements(plan, config.sources, 0, config.num_tasks, ())
    except ValueError as e:
        return [str(e)]
    for p in placed:
        try:
            tasks, _ = counts(p)
        except ValueError as e:
            errors.append(str(e))
            continue
        if p.chunk % tasks != 0:
            errors.append(
                f"leaf {p.index}: its chunk of {p.chunk} tasks is not divisible by its {tasks} tasks per yield"
            )
    return errors


def base_key(key: PRNG, index: int, seeding: Seeding) -> PRNG:
    match seeding:
        case Fresh():
            return PRNG(jax.random.fold_in(key, index))
        case Fixed(seed):
            return PRNG(jax.random.key(seed))


def materialize(config: DataConfig, plan: Plan, key: PRNG) -> list[list[PrematerializedTask]]:
    placed = placements(plan, config.sources, 0, config.num_tasks, ())
    keys = [jax.random.split(base_key(key, p.index, p.draw.seeding), 3) for p in placed]
    pools: dict[Pool, list[tuple[Dataset, Callable[[jax.Array], jax.Array]]]] = {}
    for p, (k_pool, _, _) in zip(placed, keys):
        if p.draw.pool not in pools:
            pools[p.draw.pool] = dataset_sources(
                p.draw.pool.task,
                config.root_dir,
                p.draw.pool.split == "test",
                config.label_mask_value,
                config.num_tasks,
                PRNG(k_pool),
            )
    taken: list[list[PrematerializedTask]] = []
    for p, (_, k_take, _) in zip(placed, keys):
        datasets, pools[p.draw.pool] = take_datasets(
            seed=PRNG(k_take),
            remaining=pools[p.draw.pool],
            n=p.draw.take,
            n_consume=window(p.leaf),
            x_mask=config.unlabeled_mask_value,
            y_mask=config.label_mask_value,
            augment_fn=augmentation(p.draw.augment),
            shuffle=p.draw.shuffle,
        )
        taken.append(datasets)
    return taken


def features(data: list[list[PrematerializedTask]]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    dummy = PRNG(jax.random.key(0))
    return [
        (
            datasets[0].x_epoch(datasets[0].xs[0], dummy).shape[2:],
            datasets[0].y_epoch(datasets[0].ys[0], dummy).shape[2:],
        )
        for datasets in data
    ]


def epoch(
    draw: Draw,
    datasets: list[PrematerializedTask],
    indices: jax.Array,
    tasks: int,
    examples: int,
    key: PRNG,
    x_mask: float,
    y_mask: float,
) -> tuple[jax.Array, jax.Array]:
    per_x, per_y = zip(
        *[
            task_epoch_tensor(datasets[i], examples, x_mask, y_mask, PRNG(jax.random.fold_in(key, i)), draw.shuffle)
            for i in indices.tolist()
        ]
    )
    groups = len(indices) // tasks
    return regroup_leading(jnp.stack(per_x), groups, tasks), regroup_leading(jnp.stack(per_y), groups, tasks)


def checked(block: jax.Array, placed: Placement) -> jax.Array:
    if block.shape[1] != window(placed.leaf):
        raise ValueError(
            f"leaf {placed.index}: the source yields windows of {block.shape[1]} ticks but the term's innermost "
            f"Scan is {window(placed.leaf)}; the task's sequence length must be a multiple of that Scan"
        )
    return block


def item(block: jax.Array, placed: Placement) -> jax.Array:
    axes = batched(placed.leaf.axes)
    tasks, examples = positions(placed)
    yields, time = block.shape[:2]
    split = block.reshape((yields, time) + tuple(size(axes[i]) for i in tasks + examples) + block.shape[4:])
    moved = jnp.moveaxis(split, list(range(2, 2 + len(axes))), tasks + examples)
    return moved.reshape(tuple(size(a) for a in axes) + (yields * time,) + block.shape[4:])


def leaf_stream(
    placed: Placement, datasets: list[PrematerializedTask], indices: jax.Array, key: PRNG, config: DataConfig
) -> Iterator[PyTree]:
    tasks, examples = counts(placed)
    _, _, k = jax.random.split(base_key(key, placed.index, placed.draw.seeding), 3)
    passes = map(
        lambda p: epoch(
            placed.draw,
            datasets,
            indices,
            tasks,
            examples,
            PRNG(jax.random.fold_in(k, p)),
            config.unlabeled_mask_value,
            config.label_mask_value,
        ),
        itertools.count(),
    )
    per_item = ticks((*placed.outer, *placed.leaf.axes)) // window(placed.leaf)
    return map(
        lambda block: jax.tree.map(lambda a: item(checked(a, placed), placed), block),
        rechunk_pytrees(passes, per_item),
    )


def stream(
    plan: Plan,
    sources: Sources,
    data: list[list[PrematerializedTask]],
    indices: jax.Array,
    key: PRNG,
    index: int,
    chunk: int,
    outer: tuple[Axis, ...],
    config: DataConfig,
) -> Iterator[PyTree]:
    match plan, sources:
        case Leaf() as leaf, Draw() as draw:
            return leaf_stream(Placement(leaf, draw, index, chunk, outer), data[index], indices, key, config)
        case Pair(axes, left, right), Level(train, val, replicas):
            below = (*outer, *axes)
            shape = tuple(size(a) for a in batched(axes))
            n = math.prod(shape)
            sub = narrow(replicas, chunk, n)
            return gather(
                replicas,
                shape,
                [
                    zip(
                        stream(left, train, data, c, k, index, sub, below, config),
                        stream(right, val, data, c, k, index + count(left), sub, below, config),
                    )
                    for c, k in replicate(replicas, n, indices, key)
                ],
            )
        case _:
            raise ValueError(f"sources {sources} do not mirror plan {plan}")


def create_loader(
    config: DataConfig, plan: Plan, data: list[list[PrematerializedTask]], key: PRNG, task_key: PRNG
) -> Iterator[PyTree]:
    perm = jax.random.permutation(task_key, config.num_tasks)
    return stream(plan, config.sources, data, perm, key, 0, config.num_tasks, (), config)


def yields_per_epoch(config: DataConfig, plan: Plan, data: list[list[PrematerializedTask]]) -> list[int]:
    out: list[int] = []
    for p in placements(plan, config.sources, 0, config.num_tasks, ()):
        tasks, examples = counts(p)
        first = data[p.index][0]
        num_mb = math.ceil(first.xs.shape[0] / examples)
        num_vb, time = first.x_epoch(first.xs[0], PRNG(jax.random.key(0))).shape[:2]
        per_pass = (p.chunk // tasks) * num_mb * num_vb * time
        per_yield = ticks((*p.outer, *p.leaf.axes))
        if per_pass % per_yield != 0:
            raise ValueError(
                f"leaf {p.index}: ticks per pass ({per_pass}) is not divisible by ticks per yield ({per_yield}); "
                f"an epoch boundary will not align with yield boundaries"
            )
        out.append(per_pass // per_yield)
    return out
