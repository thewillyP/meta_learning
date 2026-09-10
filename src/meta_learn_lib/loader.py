from meta_learn_lib.construct.data import Axis, Leaf, Pair, Plan, shared, size, windowed
from meta_learn_lib.experiment import DataConfig, Source, Sources, Task
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.tasks import (
    Examples,
    PrematerializedTask,
    augment,
    rechunk_pytrees,
    regroup_leading,
    dataset_sources,
    take_datasets,
    task_epoch_tensor,
)

from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def infinite_keys(key: PRNG) -> Iterator[PRNG]:
    while True:
        key, subkey = jax.random.split(key)
        yield PRNG(subkey)


@dataclass(frozen=True)
class Site:
    outer: tuple[Axis, ...]
    axes: tuple[Axis, ...]
    source: Source
    train: bool


def members(axes: tuple[Axis, ...]) -> tuple[int, ...]:
    return tuple(size(a) for a in axes if not windowed(a) and not shared(a))


def datas(axes: tuple[Axis, ...]) -> tuple[int, ...]:
    return tuple(size(a) for a in axes if shared(a))


def ticks(axes: tuple[Axis, ...]) -> int:
    return math.prod(size(a) for a in axes if windowed(a))


def batches(axes: tuple[Axis, ...]) -> tuple[int, ...]:
    return tuple(size(a) for a in axes if not windowed(a))


def window(axes: tuple[Axis, ...]) -> int:
    windows = [size(a) for a in axes if windowed(a)]
    return windows[-1] if windows else 1


def tasks_examples(axes: tuple[Axis, ...]) -> tuple[int, int]:
    match datas(axes):
        case ():
            return (1, 1)
        case (examples,):
            return (1, examples)
        case (tasks, examples):
            return (tasks, examples)
        case _:
            raise ValueError(f"a leaf carries at most two BatchData axes (tasks, examples); got {axes}")


def sites(plan: Plan, sources: Sources, outer: tuple[Axis, ...], train: bool) -> list[Site]:
    match plan, sources:
        case Leaf(axes), Source() as source:
            tasks_examples(axes)
            return [Site(outer, axes, source, train)]
        case Pair(axes, left, right), (first, second):
            return sites(left, first, (*outer, *axes), True) + sites(right, second, (*outer, *axes), False)
        case _:
            raise ValueError(f"sources {sources} do not mirror plan {plan}")


def chunked(site: Site) -> int:
    own = math.prod(members(site.axes)) if site.train else 1
    return math.prod(batches(site.outer)) * own


def validate(config: DataConfig, plan: Plan) -> list[str]:
    errors: list[str] = []
    for i, site in enumerate(sites(plan, config.sources, (), True)):
        tasks, _ = tasks_examples(site.axes)
        if config.num_tasks % chunked(site) != 0:
            errors.append(
                f"leaf {i}: num_tasks ({config.num_tasks}) not divisible by its chunk product ({chunked(site)})"
            )
            continue
        if (config.num_tasks // chunked(site)) % tasks != 0:
            errors.append(
                f"leaf {i}: chunk size ({config.num_tasks // chunked(site)}) not divisible by tasks ({tasks})"
            )
    return errors


def create_sources(config: DataConfig, plan: Plan, prng: PRNG) -> tuple[PyTree, PyTree]:
    k1, k2, _ = jax.random.split(prng, 3)
    leaves = sites(plan, config.sources, (), True)

    def keyed(k: PRNG) -> list[tuple[Site, PRNG]]:
        keys = jax.random.split(k, len(leaves))
        return [
            (site, PRNG(jax.random.key(site.source.test_seed)) if site.source.is_test else PRNG(key))
            for site, key in zip(leaves, keys)
        ]

    pairs = {(site.source.task, site.source.is_test): key for site, key in keyed(k1)}
    remaining: dict[tuple[Task, bool], list[tuple[Examples, Callable[[jax.Array], jax.Array]]]] = {
        (task, is_test): dataset_sources(task, config.root_dir, is_test, config.label_mask_value, config.num_tasks, key)
        for (task, is_test), key in pairs.items()
    }

    taken: list[list[PrematerializedTask]] = []
    for site, key in keyed(k2):
        pair = (site.source.task, site.source.is_test)
        datasets, remaining[pair] = take_datasets(
            seed=key,
            remaining=remaining[pair],
            n=site.source.num_examples_total,
            n_consume=window(site.axes),
            x_mask=config.unlabeled_mask_value,
            y_mask=config.label_mask_value,
            augment_fn=augment(site.source.task, site.source.augment),
            shuffle=site.source.shuffle,
        )
        taken.append(datasets)

    dummy = PRNG(jax.random.key(0))
    shapes = [
        (
            datasets[0].x_epoch(datasets[0].xs[0], dummy).shape[2:],
            datasets[0].y_epoch(datasets[0].ys[0], dummy).shape[2:],
        )
        for datasets in taken
    ]
    return treeify(plan, taken), treeify(plan, shapes)


def flatten(plan: Plan, tree: PyTree) -> list[PyTree]:
    match plan:
        case Leaf():
            return [tree]
        case Pair(_, left, right):
            first, second = tree
            return flatten(left, first) + flatten(right, second)


def count(plan: Plan) -> int:
    match plan:
        case Leaf():
            return 1
        case Pair(_, left, right):
            return count(left) + count(right)


def treeify[T](plan: Plan, leaves: list[T]) -> PyTree:
    match plan:
        case Leaf():
            (leaf,) = leaves
            return leaf
        case Pair(_, left, right):
            n = count(left)
            return (treeify(left, leaves[:n]), treeify(right, leaves[n:]))


def task_stream(
    indices: jax.Array,
    datasets: list[PrematerializedTask],
    tasks: int,
    examples: int,
    shuffle: bool,
    key: PRNG,
    x_mask: float,
    y_mask: float,
) -> tuple[jax.Array, jax.Array]:
    groups = len(indices) // tasks
    keys = jax.random.split(key, len(indices))
    per_x, per_y = zip(
        *[
            task_epoch_tensor(datasets[idx], examples, x_mask, y_mask, PRNG(k), shuffle)
            for idx, k in zip(indices.tolist(), keys)
        ]
    )
    return regroup_leading(jnp.stack(per_x), groups, tasks), regroup_leading(jnp.stack(per_y), groups, tasks)


def passes(
    indices: jax.Array,
    datasets: list[PrematerializedTask],
    axes: tuple[Axis, ...],
    source: Source,
    key: PRNG,
    config: DataConfig,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    tasks, examples = tasks_examples(axes)
    return map(
        lambda k: task_stream(
            indices,
            datasets,
            tasks,
            examples,
            source.shuffle,
            PRNG(k),
            config.unlabeled_mask_value,
            config.label_mask_value,
        ),
        infinite_keys(key),
    )


def arrange(block: jax.Array, axes: tuple[Axis, ...], leading: tuple[int, ...]) -> jax.Array:
    lead = len(leading)
    moved = jnp.moveaxis(block, lead, lead + 2)
    a = moved.reshape(moved.shape[:lead] + datas(axes) + moved.shape[lead + 2 :])
    ints = [i for i, ax in enumerate(axes) if not windowed(ax)]
    canonical = [j for j, i in enumerate(ints) if not shared(axes[i])] + [
        j for j, i in enumerate(ints) if shared(axes[i])
    ]
    return jnp.moveaxis(a, list(range(len(canonical))), canonical)


def flatten_units(block: jax.Array, lead: int) -> jax.Array:
    return block.reshape(block.shape[:lead] + (-1,) + block.shape[lead + 2 :])


def checked(block: jax.Array, axes: tuple[Axis, ...]) -> jax.Array:
    if block.shape[1] != window(axes):
        raise ValueError(
            f"the source yields windows of {block.shape[1]} ticks but the leaf's innermost Scan is {window(axes)}; "
            f"the task's sequence length must be a multiple of that Scan"
        )
    return block


def train_leaf(
    axes: tuple[Axis, ...],
    source: Source,
    datasets: list[PrematerializedTask],
    indices: jax.Array,
    key: PRNG,
    outer: int,
    config: DataConfig,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    _, key = jax.random.split(key)
    key = PRNG(jax.random.key(source.test_seed)) if source.is_test else PRNG(key)
    chunks = members(axes)
    n = math.prod(chunks)
    keys = jax.random.split(key, n)
    units = (outer * ticks(axes)) // window(axes)
    streams = [
        rechunk_pytrees(passes(chunk, datasets, axes, source, PRNG(k), config), units)
        for chunk, k in zip(jnp.split(indices, n), keys)
    ]

    def place(blocks: tuple[jax.Array, ...]) -> jax.Array:
        stacked = jnp.stack([checked(b, axes) for b in blocks]).reshape(chunks + blocks[0].shape)
        return arrange(flatten_units(stacked, len(chunks)), axes, chunks)

    return map(lambda blocks: jax.tree.map(lambda *bs: place(bs), *blocks), zip(*streams))


def val_leaf(
    axes: tuple[Axis, ...],
    source: Source,
    datasets: list[PrematerializedTask],
    indices: jax.Array,
    key: PRNG,
    outer: int,
    config: DataConfig,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    deals = members(axes)
    n = math.prod(deals)
    per_member = (outer * ticks(axes)) // window(axes)
    stream = rechunk_pytrees(passes(indices, datasets, axes, source, key, config), per_member * n)

    def place(block: jax.Array) -> jax.Array:
        dealt = checked(block, axes).reshape((per_member,) + deals + block.shape[1:])
        moved = jnp.moveaxis(dealt, 0, len(deals))
        return arrange(flatten_units(moved, len(deals)), axes, deals)

    return map(lambda block: jax.tree.map(place, block), stream)


def stream(
    plan: Plan,
    sources: Sources,
    datasets: PyTree,
    indices: jax.Array,
    key: PRNG,
    outer: int,
    config: DataConfig,
    train: bool,
) -> Iterator[PyTree]:
    match plan, sources:
        case Leaf(axes), Source() as source:
            make = train_leaf if train else val_leaf
            return make(axes, source, datasets, indices, key, outer, config)
        case Pair(axes, left, right), (first, second):
            chunks = batches(axes)
            n = math.prod(chunks)
            child_key, val_key = jax.random.split(key)
            match right, second:
                case Leaf(), Source(is_test=True, test_seed=test_seed):
                    val_key = jax.random.key(test_seed)
                case _:
                    pass
            val_keys = jax.random.split(val_key, n)
            child_keys = jax.random.split(child_key, n)
            first_datasets, second_datasets = datasets
            below = outer * ticks(axes)
            lefts = [
                stream(left, first, first_datasets, chunk, PRNG(k), below, config, True)
                for chunk, k in zip(jnp.split(indices, n), child_keys)
            ]
            rights = [
                stream(right, second, second_datasets, chunk, PRNG(k), below, config, False)
                for chunk, k in zip(jnp.split(indices, n), val_keys)
            ]

            def gather(items: tuple[PyTree, ...]) -> PyTree:
                return jax.tree.map(lambda *xs: jnp.stack(xs).reshape(chunks + jnp.shape(xs[0])), *items)

            def both(item: tuple[tuple[PyTree, ...], tuple[PyTree, ...]]) -> tuple[PyTree, PyTree]:
                first_items, second_items = item
                return (gather(first_items), gather(second_items))

            return map(both, zip(zip(*lefts), zip(*rights)))
        case _:
            raise ValueError(f"sources {sources} do not mirror plan {plan}")


def create_loader(config: DataConfig, plan: Plan, datasets: PyTree, prng: PRNG, task_prng: PRNG) -> Iterator[PyTree]:
    k1, _ = jax.random.split(prng, 2)
    perm = jax.random.permutation(task_prng, config.num_tasks)
    return stream(plan, config.sources, datasets, perm, PRNG(k1), 1, config, True)


def ticks_per_pass(site: Site, datasets: list[PrematerializedTask], num_tasks: int) -> int:
    tasks, examples = tasks_examples(site.axes)
    groups = (num_tasks // chunked(site)) // tasks
    first = datasets[0]
    num_mb = math.ceil(first.xs.shape[0] / examples)
    num_vb, time = first.x_epoch(first.xs[0], PRNG(jax.random.key(0))).shape[:2]
    return groups * num_mb * num_vb * time


def yields_per_epoch(config: DataConfig, plan: Plan, datasets: PyTree, leaf: int) -> int:
    site = sites(plan, config.sources, (), True)[leaf]
    per_pass = ticks_per_pass(site, flatten(plan, datasets)[leaf], config.num_tasks)
    per_yield = ticks(site.outer) * ticks(site.axes)
    if per_pass % per_yield != 0:
        raise ValueError(
            f"leaf {leaf}: ticks per pass ({per_pass}) is not divisible by ticks per yield ({per_yield}); "
            f"an epoch boundary will not align with yield boundaries"
        )
    return per_pass // per_yield
