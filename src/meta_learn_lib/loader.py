from meta_learn_lib.construct.data import Axis, Leaf, Pair, Plan, size, windowed
from meta_learn_lib.experiment import DataConfig, Source, Sources, Task
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.tasks import (
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
from torch.utils.data import Dataset


def infinite_keys(key: PRNG) -> Iterator[PRNG]:
    while True:
        key, subkey = jax.random.split(key)
        yield PRNG(subkey)


@dataclass(frozen=True)
class Site:
    outer: tuple[Axis, ...]
    axes: tuple[Axis, ...]
    source: Source


def ticks(axes: tuple[Axis, ...]) -> int:
    return math.prod(size(a) for a in axes if windowed(a))


def batches(axes: tuple[Axis, ...]) -> tuple[int, ...]:
    return tuple(size(a) for a in axes if not windowed(a))


def window(axes: tuple[Axis, ...]) -> int:
    windows = [size(a) for a in axes if windowed(a)]
    return windows[-1] if windows else 1


def sites(plan: Plan, sources: Sources, outer: tuple[Axis, ...]) -> list[Site]:
    match plan, sources:
        case Leaf(axes), Source() as source:
            return [Site(outer, axes, source)]
        case Pair(axes, left, right), (first, second):
            return sites(left, first, (*outer, *axes)) + sites(right, second, (*outer, *axes))
        case _:
            raise ValueError(f"sources {sources} do not mirror plan {plan}")


def chunked(site: Site) -> int:
    return math.prod(batches(site.outer))


def examples(site: Site) -> int:
    product = math.prod(batches(site.axes))
    if product % site.source.tasks != 0:
        raise ValueError(
            f"the leaf batches {product} examples, not a multiple of its {site.source.tasks} tasks per stream"
        )
    return product // site.source.tasks


def validate(config: DataConfig, plan: Plan) -> list[str]:
    errors: list[str] = []
    for i, site in enumerate(sites(plan, config.sources, ())):
        if config.num_tasks % chunked(site) != 0:
            errors.append(
                f"leaf {i}: num_tasks ({config.num_tasks}) not divisible by its chunk product ({chunked(site)})"
            )
            continue
        if (config.num_tasks // chunked(site)) % site.source.tasks != 0:
            errors.append(
                f"leaf {i}: chunk size ({config.num_tasks // chunked(site)}) not divisible by tasks per stream "
                f"({site.source.tasks})"
            )
        if math.prod(batches(site.axes)) % site.source.tasks != 0:
            errors.append(
                f"leaf {i}: batches {math.prod(batches(site.axes))} examples, not a multiple of tasks per stream "
                f"({site.source.tasks})"
            )
    return errors


def create_sources(config: DataConfig, plan: Plan, prng: PRNG) -> tuple[PyTree, PyTree]:
    k1, k2, _ = jax.random.split(prng, 3)
    leaves = sites(plan, config.sources, ())

    def keyed(k: PRNG) -> list[tuple[Site, PRNG]]:
        keys = jax.random.split(k, len(leaves))
        return [
            (site, PRNG(jax.random.key(site.source.test_seed)) if site.source.is_test else PRNG(key))
            for site, key in zip(leaves, keys)
        ]

    pairs = {(site.source.task, site.source.is_test): key for site, key in keyed(k1)}
    remaining: dict[tuple[Task, bool], list[tuple[Dataset, Callable[[jax.Array], jax.Array]]]] = {
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


def epoch(
    indices: jax.Array,
    datasets: list[PrematerializedTask],
    tasks: int,
    per_task: int,
    shuffle: bool,
    key: PRNG,
    x_mask: float,
    y_mask: float,
) -> tuple[jax.Array, jax.Array]:
    groups = len(indices) // tasks
    keys = jax.random.split(key, len(indices))
    per_x, per_y = zip(
        *[
            task_epoch_tensor(datasets[idx], per_task, x_mask, y_mask, PRNG(k), shuffle)
            for idx, k in zip(indices.tolist(), keys)
        ]
    )
    return regroup_leading(jnp.stack(per_x), groups, tasks), regroup_leading(jnp.stack(per_y), groups, tasks)


def passes(site: Site, datasets: list[PrematerializedTask], indices: jax.Array, key: PRNG, config: DataConfig):
    return map(
        lambda k: epoch(
            indices,
            datasets,
            site.source.tasks,
            examples(site),
            site.source.shuffle,
            PRNG(k),
            config.unlabeled_mask_value,
            config.label_mask_value,
        ),
        infinite_keys(key),
    )


def checked(block: jax.Array, axes: tuple[Axis, ...]) -> jax.Array:
    if block.shape[1] != window(axes):
        raise ValueError(
            f"the source yields windows of {block.shape[1]} ticks but the leaf's innermost Scan is {window(axes)}; "
            f"the task's sequence length must be a multiple of that Scan"
        )
    return block


def leaf_stream(
    site: Site, datasets: list[PrematerializedTask], indices: jax.Array, key: PRNG, config: DataConfig
) -> Iterator[tuple[jax.Array, jax.Array]]:
    units = (ticks(site.outer) * ticks(site.axes)) // window(site.axes)
    blocks = rechunk_pytrees(passes(site, datasets, indices, key, config), units)

    def place(block: jax.Array) -> jax.Array:
        flat = checked(block, site.axes).reshape((-1, block.shape[2] * block.shape[3]) + block.shape[4:])
        return jnp.moveaxis(flat, 1, 0).reshape(batches(site.axes) + flat.shape[:1] + flat.shape[2:])

    return map(lambda block: jax.tree.map(place, block), blocks)


def entering(plan: Plan, sources: Sources, key: PRNG) -> PRNG:
    match plan, sources:
        case Leaf(), Source(is_test=True, test_seed=test_seed):
            return PRNG(jax.random.split(jax.random.key(test_seed), 1)[0])
        case Leaf(), _:
            _, own = jax.random.split(key)
            return PRNG(jax.random.split(own, 1)[0])
        case _:
            return key


def stream(
    plan: Plan,
    sources: Sources,
    datasets: PyTree,
    indices: jax.Array,
    key: PRNG,
    outer: tuple[Axis, ...],
    config: DataConfig,
) -> Iterator[PyTree]:
    match plan, sources:
        case Leaf(axes), Source() as source:
            return leaf_stream(Site(outer, axes, source), datasets, indices, key, config)
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
            below = (*outer, *axes)
            lefts = [
                stream(left, first, first_datasets, chunk, entering(left, first, PRNG(k)), below, config)
                for chunk, k in zip(jnp.split(indices, n), child_keys)
            ]
            rights = [
                stream(right, second, second_datasets, chunk, PRNG(k), below, config)
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
    return stream(plan, config.sources, datasets, perm, entering(plan, config.sources, PRNG(k1)), (), config)


def ticks_per_pass(site: Site, datasets: list[PrematerializedTask], num_tasks: int) -> int:
    first = datasets[0]
    num_mb = math.ceil(first.xs.shape[0] / examples(site))
    num_vb, time = first.x_epoch(first.xs[0], PRNG(jax.random.key(0))).shape[:2]
    return (num_tasks // chunked(site) // site.source.tasks) * num_mb * num_vb * time


def yields_per_epoch(config: DataConfig, plan: Plan, datasets: PyTree, leaf: int) -> int:
    site = sites(plan, config.sources, ())[leaf]
    per_pass = ticks_per_pass(site, flatten(plan, datasets)[leaf], config.num_tasks)
    per_yield = ticks(site.outer) * ticks(site.axes)
    if per_pass % per_yield != 0:
        raise ValueError(
            f"leaf {leaf}: ticks per pass ({per_pass}) is not divisible by ticks per yield ({per_yield}); "
            f"an epoch boundary will not align with yield boundaries"
        )
    return per_pass // per_yield
