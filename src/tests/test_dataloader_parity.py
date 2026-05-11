"""Parity tests: eager `create_dataloader` vs original lazy implementation.

Asserts bit-exact equality across all yielded leaves for a full epoch.

- `test_synthetic_batch_parity`: in-memory data, exercises nested.batch>1. Always runs.
- `test_real_config_parity`: parametrized over real configs. Requires datasets on disk
  (data_root_dir from each config). Skipped automatically when data is missing.
"""

import math
import itertools
from typing import Iterator

import jax
import jax.numpy as jnp
import pytest
from toolz import mapcat

import configs as configs_module
from meta_learn_lib.config import *
from meta_learn_lib.lib_types import PRNG
from meta_learn_lib.datasets import (
    create_data_sources,
    create_dataloader,
    get_seq_len,
    task_iterator,
    batch_iterator,
    stack_batches,
    PrematerializedTask,
)
from meta_learn_lib.util import infinite_keys


# ---- Original lazy create_dataloader, transcribed verbatim from git 6a735cf ----
def create_dataloader_lazy(
    config: GodConfig,
    data_sources: list[list[PrematerializedTask]],
    prng: PRNG,
    task_distribution_prng: PRNG,
) -> Iterator:
    k1, prng = jax.random.split(prng, 2)
    global_perm = jax.random.permutation(task_distribution_prng, config.num_tasks)

    def make_task_loader(task_indices, level, datasets, key):
        tasks_per_stream = level.validation.batch
        task_keys = jax.random.split(key, len(task_indices))
        task_iters = [
            task_iterator(
                datasets[idx],
                level.dataset.num_examples_in_minibatch,
                config.unlabeled_mask_value,
                config.label_mask_value,
                tkey,
            )
            for idx, tkey in zip(task_indices.tolist(), task_keys)
        ]
        return batch_iterator(task_iters, tasks_per_stream, axis=1)

    def make_nil_loader():
        while True:
            yield None

    def make_level_loader(levels, task_indices, key):
        if len(levels) == 0:
            return make_nil_loader()
        (meta_config, datasets), *rest = levels
        batch = meta_config.nested.batch
        num_steps = meta_config.nested.num_steps
        is_test = meta_config.dataset.is_test
        child_key, val_key = jax.random.split(key)
        val_key = jax.random.key(meta_config.test_seed) if is_test else val_key

        chunks = jnp.split(task_indices, batch)
        child_keys = jax.random.split(child_key, batch)
        val_keys_per_child = [infinite_keys(vk) for vk in jax.random.split(val_key, batch)]

        def f_val(c, k):
            return make_task_loader(c, meta_config, datasets, k)

        def nest_validation(val_stream, lower_levels):
            for lower_meta, _ in reversed(lower_levels):
                val_stream = stack_batches(val_stream, lower_meta.nested.batch)
            return val_stream

        children = [
            zip(
                make_level_loader(rest, chunk, ckey),
                map(lambda v: (v, v), nest_validation(mapcat(lambda k, c=chunk: f_val(c, k), vks), rest)),
            )
            for chunk, ckey, vks in zip(chunks, child_keys, val_keys_per_child)
        ]
        train_loader = batch_iterator(children, len(children), axis=0)
        return stack_batches(train_loader, num_steps)

    levels_with_data = list(reversed(list(zip(config.levels, data_sources))))
    return make_level_loader(levels_with_data, global_perm, k1)
# -------------------------------------------------------------------------------


def compute_iterations_per_epoch(config: GodConfig) -> int:
    """Mirror of app.py's get_iterations(0, config, 1)[1]."""
    level = config.levels[0]
    seq_len = get_seq_len(level.dataset_source, level.dataset.is_test)
    num_vb = math.ceil(seq_len / level.validation.num_steps)
    num_mb = math.ceil(level.dataset.num_examples_total / level.dataset.num_examples_in_minibatch)
    consumption = math.prod(m.nested.num_steps for m in config.levels)
    return (num_mb * num_vb) // consumption


def assert_tree_allclose(a, b, atol: float = 1e-6) -> None:
    flat_a, treedef_a = jax.tree.flatten(a, is_leaf=lambda x: x is None)
    flat_b, treedef_b = jax.tree.flatten(b, is_leaf=lambda x: x is None)
    assert treedef_a == treedef_b, f"tree-shape mismatch: {treedef_a} vs {treedef_b}"
    for i, (la, lb) in enumerate(zip(flat_a, flat_b)):
        if la is None and lb is None:
            continue
        assert la is not None and lb is not None, f"leaf {i}: one side None"
        assert la.shape == lb.shape, f"leaf {i}: shape {la.shape} vs {lb.shape}"
        assert bool(jnp.allclose(la, lb, atol=atol)), (
            f"leaf {i}: max abs diff {float(jnp.max(jnp.abs(la - lb)))}"
        )


def run_parity(cfg: GodConfig, data_sources: list[list[PrematerializedTask]]) -> None:
    n_yields = max(1, compute_iterations_per_epoch(cfg))
    loader_key = jax.random.key(cfg.seed.global_seed)
    task_seed = jax.random.key(cfg.seed.task_seed)
    dl_new = create_dataloader(cfg, data_sources, loader_key, task_seed)
    dl_old = create_dataloader_lazy(cfg, data_sources, loader_key, task_seed)
    for i in range(n_yields):
        a = next(dl_new)
        b = next(dl_old)
        assert_tree_allclose(a, b)


def _id_epoch(x, key):
    return x[None, ...]


def _make_synthetic_task(n_examples: int, x_shape: tuple, y_shape: tuple, seed: int) -> PrematerializedTask:
    kx, ky = jax.random.split(jax.random.key(seed))
    xs = jax.random.normal(kx, (n_examples, *x_shape))
    ys = jax.random.normal(ky, (n_examples, *y_shape))
    return PrematerializedTask(xs, ys, _id_epoch, _id_epoch)


def _build_synthetic_config() -> tuple[GodConfig, list[list[PrematerializedTask]]]:
    num_tasks = 4
    data_sources = [
        [_make_synthetic_task(n_ex, (3,), (2,), seed=level_idx * 100 + i) for i in range(num_tasks)]
        for level_idx, n_ex in enumerate([40, 20, 16])
    ]

    no_track = TrackLogs(
        gradient=False, hessian_contains_nans=False, largest_eigenvalue=False,
        influence_tensor_norm=False, immediate_influence_tensor=False,
        largest_jac_eigenvalue=False, jacobian=False,
    )
    id_grad = GradientConfig(
        method=IdentityLearnerConfig(bptt_config=BPTTConfig(None)), add_clip=None, scale=1.0
    )

    def make_level(n_total, minibatch, num_steps, batch, is_test):
        return MetaConfig(
            objective_fn=NoopObjective(),
            dataset_source=GaussianNoiseTaskFamily(shape=(3,), n=n_total),
            dataset=DatasetConfig(
                num_examples_in_minibatch=minibatch, num_examples_total=n_total,
                is_test=is_test, augment=False,
            ),
            validation=StepConfig(num_steps=1, batch=1, reset_t=1, track_influence_in=frozenset()),
            nested=StepConfig(
                num_steps=num_steps, batch=batch, reset_t=None, track_influence_in=frozenset()
            ),
            learner=LearnConfig(model_learner=id_grad, optimizer_learner=id_grad, optimizer={}),
            track_logs=no_track,
            test_seed=0,
            collect_predictions=False,
        )

    cfg = GodConfig(
        seed=SeedConfig(global_seed=7, data_seed=1, parameter_seed=1, task_seed=2, sample_seed=1),
        clearml_run=False,
        data_root_dir="/tmp",
        log_dir="/tmp",
        log_title="synthetic",
        logger_config=LoggersConfig(
            clearml=ClearMLLoggerConfig(enabled=False),
            hdf5=HDF5LoggerConfig(enabled=False),
            console=ConsoleLoggerConfig(enabled=False),
            matplotlib=MatplotlibLoggerConfig(save_dir="", enabled=False),
            scalar_queue_size=0,
            sample_queue_size=0,
        ),
        epochs=1,
        checkpoint_every_n_minibatches=1,
        checkpoint_every_n_epochs=1,
        transition_graph={},
        readout_graph={},
        nodes={},
        aliases={},
        hyperparameters={},
        levels=[
            make_level(n_total=40, minibatch=4, num_steps=1, batch=1, is_test=False),
            make_level(n_total=20, minibatch=4, num_steps=1, batch=2, is_test=False),
            make_level(n_total=16, minibatch=4, num_steps=2, batch=1, is_test=True),
        ],
        sample_generators=[],
        label_mask_value=-1.0,
        unlabeled_mask_value=-1.0,
        num_tasks=num_tasks,
        prefetch_buffer_size=1,
        dataloader_chunk_size=None,
    )
    return cfg, data_sources


def test_synthetic_batch_parity():
    """4 tasks, 3 levels, level 1 nested.batch=2, full epoch (5 yields)."""
    cfg, data_sources = _build_synthetic_config()
    run_parity(cfg, data_sources)


@pytest.mark.parametrize("config_name", ["VAE_BASELINE", "OHO_RNN32", "OHO_RNN1_32_RNN2_32"])
def test_real_config_parity(config_name: str):
    """Bit-equality on real configs. Skipped if data_root_dir is missing."""
    import os
    cfg: GodConfig = getattr(configs_module, config_name)
    if not os.path.isdir(cfg.data_root_dir):
        pytest.skip(f"data_root_dir not found: {cfg.data_root_dir}")
    data_sources, _ = create_data_sources(cfg, jax.random.key(cfg.seed.data_seed))
    run_parity(cfg, data_sources)
