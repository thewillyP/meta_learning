import dataclasses
from logging import Logger
import random
import os
from typing import Callable, Iterator
import jax
import jax.numpy as jnp
import torch
import numpy as np
import toolz
import math
import equinox as eqx
import threading
import queue

from meta_learn_lib.checkpoint import CheckpointManager, CheckpointMetadata, NullCheckpointManager
from meta_learn_lib.config import *
from meta_learn_lib.create_env import create_env, create_inference_axes, create_transition_fns
from meta_learn_lib.create_interface import create_learn_interfaces, create_meta_interfaces, create_task_interfaces
from meta_learn_lib.env import *
from meta_learn_lib.inference import create_inference_and_readout
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.learning import create_meta_learner
from meta_learn_lib.lib_types import *
from meta_learn_lib.datasets import create_data_sources, create_dataloader, get_seq_len, validate_dataloader_config
from meta_learn_lib.logger import MatplotlibLogger, ThreadedScalarLogger, create_logger
from meta_learn_lib.loss_function import create_readout_loss_fns
from meta_learn_lib.sample import build_sample_runner, validate_sample_generators


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def prefetch[T](iterator: Iterator[T], buffer_size: int) -> Iterator[T]:
    q: queue.Queue[T | BaseException | object] = queue.Queue(maxsize=buffer_size)
    sentinel: object = object()

    def fill() -> None:
        try:
            for item in iterator:
                q.put(item)
        except Exception as e:
            q.put(e)
        q.put(sentinel)

    threading.Thread(target=fill, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def get_consumption(level_idx: int, config: GodConfig) -> int:
    return math.prod(config.levels[i].nested.num_steps for i in range(level_idx, len(config.levels)))


def get_iterations(level_idx: int, config: GodConfig, epochs: int) -> tuple[int, int, int]:
    """Returns (total_iterations, iterations_per_epoch, logger_capacity).

    Raises ValueError if steps_per_epoch is not divisible by consumption,
    which would mean 1 epoch does not correspond to an integer number of
    dataloader iterations.
    """
    level = config.levels[level_idx]
    seq_len = get_seq_len(level.dataset_source, level.dataset.is_test)
    num_vb = math.ceil(seq_len / level.validation.num_steps)
    num_mb = math.ceil(level.dataset.num_examples_total / level.dataset.num_examples_in_minibatch)
    consumption = get_consumption(level_idx, config)
    steps_per_epoch = num_mb * num_vb
    if steps_per_epoch % consumption != 0:
        raise ValueError(
            f"Level {level_idx}: steps_per_epoch ({steps_per_epoch} = {num_mb} minibatches * {num_vb} validation batches) "
            f"is not divisible by consumption ({consumption} = product of nested.num_steps from level {level_idx} onward). "
            f"An epoch boundary will not align with iteration boundaries."
        )
    iterations_per_epoch = steps_per_epoch // consumption
    total_iterations = iterations_per_epoch * epochs
    total_steps = steps_per_epoch * epochs
    logger_capacity = total_steps // config.checkpoint_every_n_minibatches
    return total_iterations, iterations_per_epoch, logger_capacity


def make_eval_config(config: GodConfig) -> GodConfig:
    identity_grad = GradientConfig(
        method=IdentityLearnerConfig(),
        add_clip=None,
        scale=1.0,
    )
    new_levels: list[MetaConfig] = []
    for i, level in enumerate(config.levels):
        new_learner = dataclasses.replace(
            level.learner,
            model_learner=identity_grad,
            optimizer_learner=identity_grad,
        )
        new_level = dataclasses.replace(level, learner=new_learner)
        if i == len(config.levels) - 1:
            new_level = dataclasses.replace(
                new_level,
                nested=dataclasses.replace(new_level.nested, num_steps=1),
            )
        new_levels.append(new_level)
    return dataclasses.replace(config, levels=new_levels)


def prefix_stats(stats: STAT, prefix: str) -> STAT:
    return {f"{prefix}/{k}": v for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run(
    config: GodConfig,
    env_factory: Callable[
        [
            GodConfig,
            list[tuple[tuple[int, ...], tuple[int, ...]]],
            list[dict[str, GodInterface[GodState]]],
            list[tuple[GodInterface[GodState], GodInterface[GodState]]],
        ],
        GodState,
    ],
    data_prng: PRNG,
    task_prng: PRNG,
    sample_prng: PRNG,
    total_iterations: int,
    iterations_per_epoch: int,
    logger_capacity: int,
    loggers: list[Logger],
    stat_collector: Callable[[STAT, tuple[STAT, ...]], tuple[STAT, ...]],
    stat_prefix: str,
    checkpoint_manager: CheckpointManager,
) -> tuple[GodState, tuple[STAT, ...]]:
    dataset_prng, loader_prng = jax.random.split(data_prng, 2)
    data_sources, shapes = create_data_sources(config, dataset_prng)
    dataloader = create_dataloader(config, data_sources, loader_prng, task_prng)
    x, dataloader = toolz.peek(dataloader)

    meta_interfaces, count = create_meta_interfaces(config, 0)
    learn_interfaces, count = create_learn_interfaces(config, count)
    task_interfaces, count = create_task_interfaces(config, count)

    env = env_factory(config, shapes, meta_interfaces, learn_interfaces)

    val_learn_interfaces, nest_learn_interfaces = zip(*learn_interfaces)

    inference_axes = [
        create_inference_axes(env, config, meta_interface, shape, meta_config)
        for shape, meta_interface, meta_config in zip(shapes, meta_interfaces, config.levels)
    ]

    transitions, readouts = zip(
        *map(lambda i, a: create_inference_and_readout(config, i, a), meta_interfaces, inference_axes)
    )

    transition_fns = create_transition_fns(config, shapes, meta_interfaces, val_learn_interfaces, transitions)
    loss_fns = create_readout_loss_fns(config, task_interfaces, readouts, meta_interfaces)

    meta_learner = create_meta_learner(
        config,
        shapes,
        transition_fns,
        loss_fns,
        list(val_learn_interfaces),
        list(nest_learn_interfaces),
        meta_interfaces,
        env,
    )

    arr, static = eqx.partition(env, eqx.is_array)

    # Load checkpoint if available — fresh arr serves as structure template
    start_step = 0
    loaded = checkpoint_manager.load(arr)
    if loaded is not None:
        arr, meta = loaded
        start_step = meta.global_step
        print(f"Resumed from checkpoint at step {start_step} (epoch {start_step // iterations_per_epoch})")

    consumption = get_consumption(0, config)
    scalar_logger = create_logger(
        loggers,
        len(config.levels),
        logger_capacity,
        config.checkpoint_every_n_minibatches,
        config.log_title,
        start_step * consumption,
    )

    # Advance dataloader to resume point
    if start_step > 0:
        print(f"Skipping {start_step} dataloader iterations...")
        dataloader = toolz.drop(start_step, dataloader)
    remaining = total_iterations - start_step
    dataloader = prefetch(toolz.take(remaining, dataloader), buffer_size=config.prefetch_buffer_size)

    def update_fn(data: tuple, arr: GodState) -> tuple[GodState, STAT]:
        e = eqx.combine(arr, static)
        e, stat = meta_learner(e, data)
        a, _ = eqx.partition(e, eqx.is_array)
        return a, stat

    compiled = eqx.filter_jit(update_fn, donate="all-except-first").lower(x, arr).compile()

    checkpoint_interval = iterations_per_epoch * config.checkpoint_every_n_epochs
    sample_runner = build_sample_runner(config, meta_interfaces, data_sources, sample_prng, iterations_per_epoch)

    print("Starting main loop...")

    collected: tuple[STAT, ...] = ()
    for k, data in enumerate(dataloader):
        arr, stats = compiled(data, arr)
        jax.block_until_ready(arr)
        collected = stat_collector(stats, collected)
        scalar_logger.log(prefix_stats(stats, stat_prefix))

        global_step = start_step + k + 1
        if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
            checkpoint_manager.save(arr, CheckpointMetadata(global_step=global_step))
            print(f"Checkpoint saved at step {global_step} (epoch {global_step // iterations_per_epoch})")

        step_sample_prng, sample_prng = jax.random.split(sample_prng)
        sample_runner(eqx.combine(arr, static), scalar_logger, PRNG(step_sample_prng), global_step)

    scalar_logger.flush()
    scalar_logger.shutdown()

    for logger in scalar_logger.loggers:
        match logger:
            case MatplotlibLogger():
                logger.generate_figures()

    trained_env = eqx.combine(arr, static)
    return trained_env, collected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def runApp(config: GodConfig, loggers: list[Logger], checkpoint_manager: CheckpointManager) -> None:

    if not config.clearml_run:
        return

    # RNG Stuff
    base_key = jax.random.key(config.seed.global_seed)
    keys = jax.random.split(base_key, 3)
    data_prng = PRNG(jax.random.fold_in(keys[0], config.seed.data_seed))
    env_prng = PRNG(jax.random.fold_in(keys[1], config.seed.parameter_seed))
    task_prng = PRNG(jax.random.fold_in(keys[2], config.seed.task_seed))
    sample_prng = PRNG(jax.random.key(config.seed.sample_seed))
    dataset_gen_prng, torch_prng = jax.random.split(data_prng, 2)
    torch_seed = jax.random.randint(torch_prng, shape=(), minval=0, maxval=1e6, dtype=jnp.uint32)
    torch_seed = int(torch_seed)
    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    errors = validate_dataloader_config(config)
    sample_errors = validate_sample_generators(config)
    if errors or sample_errors:
        raise ValueError("\n".join(errors + sample_errors))

    # Train
    train_dl_iters, train_iters_per_epoch, train_log_cap = get_iterations(0, config, config.epochs)

    trained_env, _ = run(
        config,
        lambda cfg, shapes, mi, li: create_env(cfg, shapes, mi, li, env_prng),
        dataset_gen_prng,
        task_prng,
        sample_prng,
        train_dl_iters,
        train_iters_per_epoch,
        train_log_cap,
        loggers,
        lambda s, acc: acc,
        "train",
        checkpoint_manager,
    )

    # Evaluate on last level's dataset
    eval_config = make_eval_config(config)
    last_idx = len(eval_config.levels) - 1
    eval_dl_iters, eval_iters_per_epoch, eval_log_cap = get_iterations(last_idx, eval_config, 1)

    _, eval_stats = run(
        eval_config,
        lambda *_: trained_env,
        dataset_gen_prng,
        task_prng,
        sample_prng,
        eval_dl_iters,
        eval_iters_per_epoch,
        eval_log_cap,
        loggers,
        lambda s, acc: acc + (s,),
        "eval",
        NullCheckpointManager(),
    )

    accumulated = jax.tree.map(lambda *xs: jnp.nanmean(jnp.stack(xs), axis=0), *eval_stats)
    accumulated = prefix_stats(accumulated, "eval_accumulated")
    acc_logger = create_logger(loggers, len(eval_config.levels), 1, 1, config.log_title, 0)
    acc_logger.log(accumulated)
    acc_logger.flush()
    acc_logger.shutdown()
