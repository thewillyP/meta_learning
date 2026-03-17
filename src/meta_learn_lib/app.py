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
import time
import equinox as eqx
import threading
import queue

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


def get_iterations(level_idx: int, config: GodConfig, epochs: int) -> int:
    level = config.levels[level_idx]
    seq_len = get_seq_len(level.dataset_source, level.dataset.is_test)
    num_vb = math.ceil(seq_len / level.validation.num_steps)
    num_mb = math.ceil(level.dataset.num_examples_total / level.dataset.num_examples_in_minibatch)
    consumption = math.prod(config.levels[i].nested.num_steps for i in range(level_idx, len(config.levels)))
    return (num_mb * num_vb * epochs) // consumption


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
    total_iterations: int,
    scalar_logger: ThreadedScalarLogger,
    stat_collector: Callable[[STAT, tuple[STAT, ...]], tuple[STAT, ...]],
    stat_prefix: str,
) -> tuple[GodState, tuple[STAT, ...]]:
    dataset_prng, loader_prng = jax.random.split(data_prng, 2)
    data_sources, shapes = create_data_sources(config, dataset_prng)
    dataloader = create_dataloader(config, data_sources, loader_prng, task_prng)
    x, dataloader = toolz.peek(dataloader)
    dataloader = prefetch(toolz.take(total_iterations, dataloader), buffer_size=2)

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

    def update_fn(data: tuple, arr: GodState) -> tuple[GodState, STAT]:
        e = eqx.combine(arr, static)
        e, stat = meta_learner(e, data)
        a, _ = eqx.partition(e, eqx.is_array)
        return a, stat

    compiled = eqx.filter_jit(update_fn, donate="all-except-first").lower(x, arr).compile()

    collected: tuple[STAT, ...] = ()
    try:
        t_prev = time.time()
        for k, data in enumerate(dataloader):
            t_data = time.time()
            arr, stats = compiled(data, arr)
            jax.block_until_ready(arr)
            t_compute = time.time()
            collected = stat_collector(stats, collected)
            scalar_logger.log(prefix_stats(stats, stat_prefix))
            t_log = time.time()
            print(f"data: {t_data - t_prev:.3f}  compute+sync: {t_compute - t_data:.3f}  log: {t_log - t_compute:.3f}")
            t_prev = t_log
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down logger...")
    finally:
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


def runApp(config: GodConfig, loggers: list[Logger]) -> None:

    if not config.clearml_run:
        return

    # RNG Stuff
    base_key = jax.random.key(config.seed.global_seed)
    keys = jax.random.split(base_key, 3)
    data_prng = PRNG(jax.random.fold_in(keys[0], config.seed.data_seed))
    env_prng = PRNG(jax.random.fold_in(keys[1], config.seed.parameter_seed))
    task_prng = PRNG(jax.random.fold_in(keys[2], config.seed.task_seed))
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
    if errors:
        raise ValueError("\n".join(errors))

    # Train
    train_iterations = get_iterations(0, config, config.epochs)
    train_logger = create_logger(
        loggers,
        len(config.levels),
        train_iterations,
        config.checkpoint_every_n_minibatches,
        config.log_title,
    )

    trained_env, _ = run(
        config,
        lambda cfg, shapes, mi, li: create_env(cfg, shapes, mi, li, env_prng),
        dataset_gen_prng,
        task_prng,
        train_iterations,
        train_logger,
        lambda s, acc: acc,
        "train",
    )

    # Evaluate on last level's dataset
    eval_config = make_eval_config(config)
    last_idx = len(eval_config.levels) - 1
    eval_iterations = get_iterations(last_idx, eval_config, 1)
    eval_logger = create_logger(
        loggers,
        len(eval_config.levels),
        eval_iterations,
        config.checkpoint_every_n_minibatches,
        config.log_title,
    )

    _, eval_stats = run(
        eval_config,
        lambda *_: trained_env,
        dataset_gen_prng,
        task_prng,
        eval_iterations,
        eval_logger,
        lambda s, acc: acc + (s,),
        "eval",
    )

    accumulated = jax.tree.map(lambda *xs: jnp.nanmean(jnp.stack(xs), axis=0), *eval_stats)
    accumulated = prefix_stats(accumulated, "eval_accumulated")
    acc_logger = create_logger(loggers, len(eval_config.levels), 1, 1, config.log_title)
    acc_logger.log(accumulated)
    acc_logger.flush()
    acc_logger.shutdown()
