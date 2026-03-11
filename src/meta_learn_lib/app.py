import copy
from logging import Logger
import random
import os
import jax
import jax.numpy as jnp
import torch
import numpy as np
import toolz
import math
import time
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.create_env import create_env, create_inference_axes, create_transition_fns
from meta_learn_lib.create_interface import create_learn_interfaces, create_meta_interfaces, create_task_interfaces
from meta_learn_lib.env import *
from meta_learn_lib.inference import create_inference_and_readout
from meta_learn_lib.learning import create_meta_learner
from meta_learn_lib.lib_types import *
from meta_learn_lib.datasets import create_data_sources, create_dataloader, validate_dataloader_config
from meta_learn_lib.logger import MatplotlibLogger, create_logger
from meta_learn_lib.loss_function import create_readout_loss_fns


def runApp(config: GodConfig, loggers: list[Logger]) -> None:

    if not config.clearml_run:
        return

    # Count total num iterations
    base = config.levels[0]
    num_vb = math.ceil(base.dataset.num_examples_total / base.dataset.num_examples_in_minibatch)
    nesting = math.prod(l.nested.num_steps for l in config.levels)
    iters_per_epoch = (base.dataset.num_examples_total * num_vb) // (base.dataset.num_examples_in_minibatch * nesting)
    total_iterations = iters_per_epoch * config.epochs

    # Logger
    scalar_logger = create_logger(
        loggers,
        len(config.levels),
        total_iterations,
        config.checkpoint_every_n_minibatches,
        config.log_title,
    )

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

    # Dataset
    dataset_prng, data_loader_prng = jax.random.split(dataset_gen_prng, 2)
    data_sources, shapes = create_data_sources(config, dataset_prng)
    dataloader = create_dataloader(config, data_sources, data_loader_prng, task_prng)

    meta_interfaces, count = create_meta_interfaces(config, 0)
    learn_interfaces, count = create_learn_interfaces(config, count)
    task_interfaces, count = create_task_interfaces(config, count)

    env = create_env(config, shapes, meta_interfaces, learn_interfaces, env_prng)

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

    x, dataloader = toolz.peek(dataloader)
    env_copy = copy.deepcopy(env)
    arr_copy, static = eqx.partition(env_copy, eqx.is_array)
    arr, _ = eqx.partition(env, eqx.is_array)

    def update_fn(data, arr):
        env = eqx.combine(arr, static)
        env, stat = meta_learner(env, data)
        arr, _ = eqx.partition(env, eqx.is_array)
        return arr, stat

    meta_learner_compiled = eqx.filter_jit(update_fn, donate="all-except-first").lower(x, arr_copy).compile()

    for k, data in enumerate(toolz.take(total_iterations, dataloader)):
        start_time = time.time()
        arr, stats = meta_learner_compiled(data, arr)
        jax.block_until_ready(arr)
        end_time = time.time()
        print(f"Iteration {k + 1}/{total_iterations} took {end_time - start_time:.2f} seconds")
        scalar_logger.log(stats)

    scalar_logger.shutdown()

    for logger in scalar_logger.loggers:
        match logger:
            case MatplotlibLogger():
                logger.generate_figures()
