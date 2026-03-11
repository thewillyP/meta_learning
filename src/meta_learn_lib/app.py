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

    config = GodConfig(
        seed=SeedConfig(global_seed=14, data_seed=1, parameter_seed=1, task_seed=1),
        clearml_run=True,
        data_root_dir="/scratch/wlp9800/datasets",
        log_dir="/scratch/wlp9800/offline_logs",
        logger_config=[ClearMLLoggerConfig()],
        epochs=100,
        checkpoint_every_n_minibatches=1,
        transition_graph={
            "x": {},
            "concat": {"x"},
            "rnn1": {"concat"},
            "rnn2": {"rnn1"},
        },
        readout_graph={
            "readout": {"rnn2"},
        },
        nodes={
            "x": UnlabeledSource(),
            "concat": Concat(),
            "rnn1": VanillaRNNLayer(
                nn_layer=NNLayer(
                    n=32,
                    activation_fn="tanh",
                    use_bias=True,
                    layer_norm=None,
                ),
                use_random_init=False,
                time_constant="meta1_rnn1_time_constant",
            ),
            "rnn2": VanillaRNNLayer(
                nn_layer=NNLayer(
                    n=32,
                    activation_fn="tanh",
                    use_bias=True,
                    layer_norm=None,
                ),
                use_random_init=False,
                time_constant="meta1_rnn2_time_constant",
            ),
            "readout": NNLayer(
                n=10,
                activation_fn="identity",
                use_bias=True,
                layer_norm=None,
            ),
        },
        hyperparameters={
            "meta1_rnn1_time_constant": HyperparameterConfig(
                value=1.0,
                kind="time_constant",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_rnn2_time_constant": HyperparameterConfig(
                value=1.0,
                kind="time_constant",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_sgd1.lr": HyperparameterConfig(
                value=0.001,
                kind="learning_rate",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_sgd1.wd": HyperparameterConfig(
                value=0.00001,
                kind="weight_decay",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_sgd1.momentum": HyperparameterConfig(
                value=0.0,
                kind="momentum",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=1,
                parametrizes_transition=True,
            ),
            "meta2_adam1.lr": HyperparameterConfig(
                value=0.001,
                kind="learning_rate",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=2,
                parametrizes_transition=True,
            ),
            "meta2_adam1.wd": HyperparameterConfig(
                value=0.0,
                kind="weight_decay",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=2,
                parametrizes_transition=True,
            ),
            "meta2_adam1.momentum": HyperparameterConfig(
                value=0.9,
                kind="momentum",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=2,
                parametrizes_transition=True,
            ),
        },
        levels=[
            MetaConfig(
                objective_fn=CrossEntropyObjective(mode="cross_entropy_with_integer_labels"),
                dataset_source=MNISTTaskFamily(
                    patch_h=1,
                    patch_w=28,
                    label_last_only=True,
                    add_spurious_pixel_to_train=False,
                    domain=frozenset({"mnist"}),
                    normalize=True,
                ),
                dataset=DatasetConfig(
                    num_examples_in_minibatch=100,
                    num_examples_total=50_000,
                    is_test=False,
                ),
                validation=StepConfig(
                    num_steps=28,
                    batch=1,
                    reset_t=28,
                    track_influence_in=frozenset({0}),
                ),
                nested=StepConfig(
                    num_steps=1,
                    batch=1,
                    reset_t=None,
                    track_influence_in=frozenset({0}),
                ),
                learner=LearnConfig(
                    model_learner=GradientConfig(
                        method=BPTTConfig(None),
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer_learner=GradientConfig(
                        method=BPTTConfig(None),
                        add_clip=HardClip(1.0),
                        scale=1.0,
                    ),
                    optimizer={
                        "meta1_sgd1": OptimizerAssignment(
                            target=frozenset({"rnn1", "rnn2", "readout"}),
                            optimizer=SGDConfig(
                                learning_rate="meta1_sgd1.lr",
                                weight_decay="meta1_sgd1.wd",
                                momentum="meta1_sgd1.momentum",
                            ),
                        ),
                    },
                ),
                track_logs=TrackLogs(
                    gradient=False,
                    hessian_contains_nans=False,
                    largest_eigenvalue=False,
                    influence_tensor=False,
                    immediate_influence_tensor=False,
                    largest_jac_eigenvalue=False,
                    jacobian=False,
                ),
                test_seed=0,
            ),
            MetaConfig(
                objective_fn=CrossEntropyObjective(mode="cross_entropy_with_integer_labels"),
                dataset_source=MNISTTaskFamily(
                    patch_h=1,
                    patch_w=28,
                    label_last_only=True,
                    add_spurious_pixel_to_train=False,
                    domain=frozenset({"mnist"}),
                    normalize=True,
                ),
                dataset=DatasetConfig(
                    num_examples_in_minibatch=100,
                    num_examples_total=10_000,
                    is_test=False,
                ),
                validation=StepConfig(
                    num_steps=28,
                    batch=1,
                    reset_t=28,
                    track_influence_in=frozenset({1}),
                ),
                nested=StepConfig(
                    num_steps=1,
                    batch=1,
                    reset_t=None,
                    track_influence_in=frozenset({1}),
                ),
                learner=LearnConfig(
                    model_learner=GradientConfig(
                        method=BPTTConfig(None),
                        add_clip=HardClip(1.0),
                        scale=1.0,
                    ),
                    optimizer_learner=GradientConfig(
                        method=RTRLConfig(
                            start_at_step=0,
                            damping=1e-4,
                        ),
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer={
                        "meta2_adam1": OptimizerAssignment(
                            target=frozenset({"meta1_sgd1.lr", "meta1_sgd1.wd", "meta1_sgd1.momentum"}),
                            optimizer=AdamConfig(
                                learning_rate="meta2_adam1.lr",
                                weight_decay="meta2_adam1.wd",
                                momentum="meta2_adam1.momentum",
                            ),
                        ),
                    },
                ),
                track_logs=TrackLogs(
                    gradient=False,
                    hessian_contains_nans=False,
                    largest_eigenvalue=False,
                    influence_tensor=False,
                    immediate_influence_tensor=False,
                    largest_jac_eigenvalue=False,
                    jacobian=False,
                ),
                test_seed=0,
            ),
            MetaConfig(
                objective_fn=CrossEntropyObjective(mode="cross_entropy_with_integer_labels"),
                dataset_source=MNISTTaskFamily(
                    patch_h=1,
                    patch_w=28,
                    label_last_only=True,
                    add_spurious_pixel_to_train=False,
                    domain=frozenset({"mnist"}),
                    normalize=True,
                ),
                dataset=DatasetConfig(
                    num_examples_in_minibatch=100,
                    num_examples_total=10_000,
                    is_test=True,
                ),
                validation=StepConfig(
                    num_steps=28,
                    batch=1,
                    reset_t=28,
                    track_influence_in=frozenset({2}),
                ),
                nested=StepConfig(
                    num_steps=100,
                    batch=1,
                    reset_t=None,
                    track_influence_in=frozenset({2}),
                ),
                learner=LearnConfig(
                    model_learner=GradientConfig(
                        method=IdentityLearnerConfig(),
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer_learner=GradientConfig(
                        method=IdentityLearnerConfig(),
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer={},
                ),
                track_logs=TrackLogs(
                    gradient=False,
                    hessian_contains_nans=False,
                    largest_eigenvalue=False,
                    influence_tensor=False,
                    immediate_influence_tensor=False,
                    largest_jac_eigenvalue=False,
                    jacobian=False,
                ),
                test_seed=0,
            ),
        ],
        label_mask_value=-1.0,
        unlabeled_mask_value=-100.0,
        num_tasks=1,
    )
    if not config.clearml_run:
        return

    # Count total num iterations
    base = config.levels[0]
    num_vb = math.ceil(base.dataset.num_examples_total / base.dataset.num_examples_in_minibatch)
    nesting = math.prod(l.nested.num_steps for l in config.levels)
    iters_per_epoch = (base.dataset.num_examples_total * num_vb) // (base.dataset.num_examples_in_minibatch * nesting)
    total_iterations = iters_per_epoch * config.epochs

    # Logger
    loggers = [MatplotlibLogger(config.log_dir)]
    scalar_logger = create_logger(loggers, len(config.levels), total_iterations, config.checkpoint_every_n_minibatches)

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
    loss_fns = create_readout_loss_fns(config, task_interfaces, readouts)

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
        break

    scalar_logger.shutdown()

    for logger in scalar_logger.loggers:
        match logger:
            case MatplotlibLogger():
                logger.generate_figures()

    # def te_inf(
    #     _env: GodState, ds: traverse[batched[tuple[jax.Array, jax.Array, jax.Array]]], mask: jax.Array
    # ) -> tuple[STAT, ...]:
    #     return identity(
    #         lambda e, d: (transitions[0](e, d), ()),
    #         model_statistics_fns[0],
    #         learn_interfaces[0],
    #         lambda x: x,
    #         lambda x: (x, mask),
    #     )(_env, ds)[1]

    # final_te_loss = 0
    # final_te_acc = 0
    # num_te_batches = 0
    # test_env = general_interfaces[0].put_current_virtual_minibatch(env, jnp.nan)
    # for te_x, te_y, te_seqs, te_mask in dataloader_te:
    #     ds = traverse(batched((te_x, te_y, te_seqs)))
    #     test_env = test_reset(test_env)
    #     stats = te_inf(test_env, ds, te_mask)
    #     te_loss, te_acc, _, _wds, __, ___, _hT, _aT, _eig = stats[0]
    #     te_loss = metric_fn(te_loss)
    #     te_acc = metric_fn(te_acc)
    #     final_te_loss += te_loss
    #     final_te_acc += te_acc
    #     num_te_batches += 1

    # context = logger.get_context()
    # logger.log_scalar(
    #     context,
    #     "final_test/loss",
    #     "final_test_loss",
    #     final_te_loss / num_te_batches,
    #     total_tr_vb * config.num_base_epochs,
    #     1,
    # )
    # logger.log_scalar(
    #     context,
    #     "final_test/accuracy",
    #     "final_test_accuracy",
    #     final_te_acc / num_te_batches,
    #     total_tr_vb * config.num_base_epochs,
    #     1,
    # )
    # logger.close_context(context)

    # logger.shutdown()


if __name__ == "__main__":
    runApp(None, None)
