import copy
import random
import os
import jax
import jax.numpy as jnp
from pyrsistent import pmap, pvector
import torch
import numpy as np
import toolz
import math
import time
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import create_inference_axes
from meta_learn_lib.create_env import create_env, create_transition_fns, env_validation_resetters
from meta_learn_lib.create_interface import create_learn_interfaces, create_meta_interfaces, create_task_interfaces
from meta_learn_lib.env import *
from meta_learn_lib.inference import create_inference_and_readout
from meta_learn_lib.learning import create_meta_learner, create_validation_learners
from meta_learn_lib.lib_types import *
from meta_learn_lib.datasets import create_data_sources, create_dataloader, validate_dataloader_config
from meta_learn_lib.loss_function import create_readout_loss_fns


def runApp(config: GodConfig) -> None:

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
                    n=128,
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
                    num_examples_total=500,
                    is_test=False,
                ),
                validation=StepConfig(
                    num_steps=28,
                    batch=2,
                    reset_t=28,
                    track_influence_in=frozenset({0, 1}),
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
                    num_examples_total=500,
                    is_test=False,
                ),
                validation=StepConfig(
                    num_steps=28,
                    batch=2,
                    reset_t=28,
                    track_influence_in=frozenset({1}),
                ),
                nested=StepConfig(
                    num_steps=1,
                    batch=2,
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
                    num_steps=1,
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
        unlabeled_mask_value=0.0,
        num_tasks=8,
    )
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

    # Dataset
    dataset_prng, data_loader_prng = jax.random.split(dataset_gen_prng, 2)
    data_sources, shapes = create_data_sources(config, dataset_prng)
    dataloader = create_dataloader(config, data_sources, data_loader_prng, task_prng)

    for x in toolz.take(1, dataloader):
        ((none, (tr_x, tr_y)), (vl_x, vl_y)), (te_x, te_y) = x
        # print(tr_x, tr_y)
        # print(tr_x.min(), tr_x.max(), tr_x.mean(), tr_x.std())
        print(none)
        print(tr_x.shape, tr_y.shape)
        print(vl_x.shape, vl_y.shape)
        print(te_x.shape, te_y.shape)

    meta_interfaces, count = create_meta_interfaces(config, 0)
    learn_interfaces, count = create_learn_interfaces(config, count)
    task_interfaces, count = create_task_interfaces(config, count)

    env = create_env(config, shapes, meta_interfaces, learn_interfaces, env_prng)
    eqx.tree_pprint(env.serialize())

    # val_learn_interfaces, nest_learn_interfaces = zip(*learn_interfaces)

    # inference_axes = map(lambda i: create_inference_axes(env, i), range(len(config.levels)))
    # transitions, readouts = zip(
    #     *map(lambda i, a: create_inference_and_readout(config, i, a), meta_interfaces, inference_axes)
    # )

    # transition_fns = create_transition_fns(config, shapes, meta_interfaces, val_learn_interfaces, transitions)
    # loss_fns = create_readout_loss_fns(config, task_interfaces, readouts)

    # meta_learner = create_meta_learner(
    #     config,
    #     shapes,
    #     transition_fns,
    #     loss_fns,
    #     list(val_learn_interfaces),
    #     list(nest_learn_interfaces),
    #     meta_interfaces,
    #     env,
    # )

    # x, dataloader = toolz.peek(dataloader)
    # arr, static = eqx.partition(env, eqx.is_array)

    # def update_fn(data, arr):
    #     env = eqx.combine(arr, static)
    #     env, stat = meta_learner(env, data)
    #     arr, _ = eqx.partition(env, eqx.is_array)
    #     return arr, stat

    # meta_learner_compiled = eqx.filter_jit(update_fn, donate="all-except-first").lower(x, arr).compile()

    # meta_learner_compiled = (
    #     eqx.filter_jit(lambda d, e: meta_learner(e, d), donate="all-except-first").lower(x, env).compile()
    # )

    # (((tr_x, tr_y, tr_seqs, tr_mask), (vl_x, vl_y, vl_seqs, vl_mask)), (te_x, te_y, te_seqs, te_mask)), dataloader = (
    #     toolz.peek(dataloader)
    # )
    # _scan_data = traverse(
    #     ((traverse(batched((tr_x, tr_y, tr_seqs))), tr_mask), (traverse(batched((vl_x, vl_y, vl_seqs))), vl_mask))
    # )
    # scan_data = traverse((_scan_data, (traverse(batched((te_x, te_y, te_seqs))), te_mask)))

    # update_fn = (
    #     eqx.filter_jit(lambda data, init_model: meta_learner(init_model, data), donate="all-except-first")
    #     .lower(scan_data, env)
    #     .compile()
    # )

    # iterations_per_epoch: int = total_tr_vb // math.prod(
    #     [l.num_virtual_minibatches_per_turn for l in config.learners.values()]
    # )

    # total_iterations = iterations_per_epoch * config.num_base_epochs

    # for k, (
    #     ((tr_x, tr_y, tr_seqs, tr_mask), (vl_x, vl_y, vl_seqs, vl_mask)),
    #     (te_x, te_y, te_seqs, te_mask),
    # ) in enumerate(toolz.take(total_iterations, dataloader)):
    #     _scan_data = traverse(
    #         ((traverse(batched((tr_x, tr_y, tr_seqs))), tr_mask), (traverse(batched((vl_x, vl_y, vl_seqs))), vl_mask))
    #     )
    #     data = traverse((_scan_data, (traverse(batched((te_x, te_y, te_seqs))), te_mask)))
    #     env, stats, te_losses = update_fn(data, env)
    #     tr_stats, vl_stats, te_stats = stats

    #     tr_loss, tr_acc, _, _wds, tr_grs, __, hT, aT, _eig = tr_stats
    #     vl_loss, vl_acc, _, _wds, __, ___, _hT, _aT, _eig = vl_stats
    #     te_loss, te_acc, _, _wds, __, ___, _hT, _aT, _eig = te_stats
    #     _, __, lrs, wds, ___, meta_grs, _hT, _aT, eig = tr_stats

    #     tr_loss = metric_fn(tr_loss, axis=2)
    #     tr_acc = metric_fn(tr_acc, axis=2)
    #     tr_grs = tr_grs[:, :, -1]
    #     vl_loss = metric_fn(vl_loss, axis=2)
    #     vl_acc = metric_fn(vl_acc, axis=2)
    #     te_loss = metric_fn(te_loss, axis=1)
    #     te_acc = metric_fn(te_acc, axis=1)
    #     lrs1 = lrs[0][:, :, -1]
    #     lrs2 = lrs[1][:, :, -1]
    #     wds1 = wds[0][:, :, -1]
    #     wds2 = wds[1][:, :, -1]
    #     meta_grs = meta_grs[:, :, -1]
    #     hT = hT[:, :, -1]
    #     aT = aT[:, :, -1]
    #     eig = eig[:, :, -1]

    #     jax.block_until_ready(env)

    #     # def corr(a, b):
    #     #     # a,b: [batch, hidden]
    #     #     a_flat = a.reshape(-1)
    #     #     b_flat = b.reshape(-1)
    #     #     a_flat = a_flat - a_flat.mean()
    #     #     b_flat = b_flat - b_flat.mean()
    #     #     return jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat))

    #     window_size = 100
    #     aT_buffer = []  # store past activations

    #     def corr_per_example_dot(a_prev, a_curr, eps=1e-8):
    #         dot = jnp.sum(a_prev * a_curr, axis=1)
    #         norm_prev = jnp.linalg.norm(a_prev, axis=1)
    #         norm_curr = jnp.linalg.norm(a_curr, axis=1)
    #         corr_per_ex = dot / (norm_prev * norm_curr + eps)
    #         return jnp.mean(corr_per_ex)

    #     context = logger.get_context()
    #     for i in range(tr_loss.shape[0]):
    #         for j in range(tr_loss.shape[1]):
    #             iteration = k * tr_loss.shape[0] * tr_loss.shape[1] + i * tr_loss.shape[1] + j + 1
    #             if iteration % config.checkpoint_every_n_minibatches == 0:
    #                 logger.log_scalar(
    #                     context,
    #                     "train/loss",
    #                     "train_loss",
    #                     tr_loss[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/accuracy",
    #                     "train_accuracy",
    #                     tr_acc[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/recurrent_learning_rate",
    #                     "train_recurrent_learning_rate",
    #                     lrs1[i, j][0],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/readout_learning_rate",
    #                     "train_readout_learning_rate",
    #                     lrs2[i, j][0],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/recurrent_weight_decay",
    #                     "train_recurrent_weight_decay",
    #                     wds1[i, j][0],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/readout_weight_decay",
    #                     "train_readout_weight_decay",
    #                     wds2[i, j][0],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/gradient_norm",
    #                     "train_gradient_norm",
    #                     jnp.linalg.norm(tr_grs[i, j]),
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "validation/loss",
    #                     "validation_loss",
    #                     vl_loss[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "validation/accuracy",
    #                     "validation_accuracy",
    #                     vl_acc[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "meta/gradient_norm",
    #                     "meta_gradient_norm",
    #                     jnp.linalg.norm(meta_grs[i, j]),
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )
    #                 logger.log_scalar(
    #                     context,
    #                     "train/final_rnn_activation_norm",
    #                     "train_final_rnn_activation_norm",
    #                     hT[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )

    #                 # # store previous activations across iterations
    #                 # if iteration == 1:
    #                 #     prev_aT = aT[i, j]  # initialize
    #                 # else:
    #                 #     aT_curr = aT[i, j]  # [batch, hidden]
    #                 #     aT_prev = prev_aT  # [batch, hidden]

    #                 #     at_corr = corr(aT_prev, aT_curr)

    #                 #     logger.log_scalar(
    #                 #         context,
    #                 #         "train/aT_correlation",
    #                 #         "train_aT_correlation",
    #                 #         at_corr,
    #                 #         iteration,
    #                 #         total_tr_vb * config.num_base_epochs,
    #                 #     )

    #                 #     prev_aT = aT_curr

    #                 aT_curr = aT[i, j]  # [batch, hidden]

    #                 # add to buffer
    #                 aT_buffer.append(aT_curr)
    #                 if len(aT_buffer) > window_size:
    #                     aT_buffer.pop(0)

    #                 # only compute correlation when we have at least 2 stored activations
    #                 if len(aT_buffer) > 1:
    #                     # correlate current activation with each previous activation in buffer
    #                     corrs = [corr_per_example_dot(prev, aT_curr) for prev in aT_buffer[:-1]]
    #                     at_corr = jnp.mean(jnp.array(corrs))

    #                     logger.log_scalar(
    #                         context,
    #                         "train/aT_correlation_window",
    #                         "train_aT_correlation_window",
    #                         at_corr,
    #                         iteration,
    #                         total_tr_vb * config.num_base_epochs,
    #                     )

    #                 logger.log_scalar(
    #                     context,
    #                     "train/largest_eigenvalue",
    #                     "train_largest_eigenvalue",
    #                     eig[i, j],
    #                     iteration,
    #                     total_tr_vb * config.num_base_epochs,
    #                 )

    #         logger.log_scalar(
    #             context, "test/loss", "test_loss", te_loss[i], iteration, total_tr_vb * config.num_base_epochs
    #         )
    #         logger.log_scalar(
    #             context, "test/accuracy", "test_accuracy", te_acc[i], iteration, total_tr_vb * config.num_base_epochs
    #         )
    #     logger.close_context(context)

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
    runApp(None)
