import copy
import random
import os
import jax
import jax.numpy as jnp
import torch
import numpy as np
import toolz
import math

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import create_axes
from meta_learn_lib.create_env import create_env
from meta_learn_lib.create_interface import (
    create_general_interfaces,
    create_learn_interfaces,
    create_transition_interfaces,
    create_validation_learn_interfaces,
)
from meta_learn_lib.datasets import create_dataloader
from meta_learn_lib.env import GodState
from meta_learn_lib.inference import create_inferences, hard_reset_inference, make_resets
from meta_learn_lib.interface import ClassificationInterface
from meta_learn_lib.learning import create_meta_learner, identity
from meta_learn_lib.lib_types import *
from meta_learn_lib.log import get_logs
from meta_learn_lib.logger import Logger
from meta_learn_lib.loss_function import make_statistics_fns
from meta_learn_lib.util import create_fractional_list


def runApp(config: GodConfig, logger: Logger) -> None:
    if not config.clearml_run:
        return

    # RNG Stuff
    base_key = jax.random.key(config.seed.global_seed)
    keys = jax.random.split(base_key, 2)
    data_prng = PRNG(jax.random.fold_in(keys[0], config.seed.data_seed))
    env_prng = PRNG(jax.random.fold_in(keys[1], config.seed.parameter_seed))
    test_prng = PRNG(jax.random.key(config.seed.test_seed))
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

    # metric function
    match config.dataset:
        case MnistConfig() | FashionMnistConfig() | CIFAR10Config() | CIFAR100Config():
            metric_fn = jnp.sum
        case DelayAddOnlineConfig():
            metric_fn = jnp.mean

    # Dataset
    percentages = create_fractional_list([x.train_percent / 100 for x in config.data.values()])
    if percentages is None:
        raise ValueError("Learner percentages must sum to 100.")

    dataloader, dataloader_te, virtual_minibatches, n_in_shape, last_unpadded_lengths, total_tr_vb = create_dataloader(
        config, percentages, dataset_gen_prng, test_prng
    )

    if config.ignore_validation_inference_recurrence:
        if not all(virtual_minibatches[k] == 1 for i, k in enumerate(sorted(virtual_minibatches.keys())) if i > 0):
            raise ValueError(
                "When ignore_validation_inference_recurrence is True, all validation datasets except the first must have num_virtual_minibatches_per_turn=1."
            )

    if total_tr_vb % math.prod([l.num_virtual_minibatches_per_turn for l in config.learners.values()]) != 0:
        raise ValueError(
            f"The total number of virtual minibatches per turn ({total_tr_vb}) must be divisible by the product of num_virtual_minibatches_per_turn for each learner ({math.prod([l.num_virtual_minibatches_per_turn for l in config.learners.values()])}). Please adjust the num_virtual_minibatches_per_turn settings in the config."
        )

    learn_interfaces = create_learn_interfaces(config)
    validation_learn_interfaces = create_validation_learn_interfaces(config, learn_interfaces)
    inference_interface = create_transition_interfaces(config)
    general_interfaces = create_general_interfaces(config)
    data_interface = ClassificationInterface[tuple[jax.Array, jax.Array, jax.Array]](
        get_input=lambda data: data[0],
        get_target=lambda data: data[1],
        get_sequence=lambda data: data[2],
    )
    data_interface_for_loss = ClassificationInterface[batched[tuple[jax.Array, jax.Array, jax.Array]]](
        get_input=lambda data: data.b[0],
        get_target=lambda data: data.b[1],
        get_sequence=lambda data: data.b[2],
    )
    env = create_env(config, n_in_shape, learn_interfaces, validation_learn_interfaces, env_prng)
    # eqx.tree_pprint(env.serialize())
    axes = create_axes(env, inference_interface)
    transitions, readouts = create_inferences(config, inference_interface, data_interface, axes)
    resets = make_resets(
        # lambda prng: reinitialize_env(env, config, n_in_shape, prng),
        lambda prng: create_env(config, n_in_shape, learn_interfaces, validation_learn_interfaces, prng),
        inference_interface,
        general_interfaces,
        validation_learn_interfaces,
        virtual_minibatches,
        learn_interfaces[0],
    )
    env0 = copy.deepcopy(env)
    test_reset = hard_reset_inference(
        lambda: env0,
        inference_interface[min(inference_interface.keys())],
    )
    model_statistics_fns = make_statistics_fns(
        config,
        readouts,
        data_interface_for_loss,
        lambda out: out.b,
        general_interfaces,
        virtual_minibatches,
        last_unpadded_lengths,
        lambda e: get_logs(config, e),
    )

    meta_learner = create_meta_learner(
        config,
        [v for _, v in sorted(transitions.items())],
        [v for _, v in sorted(model_statistics_fns.items())],
        [v for _, v in sorted(resets.items())],
        test_reset,
        [v for _, v in sorted(learn_interfaces.items())],
        [v for _, v in sorted(validation_learn_interfaces.items())],
        [v for _, v in sorted(general_interfaces.items())],
        [v for _, v in sorted(virtual_minibatches.items())],
        [v for _, v in sorted(last_unpadded_lengths.items())],
    )

    (((tr_x, tr_y, tr_seqs, tr_mask), (vl_x, vl_y, vl_seqs, vl_mask)), (te_x, te_y, te_seqs, te_mask)), dataloader = (
        toolz.peek(dataloader)
    )
    _scan_data = traverse(
        ((traverse(batched((tr_x, tr_y, tr_seqs))), tr_mask), (traverse(batched((vl_x, vl_y, vl_seqs))), vl_mask))
    )
    scan_data = traverse((_scan_data, (traverse(batched((te_x, te_y, te_seqs))), te_mask)))

    update_fn = (
        eqx.filter_jit(lambda data, init_model: meta_learner(init_model, data), donate="all-except-first")
        .lower(scan_data, env)
        .compile()
    )

    iterations_per_epoch: int = total_tr_vb // math.prod(
        [l.num_virtual_minibatches_per_turn for l in config.learners.values()]
    )

    total_iterations = iterations_per_epoch * config.num_base_epochs

    for k, (
        ((tr_x, tr_y, tr_seqs, tr_mask), (vl_x, vl_y, vl_seqs, vl_mask)),
        (te_x, te_y, te_seqs, te_mask),
    ) in enumerate(toolz.take(total_iterations, dataloader)):
        _scan_data = traverse(
            ((traverse(batched((tr_x, tr_y, tr_seqs))), tr_mask), (traverse(batched((vl_x, vl_y, vl_seqs))), vl_mask))
        )
        data = traverse((_scan_data, (traverse(batched((te_x, te_y, te_seqs))), te_mask)))
        env, stats, te_losses = update_fn(data, env)
        tr_stats, vl_stats, te_stats = stats

        tr_loss, tr_acc, _, _wds, tr_grs, __, hT, aT = tr_stats
        vl_loss, vl_acc, _, _wds, __, ___, _hT, _aT = vl_stats
        te_loss, te_acc, _, _wds, __, ___, _hT, _aT = te_stats
        _, __, lrs, wds, ___, meta_grs, _hT, _aT = tr_stats

        tr_loss = metric_fn(tr_loss, axis=2)
        tr_acc = metric_fn(tr_acc, axis=2)
        tr_grs = tr_grs[:, :, -1]
        vl_loss = metric_fn(vl_loss, axis=2)
        vl_acc = metric_fn(vl_acc, axis=2)
        te_loss = metric_fn(te_loss, axis=1)
        te_acc = metric_fn(te_acc, axis=1)
        lrs1 = lrs[0][:, :, -1]
        lrs2 = lrs[1][:, :, -1]
        wds1 = wds[0][:, :, -1]
        wds2 = wds[1][:, :, -1]
        meta_grs = meta_grs[:, :, -1]
        hT = hT[:, :, -1]
        aT = aT[:, :, -1]

        jax.block_until_ready(env)

        # def corr(a, b):
        #     # a,b: [batch, hidden]
        #     a_flat = a.reshape(-1)
        #     b_flat = b.reshape(-1)
        #     a_flat = a_flat - a_flat.mean()
        #     b_flat = b_flat - b_flat.mean()
        #     return jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat))

        window_size = 100
        aT_buffer = []  # store past activations

        def corr_per_example_dot(a_prev, a_curr, eps=1e-8):
            dot = jnp.sum(a_prev * a_curr, axis=1)
            norm_prev = jnp.linalg.norm(a_prev, axis=1)
            norm_curr = jnp.linalg.norm(a_curr, axis=1)
            corr_per_ex = dot / (norm_prev * norm_curr + eps)
            return jnp.mean(corr_per_ex)

        context = logger.get_context()
        for i in range(tr_loss.shape[0]):
            for j in range(tr_loss.shape[1]):
                iteration = k * tr_loss.shape[0] * tr_loss.shape[1] + i * tr_loss.shape[1] + j + 1
                if iteration % config.checkpoint_every_n_minibatches == 0:
                    logger.log_scalar(
                        context,
                        "train/loss",
                        "train_loss",
                        tr_loss[i, j],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/accuracy",
                        "train_accuracy",
                        tr_acc[i, j],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/recurrent_learning_rate",
                        "train_recurrent_learning_rate",
                        lrs1[i, j][0],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/readout_learning_rate",
                        "train_readout_learning_rate",
                        lrs2[i, j][0],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/recurrent_weight_decay",
                        "train_recurrent_weight_decay",
                        wds1[i, j][0],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/readout_weight_decay",
                        "train_readout_weight_decay",
                        wds2[i, j][0],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/gradient_norm",
                        "train_gradient_norm",
                        jnp.linalg.norm(tr_grs[i, j]),
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "validation/loss",
                        "validation_loss",
                        vl_loss[i, j],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "validation/accuracy",
                        "validation_accuracy",
                        vl_acc[i, j],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "meta/gradient_norm",
                        "meta_gradient_norm",
                        jnp.linalg.norm(meta_grs[i, j]),
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )
                    logger.log_scalar(
                        context,
                        "train/final_rnn_activation_norm",
                        "train_final_rnn_activation_norm",
                        hT[i, j],
                        iteration,
                        total_tr_vb * config.num_base_epochs,
                    )

                    # # store previous activations across iterations
                    # if iteration == 1:
                    #     prev_aT = aT[i, j]  # initialize
                    # else:
                    #     aT_curr = aT[i, j]  # [batch, hidden]
                    #     aT_prev = prev_aT  # [batch, hidden]

                    #     at_corr = corr(aT_prev, aT_curr)

                    #     logger.log_scalar(
                    #         context,
                    #         "train/aT_correlation",
                    #         "train_aT_correlation",
                    #         at_corr,
                    #         iteration,
                    #         total_tr_vb * config.num_base_epochs,
                    #     )

                    #     prev_aT = aT_curr

                    aT_curr = aT[i, j]  # [batch, hidden]

                    # add to buffer
                    aT_buffer.append(aT_curr)
                    if len(aT_buffer) > window_size:
                        aT_buffer.pop(0)

                    # only compute correlation when we have at least 2 stored activations
                    if len(aT_buffer) > 1:
                        # correlate current activation with each previous activation in buffer
                        corrs = [corr_per_example_dot(prev, aT_curr) for prev in aT_buffer[:-1]]
                        at_corr = jnp.mean(jnp.array(corrs))

                        logger.log_scalar(
                            context,
                            "train/aT_correlation_window",
                            "train_aT_correlation_window",
                            at_corr,
                            iteration,
                            total_tr_vb * config.num_base_epochs,
                        )

            logger.log_scalar(
                context, "test/loss", "test_loss", te_loss[i], iteration, total_tr_vb * config.num_base_epochs
            )
            logger.log_scalar(
                context, "test/accuracy", "test_accuracy", te_acc[i], iteration, total_tr_vb * config.num_base_epochs
            )
        logger.close_context(context)

    def te_inf(
        _env: GodState, ds: traverse[batched[tuple[jax.Array, jax.Array, jax.Array]]], mask: jax.Array
    ) -> tuple[STAT, ...]:
        return identity(
            lambda e, d: (transitions[0](e, d), ()),
            model_statistics_fns[0],
            learn_interfaces[0],
            lambda x: x,
            lambda x: (x, mask),
        )(_env, ds)[1]

    final_te_loss = 0
    final_te_acc = 0
    num_te_batches = 0
    test_env = general_interfaces[0].put_current_virtual_minibatch(env, jnp.nan)
    for te_x, te_y, te_seqs, te_mask in dataloader_te:
        ds = traverse(batched((te_x, te_y, te_seqs)))
        test_env = test_reset(test_env)
        stats = te_inf(test_env, ds, te_mask)
        te_loss, te_acc, _, _wds, __, ___, _hT, _aT = stats[0]
        te_loss = metric_fn(te_loss)
        te_acc = metric_fn(te_acc)
        final_te_loss += te_loss
        final_te_acc += te_acc
        num_te_batches += 1

    context = logger.get_context()
    logger.log_scalar(
        context,
        "final_test/loss",
        "final_test_loss",
        final_te_loss / num_te_batches,
        total_tr_vb * config.num_base_epochs,
        1,
    )
    logger.log_scalar(
        context,
        "final_test/accuracy",
        "final_test_accuracy",
        final_te_acc / num_te_batches,
        total_tr_vb * config.num_base_epochs,
        1,
    )
    logger.close_context(context)
