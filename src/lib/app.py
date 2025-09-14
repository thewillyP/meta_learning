import copy
from typing import Union
import clearml
import boto3
import random
import string
from cattrs import unstructure, Converter
from cattrs.strategies import configure_tagged_union
import random
import os
import jax
import jax.numpy as jnp
import torch
import numpy as np
import toolz
import math

from lib.config import *
from lib.create_axes import create_axes
from lib.create_env import create_env
from lib.create_interface import (
    create_general_interfaces,
    create_learn_interfaces,
    create_transition_interfaces,
    create_validation_learn_interfaces,
)
from lib.datasets import create_dataloader
from lib.env import GodState
from lib.inference import create_inferences, hard_reset_inference, make_resets
from lib.interface import ClassificationInterface
from lib.learning import create_meta_learner, identity
from lib.lib_types import *
from lib.log import get_logs
from lib.loss_function import make_statistics_fns
from lib.util import create_fractional_list


def runApp() -> None:
    # os.environ["CLEARML_CACHE_DIR"] = "/scratch"
    ssm = boto3.client("ssm")
    clearml.Task.set_credentials(
        api_host=ssm.get_parameter(Name="/dev/research/clearml_api_host")["Parameter"]["Value"],
        web_host=ssm.get_parameter(Name="/dev/research/clearml_web_host")["Parameter"]["Value"],
        files_host=ssm.get_parameter(Name="/dev/research/clearml_files_host")["Parameter"]["Value"],
        key=ssm.get_parameter(Name="/dev/research/clearml_api_access_key", WithDecryption=True)["Parameter"]["Value"],
        secret=ssm.get_parameter(Name="/dev/research/clearml_api_secret_key", WithDecryption=True)["Parameter"][
            "Value"
        ],
    )
    clearml.Task.set_offline(offline_mode=True)
    # names don't matter, can change in UI
    task: clearml.Task = clearml.Task.init(
        project_name="temp",
        task_name="".join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        task_type=clearml.TaskTypes.training,
    )

    # Values dont matter because can change in UI
    slurm_params = SlurmParams(
        memory="8GB",
        time="01:00:00",
        cpu=2,
        gpu=0,
        log_dir="/vast/wlp9800/logs",
        singularity_overlay="",
        singularity_binds="/scratch/wlp9800/clearml:/scratch",
        container_source=SifContainerSource(sif_path="/scratch/wlp9800/images/devenv-cpu.sif"),
        use_singularity=False,
        setup_commands="module load python/intel/3.8.6",
    )
    task.connect(unstructure(slurm_params), name="slurm")

    config = GodConfig(
        clearml_run=False,
        data_root_dir="/tmp",
        dataset=MnistConfig(784),
        # dataset=DelayAddOnlineConfig(3, 4, 1, 20, 20),
        num_base_epochs=10,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(global_seed=1, data_seed=1, parameter_seed=1, test_seed=1),
        loss_fn="cross_entropy_with_integer_labels",
        # loss_fn="mse",
        transition_function={
            # 0: GRULayer(
            #     n=128,
            #     # activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=128,
            #     use_bias=True,
            # ),
            0: NNLayer(
                n=0,
                activation_fn="tanh",
                use_bias=True,
            ),
            # 1: NNLayer(
            #     n=128,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
            # 0: LSTMLayer(
            #     n=64,
            #     use_bias=True,
            # ),
            # 1: NNLayer(
            #     n=64,
            #     activation_fn="tanh",
            #     use_bias=True,
            # ),
        },
        # readout_function=FeedForwardConfig(
        #     ffw_layers={
        #         0: NNLayer(n=10, activation_fn="identity", use_bias=True),
        #     }
        # ),
        readout_function=FeedForwardConfig(
            ffw_layers={
                0: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                1: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                2: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                3: NNLayer(n=10, activation_fn="identity", use_bias=True),
            }
        ),
        learners={
            0: LearnConfig(  # normal feedforward backprop
                learner=BPTTConfig(),
                optimizer=SGDConfig(
                    learning_rate=0.1,
                    momentum=0.0,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                learner=IdentityConfig(),
                optimizer=AdamConfig(
                    learning_rate=0.01,
                    # momentum=0.0,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=500,
            ),
        },
        data={
            0: DataConfig(
                train_percent=83.33,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=1,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=16.67,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=1,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=True,
        test_batch_size=100,
    )

    converter = Converter()
    configure_tagged_union(Union[NNLayer, GRULayer, LSTMLayer], converter)
    configure_tagged_union(Union[RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig], converter)
    configure_tagged_union(Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig], converter)
    configure_tagged_union(Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig], converter)

    _config = task.connect(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)

    if not config.clearml_run:
        return

    task.execute_remotely(queue_name="willyp", clone=False, exit_process=True)

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

        tr_loss, tr_acc, _, tr_grs, __ = tr_stats
        vl_loss, vl_acc, _, ___, __ = vl_stats
        te_loss, te_acc, _, __, ___ = te_stats
        _, __, lrs, ___, meta_grs = tr_stats

        tr_loss = jnp.sum(tr_loss, axis=2)
        tr_acc = jnp.sum(tr_acc, axis=2)
        tr_grs = tr_grs[:, :, -1]
        vl_loss = jnp.sum(vl_loss, axis=2)
        vl_acc = jnp.sum(vl_acc, axis=2)
        te_loss = jnp.sum(te_loss, axis=1)
        te_acc = jnp.sum(te_acc, axis=1)
        lrs = lrs[:, :, -1]
        meta_grs = meta_grs[:, :, -1]
        jax.block_until_ready(env)

        for i in range(tr_loss.shape[0]):
            for j in range(tr_loss.shape[1]):
                iteration = k * tr_loss.shape[0] * tr_loss.shape[1] + i * tr_loss.shape[1] + j + 1
                if iteration % config.checkpoint_every_n_minibatches == 0:
                    clearml.Logger.current_logger().report_scalar(
                        title="train/loss", series="train_loss", value=tr_loss[i, j], iteration=iteration
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="train/accuracy", series="train_accuracy", value=tr_acc[i, j], iteration=iteration
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="train/learning_rate",
                        series="train_learning_rate",
                        value=lrs[i, j][0],
                        iteration=iteration,
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="train/gradient_norm",
                        series="train_gradient_norm",
                        value=jnp.linalg.norm(tr_grs[i, j]),
                        iteration=iteration,
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="validation/loss",
                        series="validation_loss",
                        value=vl_loss[i, j],
                        iteration=iteration,
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="validation/accuracy",
                        series="validation_accuracy",
                        value=vl_acc[i, j],
                        iteration=iteration,
                    )
                    clearml.Logger.current_logger().report_scalar(
                        title="meta/gradient_norm",
                        series="meta_gradient_norm",
                        value=jnp.linalg.norm(meta_grs[i, j]),
                        iteration=iteration,
                    )

            clearml.Logger.current_logger().report_scalar(
                title="test/loss",
                series="test_loss",
                value=te_loss[i],
                iteration=iteration,
            )
            clearml.Logger.current_logger().report_scalar(
                title="test/accuracy",
                series="test_accuracy",
                value=te_acc[i],
                iteration=iteration,
            )

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
    for te_x, te_y, te_seqs, te_mask in dataloader_te:
        ds = traverse(batched((te_x, te_y, te_seqs)))
        stats = te_inf(env, ds, te_mask)
        te_loss, te_acc, _, __, ___ = stats[0]
        te_loss = jnp.sum(te_loss)
        te_acc = jnp.sum(te_acc)
        final_te_loss += te_loss
        final_te_acc += te_acc
        num_te_batches += 1

    clearml.Logger.current_logger().report_scalar(
        title="final_test/loss",
        series="final_test_loss",
        value=final_te_loss / num_te_batches,
        iteration=total_iterations,
    )
    clearml.Logger.current_logger().report_scalar(
        title="final_test/accuracy",
        series="final_test_accuracy",
        value=final_te_acc / num_te_batches,
        iteration=total_iterations,
    )
