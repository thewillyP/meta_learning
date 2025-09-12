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
        singularity_binds="",
        container_source=SifContainerSource(sif_path="/scratch/wlp9800/images/devenv-cpu.sif"),
    )
    task.connect(unstructure(slurm_params), name="slurm")

    config = GodConfig(
        clearml_run=True,
        data_root_dir="/tmp",
        dataset=MnistConfig(28),
        # dataset=DelayAddOnlineConfig(3, 4, 1, 20, 20),
        num_base_epochs=1,
        checkpoint_every_n_minibatches=1,
        seed=SeedConfig(data_seed=1, parameter_seed=1, test_seed=1),
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
                n=128,
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
                # 1: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                # 2: NNLayer(n=128, activation_fn="tanh", use_bias=True),
                1: NNLayer(n=10, activation_fn="identity", use_bias=True),
            }
        ),
        learners={
            0: LearnConfig(  # normal feedforward backprop
                learner=BPTTConfig(),
                optimizer=SGDNormalizedConfig(
                    learning_rate=0.01,
                    momentum=0.0,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
                num_virtual_minibatches_per_turn=1,
            ),
            1: LearnConfig(
                learner=RTRLConfig(),
                optimizer=AdamConfig(
                    learning_rate=0.001,
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
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=16.67,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=False,
        test_batch_size=100,
    )

    converter = Converter()
    configure_tagged_union(Union[NNLayer, GRULayer, LSTMLayer], converter)
    configure_tagged_union(Union[RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig], converter)
    configure_tagged_union(Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig], converter)
    configure_tagged_union(Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig], converter)

    _config = task.connect(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)
    # task.execute_remotely(queue_name="slurm", clone=False, exit_process=True)

    if not config.clearml_run:
        return

    # RNG Stuff
    data_prng = PRNG(jax.random.key(config.seed.data_seed))
    env_prng = PRNG(jax.random.key(config.seed.parameter_seed))
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
        vl_loss, vl_acc, _, vl_grs, __ = vl_stats
        te_loss, te_acc, _, __, ___ = te_stats
        _, __, lrs, ___, meta_grs = tr_stats

        tr_loss = jnp.sum(tr_loss, axis=2)
        tr_acc = jnp.sum(tr_acc, axis=2)
        tr_grs = tr_grs[:, :, -1]
        vl_loss = jnp.sum(vl_loss, axis=2)
        vl_acc = jnp.sum(vl_acc, axis=2)
        vl_grs = vl_grs[:, :, -1]
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
                        title="validation/gradient_norm",
                        series="validation_gradient_norm",
                        value=jnp.linalg.norm(vl_grs[i, j]),
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
    for te_x, te_y, te_seqs, te_mask in dataloader_te:
        ds = traverse(batched((te_x, te_y, te_seqs)))
        stats = te_inf(env, ds, te_mask)
        te_loss, te_acc, _, __, ___ = stats
        te_loss = jnp.sum(te_loss)
        te_acc = jnp.sum(te_acc)
        final_te_loss += te_loss
        final_te_acc += te_acc

    clearml.Logger.current_logger().report_scalar(
        title="final_test/loss",
        series="final_test_loss",
        value=final_te_loss / len(dataloader_te.dataset),
        iteration=total_iterations,
    )
    clearml.Logger.current_logger().report_scalar(
        title="final_test/accuracy",
        series="final_test_accuracy",
        value=final_te_acc / len(dataloader_te.dataset),
        iteration=total_iterations,
    )


# # def create_learner(
# #     learner: str, rtrl_use_fwd: bool, uoro_std
# # ) -> RTRL | RFLO | UORO | IdentityLearner | OfflineLearning:
# #     match learner:
# #         case "rtrl":
# #             return RTRL(rtrl_use_fwd)
# #         case "rflo":
# #             return RFLO()
# #         case "uoro":
# #             return UORO(lambda key, shape: jax.random.uniform(key, shape, minval=-uoro_std, maxval=uoro_std))
# #         case "identity":
# #             return IdentityLearner()
# #         case "bptt":
# #             return OfflineLearning()
# #         case _:
# #             raise ValueError("Invalid learner")


# # def create_rnn_learner(
# #     learner: RTRL | RFLO | UORO | IdentityLearner | OfflineLearning,
# #     lossFn: Callable[[jax.Array, jax.Array], LOSS],
# #     arch: Literal["rnn", "ffn"],
# # ) -> Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]:
# #     match arch:
# #         case "rnn":
# #             stepFn = lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
# #             readoutFn = doRnnReadout
# #         case "ffn":
# #             stepFn = (
# #                 lambda d: doFeedForwardStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
# #             )
# #             readoutFn = doFeedForwardReadout

# #     lfn = lambda a, b: lossFn(a, b.y)
# #     match learner:
# #         case OfflineLearning():
# #             bptt_library: Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]
# #             bptt_library = learner.createLearner(
# #                 stepFn,
# #                 readoutFn,
# #                 lfn,
# #                 readoutRecurrentError(readoutFn, lfn),
# #             )
# #             return bptt_library
# #         case _:
# #             # library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
# #             _library: Library[IdentityF[InputOutput], GodInterpreter, GodState, IdentityF[PREDICTION]]
# #             _library = learner.createLearner(
# #                 stepFn,
# #                 readoutFn,
# #                 lfn,
# #                 readoutRecurrentError(doRnnReadout, lfn),
# #             )
# #             library = Library[InputOutput, GodInterpreter, GodState, PREDICTION](
# #                 model=lambda d: _library.model(IdentityF(d)),
# #                 modelLossFn=lambda d: _library.modelLossFn(IdentityF(d)),
# #                 modelGradient=lambda d: _library.modelGradient(IdentityF(d)),
# #             )
# #             return foldrLibrary(library)


# # def train_loop_IO[D](
# #     tr_dataset: Traversable[Traversable[InputOutput]],
# #     vl_dataset: Traversable[Traversable[InputOutput]],
# #     to_combined_ds: Callable[[Traversable[Traversable[InputOutput]], Traversable[Traversable[InputOutput]]], D],
# #     model: Callable[[D, GodState], tuple[Traversable[AllLogs], GodState]],
# #     env: GodState,
# #     refresh_env: Callable[[GodState], GodState],
# #     config: GodConfig,
# #     checkpoint_fn: Callable[[GodState], None],
# #     log_fn: Callable[[AllLogs], None],
# #     te_loss: Callable[[GodState], LOSS],
# #     statistic: Callable[[GodState], float],
# # ) -> None:
# #     tr_dataset = PyTreeDataset(tr_dataset)
# #     vl_dataset = PyTreeDataset(vl_dataset)

# #     if config.batch_or_online == "batch":
# #         vl_batch_size = config.batch_vl
# #         vl_sampler = RandomSampler(vl_dataset)
# #         tr_dl = lambda b: DataLoader(
# #             b, batch_size=config.batch_tr, shuffle=True, collate_fn=jax_collate_fn, drop_last=True
# #         )
# #         # doesn't make sense to do this in batch case. batch=subsequence in online
# #         env = copy.replace(env, start_example=0)
# #     else:
# #         vl_batch_size = config.batch_tr  # same size -> conjoin with tr batch with to_combined_ds
# #         vl_sampler = RandomSampler(vl_dataset, replacement=True, num_samples=len(tr_dataset))
# #         tr_dl = lambda b: DataLoader(b, batch_size=config.batch_tr, shuffle=False, collate_fn=jax_collate_fn)

# #     vl_dataloader = DataLoader(
# #         vl_dataset, batch_size=vl_batch_size, sampler=vl_sampler, collate_fn=jax_collate_fn, drop_last=True
# #     )

# #     def infinite_loader(loader):
# #         while True:
# #             for batch in loader:
# #                 yield batch

# #     vl_dataloader = infinite_loader(vl_dataloader)

# #     start = time.time()
# #     num_batches_seen_so_far = 0
# #     all_logs: list[Traversable[AllLogs]] = []
# #     for epoch in range(env.start_epoch, config.num_retrain_loops):
# #         print(f"Epoch {epoch + 1}/{config.num_retrain_loops}")
# #         batched_dataset = Subset(tr_dataset, indices=range(env.start_example, len(tr_dataset)))
# #         tr_dataloader = tr_dl(batched_dataset)

# #         for i, (tr_batch, vl_batch) in enumerate(zip(tr_dataloader, vl_dataloader)):
# #             ds_batch = to_combined_ds(tr_batch, vl_batch)
# #             batch_size = len(jax.tree.leaves(ds_batch)[0])
# #             env = refresh_env(env)
# #             logs, env = model(ds_batch, env)

# #             env = eqx.tree_at(lambda t: t.start_example, env, env.start_example + batch_size)
# #             all_logs.append(logs)
# #             checkpoint_condition = num_batches_seen_so_far + i + 1
# #             if checkpoint_condition % config.checkpoint_interval == 0:
# #                 checkpoint_fn(
# #                     copy.replace(
# #                         env,
# #                         inner_prng=jax.random.key_data(env.inner_prng),
# #                         outer_prng=jax.random.key_data(env.outer_prng),
# #                     )
# #                 )

# #             print(
# #                 f"Batch {i + 1}/{len(tr_dataloader)}, Loss: {logs.value.train_loss[-1]}, LR: {logs.value.inner_learning_rate[-1]}"
# #             )

# #         # env = eqx.tree_at(lambda t: t.start_epoch, env, epoch + 1)
# #         env = eqx.tree_at(lambda t: t.start_example, env, 0)
# #         num_batches_seen_so_far += len(tr_dataloader)

# #     end = time.time()
# #     print(f"Training time: {end - start} seconds")

# #     total_logs: Traversable[AllLogs] = jax.tree.map(lambda *xs: jnp.concatenate(xs), *all_logs)

# #     log_fn(total_logs.value)
# #     checkpoint_fn(
# #         copy.replace(
# #             env,
# #             inner_prng=jax.random.key_data(env.inner_prng),
# #             outer_prng=jax.random.key_data(env.outer_prng),
# #         )
# #     )

# #     def safe_norm(x):
# #         return jnp.linalg.norm(x) if x is not None else None

# #     # log wandb partial metrics
# #     for log_tree_ in tree_unstack_lazy(total_logs.value):
# #         log_data: AllLogs = jax.tree.map(
# #             lambda x: jnp.real(x) if x is not None and jnp.all(jnp.isfinite(x)) else None, log_tree_
# #         )
# #         wandb.log(
# #             {
# #                 "train_loss": log_data.train_loss,
# #                 "validation_loss": log_data.validation_loss,
# #                 "test_loss": log_data.test_loss,
# #                 "hyperparameters": log_data.hyperparameters,
# #                 "inner_learning_rate": log_data.inner_learning_rate,
# #                 "parameter_norm": log_data.parameter_norm,
# #                 "oho_gradient": log_data.oho_gradient,
# #                 "train_gradient": log_data.train_gradient,
# #                 "validation_gradient": log_data.validation_gradient,
# #                 "oho_gradient_norm": safe_norm(log_data.oho_gradient),
# #                 "train_gradient_norm": safe_norm(log_data.train_gradient),
# #                 "validation_gradient_norm": safe_norm(log_data.validation_gradient),
# #                 "immediate_influence_tensor_norm": log_data.immediate_influence_tensor_norm,
# #                 "inner_influence_tensor_norm": log_data.inner_influence_tensor_norm,
# #                 "outer_influence_tensor_norm": log_data.outer_influence_tensor_norm,
# #                 "largest_jacobian_eigenvalue": log_data.largest_jacobian_eigenvalue,
# #                 "largest_influence_eigenvalue": log_data.largest_hessian_eigenvalue,
# #                 "jacobian_eigenvalues": log_data.jacobian,
# #                 "rnn_activation_norm": log_data.rnn_activation_norm,
# #                 "immediate_influence_tensor": jnp.ravel(log_data.immediate_influence_tensor)
# #                 if log_data.immediate_influence_tensor is not None
# #                 else None,
# #                 "outer_influence_tensor": jnp.ravel(log_data.outer_influence_tensor)
# #                 if log_data.outer_influence_tensor is not None
# #                 else None,
# #             }
# #         )

# #     ee = te_loss(env)
# #     eee = statistic(env)
# #     print(ee)
# #     print(eee)
# #     wandb.log({"test_loss": ee, "test_statistic": eee})


# # def create_online_model(
# #     test_dataset: Traversable[InputOutput],
# #     tr_to_val_env: Callable[[GodState, PRNG], GodState],
# #     tr_to_te_env: Callable[[GodState, PRNG], GodState],
# #     lossFn: Callable[[jax.Array, jax.Array], LOSS],
# #     initEnv: GodState,
# #     innerInterpreter: GodInterpreter,
# #     outerInterpreter: GodInterpreter,
# #     config: GodConfig,
# # ) -> tuple[
# #     Callable[[Traversable[OhoData[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
# #     Callable[[GodState], LOSS],
# # ]:
# #     innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
# #     innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
# #     outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

# #     innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
# #     innerController = logGradient(innerController)
# #     innerLibrary = innerLibrary._replace(modelGradient=innerController)

# #     inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
# #     outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
# #     pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

# #     validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

# #     match outerLearner:
# #         case OfflineLearning():
# #             _outerLibrary: Library[
# #                 Traversable[OhoData[Traversable[InputOutput]]],
# #                 GodInterpreter,
# #                 GodState,
# #                 Traversable[Traversable[PREDICTION]],
# #             ]
# #             _outerLibrary = endowBilevelOptimization(
# #                 innerLibrary,
# #                 doOptimizerStep,
# #                 innerInterpreter,
# #                 outerLearner,
# #                 lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
# #                 tr_to_val_env,
# #                 pad_val_grad_by,
# #             )

# #             outerController = logGradient(_outerLibrary.modelGradient)
# #             _outerLibrary = _outerLibrary._replace(modelGradient=outerController)

# #             @do()
# #             def updateStep(oho_data: Traversable[OhoData[Traversable[InputOutput]]]):
# #                 print("recompiled")
# #                 env = yield from get(PX[GodState]())
# #                 interpreter = yield from ask(PX[GodInterpreter]())
# #                 hyperparameters = yield from interpreter.getRecurrentParam
# #                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

# #                 te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
# #                 vl, _ = _outerLibrary.modelLossFn(oho_data).func(outerInterpreter, env)
# #                 tr, _ = (
# #                     foldrLibrary(innerLibrary)
# #                     .modelLossFn(Traversable(oho_data.value.payload))
# #                     .func(innerInterpreter, env)
# #                 )

# #                 def safe_norm(x):
# #                     return jnp.linalg.norm(x) if x is not None else None

# #                 log = AllLogs(
# #                     train_loss=tr / config.tr_examples_per_epoch,
# #                     validation_loss=vl / config.vl_examples_per_epoch,
# #                     test_loss=te / config.numTe,
# #                     hyperparameters=hyperparameters,
# #                     inner_learning_rate=innerInterpreter.getLearningRate(env),
# #                     parameter_norm=safe_norm(weights),
# #                     oho_gradient=env.outerLogs.gradient,
# #                     train_gradient=env.innerLogs.gradient,
# #                     validation_gradient=env.outerLogs.validationGradient,
# #                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
# #                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
# #                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
# #                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
# #                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
# #                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
# #                     jacobian=env.innerLogs.hessian,
# #                     hessian=env.outerLogs.hessian,
# #                     rnn_activation_norm=safe_norm(env.rnnState.activation),
# #                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
# #                     if config.log_accumulate_influence
# #                     else None,
# #                 )
# #                 logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))

# #                 _ = yield from _outerLibrary.modelGradient(oho_data).flat_map(doOptimizerStep)
# #                 return pure(logs, PX[tuple[GodInterpreter, GodState]]())

# #             model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
# #             return model

# #         case _:
# #             outerLibrary: Library[
# #                 IdentityF[OhoData[Traversable[InputOutput]]],
# #                 GodInterpreter,
# #                 GodState,
# #                 IdentityF[Traversable[PREDICTION]],
# #             ]
# #             outerLibrary = endowBilevelOptimization(
# #                 innerLibrary,
# #                 doOptimizerStep,
# #                 innerInterpreter,
# #                 outerLearner,
# #                 lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
# #                 tr_to_val_env,
# #                 pad_val_grad_by,
# #             )

# #             outerController = logGradient(outerLibrary.modelGradient)
# #             outerLibrary = outerLibrary._replace(modelGradient=outerController)

# #             @do()
# #             def updateStep(oho_data: OhoData[Traversable[InputOutput]]):
# #                 print("recompiled")
# #                 env = yield from get(PX[GodState]())
# #                 interpreter = yield from ask(PX[GodInterpreter]())
# #                 hyperparameters = yield from interpreter.getRecurrentParam
# #                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

# #                 te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
# #                 vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
# #                 tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)

# #                 _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

# #                 # code smell but what can you do, no maybe monad or elvis operator...
# #                 def safe_norm(x):
# #                     return jnp.linalg.norm(x) if x is not None else None

# #                 log = AllLogs(
# #                     train_loss=tr / config.tr_examples_per_epoch,
# #                     validation_loss=vl / config.vl_examples_per_epoch,
# #                     test_loss=te / config.numTe,
# #                     hyperparameters=hyperparameters,
# #                     inner_learning_rate=innerInterpreter.getLearningRate(env),
# #                     parameter_norm=safe_norm(weights),
# #                     oho_gradient=env.outerLogs.gradient,
# #                     train_gradient=env.innerLogs.gradient,
# #                     validation_gradient=env.outerLogs.validationGradient,
# #                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
# #                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
# #                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
# #                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
# #                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
# #                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
# #                     jacobian=env.innerLogs.hessian,
# #                     hessian=env.outerLogs.hessian,
# #                     rnn_activation_norm=safe_norm(env.rnnState.activation),
# #                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
# #                     if config.log_accumulate_influence
# #                     else None,
# #                 )
# #                 return pure(log, PX[tuple[GodInterpreter, GodState]]())

# #             model = eqx.filter_jit(lambda d, e: traverseM(updateStep)(d).func(outerInterpreter, e))
# #             return (
# #                 model,
# #                 lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
# #                 innerLibrary,
# #             )


# # def create_batched_model(
# #     test_dataset: Traversable[Traversable[InputOutput]],
# #     tr_to_val_env: Callable[[GodState, PRNG], GodState],
# #     tr_to_te_env: Callable[[GodState, PRNG], GodState],
# #     lossFn: Callable[[jax.Array, jax.Array], LOSS],
# #     initEnv: GodState,
# #     innerInterpreter: GodInterpreter,
# #     outerInterpreter: GodInterpreter,
# #     config: GodConfig,
# # ) -> tuple[
# #     Callable[[OhoData[Traversable[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
# #     Callable[[GodState], LOSS],
# # ]:
# #     innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
# #     innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
# #     outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

# #     innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
# #     innerLibrary = innerLibrary._replace(modelGradient=innerController)
# #     innerLibrary = aggregateBatchedGradients(innerLibrary, batch_env_form)
# #     innerController = logGradient(innerLibrary.modelGradient)
# #     innerLibrary = innerLibrary._replace(modelGradient=innerController)

# #     inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
# #     outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
# #     pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

# #     validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

# #     match outerLearner:
# #         case OfflineLearning():
# #             raise NotImplementedError("BPTT on batched is not implemented yet")

# #         case _:
# #             outerLibrary: Library[
# #                 IdentityF[OhoData[Traversable[Traversable[InputOutput]]]],
# #                 GodInterpreter,
# #                 GodState,
# #                 IdentityF[Traversable[Traversable[PREDICTION]]],
# #             ]

# #             outerLibrary = endowBilevelOptimization(
# #                 innerLibrary,
# #                 doOptimizerStep,
# #                 innerInterpreter,
# #                 outerLearner,
# #                 lambda a, b: LOSS(
# #                     jnp.mean(eqx.filter_vmap(eqx.filter_vmap(lossFn))(a.value.value, b.validation.value.value.y))
# #                 ),
# #                 tr_to_val_env,
# #                 pad_val_grad_by,
# #             )

# #             outerController = logGradient(outerLibrary.modelGradient)
# #             outerLibrary = outerLibrary._replace(modelGradient=outerController)

# #             @do()
# #             def updateStep(oho_data: OhoData[Traversable[Traversable[InputOutput]]]):
# #                 print("recompiled")
# #                 env = yield from get(PX[GodState]())
# #                 interpreter = yield from ask(PX[GodInterpreter]())
# #                 hyperparameters = yield from interpreter.getRecurrentParam
# #                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

# #                 # te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
# #                 vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
# #                 tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)
# #                 te = 0

# #                 _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

# #                 # code smell but what can you do, no maybe monad or elvis operator...
# #                 def safe_norm(x):
# #                     return jnp.linalg.norm(x) if x is not None else None

# #                 log = AllLogs(
# #                     train_loss=tr / config.tr_examples_per_epoch,
# #                     validation_loss=vl / config.vl_examples_per_epoch,
# #                     test_loss=te / config.numTe,
# #                     hyperparameters=hyperparameters,
# #                     inner_learning_rate=innerInterpreter.getLearningRate(env),
# #                     parameter_norm=safe_norm(weights),
# #                     oho_gradient=env.outerLogs.gradient,
# #                     train_gradient=env.innerLogs.gradient,
# #                     validation_gradient=env.outerLogs.validationGradient,
# #                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
# #                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
# #                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
# #                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
# #                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
# #                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
# #                     jacobian=env.innerLogs.hessian,
# #                     hessian=env.outerLogs.hessian,
# #                     rnn_activation_norm=safe_norm(env.rnnState.activation),
# #                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
# #                     if config.log_accumulate_influence
# #                     else None,
# #                 )
# #                 logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))
# #                 return pure(logs, PX[tuple[GodInterpreter, GodState]]())

# #             model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
# #             return (
# #                 model,
# #                 lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
# #                 innerLibrary,
# #             )
