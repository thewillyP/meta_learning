from typing import Callable, Iterator, Union
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
import torchvision
import time
from toolz import take, mapcat

from lib.config import *
from lib.create_axes import create_axes
from lib.create_env import create_env
from lib.create_interface import create_learn_interfaces, create_transition_interfaces
from lib.datasets import (
    create_multi_epoch_dataloader,
    flatten_and_cast,
    generate_add_task_dataset,
    standard_dataloader,
    target_transform,
)
from lib.inference import create_inferences
from lib.interface import ClassificationInterface
from lib.lib_types import *
from lib.util import (
    create_fractional_list,
    infinite_keys,
    reshape_timeseries,
    subset_n,
)


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
        dataset=MnistConfig(n_in=28),
        num_base_epochs=1,
        checkpoint_every_n_minibatches=1_000,
        seed=SeedConfig(data_seed=1, parameter_seed=1, test_seed=1),
        lossFn="cross_entropy_with_integer_labels",
        transition_function={
            0: NNLayer(
                n=32,
                activation_fn="tanh",
                use_bias=True,
            ),
            1: NNLayer(
                n=32,
                activation_fn="tanh",
                use_bias=True,
            ),
        },
        readout_function=FeedForwardConfig(ffw_layers={0: NNLayer(n=10, activation_fn="identity", use_bias=True)}),
        learners={
            0: LearnConfig(  # normal feedforward backprop
                learner=BPTTConfig(),
                optimizer=SGDConfig(
                    learning_rate=0.01,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
            ),
            1: LearnConfig(  # normal OHO
                learner=RTRLConfig(),
                optimizer=SGDConfig(
                    learning_rate=0.01,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
                track_logs=True,
                track_special_logs=False,
            ),
        },
        data={
            0: DataConfig(
                train_percent=80,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
            1: DataConfig(
                train_percent=20,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=28,
                num_times_to_avg_in_timeseries=1,
            ),
        },
        num_virtual_minibatches_per_turn=10,
        ignore_validation_inference_recurrence=True,
        readout_uses_input_data=False,
        num_minibatches_in_epoch=100,
    )

    converter = Converter()
    configure_tagged_union(Union[RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig], converter)
    configure_tagged_union(Union[SGDConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig], converter)
    configure_tagged_union(Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig], converter)

    _config = task.connect(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)
    task.execute_remotely(queue_name="slurm", clone=False, exit_process=True)

    if not config.clearml_run:
        return

    # RNG Stuff
    data_prng = PRNG(jax.random.key(config.seed.data_seed))
    env_prng = PRNG(jax.random.key(config.seed.parameter_seed))
    test_prng = PRNG(jax.random.key(config.seed.test_seed))
    dataloader_prng, dataset_gen_prng, torch_prng = jax.random.split(data_prng, 3)
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

    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            data_size = dict(zip(config.data.keys(), subset_n(n, percentages)))
            dataset_te = generate_add_task_dataset(nTest, t1, t2, tau_task, test_prng)
            n_in_shape = dataset_te[0].shape[1:]

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            for idx, (i, data_config) in enumerate(sorted(config.data.items())):
                data_prng, dataset_gen_prng = jax.random.split(dataset_gen_prng, 2)
                X_vl, Y_vl = generate_add_task_dataset(data_size[i], t1, t2, tau_task, data_prng)
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]

                def get_dataloader(rng: PRNG, X_vl=X_vl, Y_vl=Y_vl, data_config=data_config):
                    return standard_dataloader(
                        X_vl, Y_vl, X_vl.shape[0], data_config.num_examples_in_minibatch, X_vl.shape[1], rng
                    )

                datasets[idx] = get_dataloader
                # 1. check when to reset after consume concrete example
                # 2. check when is last padded minibatch

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
            n_in_shape = (n_in,)

            match config.dataset:
                case MnistConfig():
                    dataset_factory = torchvision.datasets.MNIST
                case FashionMnistConfig():
                    dataset_factory = torchvision.datasets.FashionMNIST

            dataset = dataset_factory(root=f"{config.data_root_dir}/data", train=True, download=True)
            dataset_te = dataset_factory(root=f"{config.data_root_dir}/data", train=False, download=True)
            xs = jax.vmap(flatten_and_cast, in_axes=(0, None))(dataset.data.numpy(), n_in)
            ys = jax.vmap(target_transform, in_axes=(0, None))(dataset.targets.numpy(), xs.shape[1])
            xs_te = jax.vmap(flatten_and_cast, in_axes=(0, None))(dataset_te.data.numpy(), n_in)
            ys_te = jax.vmap(target_transform, in_axes=(0, None))(dataset_te.targets.numpy(), xs_te.shape[1])
            dataset_te = (xs_te, ys_te)

            perm = jax.random.permutation(dataset_gen_prng, len(xs))
            split_indices = jnp.cumsum(jnp.array(subset_n(len(xs), percentages)))[:-1]
            val_indices = jnp.split(perm, split_indices)

            datasets: dict[int, Callable[[PRNG], Iterator[tuple[jax.Array, jax.Array]]]] = {}
            virtual_minibatches: dict[int, int] = {}
            for idx, ((i, data_config), val_idx) in enumerate(zip(sorted(config.data.items()), val_indices)):
                X_vl = xs[val_idx]
                Y_vl = ys[val_idx]
                n_consume = data_config.num_steps_in_timeseries * data_config.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                virtual_minibatches[idx] = X_vl.shape[1]

                def get_dataloader(rng: PRNG, X_vl=X_vl, Y_vl=Y_vl, data_config=data_config):
                    return standard_dataloader(
                        X_vl, Y_vl, X_vl.shape[0], data_config.num_examples_in_minibatch, X_vl.shape[1], rng
                    )

                datasets[idx] = get_dataloader

        case _:
            raise ValueError("Invalid dataset")

    if config.ignore_validation_inference_recurrence:
        if not all(virtual_minibatches[k] == 1 for i, k in enumerate(sorted(virtual_minibatches.keys())) if i > 0):
            raise ValueError(
                "When ignore_validation_inference_recurrence is True, all validation datasets except the first must have num_virtual_minibatches_per_turn=1."
            )

    subkeys = jax.random.split(dataloader_prng, len(datasets))
    first_iterator = datasets[0](subkeys[0])
    cycling_iterators = [mapcat(datasets[i], infinite_keys(subkeys[i])) for i in range(1, len(datasets))]
    first_iterator = create_multi_epoch_dataloader(first_iterator, config.num_minibatches_in_epoch)
    cycling_iterators = [create_multi_epoch_dataloader(it, config.num_minibatches_in_epoch) for it in cycling_iterators]
    the_dataloader = zip(first_iterator, *cycling_iterators)

    for (x1, y1), (x2, y2) in take(3, the_dataloader):
        print(f"First dataset: {x1.shape}, {y1.shape}")
        print(f"Second dataset: {x2.shape}, {y2.shape}")

    return

    learn_interfaces = create_learn_interfaces(config)
    inference_interface = create_transition_interfaces(config)
    data_interface = ClassificationInterface[tuple[jax.Array, jax.Array]](
        get_input=lambda data: data[0],
        get_target=lambda data: data[1],
    )
    env = create_env(config, n_in_shape, learn_interfaces, env_prng)
    axes = create_axes(env, inference_interface)
    # pretty print env
    print(env)
    for _env in axes.values():
        print(f"Function for axes")
        print(_env)

    inferences = create_inferences(config, inference_interface, data_interface, axes)
    # make test data
    x = jnp.ones((100, 5, n_in_shape[0]))
    y = jnp.ones((100, 5, 10))

    arr, static = eqx.partition(env, eqx.is_array)

    def make_step(carry, data):
        model = eqx.combine(carry, static)
        update_model, out = inferences[0](model, data)
        carry, _ = eqx.partition(update_model, eqx.is_array)
        return carry, out

    # Speed comparison with PyTorch RNN
    print("\n" + "=" * 50)
    print("SPEED COMPARISON: JAX vs PyTorch RNN")
    print("=" * 50)

    batch_size, seq_len, input_size = x.shape
    hidden_size = 32  # From your config
    output_size = 10

    # Create PyTorch RNN model with 2 layers
    class PyTorchRNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
            self.linear = torch.nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.linear(out)

    pytorch_model = PyTorchRNN(input_size, hidden_size, output_size)
    pytorch_model.eval()

    # Convert JAX data to PyTorch tensors
    x_torch = torch.from_numpy(np.array(x)).float()

    # Create data for scan (1000 copies of the same data)
    scan_data = jax.tree.map(lambda x: jnp.repeat(x[None], 10000, axis=0), batched(traverse((x, y))))

    # Create and compile the scan function
    jax_scan_fn = (
        eqx.filter_jit(lambda data, init_model: jax.lax.scan(make_step, init_model, data), donate="all-except-first")
        .lower(scan_data, arr)
        .compile()
    )

    # Create PyTorch batch function for fair comparison
    def pytorch_batch_forward(model, x_batch):
        outputs = []
        for x in x_batch:
            with torch.no_grad():
                out = model(x)
                outputs.append(out)
        return torch.stack(outputs)

    # Create batch data for PyTorch
    x_torch_batch = x_torch.unsqueeze(0).repeat(10000, 1, 1, 1)

    # Warmup runs
    print("Warming up...")
    for _ in range(5):
        arr, all_outputs = jax_scan_fn(scan_data, arr)
        jax.block_until_ready(all_outputs)

        pytorch_batch_outputs = pytorch_batch_forward(pytorch_model, x_torch_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # JAX timing with scan
    print("Timing JAX scan inference...")
    jax_times = []
    for _ in range(5):
        start_time = time.time()
        arr, all_outputs = jax_scan_fn(scan_data, arr)
        jax.block_until_ready(all_outputs)
        end_time = time.time()
        jax_times.append(end_time - start_time)

    # PyTorch timing with batch
    print("Timing PyTorch batch inference...")
    pytorch_times = []
    for _ in range(5):
        start_time = time.time()
        pytorch_batch_outputs = pytorch_batch_forward(pytorch_model, x_torch_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        pytorch_times.append(end_time - start_time)

    # Results
    jax_mean = np.mean(jax_times) * 1000  # Convert to ms
    jax_std = np.std(jax_times) * 1000
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000

    # Per-step timing
    jax_per_step = jax_mean / 10000
    pytorch_per_step = pytorch_mean / 10000

    print(f"\nResults (10 runs of 1000 steps each):")
    print(f"JAX Scan Total:      {jax_mean:.3f} ± {jax_std:.3f} ms")
    print(f"PyTorch Batch Total: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    print(f"JAX per step:        {jax_per_step:.6f} ms")
    print(f"PyTorch per step:    {pytorch_per_step:.6f} ms")
    print(
        f"Speedup factor:      {pytorch_mean / jax_mean:.2f}x {'(JAX faster)' if jax_mean < pytorch_mean else '(PyTorch faster)'}"
    )

    print(f"\nData shape: {x.shape}")
    print(f"Model size: {input_size} -> {hidden_size} (2 layers) -> {output_size}")
    print(f"Total operations: 1000 steps x {batch_size} batch size")

    # inferences = create_inferences(config, inference_interface, data_interface, axes)
    # # make test data
    # x = jnp.ones((100, 5, n_in_shape[0]))
    # y = jnp.ones((100, 5, 10))
    # # inference = (
    # #     eqx.filter_jit(lambda x, y: inferences[0](y, x), donate="all-except-first")
    # #     .lower(batched(traverse((x, y))), env)
    # #     .compile()
    # # )

    # # env, out = inference(env, batched(traverse((x, y))))
    # # print("Inference")
    # # print(out)
    # # print(env)
    # # 1. create inference function dict[int, Callable] for each level
    # # 2. create meta learning function that hierarchically takes dict[int, Callable]
    # # 3. inrepreter config to get dict[int, Callable] that will be passed into the folds above
    # # 4. resetting?
    # # 5. data loading?
    # # 6. combining vl data with prev data when building meta learning function?

    # flat_model, treedef_model = jax.tree_util.tree_flatten(env)

    # def make_step(data, flat_model):
    #     model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    #     update_model, out = inferences[0](model, data)
    #     flat_update_model = jax.tree_util.tree_leaves(update_model)
    #     return flat_update_model, out

    # inference = (
    #     eqx.filter_jit(make_step, donate="all-except-first").lower(batched(traverse((x, y))), flat_model).compile()
    # )

    # # Speed comparison with PyTorch RNN
    # print("\n" + "=" * 50)
    # print("SPEED COMPARISON: JAX vs PyTorch RNN")
    # print("=" * 50)

    # batch_size, seq_len, input_size = x.shape
    # hidden_size = 32  # From your config
    # output_size = 10

    # # Create PyTorch RNN model with 2 layers
    # class PyTorchRNN(torch.nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super().__init__()
    #         self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
    #         self.linear = torch.nn.Linear(hidden_size, output_size)

    #     def forward(self, x):
    #         out, _ = self.rnn(x)
    #         return self.linear(out)

    # pytorch_model = PyTorchRNN(input_size, hidden_size, output_size)
    # pytorch_model.eval()

    # # Convert JAX data to PyTorch tensors
    # x_torch = torch.from_numpy(np.array(x)).float()

    # # Warmup runs
    # print("Warming up...")
    # for _ in range(10):
    #     flat_model, _ = inference(batched(traverse((x, y))), flat_model)
    #     with torch.no_grad():
    #         _ = pytorch_model(x_torch)

    # # JAX timing
    # print("Timing JAX inference...")
    # jax_times = []
    # for _ in range(1000):
    #     start_time = time.time()
    #     flat_model, out_jax = inference(batched(traverse((x, y))), flat_model)
    #     jax.block_until_ready(out_jax)
    #     end_time = time.time()
    #     jax_times.append(end_time - start_time)

    # # PyTorch timing
    # print("Timing PyTorch RNN...")
    # pytorch_times = []
    # with torch.no_grad():
    #     for _ in range(1000):
    #         start_time = time.time()
    #         out_torch = pytorch_model(x_torch)
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         end_time = time.time()
    #         pytorch_times.append(end_time - start_time)

    # # Results
    # jax_mean = np.mean(jax_times) * 1000  # Convert to ms
    # jax_std = np.std(jax_times) * 1000
    # pytorch_mean = np.mean(pytorch_times) * 1000
    # pytorch_std = np.std(pytorch_times) * 1000

    # print(f"\nResults (100 runs):")
    # print(f"JAX Inference:    {jax_mean:.3f} ± {jax_std:.3f} ms")
    # print(f"PyTorch RNN:      {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    # print(
    #     f"Speedup factor:   {pytorch_mean / jax_mean:.2f}x {'(JAX faster)' if jax_mean < pytorch_mean else '(PyTorch faster)'}"
    # )

    # print(f"\nData shape: {x.shape}")
    # print(f"Model size: {input_size} -> {hidden_size} (2 layers) -> {output_size}")


# def create_learner(
#     learner: str, rtrl_use_fwd: bool, uoro_std
# ) -> RTRL | RFLO | UORO | IdentityLearner | OfflineLearning:
#     match learner:
#         case "rtrl":
#             return RTRL(rtrl_use_fwd)
#         case "rflo":
#             return RFLO()
#         case "uoro":
#             return UORO(lambda key, shape: jax.random.uniform(key, shape, minval=-uoro_std, maxval=uoro_std))
#         case "identity":
#             return IdentityLearner()
#         case "bptt":
#             return OfflineLearning()
#         case _:
#             raise ValueError("Invalid learner")


# def create_rnn_learner(
#     learner: RTRL | RFLO | UORO | IdentityLearner | OfflineLearning,
#     lossFn: Callable[[jax.Array, jax.Array], LOSS],
#     arch: Literal["rnn", "ffn"],
# ) -> Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]:
#     match arch:
#         case "rnn":
#             stepFn = lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
#             readoutFn = doRnnReadout
#         case "ffn":
#             stepFn = (
#                 lambda d: doFeedForwardStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
#             )
#             readoutFn = doFeedForwardReadout

#     lfn = lambda a, b: lossFn(a, b.y)
#     match learner:
#         case OfflineLearning():
#             bptt_library: Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]
#             bptt_library = learner.createLearner(
#                 stepFn,
#                 readoutFn,
#                 lfn,
#                 readoutRecurrentError(readoutFn, lfn),
#             )
#             return bptt_library
#         case _:
#             # library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
#             _library: Library[IdentityF[InputOutput], GodInterpreter, GodState, IdentityF[PREDICTION]]
#             _library = learner.createLearner(
#                 stepFn,
#                 readoutFn,
#                 lfn,
#                 readoutRecurrentError(doRnnReadout, lfn),
#             )
#             library = Library[InputOutput, GodInterpreter, GodState, PREDICTION](
#                 model=lambda d: _library.model(IdentityF(d)),
#                 modelLossFn=lambda d: _library.modelLossFn(IdentityF(d)),
#                 modelGradient=lambda d: _library.modelGradient(IdentityF(d)),
#             )
#             return foldrLibrary(library)


# def train_loop_IO[D](
#     tr_dataset: Traversable[Traversable[InputOutput]],
#     vl_dataset: Traversable[Traversable[InputOutput]],
#     to_combined_ds: Callable[[Traversable[Traversable[InputOutput]], Traversable[Traversable[InputOutput]]], D],
#     model: Callable[[D, GodState], tuple[Traversable[AllLogs], GodState]],
#     env: GodState,
#     refresh_env: Callable[[GodState], GodState],
#     config: GodConfig,
#     checkpoint_fn: Callable[[GodState], None],
#     log_fn: Callable[[AllLogs], None],
#     te_loss: Callable[[GodState], LOSS],
#     statistic: Callable[[GodState], float],
# ) -> None:
#     tr_dataset = PyTreeDataset(tr_dataset)
#     vl_dataset = PyTreeDataset(vl_dataset)

#     if config.batch_or_online == "batch":
#         vl_batch_size = config.batch_vl
#         vl_sampler = RandomSampler(vl_dataset)
#         tr_dl = lambda b: DataLoader(
#             b, batch_size=config.batch_tr, shuffle=True, collate_fn=jax_collate_fn, drop_last=True
#         )
#         # doesn't make sense to do this in batch case. batch=subsequence in online
#         env = copy.replace(env, start_example=0)
#     else:
#         vl_batch_size = config.batch_tr  # same size -> conjoin with tr batch with to_combined_ds
#         vl_sampler = RandomSampler(vl_dataset, replacement=True, num_samples=len(tr_dataset))
#         tr_dl = lambda b: DataLoader(b, batch_size=config.batch_tr, shuffle=False, collate_fn=jax_collate_fn)

#     vl_dataloader = DataLoader(
#         vl_dataset, batch_size=vl_batch_size, sampler=vl_sampler, collate_fn=jax_collate_fn, drop_last=True
#     )

#     def infinite_loader(loader):
#         while True:
#             for batch in loader:
#                 yield batch

#     vl_dataloader = infinite_loader(vl_dataloader)

#     start = time.time()
#     num_batches_seen_so_far = 0
#     all_logs: list[Traversable[AllLogs]] = []
#     for epoch in range(env.start_epoch, config.num_retrain_loops):
#         print(f"Epoch {epoch + 1}/{config.num_retrain_loops}")
#         batched_dataset = Subset(tr_dataset, indices=range(env.start_example, len(tr_dataset)))
#         tr_dataloader = tr_dl(batched_dataset)

#         for i, (tr_batch, vl_batch) in enumerate(zip(tr_dataloader, vl_dataloader)):
#             ds_batch = to_combined_ds(tr_batch, vl_batch)
#             batch_size = len(jax.tree.leaves(ds_batch)[0])
#             env = refresh_env(env)
#             logs, env = model(ds_batch, env)

#             env = eqx.tree_at(lambda t: t.start_example, env, env.start_example + batch_size)
#             all_logs.append(logs)
#             checkpoint_condition = num_batches_seen_so_far + i + 1
#             if checkpoint_condition % config.checkpoint_interval == 0:
#                 checkpoint_fn(
#                     copy.replace(
#                         env,
#                         inner_prng=jax.random.key_data(env.inner_prng),
#                         outer_prng=jax.random.key_data(env.outer_prng),
#                     )
#                 )

#             print(
#                 f"Batch {i + 1}/{len(tr_dataloader)}, Loss: {logs.value.train_loss[-1]}, LR: {logs.value.inner_learning_rate[-1]}"
#             )

#         # env = eqx.tree_at(lambda t: t.start_epoch, env, epoch + 1)
#         env = eqx.tree_at(lambda t: t.start_example, env, 0)
#         num_batches_seen_so_far += len(tr_dataloader)

#     end = time.time()
#     print(f"Training time: {end - start} seconds")

#     total_logs: Traversable[AllLogs] = jax.tree.map(lambda *xs: jnp.concatenate(xs), *all_logs)

#     log_fn(total_logs.value)
#     checkpoint_fn(
#         copy.replace(
#             env,
#             inner_prng=jax.random.key_data(env.inner_prng),
#             outer_prng=jax.random.key_data(env.outer_prng),
#         )
#     )

#     def safe_norm(x):
#         return jnp.linalg.norm(x) if x is not None else None

#     # log wandb partial metrics
#     for log_tree_ in tree_unstack_lazy(total_logs.value):
#         log_data: AllLogs = jax.tree.map(
#             lambda x: jnp.real(x) if x is not None and jnp.all(jnp.isfinite(x)) else None, log_tree_
#         )
#         wandb.log(
#             {
#                 "train_loss": log_data.train_loss,
#                 "validation_loss": log_data.validation_loss,
#                 "test_loss": log_data.test_loss,
#                 "hyperparameters": log_data.hyperparameters,
#                 "inner_learning_rate": log_data.inner_learning_rate,
#                 "parameter_norm": log_data.parameter_norm,
#                 "oho_gradient": log_data.oho_gradient,
#                 "train_gradient": log_data.train_gradient,
#                 "validation_gradient": log_data.validation_gradient,
#                 "oho_gradient_norm": safe_norm(log_data.oho_gradient),
#                 "train_gradient_norm": safe_norm(log_data.train_gradient),
#                 "validation_gradient_norm": safe_norm(log_data.validation_gradient),
#                 "immediate_influence_tensor_norm": log_data.immediate_influence_tensor_norm,
#                 "inner_influence_tensor_norm": log_data.inner_influence_tensor_norm,
#                 "outer_influence_tensor_norm": log_data.outer_influence_tensor_norm,
#                 "largest_jacobian_eigenvalue": log_data.largest_jacobian_eigenvalue,
#                 "largest_influence_eigenvalue": log_data.largest_hessian_eigenvalue,
#                 "jacobian_eigenvalues": log_data.jacobian,
#                 "rnn_activation_norm": log_data.rnn_activation_norm,
#                 "immediate_influence_tensor": jnp.ravel(log_data.immediate_influence_tensor)
#                 if log_data.immediate_influence_tensor is not None
#                 else None,
#                 "outer_influence_tensor": jnp.ravel(log_data.outer_influence_tensor)
#                 if log_data.outer_influence_tensor is not None
#                 else None,
#             }
#         )

#     ee = te_loss(env)
#     eee = statistic(env)
#     print(ee)
#     print(eee)
#     wandb.log({"test_loss": ee, "test_statistic": eee})


# def create_online_model(
#     test_dataset: Traversable[InputOutput],
#     tr_to_val_env: Callable[[GodState, PRNG], GodState],
#     tr_to_te_env: Callable[[GodState, PRNG], GodState],
#     lossFn: Callable[[jax.Array, jax.Array], LOSS],
#     initEnv: GodState,
#     innerInterpreter: GodInterpreter,
#     outerInterpreter: GodInterpreter,
#     config: GodConfig,
# ) -> tuple[
#     Callable[[Traversable[OhoData[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
#     Callable[[GodState], LOSS],
# ]:
#     innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
#     innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
#     outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

#     innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
#     innerController = logGradient(innerController)
#     innerLibrary = innerLibrary._replace(modelGradient=innerController)

#     inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
#     outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
#     pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

#     validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

#     match outerLearner:
#         case OfflineLearning():
#             _outerLibrary: Library[
#                 Traversable[OhoData[Traversable[InputOutput]]],
#                 GodInterpreter,
#                 GodState,
#                 Traversable[Traversable[PREDICTION]],
#             ]
#             _outerLibrary = endowBilevelOptimization(
#                 innerLibrary,
#                 doOptimizerStep,
#                 innerInterpreter,
#                 outerLearner,
#                 lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
#                 tr_to_val_env,
#                 pad_val_grad_by,
#             )

#             outerController = logGradient(_outerLibrary.modelGradient)
#             _outerLibrary = _outerLibrary._replace(modelGradient=outerController)

#             @do()
#             def updateStep(oho_data: Traversable[OhoData[Traversable[InputOutput]]]):
#                 print("recompiled")
#                 env = yield from get(PX[GodState]())
#                 interpreter = yield from ask(PX[GodInterpreter]())
#                 hyperparameters = yield from interpreter.getRecurrentParam
#                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

#                 te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
#                 vl, _ = _outerLibrary.modelLossFn(oho_data).func(outerInterpreter, env)
#                 tr, _ = (
#                     foldrLibrary(innerLibrary)
#                     .modelLossFn(Traversable(oho_data.value.payload))
#                     .func(innerInterpreter, env)
#                 )

#                 def safe_norm(x):
#                     return jnp.linalg.norm(x) if x is not None else None

#                 log = AllLogs(
#                     train_loss=tr / config.tr_examples_per_epoch,
#                     validation_loss=vl / config.vl_examples_per_epoch,
#                     test_loss=te / config.numTe,
#                     hyperparameters=hyperparameters,
#                     inner_learning_rate=innerInterpreter.getLearningRate(env),
#                     parameter_norm=safe_norm(weights),
#                     oho_gradient=env.outerLogs.gradient,
#                     train_gradient=env.innerLogs.gradient,
#                     validation_gradient=env.outerLogs.validationGradient,
#                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
#                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
#                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
#                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
#                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
#                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
#                     jacobian=env.innerLogs.hessian,
#                     hessian=env.outerLogs.hessian,
#                     rnn_activation_norm=safe_norm(env.rnnState.activation),
#                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
#                     if config.log_accumulate_influence
#                     else None,
#                 )
#                 logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))

#                 _ = yield from _outerLibrary.modelGradient(oho_data).flat_map(doOptimizerStep)
#                 return pure(logs, PX[tuple[GodInterpreter, GodState]]())

#             model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
#             return model

#         case _:
#             outerLibrary: Library[
#                 IdentityF[OhoData[Traversable[InputOutput]]],
#                 GodInterpreter,
#                 GodState,
#                 IdentityF[Traversable[PREDICTION]],
#             ]
#             outerLibrary = endowBilevelOptimization(
#                 innerLibrary,
#                 doOptimizerStep,
#                 innerInterpreter,
#                 outerLearner,
#                 lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
#                 tr_to_val_env,
#                 pad_val_grad_by,
#             )

#             outerController = logGradient(outerLibrary.modelGradient)
#             outerLibrary = outerLibrary._replace(modelGradient=outerController)

#             @do()
#             def updateStep(oho_data: OhoData[Traversable[InputOutput]]):
#                 print("recompiled")
#                 env = yield from get(PX[GodState]())
#                 interpreter = yield from ask(PX[GodInterpreter]())
#                 hyperparameters = yield from interpreter.getRecurrentParam
#                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

#                 te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
#                 vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
#                 tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)

#                 _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

#                 # code smell but what can you do, no maybe monad or elvis operator...
#                 def safe_norm(x):
#                     return jnp.linalg.norm(x) if x is not None else None

#                 log = AllLogs(
#                     train_loss=tr / config.tr_examples_per_epoch,
#                     validation_loss=vl / config.vl_examples_per_epoch,
#                     test_loss=te / config.numTe,
#                     hyperparameters=hyperparameters,
#                     inner_learning_rate=innerInterpreter.getLearningRate(env),
#                     parameter_norm=safe_norm(weights),
#                     oho_gradient=env.outerLogs.gradient,
#                     train_gradient=env.innerLogs.gradient,
#                     validation_gradient=env.outerLogs.validationGradient,
#                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
#                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
#                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
#                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
#                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
#                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
#                     jacobian=env.innerLogs.hessian,
#                     hessian=env.outerLogs.hessian,
#                     rnn_activation_norm=safe_norm(env.rnnState.activation),
#                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
#                     if config.log_accumulate_influence
#                     else None,
#                 )
#                 return pure(log, PX[tuple[GodInterpreter, GodState]]())

#             model = eqx.filter_jit(lambda d, e: traverseM(updateStep)(d).func(outerInterpreter, e))
#             return (
#                 model,
#                 lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
#                 innerLibrary,
#             )


# def create_batched_model(
#     test_dataset: Traversable[Traversable[InputOutput]],
#     tr_to_val_env: Callable[[GodState, PRNG], GodState],
#     tr_to_te_env: Callable[[GodState, PRNG], GodState],
#     lossFn: Callable[[jax.Array, jax.Array], LOSS],
#     initEnv: GodState,
#     innerInterpreter: GodInterpreter,
#     outerInterpreter: GodInterpreter,
#     config: GodConfig,
# ) -> tuple[
#     Callable[[OhoData[Traversable[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
#     Callable[[GodState], LOSS],
# ]:
#     innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
#     innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
#     outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

#     innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
#     innerLibrary = innerLibrary._replace(modelGradient=innerController)
#     innerLibrary = aggregateBatchedGradients(innerLibrary, batch_env_form)
#     innerController = logGradient(innerLibrary.modelGradient)
#     innerLibrary = innerLibrary._replace(modelGradient=innerController)

#     inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
#     outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
#     pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

#     validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

#     match outerLearner:
#         case OfflineLearning():
#             raise NotImplementedError("BPTT on batched is not implemented yet")

#         case _:
#             outerLibrary: Library[
#                 IdentityF[OhoData[Traversable[Traversable[InputOutput]]]],
#                 GodInterpreter,
#                 GodState,
#                 IdentityF[Traversable[Traversable[PREDICTION]]],
#             ]

#             outerLibrary = endowBilevelOptimization(
#                 innerLibrary,
#                 doOptimizerStep,
#                 innerInterpreter,
#                 outerLearner,
#                 lambda a, b: LOSS(
#                     jnp.mean(eqx.filter_vmap(eqx.filter_vmap(lossFn))(a.value.value, b.validation.value.value.y))
#                 ),
#                 tr_to_val_env,
#                 pad_val_grad_by,
#             )

#             outerController = logGradient(outerLibrary.modelGradient)
#             outerLibrary = outerLibrary._replace(modelGradient=outerController)

#             @do()
#             def updateStep(oho_data: OhoData[Traversable[Traversable[InputOutput]]]):
#                 print("recompiled")
#                 env = yield from get(PX[GodState]())
#                 interpreter = yield from ask(PX[GodInterpreter]())
#                 hyperparameters = yield from interpreter.getRecurrentParam
#                 weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

#                 # te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
#                 vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
#                 tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)
#                 te = 0

#                 _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

#                 # code smell but what can you do, no maybe monad or elvis operator...
#                 def safe_norm(x):
#                     return jnp.linalg.norm(x) if x is not None else None

#                 log = AllLogs(
#                     train_loss=tr / config.tr_examples_per_epoch,
#                     validation_loss=vl / config.vl_examples_per_epoch,
#                     test_loss=te / config.numTe,
#                     hyperparameters=hyperparameters,
#                     inner_learning_rate=innerInterpreter.getLearningRate(env),
#                     parameter_norm=safe_norm(weights),
#                     oho_gradient=env.outerLogs.gradient,
#                     train_gradient=env.innerLogs.gradient,
#                     validation_gradient=env.outerLogs.validationGradient,
#                     immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
#                     outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
#                     outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
#                     inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
#                     largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
#                     largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
#                     jacobian=env.innerLogs.hessian,
#                     hessian=env.outerLogs.hessian,
#                     rnn_activation_norm=safe_norm(env.rnnState.activation),
#                     immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
#                     if config.log_accumulate_influence
#                     else None,
#                 )
#                 logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))
#                 return pure(logs, PX[tuple[GodInterpreter, GodState]]())

#             model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
#             return (
#                 model,
#                 lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
#                 innerLibrary,
#             )
