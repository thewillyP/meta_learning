from typing import Literal, Union
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
import copy
import torchvision

from lib.config import *
from lib.datasets import flatten_and_cast, generate_add_task_dataset, standard_dataloader, target_transform
from lib.env import RNN, CustomSequential, General, GodState, Inference, Learning
from lib.lib_types import *
from lib.util import create_fractional_list, get_activation_fn, reshape_timeseries, subset_n


def create_env(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> GodState:
    env = GodState(
        states=None,
        base_inference=None,
        start_epoch=0,
    )

    transition_inference: dict[int, Inference] = {}
    for i, transition_fn in config.transition_function.items():
        match transition_fn:
            case NNLayer(n_h, activation_fn, use_bias):
                n_in, _ = n_in_shape
                prgn1, prng2, prng3, prng = jax.random.split(prng, 4)
                activation = ACTIVATION(jax.random.normal(prgn1, (n_h,)))
                W_in = jax.random.normal(prng2, (n_h, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(prng3, (n_h, n_h)))[0]
                w_rec = jnp.hstack([W_rec, W_in])
                b_rec = jnp.zeros((n_h, 1)) if use_bias else None

                transition_inference[i] = Inference(
                    rnn=RNN(
                        activation=activation,
                        w_rec=w_rec,
                        b_rec=b_rec,
                        n_h=n_h,
                        n_in=n_in,
                        activation_fn=activation_fn,
                    )
                )

            case _:
                raise ValueError("Unsupported transition function")

    match config.readout_function:
        case FeedForwardConfig(ffw_layers):
            layers = []
            n_in, _ = n_in_shape
            for i, layer in ffw_layers.items():
                match layer:
                    case NNLayer(n, activation_fn, use_bias):
                        layers.append((n, use_bias, get_activation_fn(activation_fn)))
            prng1, prng = jax.random.split(prng)
            readout_fn = Inference(readout_fn=CustomSequential(layers, n_in, prng1))
        case _:
            raise ValueError("Unsupported readout function")

    env = copy.replace(env, base_inference=(transition_inference, readout_fn))

    states: dict[int, tuple[General, tuple[dict[int, Inference], Inference], Learning]] = {}
    for i,


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
                train_percent=80,
                log_influence_tensor=False,
                log_immediate_influence_tensor=False,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=1,
                num_times_to_avg_in_timeseries=1,
                learner=BPTTConfig(),
                optimizer=SGDConfig(
                    learning_rate=0.01,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
            ),
            1: LearnConfig(  # normal OHO
                train_percent=20,
                log_influence_tensor=False,
                log_immediate_influence_tensor=False,
                num_examples_in_minibatch=100,
                num_steps_in_timeseries=1,
                num_times_to_avg_in_timeseries=1,
                learner=RTRLConfig(),
                optimizer=SGDConfig(
                    learning_rate=0.01,
                ),
                hyperparameter_parametrization="softplus",
                lanczos_iterations=0,
            ),
        },
        num_virtual_minibatches_per_turn=10,
    )

    converter = Converter()
    configure_tagged_union(Union[RTRLConfig, BPTTConfig, IdentityConfig, RFLOConfig, UOROConfig], converter)
    configure_tagged_union(
        Union[SGDConfig, SGDPositiveConfig, SGDNormalizedConfig, SGDClipConfig, AdamConfig], converter
    )
    configure_tagged_union(Union[MnistConfig, FashionMnistConfig, DelayAddOnlineConfig], converter)

    _config = task.connect(converter.unstructure(config), name="config")
    config = converter.structure(_config, GodConfig)
    task.execute_remotely(queue_name="slurm", clone=False, exit_process=True)

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
    percentages = create_fractional_list([x.train_percent / 100 for x in config.learners.values()])
    if percentages is None:
        raise ValueError("Learner percentages must sum to 100.")

    match config.dataset:
        case DelayAddOnlineConfig(t1, t2, tau_task, n, nTest):
            data_size = dict(zip(config.learners.keys(), subset_n(n, percentages)))
            dataset_te = generate_add_task_dataset(nTest, t1, t2, tau_task, test_prng)

            datasets = {}
            for i, learner in config.learners.items():
                data_prng, dataset_gen_prng = jax.random.split(dataset_gen_prng, 2)
                X_vl, Y_vl = generate_add_task_dataset(data_size[i], t1, t2, tau_task, data_prng)
                n_consume = learner.num_steps_in_timeseries * learner.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                num_virtual_minibatches = X_vl.shape[1]

                def get_dataloader(rng: PRNG):
                    return standard_dataloader(
                        X_vl, Y_vl, X_vl.shape[0], learner.num_examples_in_minibatch, X_vl.shape[1], rng
                    )

                datasets[i] = get_dataloader
                # 1. check when to reset after consume concrete example
                # 2. check when is last padded minibatch

        case MnistConfig(n_in) | FashionMnistConfig(n_in):
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
            split_indices = jnp.cumsum(subset_n(len(xs), percentages))[:-1]
            val_indices = jnp.split(perm, split_indices)

            datasets = {}
            for (i, learner), val_idx in zip(config.learners.items(), val_indices):
                X_vl = xs[val_idx]
                Y_vl = ys[val_idx]
                n_consume = learner.num_steps_in_timeseries * learner.num_times_to_avg_in_timeseries
                X_vl, last_unpadded_length = reshape_timeseries(X_vl, n_consume)
                Y_vl, _ = reshape_timeseries(Y_vl, n_consume)
                num_virtual_minibatches = X_vl.shape[1]

                def get_dataloader(rng: PRNG):
                    return standard_dataloader(
                        X_vl, Y_vl, X_vl.shape[0], learner.num_examples_in_minibatch, X_vl.shape[1], rng
                    )

                datasets[i] = get_dataloader

        case _:
            raise ValueError("Invalid dataset")

    # Model
    _, innerInterpreter, outerInterpreter = create_env(config, env_prng)
    load_env = eqx.filter_jit(load_env)
    match config.batch_or_online:
        case "batch":
            env_root_prng, tr_batches_prng = jax.random.split(env_prng, 2)
            tr_prngs = jax.random.split(tr_batches_prng, config.batch_tr)
            _env = load_env(config, env_root_prng)
            create_envs = eqx.filter_vmap(load_env, in_axes=(None, 0))
            create_envs = eqx.filter_jit(create_envs)
            _tr_env = create_envs(config, tr_prngs)
            tr_env = to_batched_form(_tr_env, _env)

            def combine_ds(
                tr_ds: Traversable[Traversable[InputOutput]], vl_ds: Traversable[Traversable[InputOutput]]
            ) -> OhoData[Traversable[Traversable[InputOutput]]]:
                return OhoData(payload=tr_ds, validation=vl_ds)

            def _tr_to_env(env: GodState, prng: PRNG, batch_size: int) -> GodState:
                vl_prngs = jax.random.split(prng, batch_size)
                vl_env = create_envs(config, vl_prngs)
                return to_batched_form(vl_env, env)

            def tr_to_val_env(env: GodState, prng: PRNG) -> GodState:
                return _tr_to_env(env, prng, config.batch_vl)

            def tr_to_te_env(env: GodState, prng: PRNG) -> GodState:
                return _tr_to_env(env, prng, te_dataset.value.value.x.shape[0])

            def refresh_env(env: GodState) -> GodState:
                _tr_prngs = jax.random.split(env.inner_prng[0], config.batch_tr)
                _env = create_envs(config, _tr_prngs)
                return to_batched_form(_env, env)

            model, te_lossfn, innerLibrary = create_batched_model(
                te_dataset, tr_to_val_env, tr_to_te_env, lossFn, tr_env, innerInterpreter, outerInterpreter, config
            )

            checkpoint_fn(
                copy.replace(
                    tr_env,
                    inner_prng=jax.random.key_data(tr_env.inner_prng),
                    outer_prng=jax.random.key_data(tr_env.outer_prng),
                )
            )

            def getaccuracy(env):
                def _accuracy_seq_filter_single(
                    preds: jnp.ndarray,  # [N, C]
                    labels: jnp.ndarray,  # [N, 2]
                    n: int,  # sequence number to filter
                ) -> float:
                    class_indices = labels[:, 0].astype(jnp.int32)
                    sequence_numbers = labels[:, 1].astype(jnp.int32)

                    pred_classes = jnp.argmax(preds, axis=-1)
                    correct = pred_classes == class_indices

                    # Mask: 1 where sequence == n, else 0
                    mask = (sequence_numbers == n).astype(jnp.float32)
                    correct_masked = correct.astype(jnp.float32) * mask

                    total = jnp.sum(mask)
                    correct_total = jnp.sum(correct_masked)

                    return jax.lax.cond(total > 0, lambda: correct_total / total, lambda: 0.0)

                # Vectorized over batch dimension: preds [B, N, C], labels [B, N, 2]
                batched_accuracy_with_sequence_filter = eqx.filter_vmap(
                    _accuracy_seq_filter_single,
                    in_axes=(0, 0, None),  # vmap over batch; n is shared
                )
                predictions, _ = innerLibrary.model(te_dataset).func(
                    innerInterpreter, tr_to_te_env(env, env.outer_prng)
                )
                accuracy = batched_accuracy_with_sequence_filter(
                    predictions.value.value, te_dataset.value.value.y, sequence_length - 1
                )
                return jnp.mean(accuracy)

            train_loop_IO(
                tr_dataset,
                vl_dataset,
                combine_ds,
                model,
                tr_env,
                refresh_env,
                config,
                checkpoint_fn,
                log_fn,
                te_lossfn,
                getaccuracy,
            )

        case "online":
            tr_prng, vl_prng = jax.random.split(env_prng, 2)
            tr_env = load_env(config, tr_prng)
            vl_env = load_env(config, vl_prng)
            te_dataset = tree_unstack(te_dataset)[0].value

            def combine_ds(
                tr_ds: Traversable[Traversable[InputOutput]], vl_ds: Traversable[Traversable[InputOutput]]
            ) -> Traversable[OhoData[Traversable[InputOutput]]]:
                return Traversable(OhoData(payload=tr_ds.value, validation=vl_ds.value))

            def _tr_to_env(env: GodState, _: PRNG) -> GodState:
                return to_batched_form(vl_env, env)

            def tr_to_val_env(env: GodState, prng: PRNG) -> GodState:
                return _tr_to_env(env, prng)

            def tr_to_te_env(env: GodState, prng: PRNG) -> GodState:
                return _tr_to_env(env, prng)

            def refresh_env(env: GodState) -> GodState:
                return env

            # def refresh_env(env: GodState) -> GodState:
            #     prng, env = innerInterpreter.updatePRNG().func(innerInterpreter, env)
            #     _env: GodState = load_env(config, prng)
            #     return to_batched_form(_env, env)

            model, _te_lossfn, _ = create_online_model(
                te_dataset, tr_to_val_env, tr_to_te_env, lossFn, tr_env, innerInterpreter, outerInterpreter, config
            )

            checkpoint_fn(
                copy.replace(
                    tr_env,
                    inner_prng=jax.random.key_data(tr_env.inner_prng),
                    outer_prng=jax.random.key_data(tr_env.outer_prng),
                )
            )

            train_loop_IO(
                tr_dataset,
                vl_dataset,
                combine_ds,
                model,
                tr_env,
                refresh_env,
                config,
                checkpoint_fn,
                log_fn,
                te_lossfn,
                lambda _: 0,
            )

        case _:
            raise ValueError("Invalid batch_or_online option")


def generate_unique_id():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # Microsecond precision
    short_uuid = uuid.uuid4().hex[:8]  # Random component
    return f"{timestamp}_{short_uuid}"


def save_object_as_wandb_artifact(
    obj: Any, artifact_name: str, filename: str, artifact_type: str, to_float16: bool
) -> None:
    os.makedirs("/scratch/artifacts", exist_ok=True)

    # Ensure filename ends with .pkl
    # if not filename.endswith(".dill"):
    #     filename = filename + ".dill"
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join("/scratch/artifacts", filename)

    # Reduce precision of JAX arrays if to_float16 is True
    if to_float16:
        obj = jax.tree.map(lambda x: x.astype(jnp.float16) if isinstance(x, jax.Array) else x, obj)

    joblib.dump(obj, full_path, compress=0)
    # with open(full_path, "wb") as f:
    #     dill.dump(obj, f)

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(full_path)
    wandb.log_artifact(artifact)


def save_checkpoint(obj: Any, name: str, filename: str) -> None:
    save_object_as_wandb_artifact(obj, name, filename, "checkpoint", False)


def load_artifact(artifact_name: str, filename: str) -> Any:
    api = wandb.Api()
    model_artifact = api.artifact(artifact_name)
    model_dir = model_artifact.download()

    # Ensure filename ends with .pkl
    # if not filename.endswith(".dill"):
    #     filename = filename + ".dill"
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join(model_dir, filename)
    return joblib.load(full_path)

    # with open(full_path, "rb") as f:
    #     loaded_model = dill.load(f)

    # return loaded_model


def log_jax_array(array: jax.Array, artifact_name: str):
    filename = f"{artifact_name}.npy"
    jnp.save(filename, array)
    artifact = wandb.Artifact(artifact_name, type="checkpoint")
    artifact.add_file(filename)
    wandb.log_artifact(artifact)
    os.remove(filename)


def getLossFn(config: GodConfig) -> Callable[[jax.Array, jax.Array], LOSS]:
    match config.lossFn:
        case "cross_entropy":
            return lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))
        case "cross_entropy_with_integer_labels":
            return lambda a, b: LOSS(optax.losses.softmax_cross_entropy_with_integer_labels(a, b))
        case _:
            raise ValueError("Invalid loss function")


def create_env(config: GodConfig, prng: PRNG) -> tuple[GodState, GodInterpreter, GodInterpreter]:
    # These don't care how env is created, just tags all possible parameter/states and vectorizes them
    inner_states = [lambda s: s.rnnState.activation]
    inner_params = [lambda s: s.rnnState.rnnParameter, lambda s: s.feedforwardState]
    outer_states = inner_params + [lambda s: s.innerOptState]  # VERY IMPORTANT COMES FIRST
    if config.batch_or_online == "online":
        outer_states += [
            lambda s: s.innerInfluenceTensor,
            lambda s: s.innerUoro,
            lambda s: s.rnnState.activation,
        ]
    outer_params = [lambda s: s.innerSgdParameter, lambda s: s.innerAdamParameter]

    def toArray(godState: GodState, attrbs: list[Callable[[GodState], Any]]) -> Array:
        exact = [attrb(godState) for attrb in attrbs if attrb(godState) is not None]
        return toVector(endowVector(exact))

    def fromArray(godState: GodState, attrbs: list[Callable[[GodState], Any]], value: Array) -> GodState:
        fs = [attrb for attrb in attrbs if attrb(godState) is not None]
        ys = [attrb(godState) for attrb in fs]
        xs = invmap(ys, lambda _: value)
        for x, f in zip(xs, fs):
            godState = eqx.tree_at(f, godState, x, is_leaf=lambda x: x is None)
        return godState

    inner_state_get, inner_state_set = lambda g: toArray(g, inner_states), lambda g, v: fromArray(g, inner_states, v)
    inner_param_get, inner_param_set = lambda g: toArray(g, inner_params), lambda g, v: fromArray(g, inner_params, v)
    outer_state_get, outer_state_set = lambda g: toArray(g, outer_states), lambda g, v: fromArray(g, outer_states, v)
    outer_param_get, outer_param_set = lambda g: toArray(g, outer_params), lambda g, v: fromArray(g, outer_params, v)

    def putter[T](env: GodState, f: Callable[[GodState], T], value: T) -> GodState:
        return eqx.tree_at(f, env, value, is_leaf=lambda x: x is None)

    prng1, prng2, prng3, prng4, prng5, prng6, prng7, prng8, prng9, prng10, prng11, prng12 = jax.random.split(prng, 12)
    inner_prng, outer_prng = jax.random.split(prng1, 2)
    env = GodState(
        inner_prng=inner_prng,
        outer_prng=outer_prng,
        innerTimeConstant=config.inner_time_constant,
        outerTimeConstant=config.outer_time_constant,
        start_epoch=0,
        start_example=0,
        globalLogConfig=GlobalLogConfig(
            stop_influence=False,
            log_influence=config.log_influence,
            log_accumulate_influence=config.log_accumulate_influence,
        ),
    )

    inner_log_config = LogConfig(
        log_special=config.inner_log_special,
        lanczos_iterations=config.inner_lanczos_iterations,
        log_expensive=config.inner_log_expensive if config.inner_log_expensive is not None else False,
    )
    env = copy.replace(env, innerLogConfig=inner_log_config)

    # 1) Initialize inner state and parameters

    rnn_state = RnnState(rnnConfig=None, activation=None, rnnParameter=None)
    env = putter(env, lambda s: s.rnnState, rnn_state)

    match config.architecture:
        case "rnn":
            i_std = config.initialization_std
            W_in = jax.random.normal(prng2, (config.n_h, config.n_in)) * jnp.sqrt(1 / config.n_in) * i_std
            W_rec = jnp.linalg.qr(jax.random.normal(prng3, (config.n_h, config.n_h)))[0] * i_std
            W_out = jax.random.normal(prng4, (config.n_out, config.n_h)) * jnp.sqrt(1 / config.n_h) * i_std

            b_rec = jnp.zeros((config.n_h, 1))
            b_out = jnp.zeros((config.n_out, 1))
            w_rec = jnp.hstack([W_rec, W_in, b_rec])
            w_out = jnp.hstack([W_out, b_out])
            rnn_parameter = RnnParameter(w_rec=w_rec, w_out=w_out)
            rnn_config = RnnConfig(
                n_h=config.n_h, n_in=config.n_in, n_out=config.n_out, activationFn=config.activation_fn
            )
            activation = ACTIVATION(jax.random.normal(prng5, (config.n_h,)))
            rnn_state = RnnState(rnnConfig=rnn_config, activation=activation, rnnParameter=rnn_parameter)

            env = putter(env, lambda s: s.rnnState, rnn_state)

            if config.inner_learner == "uoro":
                a_init = jax.random.normal(prng6, (rnn_config.n_h,))
                b_init = toVector(
                    endowVector(
                        RnnParameter(
                            w_rec=jax.random.normal(prng7, rnn_parameter.w_rec.shape),
                            w_out=jnp.zeros_like(rnn_parameter.w_out),
                        )
                    )
                )
                env = putter(env, lambda s: s.innerUoro, UORO_Param(A=a_init, B=b_init))
        case "ffn":

            def map2Fn(str) -> Callable[[jax.Array], jax.Array]:
                match str:
                    case "tanh":
                        return jax.nn.tanh
                    case "relu":
                        return jax.nn.relu
                    case "sigmoid":
                        return jax.nn.sigmoid
                    case "identity":
                        return lambda x: x
                    case "softmax":
                        return jax.nn.softmax
                    case _:
                        raise ValueError("Invalid activation function")

            layers = list(map(lambda x: (x[0], map2Fn(x[1])), config.ffn_layers))
            ffn = CustomSequential(layers, config.n_in, prng12)
            env = putter(env, lambda s: s.feedforwardState, ffn)

        case _:
            raise ValueError("Invalid architecture")

    rec_state_n = jnp.size(inner_state_get(env))
    rec_param_n = jnp.size(inner_param_get(env))

    # ============================
    # 2) Initialize inner optimizer

    class OptimizerConfig(eqx.Module):
        optimizer: str
        lr_reparam: str
        learning_rate: float
        clip: float
        sharpness: float

    def setup_optimizer(opt_config: OptimizerConfig, param_n: int, get_lr: Callable[[GodState, str], float]):
        def lr_2_optimizer(lr: float):
            match opt_config.optimizer:
                case "sgd" | "sgd_positive":
                    return optax.sgd(lr)
                case "sgd_normalized":
                    return normalized_sgd(lr)
                case "sgd_clipped":
                    return soft_clipped_sgd(lr, opt_config.clip, opt_config.sharpness)
                case "adam":
                    return optax.adam(lr)
                case _:
                    raise ValueError("Invalid optimizer")

        match opt_config.lr_reparam:
            case "identity":
                reparam_fn = lambda lr: lr
                reparam_inverse = lambda lr: lr
            case "softplus":
                reparam_fn = jax.nn.softplus
                reparam_inverse = lambda y: jnp.log(jnp.expm1(y))
            case _:
                raise ValueError(f"Invalid learning rate reparametrization")

        match opt_config.optimizer:
            case "sgd" | "sgd_positive" | "sgd_normalized" | "sgd_clipped":
                param = SgdParameter(learning_rate=reparam_inverse(jnp.array(opt_config.learning_rate)))
            case "adam":
                param = AdamParameter(learning_rate=reparam_inverse(jnp.array(opt_config.learning_rate)))
            case _:
                raise ValueError(f"Invalid optimizer")

        optimizer_fn = lambda lr: lr_2_optimizer(reparam_fn(lr))

        optimizer = optimizer_fn(reparam_inverse(opt_config.learning_rate))
        opt_state = optimizer.init(jnp.empty((param_n,)))

        return param, opt_state, lambda s: optimizer_fn(get_lr(s, opt_config.optimizer)), reparam_fn

    def inner_get_lr(s: GodState, opt: str):
        match opt:
            case "sgd" | "sgd_positive" | "sgd_normalized" | "sgd_clipped":
                return s.innerSgdParameter.learning_rate
            case "adam":
                return s.innerAdamParameter.learning_rate
            case _:
                raise ValueError("Invalid optimizer")

    inner_param, inner_opt_state, get_inner_optimizer, inner_param_fn = setup_optimizer(
        OptimizerConfig(
            optimizer=config.inner_optimizer,
            lr_reparam=config.inner_optimizer_parametrization,
            learning_rate=config.inner_learning_rate,
            clip=config.inner_clip,
            sharpness=config.inner_clip_sharpness,
        ),
        rec_param_n,
        inner_get_lr,
    )
    env = putter(env, lambda s: s.innerOptState, inner_opt_state)
    match inner_param:
        case SgdParameter():
            env = putter(env, lambda s: s.innerSgdParameter, inner_param)
        case AdamParameter():
            env = putter(env, lambda s: s.innerAdamParameter, inner_param)

    match config.inner_optimizer:
        case "sgd_positive":

            def inner_updater(params: optax.Params, updates: optax.Updates):
                return jax.tree.map(lambda p, u: jnp.maximum(p + u, 1e-6), params, updates)
        case _:
            inner_updater = optax.apply_updates

    if config.inner_learner != "bptt":
        inner_influence_tensor = JACOBIAN(jax.random.normal(prng10, (rec_state_n, rec_param_n)))
        env = putter(env, lambda s: s.innerInfluenceTensor, inner_influence_tensor)
    else:
        inner_influence_tensor = None

    innerLogs = Logs(
        gradient=jnp.zeros((rec_param_n,)),
        validationGradient=None,
        influenceTensor=inner_influence_tensor if config.log_influence else None,
        immediateInfluenceTensor=inner_influence_tensor if config.log_influence else None,
        jac_eigenvalue=0.0 if config.inner_log_special else None,
        hessian=jnp.zeros((rec_state_n, rec_state_n)) if config.inner_log_expensive else None,
    )
    env = putter(env, lambda s: s.innerLogs, innerLogs)

    # =============================
    # 3) Initialize outer state and parameters

    outer_log_config = LogConfig(
        log_special=config.outer_log_special,
        lanczos_iterations=config.outer_lanczos_iterations,
        log_expensive=config.outer_log_expensive if config.outer_log_expensive is not None else False,
    )
    env = copy.replace(env, outerLogConfig=outer_log_config)

    outer_rec_state_n = jnp.size(outer_state_get(env))
    outer_rec_param_n = jnp.size(outer_param_get(env))

    def outer_get_lr(s: GodState, opt: str):
        match opt:
            case "sgd" | "sgd_positive" | "sgd_normalized" | "sgd_clipped":
                return s.outerSgdParameter.learning_rate
            case "adam":
                return s.outerAdamParameter.learning_rate
            case _:
                raise ValueError("Invalid optimizer")

    outer_param, outer_opt_state, get_outer_optimizer, outer_param_fn = setup_optimizer(
        OptimizerConfig(
            optimizer=config.outer_optimizer,
            lr_reparam=config.outer_optimizer_parametrization,
            learning_rate=config.outer_learning_rate,
            clip=config.outer_clip,
            sharpness=config.outer_clip_sharpness,
        ),
        outer_rec_param_n,
        outer_get_lr,
    )
    env = putter(env, lambda s: s.outerOptState, outer_opt_state)
    match outer_param:
        case SgdParameter():
            env = putter(env, lambda s: s.outerSgdParameter, outer_param)
        case AdamParameter():
            env = putter(env, lambda s: s.outerAdamParameter, outer_param)

    match config.outer_optimizer:
        case "sgd_positive":

            def outer_updater(params: optax.Params, updates: optax.Updates):
                return jax.tree.map(lambda p, u: jnp.maximum(p + u, 1e-6), params, updates)
        case _:
            outer_updater = optax.apply_updates

    if config.outer_learner != "bptt":
        outer_influence_tensor = JACOBIAN(jax.random.normal(prng11, (outer_rec_state_n, outer_rec_param_n)))
        env = putter(env, lambda s: s.outerInfluenceTensor, outer_influence_tensor)
    else:
        outer_influence_tensor = None

    outerLogs = Logs(
        gradient=jnp.zeros((outer_rec_param_n,)),
        validationGradient=jnp.zeros((rec_param_n,)),
        influenceTensor=outer_influence_tensor if config.log_influence else None,
        immediateInfluenceTensor=outer_influence_tensor if config.log_influence else None,
        jac_eigenvalue=0.0 if config.outer_log_special else None,
        hessian=jnp.zeros((outer_rec_state_n, outer_rec_state_n)) if config.outer_log_expensive else None,
    )
    env = putter(env, lambda s: s.outerLogs, outerLogs)

    match config.outer_learner:
        case "uoro":
            uoro = UORO_Param(
                A=jax.random.normal(prng8, (outer_rec_state_n,)),
                B=jax.random.normal(prng9, (outer_rec_param_n,)),
            )
            env = putter(env, lambda s: s.outerUoro, uoro)
        case _:
            pass

    # =============================
    def monadic_getter[T](f: Callable[[GodState], T]) -> App[GodInterpreter, GodState, T]:
        return get(PX[GodState]()).fmap(f)

    def monadic_putter[T](f: Callable[[GodState], T]) -> Callable[[T], App[GodInterpreter, GodState, Unit]]:
        return lambda x: modifies(lambda s: eqx.tree_at(f, s, x, is_leaf=lambda x: x is None))

    # 4) Create interpreters
    innerInterpreter = GodInterpreter(
        getRecurrentState=monadic_getter(inner_state_get),
        putRecurrentState=lambda x: modifies(lambda s: inner_state_set(s, x)),
        getRecurrentParam=monadic_getter(inner_param_get),
        putRecurrentParam=lambda x: modifies(lambda s: inner_param_set(s, x)),
        getActivation=monadic_getter(lambda s: s.rnnState.activation),
        putActivation=monadic_putter(lambda s: s.rnnState.activation),
        getInfluenceTensor=monadic_getter(lambda s: s.innerInfluenceTensor),
        putInfluenceTensor=monadic_putter(lambda s: s.innerInfluenceTensor),
        getUoro=monadic_getter(lambda s: s.innerUoro),
        putUoro=monadic_putter(lambda s: s.innerUoro),
        getRnnConfig=monadic_getter(lambda s: s.rnnState.rnnConfig),
        getTimeConstant=monadic_getter(lambda s: s.innerTimeConstant),
        getLogConfig=monadic_getter(lambda s: s.innerLogConfig),
        getGlobalLogConfig=monadic_getter(lambda s: s.globalLogConfig),
        # putLogs=monadic_putter(lambda s: s.innerLogs),
        putLogs=lambda s: modifies(lambda e: eqx.tree_at(lambda t: t.innerLogs, e, eqx.combine(s, e.innerLogs))),
        getRnnParameter=monadic_getter(lambda s: s.rnnState.rnnParameter),
        putRnnParameter=monadic_putter(lambda s: s.rnnState.rnnParameter),
        getOptState=monadic_getter(lambda s: s.innerOptState),
        putOptState=monadic_putter(lambda s: s.innerOptState),
        getOptimizer=monadic_getter(get_inner_optimizer),
        getUpdater=pure(inner_updater, PX[tuple[GodInterpreter, GodState]]()),
        getLearningRate=lambda s: inner_param_fn(inner_get_lr(s, config.inner_optimizer)),
        getPRNG=monadic_getter(lambda s: s.inner_prng),
        putPRNG=monadic_putter(lambda s: s.inner_prng),
        getFeedForward=monadic_getter(lambda s: s.feedforwardState),
    )

    outerInterpreter = GodInterpreter(
        getRecurrentState=monadic_getter(outer_state_get),
        putRecurrentState=lambda x: modifies(lambda s: outer_state_set(s, x)),
        getRecurrentParam=monadic_getter(outer_param_get),
        putRecurrentParam=lambda x: modifies(lambda s: outer_param_set(s, x)),
        getActivation=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        putActivation=lambda x: pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getInfluenceTensor=monadic_getter(lambda s: s.outerInfluenceTensor),
        putInfluenceTensor=monadic_putter(lambda s: s.outerInfluenceTensor),
        getUoro=monadic_getter(lambda s: s.outerUoro),
        putUoro=monadic_putter(lambda s: s.outerUoro),
        getRnnConfig=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getTimeConstant=monadic_getter(lambda s: s.outerTimeConstant),
        getLogConfig=monadic_getter(lambda s: s.outerLogConfig),
        getGlobalLogConfig=monadic_getter(lambda s: s.globalLogConfig),
        putLogs=lambda s: modifies(lambda e: eqx.tree_at(lambda t: t.outerLogs, e, eqx.combine(s, e.outerLogs))),
        getRnnParameter=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        putRnnParameter=lambda x: pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getOptState=monadic_getter(lambda s: s.outerOptState),
        putOptState=monadic_putter(lambda s: s.outerOptState),
        getOptimizer=monadic_getter(get_outer_optimizer),
        getUpdater=pure(outer_updater, PX[tuple[GodInterpreter, GodState]]()),
        getLearningRate=lambda s: outer_param_fn(outer_get_lr(s, config.outer_optimizer)),
        getPRNG=monadic_getter(lambda s: s.outer_prng),
        putPRNG=monadic_putter(lambda s: s.outer_prng),
        getFeedForward=pure(None, PX[tuple[GodInterpreter, GodState]]()),
    )

    return env, innerInterpreter, outerInterpreter


def create_datasets(
    config: GodConfig, prng: PRNG, test_rng_key: PRNG
) -> tuple[Traversable[OhoData[Traversable[InputOutput]]], Traversable[InputOutput]]:
    # Extract parameters from config
    t1, t2 = config.ts
    N_train = config.numTr
    N_val = config.numVal
    N_test = config.numTe
    tr_series_length = config.tr_examples_per_epoch
    vl_series_length = config.vl_examples_per_epoch
    tau_task = int(1 / config.inner_time_constant) if config.tau_task else 1

    # Initialize PRNG keys
    prng1, prng2 = jax.random.split(prng, 2)

    # Generate raw datasets
    X_train, Y_train = generate_add_task_dataset(N_train, t1, t2, tau_task, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, t1, t2, tau_task, prng2)
    X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, tau_task, test_rng_key)

    # Calculate number of updates
    num_updates_train = N_train // tr_series_length
    num_updates_val = N_val // vl_series_length

    # Transform training data
    X_train = transform(X_train, num_updates_train)
    Y_train = transform(Y_train, num_updates_train)

    # Transform validation data
    X_val = transform(X_val, num_updates_val)
    Y_val = transform(Y_val, num_updates_val)

    # Repeat validation data to match training epochs if needed
    if num_updates_val < num_updates_train:
        repeats_needed = num_updates_train // num_updates_val + 1  # subsequent :num_updates_train trims down to exact
        X_val = jnp.repeat(X_val, repeats_needed, axis=0)[:num_updates_train]
        Y_val = jnp.repeat(Y_val, repeats_needed, axis=0)[:num_updates_train]
    elif num_updates_val > num_updates_train:
        X_val = X_val[:num_updates_train]
        Y_val = Y_val[:num_updates_train]

    # Create Traversable objects
    train_set = Traversable(InputOutput(x=X_train, y=Y_train))
    val_set = Traversable(InputOutput(x=X_val, y=Y_val))
    test_set = Traversable(InputOutput(x=X_test, y=Y_test))

    oho_set = Traversable(OhoData(payload=train_set, validation=val_set))

    return oho_set, test_set


def transform(arr: Array, _t: int):
    return arr.reshape((_t, -1) + arr.shape[1:])


def create_learner(
    learner: str, rtrl_use_fwd: bool, uoro_std
) -> RTRL | RFLO | UORO | IdentityLearner | OfflineLearning:
    match learner:
        case "rtrl":
            return RTRL(rtrl_use_fwd)
        case "rflo":
            return RFLO()
        case "uoro":
            return UORO(lambda key, shape: jax.random.uniform(key, shape, minval=-uoro_std, maxval=uoro_std))
        case "identity":
            return IdentityLearner()
        case "bptt":
            return OfflineLearning()
        case _:
            raise ValueError("Invalid learner")


def create_rnn_learner(
    learner: RTRL | RFLO | UORO | IdentityLearner | OfflineLearning,
    lossFn: Callable[[jax.Array, jax.Array], LOSS],
    arch: Literal["rnn", "ffn"],
) -> Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]:
    match arch:
        case "rnn":
            stepFn = lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
            readoutFn = doRnnReadout
        case "ffn":
            stepFn = (
                lambda d: doFeedForwardStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState)
            )
            readoutFn = doFeedForwardReadout

    lfn = lambda a, b: lossFn(a, b.y)
    match learner:
        case OfflineLearning():
            bptt_library: Library[Traversable[InputOutput], GodInterpreter, GodState, Traversable[PREDICTION]]
            bptt_library = learner.createLearner(
                stepFn,
                readoutFn,
                lfn,
                readoutRecurrentError(readoutFn, lfn),
            )
            return bptt_library
        case _:
            # library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
            _library: Library[IdentityF[InputOutput], GodInterpreter, GodState, IdentityF[PREDICTION]]
            _library = learner.createLearner(
                stepFn,
                readoutFn,
                lfn,
                readoutRecurrentError(doRnnReadout, lfn),
            )
            library = Library[InputOutput, GodInterpreter, GodState, PREDICTION](
                model=lambda d: _library.model(IdentityF(d)),
                modelLossFn=lambda d: _library.modelLossFn(IdentityF(d)),
                modelGradient=lambda d: _library.modelGradient(IdentityF(d)),
            )
            return foldrLibrary(library)


def train_loop_IO[D](
    tr_dataset: Traversable[Traversable[InputOutput]],
    vl_dataset: Traversable[Traversable[InputOutput]],
    to_combined_ds: Callable[[Traversable[Traversable[InputOutput]], Traversable[Traversable[InputOutput]]], D],
    model: Callable[[D, GodState], tuple[Traversable[AllLogs], GodState]],
    env: GodState,
    refresh_env: Callable[[GodState], GodState],
    config: GodConfig,
    checkpoint_fn: Callable[[GodState], None],
    log_fn: Callable[[AllLogs], None],
    te_loss: Callable[[GodState], LOSS],
    statistic: Callable[[GodState], float],
) -> None:
    tr_dataset = PyTreeDataset(tr_dataset)
    vl_dataset = PyTreeDataset(vl_dataset)

    if config.batch_or_online == "batch":
        vl_batch_size = config.batch_vl
        vl_sampler = RandomSampler(vl_dataset)
        tr_dl = lambda b: DataLoader(
            b, batch_size=config.batch_tr, shuffle=True, collate_fn=jax_collate_fn, drop_last=True
        )
        # doesn't make sense to do this in batch case. batch=subsequence in online
        env = copy.replace(env, start_example=0)
    else:
        vl_batch_size = config.batch_tr  # same size -> conjoin with tr batch with to_combined_ds
        vl_sampler = RandomSampler(vl_dataset, replacement=True, num_samples=len(tr_dataset))
        tr_dl = lambda b: DataLoader(b, batch_size=config.batch_tr, shuffle=False, collate_fn=jax_collate_fn)

    vl_dataloader = DataLoader(
        vl_dataset, batch_size=vl_batch_size, sampler=vl_sampler, collate_fn=jax_collate_fn, drop_last=True
    )

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    vl_dataloader = infinite_loader(vl_dataloader)

    start = time.time()
    num_batches_seen_so_far = 0
    all_logs: list[Traversable[AllLogs]] = []
    for epoch in range(env.start_epoch, config.num_retrain_loops):
        print(f"Epoch {epoch + 1}/{config.num_retrain_loops}")
        batched_dataset = Subset(tr_dataset, indices=range(env.start_example, len(tr_dataset)))
        tr_dataloader = tr_dl(batched_dataset)

        for i, (tr_batch, vl_batch) in enumerate(zip(tr_dataloader, vl_dataloader)):
            ds_batch = to_combined_ds(tr_batch, vl_batch)
            batch_size = len(jax.tree.leaves(ds_batch)[0])
            env = refresh_env(env)
            logs, env = model(ds_batch, env)

            env = eqx.tree_at(lambda t: t.start_example, env, env.start_example + batch_size)
            all_logs.append(logs)
            checkpoint_condition = num_batches_seen_so_far + i + 1
            if checkpoint_condition % config.checkpoint_interval == 0:
                checkpoint_fn(
                    copy.replace(
                        env,
                        inner_prng=jax.random.key_data(env.inner_prng),
                        outer_prng=jax.random.key_data(env.outer_prng),
                    )
                )

            print(
                f"Batch {i + 1}/{len(tr_dataloader)}, Loss: {logs.value.train_loss[-1]}, LR: {logs.value.inner_learning_rate[-1]}"
            )

        # env = eqx.tree_at(lambda t: t.start_epoch, env, epoch + 1)
        env = eqx.tree_at(lambda t: t.start_example, env, 0)
        num_batches_seen_so_far += len(tr_dataloader)

    end = time.time()
    print(f"Training time: {end - start} seconds")

    total_logs: Traversable[AllLogs] = jax.tree.map(lambda *xs: jnp.concatenate(xs), *all_logs)

    log_fn(total_logs.value)
    checkpoint_fn(
        copy.replace(
            env,
            inner_prng=jax.random.key_data(env.inner_prng),
            outer_prng=jax.random.key_data(env.outer_prng),
        )
    )

    def safe_norm(x):
        return jnp.linalg.norm(x) if x is not None else None

    # log wandb partial metrics
    for log_tree_ in tree_unstack_lazy(total_logs.value):
        log_data: AllLogs = jax.tree.map(
            lambda x: jnp.real(x) if x is not None and jnp.all(jnp.isfinite(x)) else None, log_tree_
        )
        wandb.log(
            {
                "train_loss": log_data.train_loss,
                "validation_loss": log_data.validation_loss,
                "test_loss": log_data.test_loss,
                "hyperparameters": log_data.hyperparameters,
                "inner_learning_rate": log_data.inner_learning_rate,
                "parameter_norm": log_data.parameter_norm,
                "oho_gradient": log_data.oho_gradient,
                "train_gradient": log_data.train_gradient,
                "validation_gradient": log_data.validation_gradient,
                "oho_gradient_norm": safe_norm(log_data.oho_gradient),
                "train_gradient_norm": safe_norm(log_data.train_gradient),
                "validation_gradient_norm": safe_norm(log_data.validation_gradient),
                "immediate_influence_tensor_norm": log_data.immediate_influence_tensor_norm,
                "inner_influence_tensor_norm": log_data.inner_influence_tensor_norm,
                "outer_influence_tensor_norm": log_data.outer_influence_tensor_norm,
                "largest_jacobian_eigenvalue": log_data.largest_jacobian_eigenvalue,
                "largest_influence_eigenvalue": log_data.largest_hessian_eigenvalue,
                "jacobian_eigenvalues": log_data.jacobian,
                "rnn_activation_norm": log_data.rnn_activation_norm,
                "immediate_influence_tensor": jnp.ravel(log_data.immediate_influence_tensor)
                if log_data.immediate_influence_tensor is not None
                else None,
                "outer_influence_tensor": jnp.ravel(log_data.outer_influence_tensor)
                if log_data.outer_influence_tensor is not None
                else None,
            }
        )

    ee = te_loss(env)
    eee = statistic(env)
    print(ee)
    print(eee)
    wandb.log({"test_loss": ee, "test_statistic": eee})


def create_online_model(
    test_dataset: Traversable[InputOutput],
    tr_to_val_env: Callable[[GodState, PRNG], GodState],
    tr_to_te_env: Callable[[GodState, PRNG], GodState],
    lossFn: Callable[[jax.Array, jax.Array], LOSS],
    initEnv: GodState,
    innerInterpreter: GodInterpreter,
    outerInterpreter: GodInterpreter,
    config: GodConfig,
) -> tuple[
    Callable[[Traversable[OhoData[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
    Callable[[GodState], LOSS],
]:
    innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
    innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
    outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

    innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
    innerController = logGradient(innerController)
    innerLibrary = innerLibrary._replace(modelGradient=innerController)

    inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
    outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
    pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

    validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

    match outerLearner:
        case OfflineLearning():
            _outerLibrary: Library[
                Traversable[OhoData[Traversable[InputOutput]]],
                GodInterpreter,
                GodState,
                Traversable[Traversable[PREDICTION]],
            ]
            _outerLibrary = endowBilevelOptimization(
                innerLibrary,
                doOptimizerStep,
                innerInterpreter,
                outerLearner,
                lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
                tr_to_val_env,
                pad_val_grad_by,
            )

            outerController = logGradient(_outerLibrary.modelGradient)
            _outerLibrary = _outerLibrary._replace(modelGradient=outerController)

            @do()
            def updateStep(oho_data: Traversable[OhoData[Traversable[InputOutput]]]):
                print("recompiled")
                env = yield from get(PX[GodState]())
                interpreter = yield from ask(PX[GodInterpreter]())
                hyperparameters = yield from interpreter.getRecurrentParam
                weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

                te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
                vl, _ = _outerLibrary.modelLossFn(oho_data).func(outerInterpreter, env)
                tr, _ = (
                    foldrLibrary(innerLibrary)
                    .modelLossFn(Traversable(oho_data.value.payload))
                    .func(innerInterpreter, env)
                )

                def safe_norm(x):
                    return jnp.linalg.norm(x) if x is not None else None

                log = AllLogs(
                    train_loss=tr / config.tr_examples_per_epoch,
                    validation_loss=vl / config.vl_examples_per_epoch,
                    test_loss=te / config.numTe,
                    hyperparameters=hyperparameters,
                    inner_learning_rate=innerInterpreter.getLearningRate(env),
                    parameter_norm=safe_norm(weights),
                    oho_gradient=env.outerLogs.gradient,
                    train_gradient=env.innerLogs.gradient,
                    validation_gradient=env.outerLogs.validationGradient,
                    immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
                    outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
                    outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
                    inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
                    largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
                    largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
                    jacobian=env.innerLogs.hessian,
                    hessian=env.outerLogs.hessian,
                    rnn_activation_norm=safe_norm(env.rnnState.activation),
                    immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
                    if config.log_accumulate_influence
                    else None,
                )
                logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))

                _ = yield from _outerLibrary.modelGradient(oho_data).flat_map(doOptimizerStep)
                return pure(logs, PX[tuple[GodInterpreter, GodState]]())

            model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
            return model

        case _:
            outerLibrary: Library[
                IdentityF[OhoData[Traversable[InputOutput]]],
                GodInterpreter,
                GodState,
                IdentityF[Traversable[PREDICTION]],
            ]
            outerLibrary = endowBilevelOptimization(
                innerLibrary,
                doOptimizerStep,
                innerInterpreter,
                outerLearner,
                lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
                tr_to_val_env,
                pad_val_grad_by,
            )

            outerController = logGradient(outerLibrary.modelGradient)
            outerLibrary = outerLibrary._replace(modelGradient=outerController)

            @do()
            def updateStep(oho_data: OhoData[Traversable[InputOutput]]):
                print("recompiled")
                env = yield from get(PX[GodState]())
                interpreter = yield from ask(PX[GodInterpreter]())
                hyperparameters = yield from interpreter.getRecurrentParam
                weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

                te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
                vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
                tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)

                _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

                # code smell but what can you do, no maybe monad or elvis operator...
                def safe_norm(x):
                    return jnp.linalg.norm(x) if x is not None else None

                log = AllLogs(
                    train_loss=tr / config.tr_examples_per_epoch,
                    validation_loss=vl / config.vl_examples_per_epoch,
                    test_loss=te / config.numTe,
                    hyperparameters=hyperparameters,
                    inner_learning_rate=innerInterpreter.getLearningRate(env),
                    parameter_norm=safe_norm(weights),
                    oho_gradient=env.outerLogs.gradient,
                    train_gradient=env.innerLogs.gradient,
                    validation_gradient=env.outerLogs.validationGradient,
                    immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
                    outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
                    outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
                    inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
                    largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
                    largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
                    jacobian=env.innerLogs.hessian,
                    hessian=env.outerLogs.hessian,
                    rnn_activation_norm=safe_norm(env.rnnState.activation),
                    immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
                    if config.log_accumulate_influence
                    else None,
                )
                return pure(log, PX[tuple[GodInterpreter, GodState]]())

            model = eqx.filter_jit(lambda d, e: traverseM(updateStep)(d).func(outerInterpreter, e))
            return (
                model,
                lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
                innerLibrary,
            )


def create_batched_model(
    test_dataset: Traversable[Traversable[InputOutput]],
    tr_to_val_env: Callable[[GodState, PRNG], GodState],
    tr_to_te_env: Callable[[GodState, PRNG], GodState],
    lossFn: Callable[[jax.Array, jax.Array], LOSS],
    initEnv: GodState,
    innerInterpreter: GodInterpreter,
    outerInterpreter: GodInterpreter,
    config: GodConfig,
) -> tuple[
    Callable[[OhoData[Traversable[Traversable[InputOutput]]], GodState], tuple[Traversable[AllLogs], GodState]],
    Callable[[GodState], LOSS],
]:
    innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
    innerLibrary = create_rnn_learner(innerLearner, lossFn, config.architecture)
    outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

    innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
    innerLibrary = innerLibrary._replace(modelGradient=innerController)
    innerLibrary = aggregateBatchedGradients(innerLibrary, batch_env_form)
    innerController = logGradient(innerLibrary.modelGradient)
    innerLibrary = innerLibrary._replace(modelGradient=innerController)

    inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
    outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
    pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

    validation_model = lambda ds: innerLibrary.modelLossFn(ds).func

    match outerLearner:
        case OfflineLearning():
            raise NotImplementedError("BPTT on batched is not implemented yet")

        case _:
            outerLibrary: Library[
                IdentityF[OhoData[Traversable[Traversable[InputOutput]]]],
                GodInterpreter,
                GodState,
                IdentityF[Traversable[Traversable[PREDICTION]]],
            ]

            outerLibrary = endowBilevelOptimization(
                innerLibrary,
                doOptimizerStep,
                innerInterpreter,
                outerLearner,
                lambda a, b: LOSS(
                    jnp.mean(eqx.filter_vmap(eqx.filter_vmap(lossFn))(a.value.value, b.validation.value.value.y))
                ),
                tr_to_val_env,
                pad_val_grad_by,
            )

            outerController = logGradient(outerLibrary.modelGradient)
            outerLibrary = outerLibrary._replace(modelGradient=outerController)

            @do()
            def updateStep(oho_data: OhoData[Traversable[Traversable[InputOutput]]]):
                print("recompiled")
                env = yield from get(PX[GodState]())
                interpreter = yield from ask(PX[GodInterpreter]())
                hyperparameters = yield from interpreter.getRecurrentParam
                weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

                # te, _ = validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))
                vl, _ = validation_model(oho_data.validation)(innerInterpreter, tr_to_val_env(env, env.outer_prng))
                tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)
                te = 0

                _ = yield from outerLibrary.modelGradient(IdentityF(oho_data)).flat_map(doOptimizerStep)

                # code smell but what can you do, no maybe monad or elvis operator...
                def safe_norm(x):
                    return jnp.linalg.norm(x) if x is not None else None

                log = AllLogs(
                    train_loss=tr / config.tr_examples_per_epoch,
                    validation_loss=vl / config.vl_examples_per_epoch,
                    test_loss=te / config.numTe,
                    hyperparameters=hyperparameters,
                    inner_learning_rate=innerInterpreter.getLearningRate(env),
                    parameter_norm=safe_norm(weights),
                    oho_gradient=env.outerLogs.gradient,
                    train_gradient=env.innerLogs.gradient,
                    validation_gradient=env.outerLogs.validationGradient,
                    immediate_influence_tensor_norm=safe_norm(env.outerLogs.immediateInfluenceTensor),
                    outer_influence_tensor_norm=safe_norm(env.outerLogs.influenceTensor),
                    outer_influence_tensor=env.outerLogs.influenceTensor if config.log_accumulate_influence else None,
                    inner_influence_tensor_norm=safe_norm(env.innerLogs.influenceTensor),
                    largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
                    largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
                    jacobian=env.innerLogs.hessian,
                    hessian=env.outerLogs.hessian,
                    rnn_activation_norm=safe_norm(env.rnnState.activation),
                    immediate_influence_tensor=env.outerLogs.immediateInfluenceTensor
                    if config.log_accumulate_influence
                    else None,
                )
                logs: Traversable[AllLogs] = Traversable(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), log))
                return pure(logs, PX[tuple[GodInterpreter, GodState]]())

            model = eqx.filter_jit(lambda d, e: updateStep(d).func(outerInterpreter, e))
            return (
                model,
                lambda env: validation_model(test_dataset)(innerInterpreter, tr_to_te_env(env, env.outer_prng))[0],
                innerLibrary,
            )


def accuracy_hard(preds: jnp.ndarray, labels: jnp.ndarray) -> float:
    pred_classes = jnp.argmax(preds, axis=-1)
    return jnp.mean(pred_classes == labels).item()


def accuracy_soft(preds: jnp.ndarray, labels: jnp.ndarray) -> float:
    pred_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(labels, axis=-1)
    return jnp.mean(pred_classes == true_classes).item()


def accuracy_with_sequence_filter(preds: jnp.ndarray, labels: jnp.ndarray, n: int) -> float:
    class_indices = labels[0]
    sequence_numbers = labels[1]
    pred_classes = jnp.argmax(preds, axis=-1)

    mask = sequence_numbers == n
    filtered_preds = pred_classes[mask]
    filtered_labels = class_indices[mask]

    if filtered_labels.size == 0:
        return float("nan")

    return jnp.mean(filtered_preds == filtered_labels).item()
