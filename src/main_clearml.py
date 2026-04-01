import json
import os
from typing import Union
import clearml
from clearml import InputModel, Model
import random
import string
from cattrs import Converter
import argparse
import time
from meta_learn_lib import app
from meta_learn_lib.checkpoint import ClearMLCheckpointManager, NullCheckpointManager
from meta_learn_lib.config import *
from meta_learn_lib.logger import ClearMLLogger, HDF5Logger, MatplotlibLogger, ConsoleLogger
from meta_learn_lib.util import setup_flattened_union
import jax.numpy as jnp

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def make_converter() -> Converter:
    """Build a cattrs Converter that handles all union types and special types like frozenset."""
    converter = Converter()

    # -- frozenset/set: serialize as sorted list so ClearML/JSON can handle it --
    converter.register_unstructure_hook(frozenset, lambda fs: sorted(fs, key=str))
    converter.register_structure_hook(frozenset, lambda val, _: frozenset(val))
    converter.register_unstructure_hook(set, lambda s: sorted(s, key=str))
    converter.register_structure_hook(set, lambda val, _: set(val))

    # -- list <-> index-keyed dict: so ClearML task.connect() flattens list elements --
    converter.register_unstructure_hook_func(
        lambda tp: getattr(tp, "__origin__", None) is list or tp is list,
        lambda lst: {str(i): converter.unstructure(v) for i, v in enumerate(lst)},
    )
    converter.register_structure_hook_func(
        lambda tp: getattr(tp, "__origin__", None) is list,
        lambda val, tp: (
            [converter.structure(val[k], tp.__args__[0]) for k in sorted(val, key=lambda k: int(k))]
            if isinstance(val, dict)
            else [converter.structure(v, tp.__args__[0]) for v in val]
        ),
    )

    # -- jnp.inf: serialize as string so JSON doesn't choke --
    original_float_unstructure = converter._unstructure_func.dispatch(float)

    def unstructure_float(val):
        if val == float("inf") or (hasattr(val, "item") and val.item() == float("inf")):
            return "inf"
        if val == float("-inf") or (hasattr(val, "item") and val.item() == float("-inf")):
            return "-inf"
        return original_float_unstructure(val)

    converter.register_unstructure_hook(float, unstructure_float)

    original_float_structure = converter._structure_func.dispatch(float)

    def structure_float(val, tp):
        if val == "inf":
            return jnp.inf
        if val == "-inf":
            return -jnp.inf
        return original_float_structure(val, tp)

    converter.register_structure_hook(float, structure_float)

    # -- HyperparameterConfig.Parametrization --
    setup_flattened_union(
        converter,
        Union[
            HyperparameterConfig.identity,
            HyperparameterConfig.softplus,
            HyperparameterConfig.relu,
            HyperparameterConfig.softrelu,
            HyperparameterConfig.silu_positive,
            HyperparameterConfig.squared,
            HyperparameterConfig.softclip,
        ],
    )

    # -- Node types --
    setup_flattened_union(
        converter,
        Union[
            NNLayer,
            VanillaRNNLayer,
            GRULayer,
            LSTMLayer,
            Scan,
            UnlabeledSource,
            LabeledSource,
            Repeat,
            Concat,
            ToEmpty,
            ReparameterizeLayer,
            MergeOutputs,
            ExtractZ,
            Reshape,
        ],
    )

    # -- GradientMethod --
    setup_flattened_union(
        converter,
        Union[
            RTRLConfig,
            RTRLFiniteHvpConfig,
            BPTTConfig,
            IdentityLearnerConfig,
            RFLOConfig,
            UOROConfig,
            UOROFiniteDiffConfig,
            ImmediateLearnerConfig,
        ],
    )

    # -- Optimizer --
    setup_flattened_union(
        converter,
        Union[SGDConfig, SGDNormalizedConfig, AdamConfig, ExponentiatedGradientConfig],
    )

    # -- Clip --
    setup_flattened_union(converter, Union[HardClip, SoftClip])
    # GradientConfig.add_clip is HardClip | SoftClip | None — cattrs sees this as
    # Union[HardClip, SoftClip, NoneType], a different type. Reuse the same hook.
    clip_hook = converter._structure_func.dispatch(Union[HardClip, SoftClip])
    converter.register_structure_hook(Union[HardClip, SoftClip, None], clip_hook)

    # -- Task / dataset source --
    setup_flattened_union(
        converter,
        Union[
            MNISTTaskFamily,
            CIFAR10TaskFamily,
            CIFAR100TaskFamily,
            DelayAddTaskFamily,
            GaussianNoiseTaskFamily,
        ],
    )

    # -- ELBOObjective inner unions (must be registered BEFORE ObjectiveFn) --
    setup_flattened_union(converter, Union[RegressionObjective, BernoulliObjective])

    # -- ObjectiveFn --
    setup_flattened_union(
        converter,
        Union[ELBOObjective, RegressionObjective, CrossEntropyObjective, BernoulliObjective],
    )

    # -- SampleInput --
    setup_flattened_union(converter, Union[GaussianSampleInput, DataSampleInput])

    return converter


def fetch_config(config_name: str | None = None, config_id: str | None = None) -> tuple[InputModel, dict]:
    """Fetch a config dict from ClearML Model Registry by name (latest) or exact model ID."""
    CONFIG_PROJECT = "oho"
    if config_id:
        model = InputModel(model_id=config_id)
    elif config_name:
        models = Model.query_models(
            project_name=CONFIG_PROJECT,
            model_name=config_name,
            tags=["config"],
        )
        if not models:
            raise ValueError(f"No config '{config_name}' found in project '{CONFIG_PROJECT}'")
        # query_models returns newest first — take the latest
        model = InputModel(model_id=models[0].id)
    else:
        raise ValueError("Must specify either config_name or config_id")

    local_path = model.get_local_copy()
    with open(local_path) as f:
        return model, json.load(f)


def main(config_name: str | None, config_id: str | None, skip_jitter: bool, resume_model_id: str | None):
    if not skip_jitter:
        _jitter_rng = random.Random()
        time.sleep(_jitter_rng.uniform(1, 60))

    os.environ["CLEARML_SET_ITERATION_OFFSET"] = "0"
    task: clearml.Task = clearml.Task.init(
        project_name="temp",
        task_name="".join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        task_type=clearml.TaskTypes.training,
        auto_resource_monitoring=False,
        output_uri=True,
    )

    slurm_params = SlurmParams(
        memory="16GB",
        time="01:00:00",
        cpu=2,
        gpu=1,
        log_dir="/scratch/wlp9800/logs",
        skip_python_env_install=True,
    )

    converter = make_converter()
    task.connect(converter.unstructure(slurm_params), name="slurm")

    # Fetch base config from registry, then connect so ClearML sweeps/UI overrides work
    config_model, config_dict = fetch_config(config_name=config_name, config_id=config_id)
    task.connect(config_model)
    _config = task.connect(config_dict, name="config")
    config = converter.structure(_config, GodConfig)

    loggers = []
    lc = config.logger_config
    if lc.clearml.enabled:
        loggers.append(ClearMLLogger(task))
    if lc.hdf5.enabled:
        loggers.append(HDF5Logger(config.log_dir, task.task_id, config.checkpoint_every_n_minibatches))
    if lc.console.enabled:
        loggers.append(ConsoleLogger())
    if lc.matplotlib.enabled:
        loggers.append(MatplotlibLogger(lc.matplotlib.save_dir))

    if config.checkpoint_every_n_epochs > 0:
        ckpt_dir = os.path.join(config.data_root_dir, "checkpoints", task.task_id)
        checkpoint_manager = ClearMLCheckpointManager(task, ckpt_dir, initial_model_id=resume_model_id)
    else:
        checkpoint_manager = NullCheckpointManager()

    app.runApp(config, loggers, checkpoint_manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config-name", default=None, help="Config name in the registry (gets latest)")
    group.add_argument("--config-id", default=None, help="Exact model ID for a pinned config")
    parser.add_argument("--skip-jitter", action="store_true", default=False)
    parser.add_argument("--resume-model-id", default=None, help="ClearML model ID to resume training from")
    args = parser.parse_args()
    main(
        config_name=args.config_name,
        config_id=args.config_id,
        skip_jitter=args.skip_jitter,
        resume_model_id=args.resume_model_id,
    )
