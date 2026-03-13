import copy
from typing import Union
import clearml
import random
import string
from cattrs import Converter
import argparse
import time
from configs import *
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import ClearMLLogger, HDF5Logger, MatplotlibLogger, PrintLogger
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
            NNLayer, VanillaRNNLayer, GRULayer, LSTMLayer, Scan, UnlabeledSource, LabeledSource, Repeat, Concat, ToEmpty
        ],
    )

    # -- GradientMethod --
    setup_flattened_union(
        converter,
        Union[RTRLConfig, RTRLFiniteHvpConfig, BPTTConfig, IdentityLearnerConfig, RFLOConfig, UOROConfig],
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
        Union[MNISTTaskFamily, CIFAR10TaskFamily, CIFAR100TaskFamily, DelayAddTaskFamily],
    )

    # -- ELBOObjective inner unions (must be registered BEFORE ObjectiveFn) --
    setup_flattened_union(converter, Union[RegressionObjective, BernoulliObjective])

    # -- ObjectiveFn --
    setup_flattened_union(
        converter,
        Union[ELBOObjective, RegressionObjective, CrossEntropyObjective, BernoulliObjective],
    )

    # -- LoggerConfig --
    setup_flattened_union(
        converter,
        Union[HDF5LoggerConfig, ClearMLLoggerConfig, PrintLoggerConfig, MatplotlibLoggerConfig],
    )

    return converter


def main(skip_jitter: bool):
    if not skip_jitter:
        _jitter_rng = random.Random()
        time.sleep(_jitter_rng.uniform(1, 60))

    task: clearml.Task = clearml.Task.init(
        project_name="temp",
        task_name="".join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        task_type=clearml.TaskTypes.training,
        auto_resource_monitoring=False,
    )

    slurm_params = SlurmParams(
        memory="8GB",
        time="01:00:00",
        cpu=2,
        gpu=0,
        log_dir="/vast/wlp9800/logs",
        skip_python_env_install=True,
    )

    converter = make_converter()
    task.connect(converter.unstructure(slurm_params), name="slurm")

    config = OHO_RNN256_V3
    config = copy.replace(config, logger_config=[ClearMLLoggerConfig()])

    def _deep_convert(obj):
        """Recursively convert sets/frozensets to sorted lists for JSON/ClearML compatibility."""
        if isinstance(obj, (set, frozenset)):
            return sorted((_deep_convert(v) for v in obj), key=str)
        if isinstance(obj, dict):
            return {k: _deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_deep_convert(v) for v in obj]
        return obj

    # Two connects: one as configuration (for full structure editing in UI),
    # one as hyperparameters (for HPO sweeps, since HPO can't add new fields)
    _config = task.connect_configuration(_deep_convert(converter.unstructure(config)), name="config")
    config = converter.structure(_config, GodConfig)

    _config = task.connect(_deep_convert(converter.unstructure(config)), name="config")
    config = converter.structure(_config, GodConfig)

    loggers = []
    for log_config in config.logger_config:
        match log_config:
            case HDF5LoggerConfig():
                logger = HDF5Logger(config.log_dir, task.task_id)
            case ClearMLLoggerConfig():
                logger = ClearMLLogger(task)
            case PrintLoggerConfig():
                logger = PrintLogger()
            case MatplotlibLoggerConfig(save_dir):
                logger = MatplotlibLogger(save_dir)
            case _:
                raise ValueError("Invalid logger configuration.")
        loggers.append(logger)

    app.runApp(config, loggers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jitter", action="store_true", default=False)
    args = parser.parse_args()
    main(skip_jitter=args.skip_jitter)
