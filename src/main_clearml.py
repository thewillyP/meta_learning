from typing import Union
import clearml
import random
import string
from cattrs import Converter
import random
import time
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import ClearMLLogger, HDF5Logger, MatplotlibLogger, MultiLogger, PrintLogger
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


def main():
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

    config = GodConfig(
        seed=SeedConfig(global_seed=14, data_seed=1, parameter_seed=1, task_seed=1),
        clearml_run=True,
        data_root_dir="/scratch/wlp9800/datasets",
        log_dir="/scratch/wlp9800/offline_logs",
        log_title="train",
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
            "meta1_sgd1_lr": HyperparameterConfig(
                value=0.001,
                kind="learning_rate",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_sgd1_wd": HyperparameterConfig(
                value=0.00001,
                kind="weight_decay",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=1,
                parametrizes_transition=True,
            ),
            "meta1_sgd1_momentum": HyperparameterConfig(
                value=0.0,
                kind="momentum",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=1,
                parametrizes_transition=True,
            ),
            "meta2_adam1_lr": HyperparameterConfig(
                value=0.001,
                kind="learning_rate",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=2,
                parametrizes_transition=True,
            ),
            "meta2_adam1_wd": HyperparameterConfig(
                value=0.0,
                kind="weight_decay",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=2,
                parametrizes_transition=True,
            ),
            "meta2_adam1_momentum": HyperparameterConfig(
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
                    track_influence_in=frozenset({0, 1}),
                ),
                learner=LearnConfig(
                    model_learner=GradientConfig(
                        method=BPTTConfig(None),
                        add_clip=HardClip(1.0),
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
                                learning_rate="meta1_sgd1_lr",
                                weight_decay="meta1_sgd1_wd",
                                momentum="meta1_sgd1_momentum",
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
                            target=frozenset({"meta1_sgd1_lr", "meta1_sgd1_wd", "meta1_sgd1_momentum"}),
                            optimizer=AdamConfig(
                                learning_rate="meta2_adam1_lr",
                                weight_decay="meta2_adam1_wd",
                                momentum="meta2_adam1_momentum",
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
    main()
