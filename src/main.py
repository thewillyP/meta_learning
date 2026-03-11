import jax.numpy as jnp
from meta_learn_lib import app
from meta_learn_lib.config import *
from meta_learn_lib.logger import *
# import jax

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def main():
    config = GodConfig(
        seed=SeedConfig(global_seed=14, data_seed=1, parameter_seed=1, task_seed=1),
        clearml_run=True,
        data_root_dir="/scratch/wlp9800/datasets",
        log_dir="/scratch/wlp9800/offline_logs",
        log_title="train",
        logger_config=[MatplotlibLoggerConfig(save_dir="/scratch/wlp9800/offline_logs")],
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

    loggers = []
    for log_config in config.logger_config:
        match log_config:
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
