"""Correctness tests for the meta-learning algorithm."""

import random
import os
from typing import Callable, Iterator, NamedTuple
import numpy as np
import pytest
import torch
import toolz
import jax
import jax.numpy as jnp
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.constants import MODEL_LEARNER, OPTIMIZER_LEARNER
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.create_env import (
    create_env,
    create_inference_axes,
    create_transition_fns,
    env_resetters,
    env_validation_resetters,
    make_tick_advancer,
)
from meta_learn_lib.create_interface import build_interfaces
from meta_learn_lib.datasets import create_data_sources, create_dataloader, validate_dataloader_config
from meta_learn_lib.env import *
from meta_learn_lib.inference import create_inference_and_readout
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.learning import create_meta_learner, create_validation_learners
from meta_learn_lib.lib_types import *
from meta_learn_lib.loss_function import create_readout_loss_fns


class Setup(NamedTuple):
    env: GodState
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]]
    interfaces: dict[S_ID, GodInterface[GodState]]
    transition_fns: list[Callable[[GodState, tuple[jax.Array, jax.Array]], tuple[GodState, STAT]]]
    loss_fns: list[Callable[[GodState, tuple[jax.Array, jax.Array]], tuple[GodState, LOSS, STAT]]]
    inference_axes: list[GodState]
    data_sample: tuple
    dataloader: Iterator
    meta_learner: Callable[[GodState, tuple], tuple[GodState, STAT]]


# ============================================================================
# Config builder
# ============================================================================

RNN_SIZE = 8
NUM_CLASSES = 10


def make_single_level_config(method: GradientMethod, model_clip: Optional[Clip] = None) -> GodConfig:
    """Single-level config. `method` is the model_learner method at level 0."""
    return GodConfig(
        seed=SeedConfig(global_seed=42, data_seed=1, parameter_seed=1, task_seed=1, sample_seed=1),
        clearml_run=True,
        data_root_dir="/scratch/wlp9800/datasets",
        log_dir="/tmp",
        log_title="test",
        logger_config=LoggersConfig(
            clearml=ClearMLLoggerConfig(enabled=False),
            hdf5=HDF5LoggerConfig(enabled=False),
            sqlite=SQLiteLoggerConfig(enabled=False),
            console=ConsoleLoggerConfig(enabled=False),
            matplotlib=MatplotlibLoggerConfig(save_dir="/tmp", enabled=False),
            scalar_queue_size=0,
            sample_queue_size=0,
        ),
        epochs=1,
        checkpoint_every_n_minibatches=1,
        checkpoint_every_n_epochs=0,
        transition_graph={
            "x": frozenset(),
            "concat": frozenset({"x"}),
            "rnn1": frozenset({"concat"}),
        },
        readout_graph={
            "readout": frozenset({"rnn1"}),
        },
        nodes={
            "x": UnlabeledSource(),
            "concat": Concat(),
            "rnn1": VanillaRNNLayer(
                nn_layer=NNLayer(
                    n=RNN_SIZE,
                    activation_fn="tanh",
                    use_bias=True,
                    init="lecun_normal",
                ),
                layer_norm=None,
                use_random_init=False,
                time_constant="hp_tc",
            ),
            "readout": NNLayer(
                n=NUM_CLASSES,
                activation_fn="identity",
                use_bias=True,
                init="lecun_normal",
            ),
        },
        aliases={},
        hyperparameters={
            "hp_tc": HyperparameterConfig(
                value=1.0,
                kind="time_constant",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=0,
                parametrizes_transition=True,
            ),
            "hp_lr": HyperparameterConfig(
                value=0.01,
                kind="learning_rate",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=0,
                parametrizes_transition=True,
            ),
            "hp_wd": HyperparameterConfig(
                value=0.0,
                kind="weight_decay",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=jnp.inf,
                level=0,
                parametrizes_transition=True,
            ),
            "hp_mom": HyperparameterConfig(
                value=0.0,
                kind="momentum",
                count=1,
                hyperparameter_parametrization=HyperparameterConfig.identity(),
                min_value=0.0,
                max_value=1.0,
                level=0,
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
                    pixel_transform="normalize",
                ),
                dataset=DatasetConfig(
                    num_examples_in_minibatch=10,
                    num_examples_total=100,
                    is_test=False,
                    augment=False,
                    shuffle=True,
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
                    track_influence_in=frozenset({0}),
                ),
                learner=LearnConfig(
                    model_learner=GradientConfig(
                        method=method,
                        add_clip=model_clip,
                        scale=1.0,
                    ),
                    optimizer_learner=GradientConfig(
                        method=RTRLConfig(
                            start_at_step=0,
                            damping=0.0,
                            beta=1.0,
                            use_finite_hvp=None,
                            influence_clip=None,
                            propagation_clip=None,
                            lr_edge_margin=None,
                            unit_circle_margin=None,
                        ),
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer={
                        "sgd1": OptimizerAssignment(
                            target=frozenset({"rnn1", "readout"}),
                            optimizer=SGDConfig(
                                learning_rate="hp_lr",
                                weight_decay="hp_wd",
                                momentum="hp_mom",
                            ),
                            add_clip=None,
                        ),
                    },
                ),
                track_logs=TrackLogs(
                    gradient=False,
                    hessian_contains_nans=False,
                    largest_eigenvalue=False,
                    influence_tensor_norm=False,
                    immediate_influence_tensor=False,
                    largest_jac_eigenvalue=False,
                    jacobian=False,
                ),
                test_seed=0,
                collect_predictions=False,
            ),
        ],
        sample_generators=[],
        label_mask_value=-1.0,
        unlabeled_mask_value=-100.0,
        num_tasks=1,
        prefetch_buffer_size=1,
        dataloader_chunk_size=None,
    )


META_INNER_STEPS = 4


def make_two_level_config(
    inner_method: GradientMethod,
    meta_method: GradientMethod,
    inner_steps: int,
    level0_clip: Optional[Clip] = None,
    level1_clip: Optional[Clip] = None,
    outer_val_clip: Optional[Clip] = None,
    track0: frozenset[int] = frozenset({0}),
    track1: frozenset[int] = frozenset({1}),
    validation_steps: int = 28,
) -> GodConfig:
    immediate = GradientConfig(method=ImmediateLearnerConfig(), add_clip=None, scale=1.0)
    level0_model = GradientConfig(method=inner_method, add_clip=level0_clip, scale=1.0)
    level1_model = GradientConfig(method=inner_method, add_clip=outer_val_clip, scale=1.0)
    meta_grad = GradientConfig(method=meta_method, add_clip=level1_clip, scale=1.0)

    def hp(value: float, kind: HyperparameterConfig.Kind, level: int, max_value: float) -> HyperparameterConfig:
        return HyperparameterConfig(
            value=value,
            kind=kind,
            count=1,
            hyperparameter_parametrization=HyperparameterConfig.identity(),
            min_value=0.0,
            max_value=max_value,
            level=level,
            parametrizes_transition=True,
        )

    def meta_level(num_steps: int, track: frozenset[int], learner: LearnConfig) -> MetaConfig:
        return MetaConfig(
            objective_fn=CrossEntropyObjective(mode="cross_entropy_with_integer_labels"),
            dataset_source=MNISTTaskFamily(
                patch_h=1,
                patch_w=28,
                label_last_only=True,
                add_spurious_pixel_to_train=False,
                pixel_transform="normalize",
            ),
            dataset=DatasetConfig(
                num_examples_in_minibatch=10,
                num_examples_total=200,
                is_test=False,
                augment=False,
                shuffle=True,
            ),
            validation=StepConfig(
                num_steps=validation_steps, batch=1, reset_t=validation_steps, track_influence_in=track
            ),
            nested=StepConfig(num_steps=num_steps, batch=1, reset_t=None, track_influence_in=track),
            learner=learner,
            track_logs=TrackLogs(
                gradient=False,
                hessian_contains_nans=False,
                largest_eigenvalue=False,
                influence_tensor_norm=False,
                immediate_influence_tensor=False,
                largest_jac_eigenvalue=False,
                jacobian=False,
            ),
            test_seed=0,
            collect_predictions=False,
        )

    return GodConfig(
        seed=SeedConfig(global_seed=42, data_seed=1, parameter_seed=1, task_seed=1, sample_seed=1),
        clearml_run=True,
        data_root_dir="/scratch/wlp9800/datasets",
        log_dir="/tmp",
        log_title="test",
        logger_config=LoggersConfig(
            clearml=ClearMLLoggerConfig(enabled=False),
            hdf5=HDF5LoggerConfig(enabled=False),
            sqlite=SQLiteLoggerConfig(enabled=False),
            console=ConsoleLoggerConfig(enabled=False),
            matplotlib=MatplotlibLoggerConfig(save_dir="/tmp", enabled=False),
            scalar_queue_size=0,
            sample_queue_size=0,
        ),
        epochs=1,
        checkpoint_every_n_minibatches=1,
        checkpoint_every_n_epochs=0,
        transition_graph={
            "x": frozenset(),
            "concat": frozenset({"x"}),
            "rnn1": frozenset({"concat"}),
        },
        readout_graph={
            "readout": frozenset({"rnn1"}),
        },
        nodes={
            "x": UnlabeledSource(),
            "concat": Concat(),
            "rnn1": VanillaRNNLayer(
                nn_layer=NNLayer(n=RNN_SIZE, activation_fn="tanh", use_bias=True, init="lecun_normal"),
                layer_norm=None,
                use_random_init=False,
                time_constant="meta1_tc",
            ),
            "readout": NNLayer(n=NUM_CLASSES, activation_fn="identity", use_bias=True, init="lecun_normal"),
        },
        aliases={},
        hyperparameters={
            "meta1_tc": hp(1.0, "time_constant", 1, 1.0),
            "meta1_lr": hp(0.05, "learning_rate", 1, jnp.inf),
            "meta1_wd": hp(0.001, "weight_decay", 1, jnp.inf),
            "meta1_mom": hp(0.0, "momentum", 1, 1.0),
            "meta2_lr": hp(1e-3, "learning_rate", 1, jnp.inf),
            "meta2_wd": hp(0.0, "weight_decay", 1, jnp.inf),
            "meta2_mom": hp(0.0, "momentum", 1, 1.0),
        },
        levels=[
            meta_level(
                num_steps=1,
                track=track0,
                learner=LearnConfig(
                    model_learner=level0_model,
                    optimizer_learner=immediate,
                    optimizer={
                        "meta1_sgd1": OptimizerAssignment(
                            target=frozenset({"rnn1", "readout"}),
                            optimizer=SGDConfig(
                                learning_rate="meta1_lr", weight_decay="meta1_wd", momentum="meta1_mom"
                            ),
                            add_clip=None,
                        ),
                    },
                ),
            ),
            meta_level(
                num_steps=inner_steps,
                track=track1,
                learner=LearnConfig(
                    model_learner=level1_model,
                    optimizer_learner=meta_grad,
                    optimizer={
                        "meta2_sgd1": OptimizerAssignment(
                            target=frozenset({"meta1_lr", "meta1_wd"}),
                            optimizer=SGDConfig(
                                learning_rate="meta2_lr", weight_decay="meta2_wd", momentum="meta2_mom"
                            ),
                            add_clip=None,
                        ),
                    },
                ),
            ),
        ],
        sample_generators=[],
        label_mask_value=-1.0,
        unlabeled_mask_value=-100.0,
        num_tasks=1,
        prefetch_buffer_size=1,
        dataloader_chunk_size=None,
    )


# ============================================================================
# Setup helper
# ============================================================================


def setup_env_and_fns(config: GodConfig):
    """Build env, interfaces, meta-learner, and one real data batch."""

    base_key = jax.random.key(config.seed.global_seed)
    keys = jax.random.split(base_key, 3)
    data_prng = PRNG(jax.random.fold_in(keys[0], config.seed.data_seed))
    env_prng = PRNG(jax.random.fold_in(keys[1], config.seed.parameter_seed))
    task_prng = PRNG(jax.random.fold_in(keys[2], config.seed.task_seed))
    dataset_gen_prng, torch_prng = jax.random.split(data_prng, 2)
    torch_seed = int(jax.random.randint(torch_prng, shape=(), minval=0, maxval=1_000_000, dtype=jnp.uint32))
    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)

    errors = validate_dataloader_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    dataset_prng, data_loader_prng = jax.random.split(dataset_gen_prng, 2)
    data_sources, shapes = create_data_sources(config, dataset_prng)
    dataloader = create_dataloader(config, data_sources, data_loader_prng, task_prng)

    interfaces = build_interfaces(config)

    env = create_env(config, shapes, env_prng)

    inference_axes = [create_inference_axes(env, config, interfaces, s, lvl) for lvl, s in enumerate(shapes)]

    transitions, readouts = zip(
        *[create_inference_and_readout(config, interfaces, lvl, ax) for lvl, ax in enumerate(inference_axes)]
    )

    transition_fns = create_transition_fns(config, shapes, interfaces, list(transitions))
    loss_fns = create_readout_loss_fns(config, interfaces, list(readouts))

    meta_learner = create_meta_learner(config, shapes, transition_fns, loss_fns, interfaces, env)

    data_sample, dataloader = toolz.peek(dataloader)

    return Setup(
        env=env,
        shapes=shapes,
        interfaces=interfaces,
        transition_fns=transition_fns,
        loss_fns=loss_fns,
        inference_axes=inference_axes,
        data_sample=data_sample,
        dataloader=dataloader,
        meta_learner=meta_learner,
    )


# ============================================================================
# TEST 1: RTRL vs BPTT at level 0
# ============================================================================


def test_rtrl_vs_bptt_level0():
    print("=" * 60)
    print("TEST 1: RTRL vs BPTT gradient equivalence (level 0)")
    print("=" * 60)

    config_bptt = make_single_level_config(BPTTConfig(truncate_at=None))
    config_rtrl = make_single_level_config(
        RTRLConfig(
            start_at_step=0,
            damping=0.0,
            beta=1.0,
            use_finite_hvp=None,
            influence_clip=None,
            propagation_clip=None,
            lr_edge_margin=None,
            unit_circle_margin=None,
        )
    )

    stuff_bptt = setup_env_and_fns(config_bptt)
    stuff_rtrl = setup_env_and_fns(config_rtrl)

    data = stuff_bptt.data_sample

    params_init_bptt = stuff_bptt.interfaces[(MODEL_LEARNER, 0)].param.get(stuff_bptt.env)
    params_init_rtrl = stuff_rtrl.interfaces[(MODEL_LEARNER, 0)].param.get(stuff_rtrl.env)
    init_diff = jnp.max(jnp.abs(params_init_bptt - params_init_rtrl))
    print(f"  Initial param diff: {init_diff:.2e}")

    env_bptt_after, stats_bptt = stuff_bptt.meta_learner(stuff_bptt.env, data)
    env_rtrl_after, stats_rtrl = stuff_rtrl.meta_learner(stuff_rtrl.env, data)

    norm_bptt = stats_bptt["level0/meta_gradient_norm"].data
    norm_rtrl = stats_rtrl["level0/meta_gradient_norm"].data
    norm_diff = jnp.abs(norm_bptt - norm_rtrl)

    print(f"  BPTT meta-gradient norm: {norm_bptt:.8f}")
    print(f"  RTRL meta-gradient norm: {norm_rtrl:.8f}")
    print(f"  Abs diff in norms:       {norm_diff:.2e}")

    val_interface_bptt = stuff_bptt.interfaces[(MODEL_LEARNER, 0)]
    val_interface_rtrl = stuff_rtrl.interfaces[(MODEL_LEARNER, 0)]
    params_bptt = val_interface_bptt.param.get(env_bptt_after)
    params_rtrl = val_interface_rtrl.param.get(env_rtrl_after)
    param_diff = jnp.max(jnp.abs(params_bptt - params_rtrl))

    print(f"  Max abs param difference: {param_diff:.2e}")

    assert norm_bptt > 1e-8, f"BPTT gradient norm is zero — test is degenerate"
    assert norm_rtrl > 1e-8, f"RTRL gradient norm is zero — test is degenerate"
    assert param_diff < 1e-10, f"RTRL/BPTT param diff too large: {param_diff:.2e}"
    assert norm_diff < 1e-10, f"RTRL/BPTT meta-gradient norm diff too large: {norm_diff:.2e}"


# ============================================================================
# TEST 1b: Validation learner gradient RTRL vs BPTT (direct comparison)
# ============================================================================


def test_validation_gradient_rtrl_vs_bptt():
    print("=" * 60)
    print("TEST 1b: Validation gradient RTRL vs BPTT (direct)")
    print("=" * 60)

    config_bptt = make_single_level_config(BPTTConfig(truncate_at=None))
    config_rtrl = make_single_level_config(
        RTRLConfig(
            start_at_step=0,
            damping=0.0,
            beta=1.0,
            use_finite_hvp=None,
            influence_clip=None,
            propagation_clip=None,
            lr_edge_margin=None,
            unit_circle_margin=None,
        )
    )

    stuff_bptt = setup_env_and_fns(config_bptt)
    stuff_rtrl = setup_env_and_fns(config_rtrl)

    vl_learners_bptt, _ = create_validation_learners(
        stuff_bptt.transition_fns, stuff_bptt.loss_fns, stuff_bptt.interfaces, config_bptt
    )
    vl_learners_rtrl, _ = create_validation_learners(
        stuff_rtrl.transition_fns, stuff_rtrl.loss_fns, stuff_rtrl.interfaces, config_rtrl
    )
    vl_bptt = vl_learners_bptt[0]
    vl_rtrl = vl_learners_rtrl[0]

    env_bptt = stuff_bptt.env
    env_rtrl = stuff_rtrl.env

    def get_axes(config, stuff):
        resetters = env_resetters(config, stuff.shapes, stuff.interfaces, [False] * len(config.levels))
        inner_resetter, _ = resetters[0]
        return diff_axes(stuff.env, inner_resetter(stuff.env, jax.random.key(0)))

    axes_bptt = get_axes(config_bptt, stuff_bptt)
    axes_rtrl = get_axes(config_rtrl, stuff_rtrl)

    data_batch = stuff_bptt.data_sample
    step_data = jax.tree.map(lambda x: x[0], data_batch)
    _, vl_data = step_data

    def run_vl(vl_fn, env, data, axes):
        def inner(env, data):
            _, grad, stats = vl_fn(env, data)
            return grad

        return eqx.filter_vmap(inner, in_axes=(axes, 0), out_axes=0)(env, data)

    grad_bptt = run_vl(vl_bptt, env_bptt, vl_data, axes_bptt).squeeze(axis=0)
    grad_rtrl = run_vl(vl_rtrl, env_rtrl, vl_data, axes_rtrl).squeeze(axis=0)

    diff = jnp.max(jnp.abs(grad_bptt - grad_rtrl))
    rel_diff = diff / jnp.maximum(jnp.max(jnp.abs(grad_bptt)), 1e-10)

    print(f"  BPTT validation gradient norm: {jnp.linalg.norm(grad_bptt):.8f}")
    print(f"  RTRL validation gradient norm: {jnp.linalg.norm(grad_rtrl):.8f}")
    print(f"  Max abs difference:            {diff:.2e}")
    print(f"  Max rel difference:            {rel_diff:.2e}")

    assert jnp.linalg.norm(grad_bptt) > 1e-8, f"BPTT gradient is zero — test is degenerate"
    assert diff < 1e-10, f"RTRL/BPTT validation gradient diff too large: {diff:.2e}"


# ============================================================================
# TEST 1b2: jacfwd vs jacrev of the validation gradient (2nd-order consistency)
# ============================================================================


def test_validation_gradient_jacobian_consistency():
    print("=" * 60)
    print("TEST 1b2: jacfwd vs jacrev of d(val_grad)/d(theta)")
    print("=" * 60)

    for label, method in [
        ("BPTT", BPTTConfig(truncate_at=None)),
        (
            "RTRL",
            RTRLConfig(
                start_at_step=0,
                damping=0.0,
                beta=1.0,
                use_finite_hvp=None,
                influence_clip=None,
                propagation_clip=None,
                lr_edge_margin=None,
                unit_circle_margin=None,
            ),
        ),
    ]:
        config = make_single_level_config(method)
        stuff = setup_env_and_fns(config)
        vl_learners, _ = create_validation_learners(stuff.transition_fns, stuff.loss_fns, stuff.interfaces, config)
        vl = vl_learners[0]
        iface = stuff.interfaces[(MODEL_LEARNER, 0)]

        resetters = env_resetters(config, stuff.shapes, stuff.interfaces, [False] * len(config.levels))
        inner_resetter, _ = resetters[0]
        axes = diff_axes(stuff.env, inner_resetter(stuff.env, jax.random.key(0)))

        step_data = jax.tree.map(lambda x: x[0], stuff.data_sample)
        _, vl_data = step_data
        theta0 = iface.param.get(stuff.env)

        def g_fn(theta):
            env_t = iface.param.put(stuff.env, theta)

            def inner(env, data):
                _, grad, _ = vl(env, data)
                return grad

            grads = eqx.filter_vmap(inner, in_axes=(axes, 0), out_axes=0)(env_t, vl_data)
            return grads.sum(axis=0)

        Jf = jax.jacfwd(g_fn)(theta0)
        Jr = jax.jacrev(g_fn)(theta0)
        diff = jnp.max(jnp.abs(Jf - Jr))
        rel = diff / jnp.maximum(jnp.max(jnp.abs(Jr)), 1e-30)
        print(f"  {label}: jacfwd vs jacrev  max abs={diff:.3e}  rel={rel:.3e}")

        if label == "BPTT":
            assert rel < 1e-8, f"BPTT 2nd-order inconsistent (control should pass): {rel:.3e}"


# ============================================================================
# Meta-hypergradient runner / divergence helpers
# ============================================================================


def run_two_level_meta(config: GodConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    stuff = setup_env_and_fns(config)
    env_after, stats = eqx.filter_jit(stuff.meta_learner)(stuff.env, stuff.data_sample)
    iface = stuff.interfaces[(OPTIMIZER_LEARNER, 1)]
    hp_init = iface.param.get(stuff.env)
    hp_after = iface.param.get(env_after)
    norm = stats["level1/meta_gradient_norm"].data
    return norm, hp_init, hp_after


def hypergradient_rel_divergence(hp_init: jax.Array, hp_after: jax.Array, hp_after_truth: jax.Array) -> jax.Array:
    step = hp_after - hp_init
    step_truth = hp_after_truth - hp_init
    return jnp.linalg.norm(step - step_truth) / jnp.maximum(jnp.linalg.norm(step_truth), 1e-30)


# ============================================================================
# TEST 1c: Meta hypergradient — RTRL-over-RTRL vs BPTT-over-BPTT
# ============================================================================


@pytest.mark.slow
def test_meta_hypergradient_rtrl_vs_bptt():
    print("=" * 60)
    print("TEST 1c: Meta hypergradient RTRL-over-RTRL vs BPTT-over-BPTT")
    print("=" * 60)

    bptt = BPTTConfig(truncate_at=None)
    rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )

    norm_bptt, hp_init_bptt, hp_after_bptt = run_two_level_meta(make_two_level_config(bptt, bptt, META_INNER_STEPS))
    norm_rtrl, hp_init_rtrl, hp_after_rtrl = run_two_level_meta(make_two_level_config(rtrl, rtrl, META_INNER_STEPS))

    init_diff = jnp.max(jnp.abs(hp_init_bptt - hp_init_rtrl))
    norm_abs_diff = jnp.abs(norm_bptt - norm_rtrl)
    norm_rel_diff = norm_abs_diff / jnp.maximum(jnp.abs(norm_bptt), 1e-12)
    hp_step = jnp.max(jnp.abs(hp_after_bptt - hp_init_bptt))
    vec_rel_diff = hypergradient_rel_divergence(hp_init_bptt, hp_after_rtrl, hp_after_bptt)

    print(f"  Initial meta-param diff: {init_diff:.2e}")
    print(f"  BPTT-over-BPTT hypergradient norm: {norm_bptt:.12f}")
    print(f"  RTRL-over-RTRL hypergradient norm: {norm_rtrl:.12f}")
    print(f"  Rel diff in norms:           {norm_rel_diff:.2e}")
    print(f"  Rel diff in vector (vs BPTT): {vec_rel_diff:.2e}")
    print(f"  Inner hp after (BPTT): {hp_after_bptt}")
    print(f"  Inner hp after (RTRL): {hp_after_rtrl}")
    print(f"  Meta step size (max |hp_after - hp_init|): {hp_step:.2e}")

    assert init_diff < 1e-12, f"Starting meta-params differ: {init_diff:.2e}"
    assert norm_bptt > 1e-8, "BPTT hypergradient norm is zero — test is degenerate"
    assert norm_rtrl > 1e-8, "RTRL hypergradient norm is zero — test is degenerate"
    assert hp_step > 1e-8, "Meta step is zero — meta optimizer did not move the hyperparameters"
    assert norm_rel_diff < 1e-4, f"RTRL/BPTT hypergradient norm rel diff too large: {norm_rel_diff:.2e}"
    assert vec_rel_diff < 1e-4, f"RTRL/BPTT hypergradient vector rel diff too large: {vec_rel_diff:.2e}"


# ============================================================================
# TEST 1d: Finite-difference RTRL-over-RTRL is close to exact BPTT-over-BPTT
# ============================================================================


@pytest.mark.slow
def test_meta_hypergradient_finite_difference_rtrl():
    print("=" * 60)
    print("TEST 1d: Finite-difference RTRL-over-RTRL vs exact BPTT-over-BPTT")
    print("=" * 60)

    eps = 1e-3
    bptt = BPTTConfig(truncate_at=None)
    fd_rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=eps,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )

    norm_truth, hp_init, hp_after_truth = run_two_level_meta(make_two_level_config(bptt, bptt, META_INNER_STEPS))
    norm_fd, _, hp_after_fd = run_two_level_meta(make_two_level_config(fd_rtrl, fd_rtrl, META_INNER_STEPS))

    vec_rel_diff = hypergradient_rel_divergence(hp_init, hp_after_fd, hp_after_truth)
    norm_rel_diff = jnp.abs(norm_fd - norm_truth) / jnp.maximum(jnp.abs(norm_truth), 1e-12)

    print(f"  eps = {eps:.0e}")
    print(f"  exact BPTT/BPTT hypergradient norm:   {norm_truth:.12f}")
    print(f"  finite-diff RTRL/RTRL hypergrad norm: {norm_fd:.12f}")
    print(f"  Rel diff in norms:            {norm_rel_diff:.2e}")
    print(f"  Rel diff in vector (vs BPTT): {vec_rel_diff:.2e}")

    assert norm_truth > 1e-8, "ground-truth hypergradient norm is zero — test is degenerate"
    assert norm_fd > 1e-8, "finite-diff hypergradient norm is zero — test is degenerate"
    assert vec_rel_diff < 5e-2, f"finite-diff RTRL hypergradient too far from BPTT: {vec_rel_diff:.2e}"


# ============================================================================
# TEST 1e: How far the finite-difference OHO hypergradient diverges from truth
# ============================================================================


@pytest.mark.slow
def test_finite_difference_oho_divergence():
    print("=" * 60)
    print("TEST 1e: Finite-difference OHO divergence from true hypergradient")
    print("=" * 60)

    bptt = BPTTConfig(truncate_at=None)
    exact_rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )
    eps_values = [1e-2, 1e-4]

    _, hp_init, hp_after_truth = run_two_level_meta(make_two_level_config(bptt, bptt, META_INNER_STEPS))

    _, _, hp_after_exact = run_two_level_meta(make_two_level_config(bptt, exact_rtrl, META_INNER_STEPS))
    div_exact = hypergradient_rel_divergence(hp_init, hp_after_exact, hp_after_truth)
    print(f"  exact RTRL OHO            rel divergence = {div_exact:.3e}")

    div_finite = {}
    for eps in eps_values:
        fd_rtrl = RTRLConfig(
            start_at_step=0,
            damping=0.0,
            beta=1.0,
            use_finite_hvp=eps,
            influence_clip=None,
            propagation_clip=None,
            lr_edge_margin=None,
            unit_circle_margin=None,
        )
        _, _, hp_after_fd = run_two_level_meta(make_two_level_config(bptt, fd_rtrl, META_INNER_STEPS))
        div = hypergradient_rel_divergence(hp_init, hp_after_fd, hp_after_truth)
        div_finite[eps] = div
        print(f"  finite-diff OHO eps={eps:.0e} rel divergence = {div:.3e}")

    worst_finite = max(div_finite.values())

    assert div_exact < 1e-4, f"exact RTRL OHO should match truth, got {div_exact:.3e}"
    assert worst_finite < 0.5, f"finite-diff OHO diverged wildly: {worst_finite:.3e}"
    assert worst_finite > div_exact, (
        f"finite-diff OHO ({worst_finite:.3e}) should be a worse approximation than exact RTRL ({div_exact:.3e})"
    )


# ============================================================================
# TEST 1f: Full inner x meta method matrix (localize the divergence)
# ============================================================================


@pytest.mark.slow
def test_meta_hypergradient_matrix():
    print("=" * 60)
    print("Meta hypergradient 2x2 matrix (rel divergence vs BPTT-over-BPTT)")
    print("=" * 60)

    bptt = BPTTConfig(truncate_at=None)
    rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )
    combos = {
        ("inner=BPTT", "meta=BPTT"): (bptt, bptt),
        ("inner=BPTT", "meta=RTRL"): (bptt, rtrl),
        ("inner=RTRL", "meta=BPTT"): (rtrl, bptt),
        ("inner=RTRL", "meta=RTRL"): (rtrl, rtrl),
    }

    results = {
        labels: run_two_level_meta(make_two_level_config(im, mm, META_INNER_STEPS))
        for labels, (im, mm) in combos.items()
    }

    _, ref_init, ref_after = results[("inner=BPTT", "meta=BPTT")]
    for (inner_lbl, meta_lbl), (norm, _, hp_after) in results.items():
        div = hypergradient_rel_divergence(ref_init, hp_after, ref_after)
        print(f"  {inner_lbl} / {meta_lbl}: norm={norm:.10f}  rel_div_vs_BPTT/BPTT={div:.3e}")

    for _, (norm, _, _) in results.items():
        assert norm > 1e-8, "degenerate run (zero hypergradient)"


# ============================================================================
# TEST 1g: RTRL-over-RTRL divergence vs inner trajectory length
# ============================================================================


@pytest.mark.slow
def test_rtrl_over_rtrl_divergence_vs_trajectory_length():
    print("=" * 60)
    print("RTRL-over-RTRL divergence vs META_INNER_STEPS (T)")
    print("=" * 60)

    bptt = BPTTConfig(truncate_at=None)
    rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )

    for T in [1, 2, 3, 4]:
        _, hp_init, hp_after_truth = run_two_level_meta(make_two_level_config(bptt, bptt, T))
        _, _, hp_after_rtrl = run_two_level_meta(make_two_level_config(rtrl, rtrl, T))
        div = hypergradient_rel_divergence(hp_init, hp_after_rtrl, hp_after_truth)
        print(f"  T={T}: rel_div(RTRL/RTRL vs BPTT/BPTT) = {div:.3e}")


# ============================================================================
# TEST 1h: stateful clip on each inner SGD step — outer RTRL tracks its EMA
# ============================================================================


@pytest.mark.slow
def test_inner_stateful_clip_tracked():
    print("=" * 60)
    print("Inner stateful SoftNormClip: inner=BPTT, meta=RTRL vs meta=BPTT")
    print("=" * 60)

    clip = SoftNormClip(
        bound=jnp.array(1.0),
        ema_decay=jnp.array(0.9),
        headroom=jnp.array(1.0),
        init_ema=jnp.array(0.1),
        eps_root=jnp.array(1e-8),
    )
    bptt = BPTTConfig(truncate_at=None)
    rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )

    norm_bptt, hp_init, hp_after_bptt = run_two_level_meta(
        make_two_level_config(bptt, bptt, META_INNER_STEPS, level0_clip=clip)
    )
    norm_rtrl, _, hp_after_rtrl = run_two_level_meta(
        make_two_level_config(bptt, rtrl, META_INNER_STEPS, level0_clip=clip)
    )
    _, _, hp_after_noclip = run_two_level_meta(make_two_level_config(bptt, bptt, META_INNER_STEPS))

    rel = hypergradient_rel_divergence(hp_init, hp_after_rtrl, hp_after_bptt)
    clip_effect = jnp.linalg.norm(hp_after_bptt - hp_after_noclip)
    print(f"  hypergrad norm BPTT={norm_bptt:.8f} RTRL={norm_rtrl:.8f}")
    print(f"  clip effect (vs no-clip): {clip_effect:.3e}")
    print(f"  meta RTRL vs BPTT rel:    {rel:.3e}")

    assert norm_bptt > 1e-8, "degenerate hypergradient"
    assert clip_effect > 1e-5, f"clip is inert — test is vacuous: {clip_effect:.3e}"
    assert rel < 1e-4, f"RTRL does not track the inner stateful clip's EMA (vs BPTT): {rel:.3e}"


# ============================================================================
# TEST 1i: outer validation clip's EMA must not enter the level-1 RTRL state
# ============================================================================


def test_outer_validation_clip_not_in_rtrl_state():
    print("=" * 60)
    print("Outer validation SoftNormClip: EMA must not leak into level-1 RTRL state")
    print("=" * 60)

    clip = SoftNormClip(
        bound=jnp.array(1.0),
        ema_decay=jnp.array(0.9),
        headroom=jnp.array(1.0),
        init_ema=jnp.array(0.1),
        eps_root=jnp.array(1e-8),
    )
    bptt = BPTTConfig(truncate_at=None)
    rtrl = RTRLConfig(
        start_at_step=0,
        damping=0.0,
        beta=1.0,
        use_finite_hvp=None,
        influence_clip=None,
        propagation_clip=None,
        lr_edge_margin=None,
        unit_circle_margin=None,
    )

    def state_dims(config):
        stuff = setup_env_and_fns(config)
        iface = stuff.interfaces[(OPTIMIZER_LEARNER, 1)]
        return iface.forward_mode_jacobian.get(stuff.env).shape, iface.state.get(stuff.env).shape

    base = state_dims(make_two_level_config(bptt, rtrl, META_INNER_STEPS))
    clipped = state_dims(make_two_level_config(bptt, rtrl, META_INNER_STEPS, outer_val_clip=clip))
    print(f"  level-1 RTRL (influence, state) dims  no clip: {base}")
    print(f"  level-1 RTRL (influence, state) dims with clip: {clipped}")

    assert base == clipped, f"outer validation clip leaked its EMA into the level-1 RTRL state: {base} vs {clipped}"


# ============================================================================
# TEST 2: Reset - state is actually reset
# ============================================================================


def test_reset_state():
    print("=" * 60)
    print("TEST 2: Reset actually resets state")
    print("=" * 60)

    config = make_single_level_config(BPTTConfig(truncate_at=None))
    stuff = setup_env_and_fns(config)
    env = stuff.env
    val_interface = stuff.interfaces[(MODEL_LEARNER, 0)]

    val_resetters = env_validation_resetters(config, stuff.shapes, stuff.interfaces)
    resetter = val_resetters[0]

    state_before = val_interface.state.get(env)
    print(f"  State norm before reset:  {jnp.linalg.norm(state_before):.6f}")

    env_corrupted = val_interface.state.put(env, state_before + 999.0)
    print(f"  State norm after corrupt: {jnp.linalg.norm(val_interface.state.get(env_corrupted)):.6f}")

    env_reset = resetter(env_corrupted, jax.random.key(77))
    state_after = val_interface.state.get(env_reset)
    print(f"  State norm after reset:   {jnp.linalg.norm(state_after):.6f}")

    assert jnp.allclose(state_after, jnp.zeros_like(state_after), atol=1e-6), (
        f"State not zeroed after reset, norm={jnp.linalg.norm(state_after):.6f}"
    )


# ============================================================================
# TEST 3: Reset - parameters survive reset
# ============================================================================


def test_reset_preserves_params():
    print("=" * 60)
    print("TEST 3: Reset preserves parameters")
    print("=" * 60)

    config = make_single_level_config(BPTTConfig(truncate_at=None))
    stuff = setup_env_and_fns(config)
    env = stuff.env
    val_interface = stuff.interfaces[(MODEL_LEARNER, 0)]

    val_resetters = env_validation_resetters(config, stuff.shapes, stuff.interfaces)
    resetter = val_resetters[0]

    param_before = val_interface.param.get(env)
    print(f"  Param norm before reset: {jnp.linalg.norm(param_before):.6f}")

    env_reset = resetter(env, jax.random.key(77))
    param_after = val_interface.param.get(env_reset)
    print(f"  Param norm after reset:  {jnp.linalg.norm(param_after):.6f}")

    diff = jnp.max(jnp.abs(param_before - param_after))
    print(f"  Max diff: {diff:.2e}")
    assert jnp.allclose(param_before, param_after, atol=1e-8), f"Params changed after reset, max diff={diff:.2e}"


# ============================================================================
# TEST 4: Reset - influence tensor is zeroed
# ============================================================================


def test_reset_zeros_influence_tensor():
    print("=" * 60)
    print("TEST 4: Reset zeros influence tensor")
    print("=" * 60)

    config = make_single_level_config(
        RTRLConfig(
            start_at_step=0,
            damping=0.0,
            beta=1.0,
            use_finite_hvp=None,
            influence_clip=None,
            propagation_clip=None,
            lr_edge_margin=None,
            unit_circle_margin=None,
        )
    )
    stuff = setup_env_and_fns(config)
    env = stuff.env
    val_interface = stuff.interfaces[(MODEL_LEARNER, 0)]

    J = val_interface.forward_mode_jacobian.get(env)
    print(f"  Influence tensor shape: {J.shape}")
    print(f"  Influence tensor norm (init): {jnp.linalg.norm(J):.6f}")

    assert J.shape[1] > 0, f"Influence tensor has zero state dim — test is vacuous"

    env_corrupted = val_interface.forward_mode_jacobian.put(env, J + 100.0)
    print(
        f"  Influence tensor norm (corrupted): {jnp.linalg.norm(val_interface.forward_mode_jacobian.get(env_corrupted)):.6f}"
    )

    val_resetters = env_validation_resetters(config, stuff.shapes, stuff.interfaces)
    resetter = val_resetters[0]
    env_reset = resetter(env_corrupted, jax.random.key(77))
    J_after = val_interface.forward_mode_jacobian.get(env_reset)
    print(f"  Influence tensor norm (after reset): {jnp.linalg.norm(J_after):.6f}")

    assert jnp.linalg.norm(J_after) < 50.0, (
        f"Influence tensor not re-initialized after reset, norm={jnp.linalg.norm(J_after):.2f}"
    )


# ============================================================================
# TEST 5: Tick counter resets
# ============================================================================


def test_tick_reset():
    print("=" * 60)
    print("TEST 5: Tick counter resets properly")
    print("=" * 60)

    config = make_single_level_config(BPTTConfig(truncate_at=None))
    stuff = setup_env_and_fns(config)
    env = stuff.env
    val_interface = stuff.interfaces[(MODEL_LEARNER, 0)]

    tick_init = val_interface.tick.get(env)
    print(f"  Tick at init: {tick_init}")

    advance = make_tick_advancer(val_interface)
    env4 = advance(advance(advance(env)))
    tick_after_3 = val_interface.tick.get(env4)
    print(f"  Tick after 3 advances: {tick_after_3}")

    val_resetters = env_validation_resetters(config, stuff.shapes, stuff.interfaces)
    resetter = val_resetters[0]
    env_reset = resetter(env4, jax.random.key(77))
    tick_after_reset = val_interface.tick.get(env_reset)
    print(f"  Tick after reset: {tick_after_reset}")

    assert int(tick_after_reset) == 0, f"Tick not reset to 0, got {tick_after_reset}"


# ============================================================================
# TEST 6: Identity learner produces zero gradient
# ============================================================================


def test_identity_learner():
    print("=" * 60)
    print("TEST 6: Identity learner produces zero meta-gradient")
    print("=" * 60)

    config = make_single_level_config(IdentityLearnerConfig(bptt_config=BPTTConfig(None)))
    stuff = setup_env_and_fns(config)

    _, stats = stuff.meta_learner(stuff.env, stuff.data_sample)
    grad_norm = stats["level0/meta_gradient_norm"].data
    print(f"  Gradient norm: {grad_norm:.2e}")

    assert grad_norm == 0.0, f"Identity learner gradient not zero, norm={grad_norm:.2e}"


# ============================================================================
# TEST 7: Tick modular arithmetic for reset
# ============================================================================


def test_reset_checker_fires_correctly():
    print("=" * 60)
    print("TEST 7: Reset fires at correct tick values")
    print("=" * 60)

    config = make_single_level_config(BPTTConfig(truncate_at=None))
    stuff = setup_env_and_fns(config)
    env = stuff.env
    val_interface = stuff.interfaces[(MODEL_LEARNER, 0)]

    advance = make_tick_advancer(val_interface)
    reset_t = 5

    tick_vals = []
    would_reset = []
    env_cur = env
    for i in range(12):
        t = val_interface.tick.get(env_cur).squeeze()
        tick_vals.append(int(t))
        would_reset.append(bool(int(t) % reset_t == 0))
        env_cur = advance(env_cur)

    print(f"  Ticks:       {tick_vals}")
    print(f"  Would reset: {would_reset}")

    assert tick_vals == list(range(12)), f"Ticks don't advance correctly: {tick_vals}"
    expected_resets = [i % reset_t == 0 for i in range(12)]
    assert would_reset == expected_resets, f"Reset condition mismatch: {would_reset} vs {expected_resets}"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    test_rtrl_vs_bptt_level0()
    test_validation_gradient_rtrl_vs_bptt()
    test_meta_hypergradient_rtrl_vs_bptt()
    test_meta_hypergradient_finite_difference_rtrl()
    test_finite_difference_oho_divergence()
    test_meta_hypergradient_matrix()
    test_rtrl_over_rtrl_divergence_vs_trajectory_length()
    test_inner_stateful_clip_tracked()
    test_outer_validation_clip_not_in_rtrl_state()
    test_reset_state()
    test_reset_preserves_params()
    test_reset_zeros_influence_tensor()
    test_tick_reset()
    test_identity_learner()
    test_reset_checker_fires_correctly()

    print("\nAll tests passed.")
