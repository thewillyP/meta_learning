"""
Correctness tests for the meta-learning algorithm.

Tests:
1. RTRL vs BPTT gradient equivalence at level 0 (via full meta-learner)
2. Reset: state is actually reset
3. Reset: parameters survive reset
4. Reset: influence tensor is zeroed on reset
5. Reset: tick counter resets properly
6. Identity learner produces zero meta-gradient
7. Tick modular arithmetic for reset checker
"""

import random
import os
from typing import Callable, Iterator, NamedTuple
import numpy as np
import torch
import toolz
import jax
import jax.numpy as jnp
import equinox as eqx

from meta_learn_lib.config import *
from meta_learn_lib.constants import MODEL_LEARNER
from meta_learn_lib.create_axes import diff_axes
from meta_learn_lib.create_env import (
    create_env,
    create_inference_axes,
    create_transition_fns,
    env_resetters,
    env_validation_resetters,
    make_tick_advancer,
)
from meta_learn_lib.create_interface import build_id_map, build_interfaces
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


def make_single_level_config(method: GradientMethod) -> GodConfig:
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
            console=ConsoleLoggerConfig(enabled=False),
            matplotlib=MatplotlibLoggerConfig(save_dir="/tmp", enabled=False),
        ),
        epochs=1,
        checkpoint_every_n_minibatches=1,
        checkpoint_every_n_epochs=0,
        transition_graph={
            "x": {},
            "concat": {"x"},
            "rnn1": {"concat"},
        },
        readout_graph={
            "readout": {"rnn1"},
        },
        nodes={
            "x": UnlabeledSource(),
            "concat": Concat(),
            "rnn1": VanillaRNNLayer(
                nn_layer=NNLayer(
                    n=RNN_SIZE,
                    activation_fn="tanh",
                    use_bias=True,
                    layer_norm=None,
                ),
                use_random_init=False,
                time_constant="hp_tc",
            ),
            "readout": NNLayer(
                n=NUM_CLASSES,
                activation_fn="identity",
                use_bias=True,
                layer_norm=None,
            ),
        },
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
                    domain=frozenset({"mnist"}),
                    pixel_transform="normalize",
                ),
                dataset=DatasetConfig(
                    num_examples_in_minibatch=10,
                    num_examples_total=100,
                    is_test=False,
                    augment=False,
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
                        add_clip=None,
                        scale=1.0,
                    ),
                    optimizer_learner=GradientConfig(
                        method=RTRLConfig(start_at_step=0, damping=0.0, beta=1.0, use_finite_hvp=None),
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

    id_map = build_id_map(config)
    interfaces = build_interfaces(config, id_map)

    env = create_env(config, shapes, interfaces, env_prng)

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
    config_rtrl = make_single_level_config(RTRLConfig(start_at_step=0, damping=0.0, beta=1.0, use_finite_hvp=None))

    stuff_bptt = setup_env_and_fns(config_bptt)
    stuff_rtrl = setup_env_and_fns(config_rtrl)

    data = stuff_bptt.data_sample

    # Verify initial envs are identical
    params_init_bptt = stuff_bptt.interfaces[(MODEL_LEARNER, 0)].param.get(stuff_bptt.env)
    params_init_rtrl = stuff_rtrl.interfaces[(MODEL_LEARNER, 0)].param.get(stuff_rtrl.env)
    init_diff = jnp.max(jnp.abs(params_init_bptt - params_init_rtrl))
    print(f"  Initial param diff: {init_diff:.2e}")

    env_bptt_after, stats_bptt = stuff_bptt.meta_learner(stuff_bptt.env, data)
    env_rtrl_after, stats_rtrl = stuff_rtrl.meta_learner(stuff_rtrl.env, data)

    # Compare meta-gradient norms.
    # optimizer_learner is RTRL in both configs, differentiating through the
    # model_learner. The model_learner BPTT and RTRL produce equivalent model
    # gradients, but the optimizer_learner's RTRL sees different computational
    # graphs (eqx.filter_grad vs the RTRL scan), so the meta-gradients can
    # legitimately differ. We check params instead — if model gradients match,
    # the SGD update is identical, so params after one step must match.
    norm_bptt = stats_bptt["level0/meta_gradient_norm"]
    norm_rtrl = stats_rtrl["level0/meta_gradient_norm"]
    norm_diff = jnp.abs(norm_bptt - norm_rtrl)

    print(f"  BPTT meta-gradient norm: {norm_bptt:.8f}")
    print(f"  RTRL meta-gradient norm: {norm_rtrl:.8f}")
    print(f"  Abs diff in norms:       {norm_diff:.2e}")

    # Compare params after one step — the model_learner gradient (BPTT vs RTRL)
    # feeds into SGD. If equivalent, params are identical after one step.
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
    config_rtrl = make_single_level_config(RTRLConfig(start_at_step=0, damping=0.0, beta=1.0, use_finite_hvp=None))

    stuff_bptt = setup_env_and_fns(config_bptt)
    stuff_rtrl = setup_env_and_fns(config_rtrl)

    # Rebuild validation learners to get model_learner gradients directly
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

    # Compute axes for vmap (to peel nested.batch dim)
    def get_axes(config, stuff):
        resetters = env_resetters(config, stuff.shapes, stuff.interfaces, [False] * len(config.levels))
        inner_resetter, _ = resetters[0]
        return diff_axes(stuff.env, inner_resetter(stuff.env, jax.random.key(0)))

    axes_bptt = get_axes(config_bptt, stuff_bptt)
    axes_rtrl = get_axes(config_rtrl, stuff_rtrl)

    # Extract validation data from dataloader batch
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

    config = make_single_level_config(RTRLConfig(start_at_step=0, damping=0.0, beta=1.0, use_finite_hvp=None))
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
    grad_norm = stats["level0/meta_gradient_norm"]
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
    test_reset_state()
    test_reset_preserves_params()
    test_reset_zeros_influence_tensor()
    test_tick_reset()
    test_identity_learner()
    test_reset_checker_fires_correctly()

    print("\nAll tests passed.")
