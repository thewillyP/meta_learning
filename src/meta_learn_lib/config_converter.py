from typing import Union
from cattrs import Converter
import jax
import jax.numpy as jnp

from meta_learn_lib.config import *
from meta_learn_lib.util import setup_flattened_union


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

    # -- jax.Array: promote to runtime placeholder so jit can pass values without baking --
    converter.register_unstructure_hook(jax.Array, lambda a: float(a))
    converter.register_structure_hook(jax.Array, lambda val, _: jnp.asarray(val))

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
            TikhonovRTRLConfig,
            PadeRTRLConfig,
            MidpointRTRLConfig,
            HeunRTRLConfig,
            ImplicitEulerRTRLConfig,
            BPTTConfig,
            IdentityLearnerConfig,
            RFLOConfig,
            UOROConfig,
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
        Union[MNISTTaskFamily, CIFAR10TaskFamily, CIFAR100TaskFamily, DelayAddTaskFamily],
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
