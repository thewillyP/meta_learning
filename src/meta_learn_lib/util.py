from typing import Callable, Union, get_args
from cattrs.strategies import configure_tagged_union
from cattrs.gen import make_dict_unstructure_fn
import jax
import jax.flatten_util
import jax.numpy as jnp
import math
import equinox as eqx
import equinox.internal as eqxi
import optax
import jax.lax as lax
from pyrsistent import PClass, thaw
from pyrsistent._pmap import PMap as PMapClass
from pyrsistent._pvector import PythonPVector

from meta_learn_lib.config import HyperparameterConfig
from meta_learn_lib.lib_types import ACTIVATION_FN, LOSS, PRNG


def deep_serialize(_, obj):
    """Recursively serialize pyrsistent objects to Python built-ins"""
    if isinstance(obj, PClass):
        serialized = obj.serialize()
        return {k: deep_serialize(_, v) for k, v in serialized.items()}
    elif isinstance(obj, PMapClass):
        thawed = thaw(obj)
        return {k: deep_serialize(_, v) for k, v in thawed.items()}
    elif isinstance(obj, PythonPVector):
        return [deep_serialize(_, v) for v in obj]
    elif isinstance(obj, dict):
        return {k: deep_serialize(_, v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_serialize(_, v) for v in obj)
    else:
        return obj


def setup_flattened_union(converter, union_type):
    union_members = get_args(union_type)

    def factory(cls, converter):
        all_fields = set()
        for member_type in union_members:
            if hasattr(member_type, "__attrs_attrs__"):  # attrs class
                all_fields.update(field.name for field in member_type.__attrs_attrs__)
            elif hasattr(member_type, "__dataclass_fields__"):  # dataclass
                all_fields.update(member_type.__dataclass_fields__.keys())

        base_fn = make_dict_unstructure_fn(cls, converter)

        def flatten_unstructure(obj):
            result = base_fn(obj)
            for field_name in all_fields:
                if field_name not in result:
                    result[field_name] = None
            return result

        return flatten_unstructure

    converter.register_unstructure_hook_factory(lambda cls: cls in union_members, factory)
    configure_tagged_union(union_type, converter)


def jvp(f, primal, tangent):
    return jax.jvp(f, (primal,), (tangent,), has_aux=True)


def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return eqx.filter_vmap(wrapper, in_axes=(None, 1), out_axes=(None, 1, None))(primal, matrix)


def reshape_timeseries(arr: jax.Array, target_time_dim: int) -> tuple[jax.Array, int]:
    num_examples, time_series_length, *rest = arr.shape

    num_virtual_minibatches = math.ceil(time_series_length / target_time_dim)
    pad_length = (-time_series_length) % num_virtual_minibatches
    new_time_dim = (time_series_length + pad_length) // num_virtual_minibatches

    if pad_length > 0:
        pad_width = [(0, 0), (0, pad_length)] + [(0, 0)] * len(rest)
        arr_padded = jnp.pad(arr, pad_width, constant_values=0)
    else:
        arr_padded = arr

    new_shape = (num_examples, num_virtual_minibatches, new_time_dim) + tuple(rest)

    # Length of non-padded portion in last minibatch
    last_minibatch_length = new_time_dim - pad_length

    return arr_padded.reshape(new_shape), last_minibatch_length


def get_activation_fn(s: ACTIVATION_FN) -> Callable[[jax.Array], jax.Array]:
    match s:
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


def get_loss_fn(s: str) -> Callable[[jax.Array, jax.Array], LOSS]:
    match s:
        case "cross_entropy":
            return lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))
        case "cross_entropy_with_integer_labels":
            return lambda a, b: LOSS(optax.losses.softmax_cross_entropy_with_integer_labels(a, b))
        case "mse":
            return lambda a, b: LOSS(optax.losses.squared_error(a, b))
        case _:
            raise ValueError("Invalid loss function")


def accuracy_hard(preds: jax.Array, labels: jax.Array) -> jax.Array:
    pred_classes = jnp.argmax(preds, axis=-1)
    return pred_classes == labels


def accuracy_soft(preds: jax.Array, labels: jax.Array) -> jax.Array:
    pred_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(labels, axis=-1)
    return pred_classes == true_classes


def accuracy_with_sequence_filter(
    preds: jnp.ndarray,  # [B, N, C]
    labels: jnp.ndarray,  # [B, N, 2]
    n: int,  # sequence number to filter
) -> jnp.ndarray:  # [B,]
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
        correct_masked: jax.Array = correct.astype(jnp.float32) * mask
        total = jnp.sum(mask)
        correct_total = jnp.sum(correct_masked)
        return jax.lax.cond(total > 0, lambda: correct_total / total, lambda: 0.0)

    # Vectorized over batch dimension: preds [B, N, C], labels [B, N, 2]
    batched_accuracy_with_sequence_filter = eqx.filter_vmap(
        _accuracy_seq_filter_single,
        in_axes=(0, 0, None),  # vmap over batch; n is shared
    )

    return batched_accuracy_with_sequence_filter(preds, labels, n)


class Vector[T](eqx.Module):
    vector: jax.Array
    to_param: Callable[[jax.Array], T] = eqx.field(static=True)


def to_vector[T](tree: T) -> Vector[T]:
    """Convert a pytree to a Vector, which contains a flattened array and non-parameter parts."""
    params, nonparams = eqx.partition(tree, eqx.is_inexact_array)
    vector, to_param = jax.flatten_util.ravel_pytree(params)
    return Vector(vector=vector, to_param=lambda a: eqx.combine(to_param(a), nonparams))


def hyperparameter_reparametrization(
    reparametrization: Union[
        HyperparameterConfig.identity,
        HyperparameterConfig.softplus,
        HyperparameterConfig.relu,
        HyperparameterConfig.softrelu,
    ],
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    def softminus(x):
        return -jax.nn.softplus(-x)

    match reparametrization:
        case HyperparameterConfig.identity():
            reparam_fn = lambda lr: lr
            reparam_inverse = lambda lr: lr
        case HyperparameterConfig.softplus():
            reparam_fn = jax.nn.softplus
            reparam_inverse = lambda y: jnp.log(jnp.expm1(y))
        case HyperparameterConfig.softrelu(clip):

            def softrelu(x, c):
                v = x - softminus(c * x) / c
                return v

            reparam_fn = lambda x: softrelu(x, clip)
            reparam_inverse = lambda y: y  # Note: not strictly correct, as softrelu is not invertible

        case HyperparameterConfig.relu():
            # def straight_through_relu(x):
            #     return x + lax.stop_gradient(jnp.maximum(x, 0.0) - x)

            reparam_fn = lambda x: jnp.maximum(x, 0.0)
            # reparam_fn = straight_through_relu
            reparam_inverse = lambda y: y  # Note: not strictly correct, as relu is not invertible
        case HyperparameterConfig.silu_positive(scale):

            @jax.jit
            def lambertw_jax(x, max_iters=20):
                """JAX-compatible Lambert W approximation (principal branch W₀)."""
                # Initial guess: good heuristic for principal branch
                w = jnp.log1p(x)
                for _ in range(max_iters):
                    ew = jnp.exp(w)
                    wew = w * ew
                    w -= (wew - x) / (ew * (w + 1) - (w + 2) * (wew - x) / (2 * w + 2))
                return w

            bias = jnp.real(lambertw_jax(1 / jnp.e))  # ≈ 0.2784645

            def silu_positive(x, s):
                return x * jax.nn.sigmoid(s * x) + bias

            def silu_positive_inverse(y, s):
                z = s * (y - bias)
                w = lambertw_jax(z * jnp.exp(-z))
                return (z + w) / s

            reparam_fn = lambda x: silu_positive(x, scale)
            reparam_inverse = lambda y: silu_positive_inverse(y, scale)

        case HyperparameterConfig.squared(scale):
            reparam_fn = lambda x: (scale * x) ** 2
            reparam_inverse = lambda y: jnp.sqrt(y) / scale

        case HyperparameterConfig.softclip(a, b, clip):

            def softclip(x):
                """
                Clipping with softplus and softminus, with paramterized corner sharpness.
                Set either (or both) endpoint to None to indicate no clipping at that end.
                """
                # when clipping at both ends, make c dimensionless w.r.t. b - a / 2
                c = clip
                if a is not None and b is not None:
                    c /= (b - a) / 2

                v = x
                if a is not None:
                    v = v - softminus(c * (x - a)) / c
                if b is not None:
                    v = v - jax.nn.softplus(c * (x - b)) / c
                return v

            reparam_fn = softclip
            reparam_inverse = lambda y: y  # Note: not strictly correct, as softclip is not invertible
            # reparam_inverse = lambda y: jnp.minimum(
            #     jnp.maximum(y, a if a is not None else -jnp.inf), b if b is not None else jnp.inf
            # )  # Note: not strictly correct, as softclip is not invertible

        case _:
            raise ValueError("Invalid hyperparameter reparametrization")

    return reparam_fn, reparam_inverse


def infinite_keys(key: PRNG):
    """Generate infinite stream of JAX keys."""
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def filter_cond(pred, true_fun, false_fun, *operands):
    """Taken from https://github.com/patrick-kidger/equinox/issues/709"""
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)
