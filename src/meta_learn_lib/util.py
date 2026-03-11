from typing import Callable, get_args
from cattrs.gen import make_dict_unstructure_fn
import jax
import jax.flatten_util
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
from pyrsistent import PClass, thaw
from pyrsistent._pmap import PMap as PMapClass
from pyrsistent._pvector import PythonPVector

from meta_learn_lib.config import HyperparameterConfig
from meta_learn_lib.lib_types import ACTIVATION_FN, PRNG


def deep_serialize(_, obj):
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
    elif isinstance(obj, list):
        return [deep_serialize(_, v) for v in obj]
    elif isinstance(obj, tuple) and type(obj) is tuple:
        return tuple(deep_serialize(_, v) for v in obj)
    else:
        return obj


def setup_flattened_union(converter, union_type):
    union_members = get_args(union_type)

    # --- Unstructure: pad missing fields with None, add _type tag ---
    def factory(cls, converter):
        all_fields = set()
        for member_type in union_members:
            if hasattr(member_type, "__attrs_attrs__"):
                all_fields.update(field.name for field in member_type.__attrs_attrs__)
            elif hasattr(member_type, "__dataclass_fields__"):
                all_fields.update(member_type.__dataclass_fields__.keys())

        base_fn = make_dict_unstructure_fn(cls, converter)

        def flatten_unstructure(obj):
            result = base_fn(obj)
            result["_type"] = cls.__name__
            for field_name in all_fields:
                if field_name not in result:
                    result[field_name] = None
            return result

        return flatten_unstructure

    converter.register_unstructure_hook_factory(lambda cls: cls in union_members, factory)

    # --- Structure: dispatch on _type tag, only pass own fields ---
    name_to_cls = {}
    for cls in union_members:
        name_to_cls[cls.__name__] = cls
        name_to_cls[cls.__qualname__] = cls

    def structure_by_type_tag(val, _tp):
        if val is None:
            return None
        if not isinstance(val, dict):
            raise ValueError(f"Expected dict for union, got {type(val)}: {val!r}")
        tag = val.get("_type")
        if tag and tag in name_to_cls:
            cls = name_to_cls[tag]
            if hasattr(cls, "__dataclass_fields__"):
                own_fields = set(cls.__dataclass_fields__.keys())
            elif hasattr(cls, "__attrs_attrs__"):
                own_fields = {a.name for a in cls.__attrs_attrs__}
            else:
                own_fields = set()
            cleaned = {k: v for k, v in val.items() if k in own_fields}
            return converter.structure(cleaned, cls)
        raise ValueError(f"Unknown or missing _type tag in {val!r}")

    converter.register_structure_hook(union_type, structure_by_type_tag)


def jvp(f, primal, tangent):
    return jax.jvp(f, (primal,), (tangent,), has_aux=True)


def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return eqx.filter_vmap(wrapper, in_axes=(None, 1), out_axes=(None, 1, None))(primal, matrix)


def identity_fn(x: jax.Array) -> jax.Array:
    return x


def get_activation_fn(s: ACTIVATION_FN) -> Callable[[jax.Array], jax.Array]:
    match s:
        case "tanh":
            return jax.tree_util.Partial(jax.nn.tanh)
        case "relu":
            return jax.tree_util.Partial(jax.nn.relu)
        case "sigmoid":
            return jax.tree_util.Partial(jax.nn.sigmoid)
        case "identity":
            return jax.tree_util.Partial(identity_fn)
        case "softmax":
            return jax.tree_util.Partial(jax.nn.softmax)
        case _:
            raise ValueError("Invalid activation function")


def accuracy(preds: jax.Array, labels: jax.Array) -> jax.Array:
    """preds are logits, labels are one-hot or soft distributions."""
    return jnp.argmax(preds, axis=-1) == jnp.argmax(labels, axis=-1)


class Vector[T](eqx.Module):
    vector: jax.Array
    to_param: Callable[[jax.Array], T] = eqx.field(static=True)


def to_vector[T](tree: T) -> Vector[T]:
    """Convert a pytree to a Vector, which contains a flattened array and non-parameter parts."""
    params, nonparams = eqx.partition(tree, eqx.is_inexact_array)
    vector, to_param = jax.flatten_util.ravel_pytree(params)
    return Vector(vector=vector, to_param=lambda a: eqx.combine(to_param(a), nonparams))


def softminus(x: jax.Array) -> jax.Array:
    return -jax.nn.softplus(-x)


def softclip(x: jax.Array, a: float | None, b: float | None, sharpness: float) -> jax.Array:
    """Soft clipping with parameterized corner sharpness. Set either endpoint to None to disable."""
    c = sharpness / ((b - a) / 2) if a is not None and b is not None else sharpness
    v = x
    if a is not None:
        v = v - softminus(c * (x - a)) / c
    if b is not None:
        v = v - jax.nn.softplus(c * (x - b)) / c
    return v


def hyperparameter_reparametrization(
    reparametrization: HyperparameterConfig.Parametrization,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    match reparametrization:
        case HyperparameterConfig.identity():
            reparam_fn = lambda lr: lr
            reparam_inverse = lambda lr: lr
        case HyperparameterConfig.softplus():
            reparam_fn = jax.nn.softplus
            reparam_inverse = lambda y: jnp.log(jnp.expm1(y))
        case HyperparameterConfig.softrelu(clip):

            def softrelu(x, c):
                return x - softminus(c * x) / c

            reparam_fn = lambda x: softrelu(x, clip)
            reparam_inverse = lambda y: y  # Note: softrelu is not invertible

        case HyperparameterConfig.relu():
            reparam_fn = lambda x: jnp.maximum(x, 0.0)
            reparam_inverse = lambda y: y  # Note: relu is not invertible

        case HyperparameterConfig.silu_positive(scale):

            @jax.jit
            def lambertw_jax(x, max_iters=20):
                """JAX-compatible Lambert W approximation (principal branch W₀)."""
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
            reparam_fn = lambda x: softclip(x, a, b, clip)
            reparam_inverse = lambda y: y  # Note: softclip is not invertible

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
