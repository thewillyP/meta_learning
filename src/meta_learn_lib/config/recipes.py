from dataclasses import dataclass
from functools import singledispatch
from jaxtyping import PyTree

import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Constant:
    value: float


@dataclass(frozen=True)
class Normal:
    scale: float


@dataclass(frozen=True)
class Array:
    shape: tuple[int, ...]
    dtype: str
    init: Constant | Normal


@dataclass(frozen=True)
class Product:
    parts: tuple["Recipe", ...]


@dataclass(frozen=True)
class Optimized[O]:
    optimizer: O
    parameters: "Recipe"


@dataclass(frozen=True)
class Copies:
    body: "Recipe"
    size: int


type Recipe = Array | Product | Optimized | Copies
type ArrayTree = PyTree[jax.Array]
type Spec = PyTree[jax.ShapeDtypeStruct]


@singledispatch
def optimizer_initial(op: object, parameters: ArrayTree) -> ArrayTree:
    raise TypeError(f"no optimizer initialization for {type(op).__name__}")


def allocate(recipe: Recipe, key: jax.Array) -> ArrayTree:
    match recipe:
        case Array(shape, dtype, Constant(value)):
            return jnp.full(shape, value, dtype=dtype)
        case Array(shape, dtype, Normal(scale)):
            return scale * jax.random.normal(key, shape, dtype=dtype)
        case Product(parts):
            return tuple(
                allocate(part, subkey) for part, subkey in zip(parts, jax.random.split(key, len(parts)), strict=True)
            )
        case Optimized(op, parameters):
            return optimizer_initial(op, allocate(parameters, key))
        case Copies(body, size):
            return eqx.filter_vmap(lambda k: allocate(body, k))(jax.random.split(key, size))
    raise TypeError(f"unknown recipe: {recipe!r}")


def specification(recipe: Recipe) -> Spec:
    match recipe:
        case Array(shape, dtype):
            return jax.ShapeDtypeStruct(shape, dtype)
        case Product(parts):
            return tuple(map(specification, parts))
        case Optimized(op, parameters):
            return eqx.filter_eval_shape(lambda p: optimizer_initial(op, p), specification(parameters))
        case Copies(body, size):
            return stack(specification(body), size)
    raise TypeError(f"unknown recipe: {recipe!r}")


def stack(spec: Spec, size: int) -> Spec:
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct((size,) + x.shape, x.dtype), spec)


def unstack(spec: Spec, size: int) -> Spec:
    def remove(x):
        if not x.shape or x.shape[0] != size:
            raise ValueError(f"expected leading axis {size}, received {x.shape}")
        return jax.ShapeDtypeStruct(x.shape[1:], x.dtype)

    return jax.tree.map(remove, spec)


def require(actual: Spec, expected: Spec) -> None:
    if not eqx.tree_equal(actual, expected):
        raise ValueError(f"incompatible interfaces: {actual!r} and {expected!r}")


def reconcile(left: Recipe | None, right: Recipe | None) -> Recipe | None:
    if left is None:
        return right
    if right is None:
        return left
    if left != right:
        raise ValueError(f"incompatible requirements for a shared binding: {left!r} and {right!r}")
    return left
