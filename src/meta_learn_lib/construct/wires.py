from meta_learn_lib.category.lens import Lens
from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.record import plus, zero_like
from meta_learn_lib.construct.config import Shape

from typing import Callable
import jax
from jaxtyping import PyTree


def names(shape: Shape) -> tuple[str, ...]:
    match shape:
        case str():
            return (shape,)
        case ():
            return ()
        case (a, b):
            return names(a) + names(b)


def bound(shape: Shape, value: PyTree) -> dict[str, PyTree]:
    match shape:
        case str():
            return {shape: value}
        case ():
            return {}
        case (a, b):
            x, y = value
            left, right = bound(a, x), bound(b, y)
            return left | right | {k: plus(left[k], right[k]) for k in left.keys() & right.keys()}


def built(shape: Shape, parts: dict[str, PyTree], absent: Callable[[str], PyTree]) -> PyTree:
    match shape:
        case str():
            return parts[shape] if shape in parts else absent(shape)
        case ():
            return Unit()
        case (a, b):
            return (built(a, parts, absent), built(b, parts, absent))


def rewired(before: Shape, after: Shape) -> Lens[PyTree, PyTree, PyTree, PyTree]:
    missing = tuple(name for name in names(after) if name not in names(before))
    if missing:
        raise ValueError(f"there is no {missing} in {names(before)}")
    if len(set(names(before))) < len(names(before)):
        raise ValueError(f"a name is used twice in {names(before)}")

    def run(value: PyTree) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
        there = bound(before, value)
        return built(after, there, there.__getitem__), lambda d: built(
            before, bound(after, d), lambda name: jax.tree.map(zero_like, there[name])
        )

    return Lens(run)
