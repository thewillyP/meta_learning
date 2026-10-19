from dataclasses import dataclass
from functools import reduce
from typing import Callable

from ..category.lens import Lens, wiring
from ..category.lib_types import Proxy
from ..category.optics import Focus, first, second


@dataclass(frozen=True)
class Ref:
    name: str


type Refs = tuple[Ref, ...]
type Values[A] = tuple[A, ...]
type Parts[A] = tuple[Values[A], Values[A]]
type Partition[A, DA] = Lens[Values[A], Values[DA], Parts[A], Parts[DA]]
type Assembly[A, DA] = Lens[Parts[A], Parts[DA], Values[A], Values[DA]]


@dataclass(frozen=True)
class Layout:
    refs: Refs


@dataclass(frozen=True)
class Requested:
    source: Layout
    refs: Refs


@dataclass(frozen=True)
class Selection(Requested): ...


def layout(occurrences: Refs) -> Layout:
    return Layout(tuple(dict.fromkeys(occurrences)))


def request(source: Layout, refs: Refs) -> Requested:
    if missing := tuple(ref for ref in refs if ref not in source.refs):
        raise ValueError(f"bindings are not exposed in this scope: {missing}")
    return Requested(source, refs)


def select(requested: Requested) -> Selection:
    if len(set(requested.refs)) != len(requested.refs):
        raise ValueError("a selection cannot contain the same binding twice")
    return Selection(requested.source, requested.refs)


def remaining(selected: Selection) -> Layout:
    return Layout(tuple(ref for ref in selected.source.refs if ref not in selected.refs))


def route[A, DA](
    requested: Requested,
    zero: Callable[[A], DA],
    add: Callable[[DA, DA], DA],
) -> Lens[Values[A], Values[DA], Values[A], Values[DA]]:
    source, refs = requested.source, requested.refs
    positions = {ref: index for index, ref in enumerate(source.refs)}
    indices = tuple(positions[ref] for ref in refs)
    uses = tuple(tuple(i for i, ref in enumerate(refs) if ref == binding) for binding in source.refs)

    def run(values: Values[A]):
        def backward(feedback: Values[DA]) -> Values[DA]:
            return tuple(
                reduce(add, (feedback[i] for i in group), zero(values[position])) for position, group in enumerate(uses)
            )

        return tuple(values[i] for i in indices), backward

    return Lens(run)


def split_values[A](values: Values[A], size: int) -> Parts[A]:
    return values[:size], values[size:]


def join_values[A](parts: Parts[A]) -> Values[A]:
    left, right = parts
    return left + right


def split[A, DA](size: int) -> Partition[A, DA]:
    return wiring(lambda values: split_values(values, size), join_values)


def merge[A, DA](size: int) -> Assembly[A, DA]:
    return wiring(join_values, lambda feedback: split_values(feedback, size))


def paired[A, DA](
    source: Layout,
    left: Refs,
    right: Refs,
    zero: Callable[[A], DA],
    add: Callable[[DA, DA], DA],
) -> Partition[A, DA]:
    return route(request(source, left + right), zero, add) >> split(len(left))


def parse_selection[A, DA](requested: Selection) -> tuple[Partition[A, DA], Assembly[A, DA]]:
    source, refs = requested.source, requested.refs
    rest = remaining(requested).refs
    ordered = refs + rest
    indices = tuple(source.refs.index(ref) for ref in ordered)
    inverse = tuple(ordered.index(ref) for ref in source.refs)
    size = len(refs)

    def separate[T](values: Values[T]) -> Parts[T]:
        values = tuple(values[i] for i in indices)
        return split_values(values, size)

    def join[T](parts: Parts[T]) -> Values[T]:
        values = join_values(parts)
        return tuple(values[i] for i in inverse)

    return wiring(separate, join), wiring(join, separate)


def selection[A](parts: Partition[A, A]) -> Focus[Values[A], Values[A]]:
    return parts >> first(Proxy[Parts[A]]())


def complement[A](parts: Partition[A, A]) -> Focus[Values[A], Values[A]]:
    return parts >> second(Proxy[Parts[A]]())
