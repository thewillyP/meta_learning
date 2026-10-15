from collections.abc import Hashable, Iterable
from typing import Protocol, Self

import jax
from pyrsistent import PMap, plist, pmap, pvector
from pyrsistent._pvector import PythonPVector, python_pvector


class _OrderedKey(Hashable, Protocol):
    def __lt__(self, other: Self, /) -> bool: ...


def _sequence_flatten[A](values: Iterable[A]) -> tuple[tuple[tuple[Hashable, A], ...], None]:
    return tuple((jax.tree_util.SequenceKey(i), value) for i, value in enumerate(values)), None


def _map_flatten[K: _OrderedKey, V](values: PMap[K, V]) -> tuple[tuple[tuple[Hashable, V], ...], tuple[K, ...]]:
    keys = tuple(sorted(values))
    return tuple((jax.tree_util.DictKey(key), values[key]) for key in keys), keys


def _map_unflatten[K: Hashable, V](keys: tuple[K, ...], values: Iterable[V]) -> PMap[K, V]:
    return pmap(zip(keys, values, strict=True))


jax.tree_util.register_pytree_with_keys(type(plist()), _sequence_flatten, lambda _, values: plist(values))
jax.tree_util.register_pytree_with_keys(type(plist((None,))), _sequence_flatten, lambda _, values: plist(values))
jax.tree_util.register_pytree_with_keys(type(pmap()), _map_flatten, _map_unflatten)
jax.tree_util.register_pytree_with_keys(PythonPVector, _sequence_flatten, lambda _, values: python_pvector(values))

if type(pvector()) is not PythonPVector:
    jax.tree_util.register_pytree_with_keys(type(pvector()), _sequence_flatten, lambda _, values: pvector(values))
