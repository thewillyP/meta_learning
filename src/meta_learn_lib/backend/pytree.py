from collections.abc import Callable, Hashable, Iterable
from typing import Protocol, Self, dataclass_transform, overload
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


class Focus[S, A]:
    def __init__(self, get: Callable[[S], A], put: Callable[[S, A], S]):
        self.get = get
        self.put = put

    def set(self, s: S, a: A) -> S:
        return self.put(s, a)

    def over(self, s: S, f: Callable[[A], A]) -> S:
        return self.put(s, f(self.get(s)))

    def __rshift__[B](self, other: "Focus[A, B]") -> "Focus[S, B]":
        return Focus(lambda s: other.get(self.get(s)), lambda s, b: self.put(s, other.put(self.get(s), b)))

    def __getitem__[K, V](self: "Focus[S, PMap[K, V]]", key: K) -> "Focus[S, V]":
        return self >> Focus(lambda m: m[key], lambda m, v: m.set(key, v))


class Field[A]:
    def __init__(self, default: A):
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__[S: Record](self, obj: None, owner: type[S]) -> Focus[S, A]: ...
    @overload
    def __get__(self, obj: object, owner: type) -> A: ...
    def __get__(self, obj: object | None, owner: type) -> object:
        if obj is None:
            return Focus(lambda s: s.__dict__[self.name], lambda s, a: s.__replace__(**{self.name: a}))
        return obj.__dict__[self.name]

    def __set__(self, obj: object, value: A) -> None:
        raise AttributeError(self.name)


@dataclass_transform(kw_only_default=True, frozen_default=True, field_specifiers=(Field,))
class Record:
    def __init__(self, **kw: object):
        unknown = kw.keys() - type(self).__annotations__.keys()
        if unknown:
            raise AttributeError(f"{sorted(unknown)} are not among the specified fields for {type(self).__name__}")
        for name in type(self).__annotations__:
            self.__dict__[name] = kw[name] if name in kw else vars(type(self))[name].default

    def __replace__(self, **kw: object) -> Self:
        unknown = kw.keys() - self.__dict__.keys()
        if unknown:
            raise AttributeError(f"{sorted(unknown)} are not among the specified fields for {type(self).__name__}")
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__.update(kw)
        return new

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"


def register_record[C: Record](cls: type[C]) -> type[C]:
    names = tuple(sorted(cls.__annotations__))

    def flatten(obj: C) -> tuple[tuple[tuple[Hashable, object], ...], None]:
        return tuple((jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in names), None

    jax.tree_util.register_pytree_with_keys(
        cls, flatten, lambda _, values: cls(**dict(zip(names, values, strict=True)))
    )
    return cls
