from meta_learn_lib.category.lens import Lens, identity
from meta_learn_lib.category.lib_types import Proxy
from typing import Callable
import jax
import jax.numpy as jnp

# ── a record: what is on a port, by name ──────────────────────────────────────

type Record[V] = dict[str, V]


def select[V](keys: frozenset[str], r: Record[V]) -> Record[V]:
    return {k: r[k] for k in sorted(keys)}


# ── lenses on records ─────────────────────────────────────────────────────────


def split[V1, V2](
    a: frozenset[str],
    b: frozenset[str],
) -> Lens[Record[V1], Record[V2], tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]]]:
    def back(d: tuple[Record[V2], Record[V2]]) -> Record[V2]:
        da, db = d
        both: Record[V2] = {k: jax.tree.map(jnp.add, da[k], db[k]) for k in sorted(a & b)}
        return select(a - b, da) | select(b - a, db) | both

    return Lens(lambda r: ((select(a, r), select(b, r)), back))


def merge[V1, V2](
    a: frozenset[str],
    b: frozenset[str],
) -> Lens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], Record[V1], Record[V2]]:
    def run(
        ab: tuple[Record[V1], Record[V1]],
    ) -> tuple[Record[V1], Callable[[Record[V2]], tuple[Record[V2], Record[V2]]]]:
        ra, rb = ab
        return select(a, ra) | select(b, rb), lambda d: (select(a, d), select(b, d))

    return Lens(run)


def to_entry[V1, V2](name: str) -> Lens[Record[V1], Record[V2], V1, V2]:
    return Lens(lambda r: (r[name], lambda d: {name: d}))


def to_value[V1, V2](
    name: str, rest: frozenset[str]
) -> Lens[Record[V1], Record[V2], tuple[V1, Record[V1]], tuple[V2, Record[V2]]]:
    return split(frozenset({name}), rest) >> (to_entry(name) @ identity(Proxy[tuple[Record[V1], Record[V2]]]()))


def to_pair[V1, V2](a: str, b: str) -> Lens[Record[V1], Record[V2], tuple[V1, V1], tuple[V2, V2]]:
    return to_value(a, frozenset({b})) >> (identity(Proxy[tuple[V1, V2]]()) @ to_entry(b))


def to_triple[V1, V2](
    a: str, b: str, c: str
) -> Lens[Record[V1], Record[V2], tuple[V1, tuple[V1, V1]], tuple[V2, tuple[V2, V2]]]:
    return to_value(a, frozenset({b, c})) >> (identity(Proxy[tuple[V1, V2]]()) @ to_pair(b, c))
