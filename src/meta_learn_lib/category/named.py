from meta_learn_lib.category.lens import Lens, Proxy, drop, identity, unit_elim, unit_intro
from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category import paralens
from meta_learn_lib.category.paralens import ParaLens, reparam
from meta_learn_lib.category.record import Record, merge, split, to_value

from dataclasses import dataclass
from typing import Callable


# ── a parametric lens whose port is a record ──────────────────────────────────


@dataclass(frozen=True)
class Named[V1, V2, X1, X2, Y1, Y2]:
    keys: frozenset[str]
    lens: ParaLens[Record[V1], Record[V2], X1, X2, Y1, Y2]

    def __rshift__[A1, A2, B1, B2, C1, C2](
        f: "Named[V1, V2, A1, A2, B1, B2]",
        g: "Named[V1, V2, B1, B2, C1, C2]",
    ) -> "Named[V1, V2, A1, A2, C1, C2]":
        return together(f.keys, g.keys, f.lens >> g.lens)

    def __matmul__[A1, A2, B1, B2, C1, C2, D1, D2](
        f: "Named[V1, V2, A1, A2, B1, B2]",
        g: "Named[V1, V2, C1, C2, D1, D2]",
    ) -> "Named[V1, V2, tuple[A1, C1], tuple[A2, C2], tuple[B1, D1], tuple[B2, D2]]":
        return together(f.keys, g.keys, f.lens @ g.lens)


# ── between one record and a pair of records ──────────────────────────────────


def together[V1, V2, A1, A2, B1, B2](
    a: frozenset[str],
    b: frozenset[str],
    f: ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], A1, A2, B1, B2],
) -> Named[V1, V2, A1, A2, B1, B2]:
    return Named(a | b, reparam(split(a, b), f))


def from_unmerged[V1, V2, A1, A2, B1, B2](
    names: frozenset[str],
    f: Named[V1, V2, A1, A2, B1, B2],
) -> ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], A1, A2, B1, B2]:
    return reparam(merge(names, f.keys - names), f.lens)


def known(names: tuple[str, ...], keys: frozenset[str]) -> frozenset[str]:
    if not frozenset(names) <= keys:
        raise ValueError(f"nothing called {sorted(frozenset(names) - keys)} here; the names are {sorted(keys)}")
    return frozenset(names)


# ── the operations of paralens.py: the port is untouched ──────────────────────


def join[V1, V2, A1, A2, B1, B2](
    name: str,
    f: Named[V1, V2, tuple[V1, A1], tuple[V2, A2], B1, B2],
) -> Named[V1, V2, A1, A2, B1, B2]:
    return Named(f.keys | {name}, reparam(to_value(name, f.keys), paralens.join(f.lens)))


def first[V1, V2, A1, A2, B1, B2, C1, C2](
    f: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, tuple[A1, C1], tuple[A2, C2], tuple[B1, C1], tuple[B2, C2]]:
    return Named(f.keys, paralens.first(f.lens))


def second[V1, V2, A1, A2, B1, B2, C1, C2](
    f: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, tuple[C1, A1], tuple[C2, A2], tuple[C1, B1], tuple[C2, B2]]:
    return Named(f.keys, paralens.second(f.lens))


def pre[V1, V2, A1, A2, B1, B2, Z1, Z2](
    l: Lens[Z1, Z2, A1, A2],
    f: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, Z1, Z2, B1, B2]:
    return Named(f.keys, paralens.pre(l, f.lens))


def post[V1, V2, A1, A2, B1, B2, Z1, Z2](
    l: Lens[B1, B2, Z1, Z2],
    f: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, A1, A2, Z1, Z2]:
    return Named(f.keys, paralens.post(l, f.lens))


def fresh[V, Y1, Y2](
    name: str,
    f: Named[V, V, V, V, tuple[V, Y1], tuple[V, Y2]],
) -> Named[V, V, Unit, Unit, tuple[Unit, Y1], tuple[Unit, Y2]]:
    gives = drop(Proxy[V]()) @ identity(Proxy[tuple[Y1, Y2]]())
    return post(gives, join(name, pre(unit_elim(Proxy[tuple[V, V]]()), f)))


# ── the operations that sort the port: your function, between a merge and a split ──


def copies[V1, V2, A1, A2, B1, B2, C1, C2, D1, D2](
    wrap: Callable[
        [ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], A1, A2, B1, B2]],
        ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], C1, C2, D1, D2],
    ],
    over: tuple[str, ...],
    cell: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, C1, C2, D1, D2]:
    each = known(over, cell.keys)
    shared = cell.keys - each
    return together(shared, each, wrap(from_unmerged(shared, cell)))


def learned[V1, V2, A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2](
    learn: Callable[
        [
            ParaLens[Record[V1], Record[V2], A1, A2, B1, B2],
            ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], C1, C2, D1, D2],
        ],
        ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], E1, E2, F1, F2],
    ],
    opt: Named[V1, V2, A1, A2, B1, B2],
    names: tuple[str, ...],
    body: Named[V1, V2, C1, C2, D1, D2],
) -> Named[V1, V2, E1, E2, F1, F2]:
    theta = known(names, body.keys)
    return together(opt.keys, body.keys - theta, learn(opt.lens, from_unmerged(theta, body)))


def online[V1, V2, A1, A2, B1, B2, C1, C2, D1, D2](
    wrap: Callable[
        [ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], A1, A2, B1, B2]],
        ParaLens[
            tuple[Record[V1], tuple[Unit, Record[V1]]], tuple[Record[V2], tuple[Unit, Record[V2]]], C1, C2, D1, D2
        ],
    ],
    names: tuple[str, ...],
    body: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, C1, C2, D1, D2]:
    theta = known(names, body.keys)
    rest = body.keys - theta
    sorted_port = split(theta, rest) >> (
        identity(Proxy[tuple[Record[V1], Record[V2]]]()) @ unit_intro(Proxy[tuple[Record[V1], Record[V2]]]())
    )
    return Named(body.keys, reparam(sorted_port, wrap(from_unmerged(theta, body))))


def online_with[V1, V2, A1, A2, B1, B2, C1, C2, D1, D2](
    wrap: Callable[
        [ParaLens[tuple[Record[V1], Record[V1]], tuple[Record[V2], Record[V2]], A1, A2, B1, B2]],
        ParaLens[tuple[Record[V1], tuple[V1, Record[V1]]], tuple[Record[V2], tuple[V2, Record[V2]]], C1, C2, D1, D2],
    ],
    name: str,
    names: tuple[str, ...],
    body: Named[V1, V2, A1, A2, B1, B2],
) -> Named[V1, V2, C1, C2, D1, D2]:
    theta = known(names, body.keys)
    rest = body.keys - theta
    sorted_port = split(theta, rest | {name}) >> (
        identity(Proxy[tuple[Record[V1], Record[V2]]]()) @ to_value(name, rest)
    )
    return Named(body.keys | {name}, reparam(sorted_port, wrap(from_unmerged(theta, body))))
