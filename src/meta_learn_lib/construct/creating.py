from meta_learn_lib.category.lens import Lens
from meta_learn_lib.category.named import Reading
from meta_learn_lib.category.paralens import ParaLens
from meta_learn_lib.construct import config
from meta_learn_lib.construct.leaves import initializer

from dataclasses import dataclass
from typing import Callable, Iterator, overload
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from plum import dispatch

Loose = ParaLens[PyTree, PyTree, PyTree, PyTree, PyTree, PyTree]


# ── the reading: sizes go forwards, start values come back ────────────────────


@dataclass(frozen=True)
class Creating(Reading[PyTree, PyTree]): ...


# ── sizes: what the creating reading sends forwards ───────────────────────────


@dataclass(frozen=True)
class Unknown:
    def __iter__(self) -> "Iterator[Unknown]":
        yield self
        yield self

    def __getitem__(self, half: int) -> "Unknown":
        return self


@dataclass(frozen=True)
class Size:
    dims: tuple[int, ...]


# ── start values: what the creating reading sends back ────────────────────────


@dataclass(frozen=True)
class Unsaid:
    def __iter__(self) -> "Iterator[Unsaid]":
        yield self
        yield self

    def __getitem__(self, half: int) -> "Unsaid":
        return self


@dataclass(frozen=True)
class Starts:
    how: config.Init
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Value:
    value: float


@dataclass(frozen=True)
class SideBySide:
    left: Starts
    right: Starts


@dataclass(frozen=True)
class Each:
    n: int
    inside: "Said"


@dataclass(frozen=True)
class OptState:
    make: Callable[[PyTree], optax.GradientTransformation]
    rates: PyTree
    theta: PyTree


@dataclass(frozen=True)
class ZeroRows:
    state: PyTree
    theta: PyTree


@dataclass(frozen=True)
class ZerosFlat:
    of: PyTree


@dataclass(frozen=True)
class Key: ...


Said = Starts | Value | SideBySide | Each | OptState | ZeroRows | ZerosFlat | Key


# ── a tree of start values becomes a tree of arrays ───────────────────────────


def count(tree: PyTree) -> int:
    flat, _ = jax.flatten_util.ravel_pytree(eqx.filter(tree, eqx.is_inexact_array))
    return flat.size


@overload
def made(s: Starts, key: jax.Array) -> PyTree:
    return initializer(s.how)(key, s.shape)


@overload
def made(s: Value, key: jax.Array) -> PyTree:
    return jnp.asarray(s.value)


@overload
def made(s: SideBySide, key: jax.Array) -> PyTree:
    k1, k2 = jax.random.split(key)
    return jnp.hstack([made(s.left, k1), made(s.right, k2)])


@overload
def made(s: Each, key: jax.Array) -> PyTree:
    return jax.vmap(lambda k: made(s.inside, k))(jax.random.split(key, s.n))


@overload
def made(s: OptState, key: jax.Array) -> PyTree:
    return s.make(created(s.rates, key)).init(created(s.theta, key))


@overload
def made(s: ZeroRows, key: jax.Array) -> PyTree:
    return jnp.zeros((count(created(s.state, key)), count(created(s.theta, key))))


@overload
def made(s: ZerosFlat, key: jax.Array) -> PyTree:
    return jnp.zeros((count(created(s.of, key)),))


@overload
def made(s: Key, key: jax.Array) -> PyTree:
    return key


@overload
def made(s: Unsaid, key: jax.Array) -> PyTree:
    return s


@dispatch
def made(s: Said | Unsaid, key: jax.Array) -> PyTree:
    raise NotImplementedError


def created(starts: PyTree, key: jax.Array) -> PyTree:
    leaves, tree = jax.tree.flatten(starts)
    keys = jax.random.split(key, max(len(leaves), 1))
    return jax.tree.unflatten(tree, [made(leaf, k) for leaf, k in zip(leaves, keys)])


# ── leading axes: taken off on the way into copies, put on on the way out ─────


@overload
def without_axis(s: Size) -> Size | Unknown:
    return Size(s.dims[1:])


@overload
def without_axis(s: Unknown) -> Size | Unknown:
    return s


@dispatch
def without_axis(s: Size | Unknown) -> Size | Unknown:
    raise NotImplementedError


@overload
def with_axis(n: int, s: Size) -> Size | Unknown:
    return Size((n, *s.dims))


@overload
def with_axis(n: int, s: Unknown) -> Size | Unknown:
    return s


@dispatch
def with_axis(n: int, s: Size | Unknown) -> Size | Unknown:
    raise NotImplementedError


@overload
def axis(s: Size) -> tuple[int, ...]:
    return s.dims[:1]


@overload
def axis(s: Unknown) -> tuple[int, ...]:
    return ()


@dispatch
def axis(s: Size | Unknown) -> tuple[int, ...]:
    raise NotImplementedError


@overload
def in_each(n: int, s: Unsaid) -> Each | Unsaid:
    return s


@overload
def in_each(n: int, s: Starts | Value | SideBySide | Each | OptState | ZeroRows | ZerosFlat | Key) -> Each | Unsaid:
    return Each(n, s)


@dispatch
def in_each(n: int, s: Said | Unsaid) -> Each | Unsaid:
    raise NotImplementedError


@overload
def of_one(s: Each) -> PyTree:
    return s.inside


@overload
def of_one(s: Unsaid | Starts | Value | SideBySide | OptState | ZeroRows | ZerosFlat | Key) -> PyTree:
    return s


@dispatch
def of_one(s: Said | Unsaid) -> PyTree:
    raise NotImplementedError


def copies(n: tuple[int, ...], starts: PyTree) -> PyTree:
    """The start of one copy, as the start of n of them. n is () when the number of copies is not known."""
    for k in n:
        starts = jax.tree.map(lambda s: in_each(k, s), starts)
    return starts


def lengths(sizes: PyTree) -> tuple[int, ...]:
    """The leading axis of what is fed to copies in a row, when any of it says."""
    known = tuple(n for s in jax.tree.leaves(sizes) for n in axis(s))
    return known[:1]


def stacked(n: tuple[int, ...], sizes: PyTree) -> PyTree:
    for k in n:
        return jax.tree.map(lambda s: with_axis(k, s), sizes)
    return jax.tree.map(lambda s: Unknown(), sizes)


# ── what comes back in this reading: nothing said, and one place said twice ───

@overload
def zero_like(leaf: Size) -> Unsaid:
    return Unsaid()


@overload
def zero_like(leaf: Unknown) -> Unsaid:
    return Unsaid()


@dispatch
def zero_like(leaf: Size | Unknown) -> Unsaid:
    raise NotImplementedError


@overload
def plus(a: Unsaid, b: Unsaid) -> Unsaid | Said:
    return a


@overload
def plus(a: Unsaid, b: Said) -> Unsaid | Said:
    return b


@overload
def plus(a: Said, b: Unsaid) -> Unsaid | Said:
    return a


@overload
def plus(a: Said, b: Said) -> Unsaid | Said:
    return a


@dispatch
def plus(a: Unsaid | Said, b: Unsaid | Said) -> Unsaid | Said:
    raise NotImplementedError


def sized(keys: frozenset[str], known: dict[str, tuple[int, ...]]) -> dict[str, Size | Unknown]:
    return {k: Size(known[k]) if k in known else Unknown() for k in sorted(keys)}


# ── the six wrappers, read on sizes and start values ──────────────────────────


def map_accum(cell: Loose) -> Loose:
    """Copies in a row. The state one copy gives out starts the next, so one copy speaks for all."""

    def run(port_s: PyTree) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
        (shared, rows), s = port_s
        n = lengths(rows)
        (s1, y), back_cell = cell.arrow.run(((shared, jax.tree.map(without_axis, rows)), s))

        def rev(start: PyTree) -> PyTree:
            s1_start, _ = start
            (shared_start, row_start), s_start = back_cell((s1_start, Unsaid()))
            return (shared_start, copies(n, row_start)), s_start

        return (s1, stacked(n, y)), rev

    return ParaLens(Lens(run))


def batch(n: int, cell: Loose) -> Loose:
    """Copies side by side. Each copy has its own input, output and part of the port."""

    def run(port_a: PyTree) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
        (shared, own), a = port_a
        b, back_cell = cell.arrow.run(((shared, jax.tree.map(without_axis, own)), jax.tree.map(without_axis, a)))

        def rev(start: PyTree) -> PyTree:
            (shared_start, own_start), a_start = back_cell(jax.tree.map(of_one, start))
            return (shared_start, copies((n,), own_start)), copies((n,), a_start)

        return stacked((n,), b), rev

    return ParaLens(Lens(run))


def put(body: Loose) -> Loose:
    """What was on the port and is now on the wire starts as the port said."""

    def run(port_sigma_s: PyTree) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
        d, (sigma, s) = port_sigma_s
        (s1, y), back_body = body.arrow.run(((sigma, d), s))

        def rev(start: PyTree) -> PyTree:
            (_, s1_start), _ = start
            (sigma_start, d_start), s_start = back_body((s1_start, Unsaid()))
            return d_start, (sigma_start, s_start)

        return ((sigma, s1), y), rev

    return ParaLens(Lens(run))


def online(model: Loose, own_port: PyTree, own_state: Callable[[PyTree, PyTree], PyTree]) -> Loose:
    """The RTRL family. The model underneath starts as it says. The wrapper adds how its own
    port entry starts, and how its own state starts given the model's state and theta."""

    def run(port_sw: PyTree) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
        (theta, (_, d)), (s, _) = port_sw
        (s1, y), back_model = model.arrow.run(((theta, d), s))

        def rev(start: PyTree) -> PyTree:
            (s1_start, _), y_start = start
            (theta_start, d_start), s_start = back_model((s1_start, y_start))
            return (theta_start, (own_port, d_start)), (s_start, own_state(s_start, theta_start))

        return ((s1, Unknown()), y), rev

    return ParaLens(Lens(run))
