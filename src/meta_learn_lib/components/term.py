from dataclasses import dataclass
from typing import Callable

from ..category.lens import Lens
from ..category.lib_types import Monoid
from ..category.writer import Writer
from ..lib_types import PRNG
from ..utility.distributions import SAMPLER
from .env import Env, Log, Par, St
from .optimizer import Optimizer


class Term[X, Y]:
    def __rshift__[A, B, C](f: "Term[A, B]", g: "Term[B, C]") -> "Term[A, C]":
        return Then(f, g)

    def __matmul__[A, B, C, D](f: "Term[A, B]", g: "Term[C, D]") -> "Term[tuple[A, C], tuple[B, D]]":
        return Tensor(f, g)

    def rename(self, transpositions: tuple[tuple[str, str], ...]) -> "Term[X, Y]":
        return Rename(self, transpositions)

    def reparam(
        self, lens: Lens[Log, Par, Par, Par, Par], monoid: Monoid[Par, Par], seed: Callable[[Env], Env]
    ) -> "Term[X, Y]":
        return Reparam(self, lens, monoid, seed)

    def scan(self, to_reset: frozenset[str], to_scan: frozenset[str], length: int | None) -> "Term[X, Y]":
        return Scan(self, to_reset, to_scan, length)

    def batch(self, to_map: frozenset[str], size: int | None) -> "Term[X, Y]":
        return Batch(self, to_map, size)

    def learn(self, opt: Optimizer) -> "Term[X, Y]":
        return Learn(opt, self)

    def rtrl(self, targets: frozenset[str], influence: str) -> "Term[X, Y]":
        return Rtrl(self, targets, influence)

    def uoro(self, targets: frozenset[str], distribution: SAMPLER, aux: str, prng: str) -> "Term[X, Y]":
        return Uoro(self, targets, distribution, aux, prng)

    def rflo(self, targets: frozenset[str], influence: str, decay: str) -> "Term[X, Y]":
        return Rflo(self, targets, influence, decay)


@dataclass(frozen=True)
class Cell[X, Y](Term[X, Y]):
    step: Callable[[Par, St, X], Writer[Log, tuple[St, Y]]]
    init: Callable[[PRNG], Env]


@dataclass(frozen=True)
class Lift[X, Y](Term[X, Y]):
    lens: Lens[Log, X, X, Y, Y]


@dataclass(frozen=True)
class Then[X, Y, Z](Term[X, Z]):
    f: Term[X, Y]
    g: Term[Y, Z]


@dataclass(frozen=True)
class Tensor[X, Y, U, V](Term[tuple[X, U], tuple[Y, V]]):
    f: Term[X, Y]
    g: Term[U, V]


@dataclass(frozen=True)
class Rename[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    transpositions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Reparam[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    lens: Lens[Log, Par, Par, Par, Par]
    monoid: Monoid[Par, Par]
    seed: Callable[[Env], Env]


@dataclass(frozen=True)
class Scan[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    to_reset: frozenset[str]
    to_scan: frozenset[str]
    length: int | None


@dataclass(frozen=True)
class Batch[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    to_map: frozenset[str]
    size: int | None


@dataclass(frozen=True)
class Learn[X, Y](Term[X, Y]):
    opt: Optimizer
    body: Term[X, Y]


@dataclass(frozen=True)
class Rtrl[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    targets: frozenset[str]
    influence: str


@dataclass(frozen=True)
class Uoro[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    targets: frozenset[str]
    distribution: SAMPLER
    aux: str
    prng: str


@dataclass(frozen=True)
class Rflo[X, Y](Term[X, Y]):
    cell: Term[X, Y]
    targets: frozenset[str]
    influence: str
    decay: str
