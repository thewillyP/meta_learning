from typing import overload
from plum import dispatch

from ..algorithms.learning import rflo, rtrl, uoro
from ..category.lib_types import Proxy
from ..category.mealy import Mealy, lift
from ..components.env import Env
from ..components.term import Batch, Cell, Learn, Lift, Rename, Reparam, Rflo, Rtrl, Scan, Tensor, Term, Then, Uoro
from .jax import batch, learn, mealy_autodiff, scan
from .record import recordwise, rename


@overload
def build[X, Y](t: Cell[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return mealy_autodiff(t.step, recordwise())


@overload
def build[X, Y](t: Lift[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return lift(Proxy[tuple[Env, Env]](), t.lens, recordwise())


@overload
def build[X, Y, Z](t: Then[X, Y, Z]) -> Mealy[Env, Env, Env, Env, Env, X, X, Z, Z]:
    return build(t.f) >> build(t.g)


@overload
def build[X, Y, U, V](
    t: Tensor[X, Y, U, V],
) -> Mealy[Env, Env, Env, Env, Env, tuple[X, U], tuple[X, U], tuple[Y, V], tuple[Y, V]]:
    return build(t.f) @ build(t.g)


@overload
def build[X, Y](t: Rename[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return Mealy(rename(build(t.cell).step, t.transpositions))


@overload
def build[X, Y](t: Reparam[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return build(t.cell).reparam(t.lens, t.monoid)


@overload
def build[X, Y](t: Scan[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return scan(build(t.cell), t.to_reset, t.to_scan, t.length)


@overload
def build[X, Y](t: Batch[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return batch(build(t.cell), t.to_map, t.size)


@overload
def build[X, Y](t: Learn[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return learn(t.opt, build(t.body))


@overload
def build[X, Y](t: Rtrl[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return rtrl(build(t.cell), t.targets, t.influence)


@overload
def build[X, Y](t: Uoro[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return uoro(build(t.cell), t.targets, t.distribution, t.aux, t.prng)


@overload
def build[X, Y](t: Rflo[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    return rflo(build(t.cell), t.targets, t.influence, t.decay)


@overload
def build[X, Y](t: Term[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    raise NotImplementedError(type(t).__name__)


@dispatch
def build[X, Y](t: Term[X, Y]) -> Mealy[Env, Env, Env, Env, Env, X, X, Y, Y]:
    raise NotImplementedError
