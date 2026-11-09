from dataclasses import dataclass
from functools import reduce
from typing import Callable, Generator
from pyrsistent import PList, PVector, plist, pvector
import equinox as eqx

from .lib_types import Proxy


class Writer[E, A](eqx.Module):
    value: A
    log: PVector[E]

    def __iter__(self) -> Generator[None, None, A]: ...  # typing only: `yield from w` has the type of w.value

    def flat_map[B](self, f: Callable[[A], "Writer[E, B]"]) -> "Writer[E, B]":
        w = f(self.value)
        return Writer(w.value, self.log + w.log)


def pure[E, A](_: Proxy[E], a: A) -> Writer[E, A]:
    return Writer(a, pvector())


def tell[E](e: E) -> Writer[E, None]:
    return Writer(None, pvector([e]))


def sequence[E, A](ws: PList[Writer[E, A]]) -> Writer[E, PList[A]]:
    def step(rest: Writer[E, PList[A]], w: Writer[E, A]) -> Writer[E, PList[A]]:
        return w.flat_map(lambda a: rest.flat_map(lambda xs: pure(Proxy[E](), xs.cons(a))))

    return reduce(step, ws.reverse(), pure(Proxy[E](), plist()))
