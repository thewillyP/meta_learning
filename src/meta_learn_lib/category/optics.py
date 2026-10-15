from dataclasses import dataclass
from functools import reduce
from itertools import chain, islice
from typing import Callable

from pyrsistent import PList, plist

from .lens import Lens, fst, sequence, snd, wiring
from .lib_types import Proxy


type Focus[W, S] = Lens[W, W, S, S]


def focus[W, S](get: Callable[[W], S], put: Callable[[W, S], W]) -> Focus[W, S]:
    return Lens(lambda whole: (get(whole), lambda part: put(whole, part)))


def key[K, V](name: K) -> Focus[dict[K, V], V]:
    return focus(lambda values: values[name], lambda values, value: values | {name: value})


def first[A, B](_: Proxy[tuple[A, B]]) -> Focus[tuple[A, B], A]:
    return fst(Proxy[tuple[A, A, B, B]](), lambda b: b)


def second[A, B](_: Proxy[tuple[A, B]]) -> Focus[tuple[A, B], B]:
    return snd(Proxy[tuple[A, A, B, B]](), lambda a: a)


def over[W, S](at: Focus[W, S], operation: Callable[[S], S], whole: W) -> W:
    part, replace = at.run(whole)
    return replace(operation(part))


def flatten[A, B]() -> Lens[PList[PList[A]], PList[PList[B]], PList[A], PList[B]]:
    def run(groups: PList[PList[A]]) -> tuple[PList[A], Callable[[PList[B]], PList[PList[B]]]]:
        sizes = tuple(map(len, groups))

        def backward(values: PList[B]) -> PList[PList[B]]:
            if len(values) != (count := sum(sizes)):
                raise ValueError(f"expected {count} positions, received {len(values)}")
            it = iter(values)
            return plist(plist(islice(it, n)) for n in sizes)

        return plist(chain.from_iterable(groups)), backward

    return Lens(run)


@dataclass(frozen=True)
class Traversal[W, T, A, B]:
    parts: Lens[W, T, PList[A], PList[B]]

    def then[C, D](self, inner: "Traversal[A, B, C, D]") -> "Traversal[W, T, C, D]":
        return Traversal(self.parts >> sequence(inner.parts) >> flatten())

    def __rshift__[U, V, X, Y, Z, R](
        f: "Traversal[U, V, X, Y]",
        g: "Traversal[X, Y, Z, R]",
    ) -> "Traversal[U, V, Z, R]":
        return f.then(g)

    def interpret[R](
        self,
        interpreter: Callable[[PList[A], Callable[[PList[B]], T]], R],
        whole: W,
    ) -> R:
        selected, rebuild = self.parts.run(whole)
        return interpreter(selected, rebuild)

    def map(self, operation: Callable[[A], B], whole: W) -> T:
        return self.interpret(lambda values, rebuild: rebuild(plist(map(operation, values))), whole)

    def map_accum[S](
        self,
        step: Callable[[S, A], tuple[S, B]],
        initial: S,
        whole: W,
    ) -> tuple[S, T]:
        def advance(carry: tuple[S, PList[B]], value: A) -> tuple[S, PList[B]]:
            state, results = carry
            next_state, result = step(state, value)
            return next_state, results.cons(result)

        def run(values: PList[A], rebuild: Callable[[PList[B]], T]) -> tuple[S, T]:
            start: tuple[S, PList[B]] = initial, plist()
            final, results = reduce(advance, values, start)
            return final, rebuild(results.reverse())

        return self.interpret(run, whole)

    def lift[DW, DT, DA, DB](
        self,
        local: Lens[A, DA, B, DB],
        feedback: "Traversal[DT, DW, DB, DA]",
        repeat: Callable[[Lens[A, DA, B, DB]], Lens[PList[A], PList[DA], PList[B], PList[DB]]],
    ) -> Lens[W, DW, T, DT]:
        def run(values: PList[A], rebuild: Callable[[PList[B]], T]) -> tuple[T, Callable[[DT], DW]]:
            results, back = repeat(local).run(values)

            def backward(dt: DT) -> DW:
                seeds, restore = feedback.parts.run(dt)
                return restore(back(seeds))

            return rebuild(results), backward

        return Lens(lambda whole: self.interpret(run, whole))


def select[W, T, A, B](optic: Lens[W, T, A, B]) -> Traversal[W, T, A, B]:
    def single(values: PList[B]) -> B:
        [value] = values
        return value

    singleton: Lens[A, B, PList[A], PList[B]] = wiring(lambda value: plist((value,)), single)
    return Traversal(optic >> singleton)


def each[A, B]() -> Traversal[PList[A], PList[B], A, B]:
    def run(values: PList[A]) -> tuple[PList[A], Callable[[PList[B]], PList[B]]]:
        def backward(replacements: PList[B]) -> PList[B]:
            return plist(value for _, value in zip(values, replacements, strict=True))

        return values, backward

    return Traversal(Lens(run))


def optional[A, B]() -> Traversal[A | None, B | None, A, B]:
    def run(value: A | None) -> tuple[PList[A], Callable[[PList[B]], B | None]]:
        def backward(values: PList[B]) -> B | None:
            match value, values:
                case None, []:
                    return None
                case None, _:
                    raise ValueError(f"expected 0 positions, received {len(values)}")
                case _, [result]:
                    return result
                case _, _:
                    raise ValueError(f"expected 1 position, received {len(values)}")

        return (plist() if value is None else plist((value,))), backward

    return Traversal(Lens(run))
