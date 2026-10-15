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


def flatten[A, B]() -> Lens[list[list[A]], list[list[B]], list[A], list[B]]:
    def run(groups: list[list[A]]) -> tuple[list[A], Callable[[list[B]], list[list[B]]]]:
        sizes = tuple(map(len, groups))

        def backward(values: list[B]) -> list[list[B]]:
            if len(values) != (count := sum(sizes)):
                raise ValueError(f"expected {count} positions, received {len(values)}")
            it = iter(values)
            return [list(islice(it, n)) for n in sizes]

        return list(chain.from_iterable(groups)), backward

    return Lens(run)


@dataclass(frozen=True)
class Traversal[W, T, A, B]:
    parts: Lens[W, T, list[A], list[B]]

    def then[C, D](self, inner: "Traversal[A, B, C, D]") -> "Traversal[W, T, C, D]":
        return Traversal(self.parts >> sequence(inner.parts) >> flatten())

    def __rshift__[U, V, X, Y, Z, R](
        f: "Traversal[U, V, X, Y]",
        g: "Traversal[X, Y, Z, R]",
    ) -> "Traversal[U, V, Z, R]":
        return f.then(g)

    def interpret[R](
        self,
        interpreter: Callable[[list[A], Callable[[list[B]], T]], R],
        whole: W,
    ) -> R:
        selected, rebuild = self.parts.run(whole)
        return interpreter(selected, rebuild)

    def map(self, operation: Callable[[A], B], whole: W) -> T:
        return self.interpret(lambda values, rebuild: rebuild(list(map(operation, values))), whole)

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

        def run(values: list[A], rebuild: Callable[[list[B]], T]) -> tuple[S, T]:
            start: tuple[S, PList[B]] = initial, plist()
            final, results = reduce(advance, values, start)
            return final, rebuild(list(results.reverse()))

        return self.interpret(run, whole)

    def lift[DW, DT, DA, DB](
        self,
        local: Lens[A, DA, B, DB],
        feedback: "Traversal[DT, DW, DB, DA]",
        repeat: Callable[[Lens[A, DA, B, DB]], Lens[list[A], list[DA], list[B], list[DB]]],
    ) -> Lens[W, DW, T, DT]:
        def run(values: list[A], rebuild: Callable[[list[B]], T]) -> tuple[T, Callable[[DT], DW]]:
            results, back = repeat(local).run(values)

            def backward(dt: DT) -> DW:
                seeds, restore = feedback.parts.run(dt)
                return restore(back(seeds))

            return rebuild(results), backward

        return Lens(lambda whole: self.interpret(run, whole))


def select[W, T, A, B](optic: Lens[W, T, A, B]) -> Traversal[W, T, A, B]:
    def single(values: list[B]) -> B:
        [value] = values
        return value

    singleton: Lens[A, B, list[A], list[B]] = wiring(lambda value: [value], single)
    return Traversal(optic >> singleton)


def each[A, B]() -> Traversal[list[A], list[B], A, B]:
    def run(values: list[A]) -> tuple[list[A], Callable[[list[B]], list[B]]]:
        def backward(replacements: list[B]) -> list[B]:
            return [value for _, value in zip(values, replacements, strict=True)]

        return values, backward

    return Traversal(Lens(run))


def optional[A, B]() -> Traversal[A | None, B | None, A, B]:
    def run(value: A | None) -> tuple[list[A], Callable[[list[B]], B | None]]:
        def backward(values: list[B]) -> B | None:
            match value, values:
                case None, []:
                    return None
                case None, _:
                    raise ValueError(f"expected 0 positions, received {len(values)}")
                case _, [result]:
                    return result
                case _, _:
                    raise ValueError(f"expected 1 position, received {len(values)}")

        return ([] if value is None else [value]), backward

    return Traversal(Lens(run))
