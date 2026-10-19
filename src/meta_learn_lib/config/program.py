from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from plum import dispatch

from ..algorithms import combinators
from ..category.lens import Lens, assocL, assocR, identity, swap
from ..category.lib_types import Proxy, Unit
from ..category.mealy import Mealy, lift
from ..category.optics import Focus
from ..category.paralens import ParaLens
from ..jax_backend import autodiff, mealy_autodiff
from ..lib_types import PRNG
from .initialization import Initialization, compile_initial


@dataclass(frozen=True)
class Program[P, DP, S, DS, X, DX, Y, DY]: ...


@dataclass(frozen=True)
class Call[P, DP, S, DS, X, DX, Y, DY](Program[P, DP, S, DS, X, DX, Y, DY]):
    body: Mealy[P, DP, S, DS, X, DX, Y, DY]


@dataclass(frozen=True)
class Create[C, A](Program[Unit, Unit, Unit, Unit, tuple[C, PRNG], tuple[C, None], A, A]):
    initial: Initialization[C, A]


@dataclass(frozen=True)
class Scope[Q, DQ, T, DT, C, DC, P, DP, S, DS, X, DX, Y, DY](
    Program[tuple[Q, P], tuple[DQ, DP], T, DT, tuple[C, X], tuple[DC, DX], Y, DY]
):
    initial: Program[Q, DQ, T, DT, C, DC, S, DS]
    body: Program[P, DP, S, DS, X, DX, Y, DY]
    close: Callable[[S], DS]


@dataclass(frozen=True)
class Then[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, Z, DZ](
    Program[tuple[P, Q], tuple[DP, DQ], tuple[S, T], tuple[DS, DT], X, DX, Z, DZ]
):
    first: Program[P, DP, S, DS, X, DX, Y, DY]
    second: Program[Q, DQ, T, DT, Y, DY, Z, DZ]


@dataclass(frozen=True)
class ThenShared[P, DP, Q, DQ, S, DS, X, DX, Y, DY, Z, DZ](Program[tuple[P, Q], tuple[DP, DQ], S, DS, X, DX, Z, DZ]):
    first: Program[P, DP, S, DS, X, DX, Y, DY]
    second: Program[Q, DQ, S, DS, Y, DY, Z, DZ]


@dataclass(frozen=True)
class Beside[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, U, DU, V, DV](
    Program[
        tuple[P, Q],
        tuple[DP, DQ],
        tuple[S, T],
        tuple[DS, DT],
        tuple[X, U],
        tuple[DX, DU],
        tuple[Y, V],
        tuple[DY, DV],
    ]
):
    first: Program[P, DP, S, DS, X, DX, Y, DY]
    second: Program[Q, DQ, T, DT, U, DU, V, DV]


@dataclass(frozen=True)
class First[P, DP, S, DS, X, DX, Y, DY, C, DC](
    Program[P, DP, S, DS, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]
):
    body: Program[P, DP, S, DS, X, DX, Y, DY]


@dataclass(frozen=True)
class Pre[P, DP, S, DS, W, DW, X, DX, Y, DY](Program[P, DP, S, DS, W, DW, Y, DY]):
    body: Program[P, DP, S, DS, X, DX, Y, DY]
    inputs: Lens[W, DW, X, DX]


@dataclass(frozen=True)
class Post[P, DP, S, DS, X, DX, Y, DY, Z, DZ](Program[P, DP, S, DS, X, DX, Z, DZ]):
    body: Program[P, DP, S, DS, X, DX, Y, DY]
    outputs: Lens[Y, DY, Z, DZ]


@dataclass(frozen=True)
class Reparam[Q, DQ, P, DP, S, DS, X, DX, Y, DY](Program[Q, DQ, S, DS, X, DX, Y, DY]):
    body: Program[P, DP, S, DS, X, DX, Y, DY]
    parameters: Lens[Q, DQ, P, DP]


@dataclass(frozen=True)
class Join[Q, DQ, P, DP, S, DS, X, DX, Y, DY](Program[tuple[Q, P], tuple[DQ, DP], S, DS, X, DX, Y, DY]):
    body: Program[P, DP, S, DS, tuple[Q, X], tuple[DQ, DX], Y, DY]


@dataclass(frozen=True)
class Unjoin[Q, DQ, P, DP, S, DS, X, DX, Y, DY](Program[P, DP, S, DS, tuple[Q, X], tuple[DQ, DX], Y, DY]):
    body: Program[tuple[Q, P], tuple[DQ, DP], S, DS, X, DX, Y, DY]


@dataclass(frozen=True)
class Within[P, DP, S, DS, W, DW, X, DX, Y, DY](Program[P, DP, W, DW, X, DX, Y, DY]):
    body: Program[P, DP, S, DS, X, DX, Y, DY]
    state: Focus[W, S]
    feedback: Focus[DW, DS]


@dataclass(frozen=True)
class Scan[P, S, DS, X, DX, Y, DY](Program[P, P, S, DS, X, DX, Y, DY]):
    body: Program[P, P, S, DS, X, DX, Y, DY]
    length: int


@dataclass(frozen=True)
class Batch[Z, P, DP, S, DS, X, DX, Y, DY](Program[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY]):
    body: Program[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY]
    size: int


@dataclass(frozen=True)
class Learn[H, DH, Z, P, DP, R, DR, S, X, DX, Y](
    Program[tuple[H, R], tuple[H, R], tuple[Z, S], tuple[Z, S], X, X, Y, Y]
):
    optimizer: ParaLens[H, DH, Z, Z, P, DP]
    body: Program[tuple[P, R], tuple[DP, DR], S, S, X, DX, Y, Y]


@dataclass(frozen=True)
class Read[W, A](Program[Unit, Unit, W, W, Unit, Unit, A, A]):
    at: Focus[W, A]


@dataclass(frozen=True)
class Write[W, A](Program[Unit, Unit, W, W, A, A, Unit, Unit]):
    at: Focus[W, A]


@overload
def execute[P, DP, S, DS, X, DX, Y, DY](term: Call[P, DP, S, DS, X, DX, Y, DY]) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    return term.body


@overload
def execute[C, A](term: Create[C, A]) -> Mealy[Unit, Unit, Unit, Unit, tuple[C, PRNG], tuple[C, None], A, A]:
    def initialize(context_key: tuple[C, PRNG]) -> A:
        context, key = context_key
        return compile_initial(term.initial)(context, key)

    return lift(autodiff(initialize))


@overload
def execute[Q, DQ, T, DT, C, DC, P, DP, S, DS, X, DX, Y, DY](
    term: Scope[Q, DQ, T, DT, C, DC, P, DP, S, DS, X, DX, Y, DY],
) -> Mealy[tuple[Q, P], tuple[DQ, DP], T, DT, tuple[C, X], tuple[DC, DX], Y, DY]:
    start = execute(term.initial).first(Proxy[tuple[X, DX]]())
    local = execute(term.body).local_state(identity(Proxy[tuple[S, DS]]()), term.close).unjoin()
    return Mealy(start.step >> local.second(Proxy[tuple[T, DT]]()))


@overload
def execute[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, Z, DZ](
    term: Then[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, Z, DZ],
) -> Mealy[tuple[P, Q], tuple[DP, DQ], tuple[S, T], tuple[DS, DT], X, DX, Z, DZ]:
    return execute(term.first) >> execute(term.second)


@overload
def execute[P, DP, Q, DQ, S, DS, X, DX, Y, DY, Z, DZ](
    term: ThenShared[P, DP, Q, DQ, S, DS, X, DX, Y, DY, Z, DZ],
) -> Mealy[tuple[P, Q], tuple[DP, DQ], S, DS, X, DX, Z, DZ]:
    return execute(term.first).then_shared(execute(term.second))


@overload
def execute[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, U, DU, V, DV](
    term: Beside[P, DP, Q, DQ, S, DS, T, DT, X, DX, Y, DY, U, DU, V, DV],
) -> Mealy[
    tuple[P, Q],
    tuple[DP, DQ],
    tuple[S, T],
    tuple[DS, DT],
    tuple[X, U],
    tuple[DX, DU],
    tuple[Y, V],
    tuple[DY, DV],
]:
    return execute(term.first) @ execute(term.second)


@overload
def execute[P, DP, S, DS, X, DX, Y, DY, C, DC](
    term: First[P, DP, S, DS, X, DX, Y, DY, C, DC],
) -> Mealy[P, DP, S, DS, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]:
    return execute(term.body).first(Proxy[tuple[C, DC]]())


@overload
def execute[P, DP, S, DS, W, DW, X, DX, Y, DY](
    term: Pre[P, DP, S, DS, W, DW, X, DX, Y, DY],
) -> Mealy[P, DP, S, DS, W, DW, Y, DY]:
    return execute(term.body).pre(term.inputs)


@overload
def execute[P, DP, S, DS, X, DX, Y, DY, Z, DZ](
    term: Post[P, DP, S, DS, X, DX, Y, DY, Z, DZ],
) -> Mealy[P, DP, S, DS, X, DX, Z, DZ]:
    return execute(term.body).post(term.outputs)


@overload
def execute[Q, DQ, P, DP, S, DS, X, DX, Y, DY](
    term: Reparam[Q, DQ, P, DP, S, DS, X, DX, Y, DY],
) -> Mealy[Q, DQ, S, DS, X, DX, Y, DY]:
    return execute(term.body).reparam(term.parameters)


@overload
def execute[Q, DQ, P, DP, S, DS, X, DX, Y, DY](
    term: Join[Q, DQ, P, DP, S, DS, X, DX, Y, DY],
) -> Mealy[tuple[Q, P], tuple[DQ, DP], S, DS, X, DX, Y, DY]:
    inputs = (
        assocR(Proxy[tuple[Q, DQ, S, DS, X, DX]]())
        >> swap(Proxy[tuple[Q, DQ, S, DS]]()).first(Proxy[tuple[X, DX]]())
        >> assocL(Proxy[tuple[S, DS, Q, DQ, X, DX]]())
    )
    return Mealy(execute(term.body).step.pre(inputs).join())


@overload
def execute[Q, DQ, P, DP, S, DS, X, DX, Y, DY](
    term: Unjoin[Q, DQ, P, DP, S, DS, X, DX, Y, DY],
) -> Mealy[P, DP, S, DS, tuple[Q, X], tuple[DQ, DX], Y, DY]:
    inputs = (
        assocR(Proxy[tuple[S, DS, Q, DQ, X, DX]]())
        >> swap(Proxy[tuple[S, DS, Q, DQ]]()).first(Proxy[tuple[X, DX]]())
        >> assocL(Proxy[tuple[Q, DQ, S, DS, X, DX]]())
    )
    return Mealy(execute(term.body).step.unjoin().pre(inputs))


@overload
def execute[P, DP, S, DS, W, DW, X, DX, Y, DY](
    term: Within[P, DP, S, DS, W, DW, X, DX, Y, DY],
) -> Mealy[P, DP, W, DW, X, DX, Y, DY]:
    return execute(term.body).within(term.state, term.feedback)


@overload
def execute[P, S, DS, X, DX, Y, DY](
    term: Scan[P, S, DS, X, DX, Y, DY],
) -> Mealy[P, P, S, DS, X, DX, Y, DY]:
    return combinators.scan_shared(execute(term.body), term.length)


@overload
def execute[Z, P, DP, S, DS, X, DX, Y, DY](
    term: Batch[Z, P, DP, S, DS, X, DX, Y, DY],
) -> Mealy[tuple[Z, P], tuple[Z, DP], S, DS, X, DX, Y, DY]:
    return combinators.batch(execute(term.body), term.size)


@overload
def execute[H, DH, Z, P, DP, R, DR, S, X, DX, Y](
    term: Learn[H, DH, Z, P, DP, R, DR, S, X, DX, Y],
) -> Mealy[tuple[H, R], tuple[H, R], tuple[Z, S], tuple[Z, S], X, X, Y, Y]:
    return combinators.learn(term.optimizer, execute(term.body))


@overload
def execute[W, A](term: Read[W, A]) -> Mealy[Unit, Unit, W, W, Unit, Unit, A, A]:
    def read(parameters: Unit, state: W, inputs: Unit) -> tuple[W, A]:
        return state, term.at.get(state)

    return mealy_autodiff(read)


@overload
def execute[W, A](term: Write[W, A]) -> Mealy[Unit, Unit, W, W, A, A, Unit, Unit]:
    def write(parameters: Unit, state: W, value: A) -> tuple[W, Unit]:
        _, replace = term.at.run(state)
        return replace(value), ()

    return mealy_autodiff(write)


@overload
def execute[P, DP, S, DS, X, DX, Y, DY](term: Program[P, DP, S, DS, X, DX, Y, DY]) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    raise NotImplementedError(f"no execution interpretation for {type(term).__name__}")


@dispatch
def execute[P, DP, S, DS, X, DX, Y, DY](term: Program[P, DP, S, DS, X, DX, Y, DY]) -> Mealy[P, DP, S, DS, X, DX, Y, DY]:
    raise NotImplementedError(f"no execution interpretation for {type(term).__name__}")
