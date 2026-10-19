from collections.abc import Callable
from dataclasses import dataclass

from ..category.lens import Lens, assocL, identity, swap, with_unit, wiring
from ..category.lib_types import Proxy, Unit
from ..category.mealy import lift
from ..category.paralens import ParaLens
from ..lib_types import PRNG
from ..jax_backend import add_cotangents, zero_cotangent_like
from . import initialization as init
from . import program as prog
from .bindings import (
    Assembly,
    Layout,
    Parts,
    Refs,
    Values,
    layout,
    merge,
    paired,
    parse_selection,
    remaining,
    request,
    route,
    select,
    split,
)


def combine[A, DA, S, DS, X, DX, Y, DY](
    left: Layout,
    right: Layout,
    body: prog.Program[Parts[A], Parts[DA], S, DS, X, DX, Y, DY],
) -> "Named[A, DA, S, DS, X, DX, Y, DY]":
    source = layout(left.refs + right.refs)
    return Named(
        source,
        prog.Reparam(body, paired(source, left.refs, right.refs, zero_cotangent_like, add_cotangents)),
    )


@dataclass(frozen=True)
class Named[A, DA, S, DS, X, DX, Y, DY]:
    parameters: Layout
    body: prog.Program[Values[A], Values[DA], S, DS, X, DX, Y, DY]

    def then[T, DT, Z, DZ](
        self, other: "Named[A, DA, T, DT, Y, DY, Z, DZ]"
    ) -> "Named[A, DA, tuple[S, T], tuple[DS, DT], X, DX, Z, DZ]":
        return combine(self.parameters, other.parameters, prog.Then(self.body, other.body))

    def then_shared[Z, DZ](self, other: "Named[A, DA, S, DS, Y, DY, Z, DZ]") -> "Named[A, DA, S, DS, X, DX, Z, DZ]":
        return combine(self.parameters, other.parameters, prog.ThenShared(self.body, other.body))

    def tensor[T, DT, U, DU, V, DV](
        self, other: "Named[A, DA, T, DT, U, DU, V, DV]"
    ) -> "Named[A, DA, tuple[S, T], tuple[DS, DT], tuple[X, U], tuple[DX, DU], tuple[Y, V], tuple[DY, DV]]":
        return combine(self.parameters, other.parameters, prog.Beside(self.body, other.body))

    def __rshift__[P, DP, SF, DSF, SG, DSG, A, DA, B, DB, C, DC](
        f: "Named[P, DP, SF, DSF, A, DA, B, DB]",
        g: "Named[P, DP, SG, DSG, B, DB, C, DC]",
    ) -> "Named[P, DP, tuple[SF, SG], tuple[DSF, DSG], A, DA, C, DC]":
        return f.then(g)

    def __matmul__[P, DP, SF, DSF, SG, DSG, A, DA, B, DB, C, DC, D, DD](
        f: "Named[P, DP, SF, DSF, A, DA, B, DB]",
        g: "Named[P, DP, SG, DSG, C, DC, D, DD]",
    ) -> "Named[P, DP, tuple[SF, SG], tuple[DSF, DSG], tuple[A, C], tuple[DA, DC], tuple[B, D], tuple[DB, DD]]":
        return f.tensor(g)

    def pre[W, DW](self, inputs: Lens[W, DW, X, DX]) -> "Named[A, DA, S, DS, W, DW, Y, DY]":
        return Named(self.parameters, prog.Pre(self.body, inputs))

    def post[Z, DZ](self, outputs: Lens[Y, DY, Z, DZ]) -> "Named[A, DA, S, DS, X, DX, Z, DZ]":
        return Named(self.parameters, prog.Post(self.body, outputs))

    def first[C, DC](
        self, _: Proxy[tuple[C, DC]]
    ) -> "Named[A, DA, S, DS, tuple[X, C], tuple[DX, DC], tuple[Y, C], tuple[DY, DC]]":
        return Named(self.parameters, prog.First(self.body))

    def scope[T, DT, C, DC](
        self,
        initial: "Named[A, DA, T, DT, C, DC, S, DS]",
        close: Callable[[S], DS],
    ) -> "Named[A, DA, T, DT, tuple[C, X], tuple[DC, DX], Y, DY]":
        """Initialize this body's local state; retain the initializer's state."""
        return combine(initial.parameters, self.parameters, prog.Scope(initial.body, self.body, close))

    def provide[T, DT, C, DC](
        self,
        refs: Refs,
        initial: "Named[A, DA, T, DT, C, DC, Values[A], Values[DA]]",
    ) -> "Named[A, DA, tuple[T, S], tuple[DT, DS], tuple[C, X], tuple[DC, DX], Y, DY]":
        """Supply selected bindings for this invocation; retain other ports."""
        selected = select(request(self.parameters, refs))
        assemble: Assembly[A, DA]
        _, assemble = parse_selection(selected)
        rest = remaining(selected)
        producing = initial.first(Proxy[tuple[X, DX]]())
        supplied = Named(rest, prog.Unjoin(prog.Reparam(self.body, assemble)))
        return producing >> supplied

    def start[T, DT, C, DC](
        self,
        initial: "Named[A, DA, T, DT, C, DC, tuple[Values[A], S], tuple[Values[DA], DS]]",
        close: Callable[[S], DS],
    ) -> "Named[A, DA, T, DT, tuple[C, X], tuple[DC, DX], Y, DY]":
        """Supply (parameters, state) together, in this body's parameter order."""
        local = prog.Scope(prog.Call(lift(identity(Proxy[tuple[S, DS]]()))), self.body, close)
        supplied = prog.Unjoin(prog.Reparam(local, swap(Proxy())))
        combined = prog.Then(prog.First(initial.body), prog.Pre(supplied, assocL(Proxy())))
        # Remove the empty parameter and persistent-state factors of supplied.
        combined = prog.Reparam(combined, with_unit(Proxy()))
        combined = prog.Within(combined, with_unit(Proxy()), with_unit(Proxy()))
        return Named(initial.parameters, combined)

    def scan(self: "Named[A, A, S, DS, X, DX, Y, DY]", shared: Refs, length: int) -> "Named[A, A, S, DS, X, DX, Y, DY]":
        """Share listed bindings; slice all others along the time axis."""
        selected = select(request(self.parameters, shared))
        partition, assemble = parse_selection(selected)
        return Named(
            self.parameters,
            prog.Reparam(
                prog.ScanMixed(prog.Reparam(self.body, assemble), length),
                partition,
            ),
        )

    def learn[Z](
        self: "Named[A, DA, S, S, X, DX, Y, Y]",
        opt: "Optimizer[A, DA, Z]",
    ) -> "Named[A, A, tuple[Z, S], tuple[Z, S], X, X, Y, Y]":
        """Learn selected bindings; expose controls and remaining bindings.

        This objective must already provide the intended loss feedback.
        """
        selected = select(request(self.parameters, opt.targets))
        _, assemble = parse_selection(selected)
        body = prog.Learn(opt.rule, prog.Reparam(self.body, assemble))
        return combine(opt.controls, remaining(selected), body)


def use[A, DA, S, DS, X, DX, Y, DY](
    refs: Refs, body: prog.Program[Values[A], Values[DA], S, DS, X, DX, Y, DY]
) -> Named[A, DA, S, DS, X, DX, Y, DY]:
    """Label a positional parameter tuple; repeated refs explicitly share values."""
    source = layout(refs)
    return Named(source, prog.Reparam(body, route(request(source, refs), zero_cotangent_like, add_cotangents)))


def create[A, Y](refs: Refs, initial: init.Initialization[Values[A], Y]) -> Named[A, A, Unit, Unit, PRNG, None, Y, Y]:
    """Read named dependencies into an existing initialization recipe."""
    return use(refs, prog.Reparam(prog.Join(prog.Create(initial)), with_unit(Proxy())))


@dataclass(frozen=True)
class Optimizer[A, DA, Z]:
    targets: Refs
    controls: Layout
    rule: ParaLens[Values[A], Unit, Z, Z, Values[A], Values[DA]]
    initial: init.Initialization[tuple[Values[A], Values[A]], Z]

    def tensor[T](self, other: "Optimizer[A, DA, T]") -> "Optimizer[A, DA, tuple[Z, T]]":
        source = layout(self.controls.refs + other.controls.refs)
        inputs = paired(source, self.controls.refs, other.controls.refs, zero_cotangent_like, add_cotangents)
        outputs = merge(len(self.targets))
        rule = (self.rule @ other.rule).post(outputs).reparam(wiring(inputs.get, lambda _: ()))

        def left(context: tuple[Values[A], Values[A]]) -> tuple[Values[A], Values[A]]:
            controls, targets = context
            left_controls, _ = inputs.get(controls)
            left_targets, _ = split(len(self.targets)).get(targets)
            return left_controls, left_targets

        def right(context: tuple[Values[A], Values[A]]) -> tuple[Values[A], Values[A]]:
            controls, targets = context
            _, right_controls = inputs.get(controls)
            _, right_targets = split(len(self.targets)).get(targets)
            return right_controls, right_targets

        initial = init.Pair(init.Then(init.Apply(left), self.initial), init.Then(init.Apply(right), other.initial))
        return Optimizer(self.targets + other.targets, source, rule, initial)

    def __matmul__[A, DA, Z, T](
        f: "Optimizer[A, DA, Z]",
        g: "Optimizer[A, DA, T]",
    ) -> "Optimizer[A, DA, tuple[Z, T]]":
        return f.tensor(g)

    def initialize(self) -> Named[A, A, Unit, Unit, PRNG, None, Z, Z]:
        """Produce optimizer state from currently available controls and targets."""
        return create(
            self.controls.refs + self.targets,
            init.Then(init.Apply(split(len(self.controls.refs)).get), self.initial),
        )


def optimizer[A, DA, DH, Z](
    targets: Refs,
    controls: Refs,
    rule: ParaLens[Values[A], DH, Z, Z, Values[A], Values[DA]],
    initial: init.Initialization[tuple[Values[A], Values[A]], Z],
) -> Optimizer[A, DA, Z]:
    """Name an optimizer's target tuple and control tuple, with its state recipe.

    Learn differentiates the completed update. Its implementation discards the raw
    optimizer reply on the control port; that reply is NOT the hypergradient.
    """
    selected = select(request(layout(targets), targets))
    source = layout(controls)
    inputs = route(request(source, controls), zero_cotangent_like, add_cotangents)

    def context(values: tuple[Values[A], Values[A]]) -> tuple[Values[A], Values[A]]:
        controls, targets = values
        return inputs.get(controls), targets

    return Optimizer(
        selected.refs,
        source,
        rule.reparam(wiring(inputs.get, lambda _: ())),
        init.Then(init.Apply(context), initial),
    )
