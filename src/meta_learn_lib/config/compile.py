"""Lower the data language directly to core Mealy machines.

Names, shape checks, and dependency ordering disappear during compilation.
Runtime state is (key, values); lenses address positions in values.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import overload

import equinox as eqx
import jax
from jaxtyping import PyTree
from plum import dispatch

from ..algorithms import combinators, optimizers
from ..category.lib_types import Proxy, Unit
from ..category.mealy import Mealy
from ..category.optics import Focus, focus, second
from ..category.paralens import ParaLens
from ..backend.jax_backend import (
    autodiff,
    learning_rate_log,
    mealy_autodiff,
    zero_cotangent_like,
)
from ..lib_types import PRNG
from . import initialization as init
from . import terms as s
from .bindings import (
    Assembly,
    Layout,
    Partition,
    Ref,
    layout,
    parse_selection,
    remaining,
    request,
    select,
    selection,
)
from .named import Optimizer, optimizer
from .resolve import (
    Algorithm,
    Catalog,
    Shape,
    Values,
    activation,
    binding_shape,
    called,
    component_refs,
    dependencies,
    infer,
    loss,
    reduction,
    required,
    source,
    unstack,
    written,
)

type Frame = tuple[PRNG, Values]
type Cell[X, Y] = Mealy[Unit, Unit, Frame, Frame, X, X, Y, Y]
Tree = PyTree[jax.Array]
type Context = tuple[Tree, Tree, Values]


def abstract(tree: Tree) -> Shape:
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype) if eqx.is_array(a) else a, tree)


def value_focus(source: Layout, refs: tuple[Ref, ...]) -> Focus[Frame, Values]:
    partition, _ = parse_selection(select(request(source, refs)))
    return second(Proxy[Frame]()) >> selection(partition)


def state_ports[X, Y](
    body: Cell[X, Y], exposed: Layout, changing: tuple[Ref, ...]
) -> tuple[Mealy[Values, Values, Frame, Frame, X, X, Y, Y], Partition[Tree, Tree], Assembly[Tree, Tree]]:
    """Move unchanged bindings to the shared parameter port, outside scan carry."""
    partition, assemble = parse_selection(select(request(exposed, changing)))

    def mount(parameters_sx):
        parameters, ((key, state), x) = parameters_sx
        return (), ((key, assemble.get((state, parameters))), x)

    def unmount(sy):
        (key, values), y = sy
        state, _ = partition.get(values)
        return (key, state), y

    cell = Mealy(ParaLens(autodiff(mount) >> body.step.arrow >> autodiff(unmount)))
    return cell, partition, assemble


def then[X, Y, Z](left: Cell[X, Y], right: Cell[Y, Z]) -> Cell[X, Z]:
    return left.then_shared(right).reparam(autodiff(lambda unit: ((), ())))


def pure[X, Y](function: Callable[[X], Y]) -> Cell[X, Y]:
    return mealy_autodiff(lambda unit, state, x: (state, function(x)))


@overload
def lower(
    term: s.Call, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    body, stack = called(term, catalog, stack)
    return lower(body, catalog, exposed, inputs, stack)


@overload
def lower(
    term: s.Read, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    at = value_focus(exposed, catalog.expand((term.ref,)))
    return mealy_autodiff(lambda unit, state, x: (state, at.get(state)[0]))


@overload
def lower(
    term: s.Activation, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    return pure(activation(term))


@overload
def lower(
    term: s.Linear, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    at = value_focus(exposed, component_refs(term, catalog))

    def step(unit: Unit, values: Values, x: jax.Array):
        (layer,) = values
        return values, layer(x)

    return mealy_autodiff(step).within(at, at)


@overload
def lower(
    term: s.Bias, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    at = value_focus(exposed, component_refs(term, catalog))
    return mealy_autodiff(lambda unit, values, x: (values, x + values[0])).within(at, at)


@overload
def lower(
    term: s.Value, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    at = value_focus(exposed, component_refs(term, catalog))
    return mealy_autodiff(lambda unit, state, x: (state, at.get(state)[0]))


@overload
def lower(
    term: s.Rnn, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    refs = component_refs(term, catalog)
    unique = layout(refs)
    at = value_focus(exposed, unique.refs)
    positions = tuple(unique.refs.index(ref) for ref in refs)
    hidden_position = positions[2]
    activate = activation(term.act)
    if isinstance(term.time_constant, (float, int)):
        rate = lambda values: term.time_constant
    else:
        rate = lambda values: values[positions[3]]

    def step(unit: Unit, values: Values, x: jax.Array):
        recurrent, incoming, hidden = tuple(values[i] for i in positions[:3])
        tau = rate(values)
        next_hidden = (1 - tau) * hidden + tau * activate(recurrent(hidden) + incoming(x))
        updated = tuple(next_hidden if i == hidden_position else value for i, value in enumerate(values))
        return updated, next_hidden

    return mealy_autodiff(step).within(at, at)


@overload
def lower(
    term: s.Sequence, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    cell = pure(lambda x: (x, x, ()))
    previous, names, results = inputs, (), ()
    for step in term.steps:
        get, shape = source(step.input, inputs, previous, names, results)
        body = lower(step.body, catalog, exposed, shape, stack)
        cell = then(cell, sequence_step(body, get, step.name is not None))
        previous = infer(step.body, catalog, shape, stack)
        if step.name is not None:
            names, results = names + (step.name,), results + (previous,)
    return cell.post(autodiff(lambda context: context[1]))


def sequence_step(body: Cell[Tree, Tree], get: Callable[[Context], Tree], remember: bool) -> Cell[Context, Context]:
    def finish(result):
        value, (original, _, results) = result
        return original, value, results + (value,) if remember else results

    return body.first(Proxy()).pre(autodiff(lambda context: (get(context), context))).post(autodiff(finish))


@overload
def lower(
    term: s.Scan, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    body = lower(term.body, catalog, exposed, unstack(inputs, term.steps), stack)
    updates = written(term.body, catalog, stack)
    changing = tuple(ref for ref in exposed.refs if ref in updates)
    body, partition, assemble = state_ports(body, exposed, changing)
    scanned = combinators.scan_shared(body, term.steps)

    def prepare(unit_sx):
        _, ((key, values), xs) = unit_sx
        state, parameters = partition.get(values)
        return (parameters, ((key, state), xs)), parameters

    def finish(result):
        ((key, state), ys), parameters = result
        return (key, assemble.get((state, parameters))), ys

    return Mealy(ParaLens(autodiff(prepare) >> scanned.step.arrow.first(Proxy()) >> autodiff(finish)))


@overload
def lower(
    term: s.Batch, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    if updates := written(term.body, catalog, stack):
        raise ValueError(f"batch branches cannot write shared outer bindings: {updates}; give each branch a scope")
    body = lower(term.body, catalog, exposed, unstack(inputs, term.size), stack)
    size = term.size

    branch, _, _ = state_ports(body, exposed, ())
    parallel = combinators.batch_data(branch, size)

    def prepare(unit_sx):
        _, ((key, values), xs) = unit_sx
        next_key, branch_key = jax.random.split(key)
        return (values, ((jax.random.split(branch_key, size), ()), xs)), (PRNG(next_key), values)

    def finish(result):
        (_, ys), state = result
        return state, ys

    return Mealy(ParaLens(autodiff(prepare) >> parallel.step.arrow.first(Proxy()) >> autodiff(finish)))


@overload
def lower(
    term: s.Reduce, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    body = lower(term.body, catalog, exposed, inputs, stack)
    return body.post(autodiff(reduction(term.operation)))


@overload
def lower(
    term: s.Supervised, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    body = lower(term.body, catalog, exposed, inputs["features"], stack)
    measure = loss(term.loss)
    reduce_loss = reduction(term.reduction)
    return (
        body.first(Proxy())
        .pre(autodiff(lambda x: (x["features"], x["target"])))
        .post(autodiff(lambda prediction_target: reduce_loss(measure(prediction_target[0], prediction_target[1]))))
    )


def initializer(
    ref: Ref, catalog: Catalog, exposed: Layout, inputs: Shape
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    if ref in catalog.memories:
        algorithm = catalog.memories[ref]
        controls = value_focus(exposed, algorithm.controls)
        targets = value_focus(exposed, algorithm.targets)
        make = algorithm.make
        return mealy_autodiff(lambda unit, state, x: (state, make(controls.get(state)).init(targets.get(state))))
    return initialize_recipe(catalog.recipes[ref], ref, catalog, exposed, inputs)


@overload
def initialize_recipe(
    recipe: s.Computed, ref: Ref, catalog: Catalog, exposed: Layout, inputs: Shape
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    catalog.constrain(ref, infer(recipe.body, catalog, inputs, ()))
    return lower(recipe.body, catalog, exposed, inputs, ())


@overload
def initialize_recipe(
    recipe: init.Init, ref: Ref, catalog: Catalog, exposed: Layout, inputs: Shape
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    shape = binding_shape(ref, catalog, inputs, ())
    return initialize_recipe(init.Array(shape.shape, str(shape.dtype), recipe), ref, catalog, exposed, inputs)


@overload
def initialize_recipe(
    recipe: init.Initialization, ref: Ref, catalog: Catalog, exposed: Layout, inputs: Shape
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    make = init.compile_initial(recipe)

    def step(unit: Unit, state: Frame, x: Tree):
        key, values = state
        next_key, draw_key = jax.random.split(key)
        return (PRNG(next_key), values), make((), PRNG(draw_key))

    return mealy_autodiff(step)


@dispatch
def initialize_recipe(
    recipe: init.Initialization, ref: Ref, catalog: Catalog, exposed: Layout, inputs: Shape
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]: ...


def enter_scope(
    term: s.Scope, catalog: Catalog, exposed: Layout, inputs: Shape
) -> tuple[Layout, Callable[[Frame, Tree], Frame], Callable[[Frame, Frame], Frame]]:
    local = layout(catalog.expand(term.initialize))
    kept = remaining(select(request(exposed, tuple(r for r in local.refs if r in exposed.refs))))
    keep = value_focus(exposed, kept.refs)
    current = kept
    pending = list(local.refs)
    cells = []
    while pending:
        ready = next((r for r in pending if dependencies(r, catalog, ()) <= set(current.refs)), None)
        if ready is None:
            missing = {
                r.name: tuple(d.name for d in dependencies(r, catalog, ()) if d not in current.refs) for r in pending
            }
            raise ValueError(f"missing or cyclic initialization dependencies: {missing}")
        cell = initializer(ready, catalog, current, inputs)
        frame_shape = (
            jax.ShapeDtypeStruct((), jax.random.key(0).dtype),
            tuple(catalog.shapes[r] for r in current.refs),
        )
        _, shape = eqx.filter_eval_shape(cell.step.arrow.get, ((), (frame_shape, inputs)))
        if ready in catalog.memories:
            catalog.shapes[ready] = shape
        else:
            catalog.constrain(ready, shape)
        cells.append(cell)
        current = layout(current.refs + (ready,))
        pending.remove(ready)
    initializers = tuple(cells)

    def enter(state: Frame, x: Tree) -> Frame:
        state = state[0], keep.get(state)
        for cell in initializers:
            (key, values), value = cell.step.arrow.get(((), (state, x)))
            state = key, values + (value,)
        return state

    restore = value_focus(current, kept.refs)

    def leave(original: Frame, final: Frame) -> Frame:
        _, replace = keep.run(original)
        restored = replace(restore.get(final))
        return final[0], restored[1]

    return current, enter, leave


@overload
def lower(
    term: s.Scope, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    inner, enter, leave = enter_scope(term, catalog, exposed, inputs)
    body = lower(term.body, catalog, inner, inputs, stack)
    # The local state is closed by the existing lifetime combinator. Expose its
    # final values as output so surviving outer bindings can be restored.
    observable = Mealy(body.step.post(autodiff(lambda sy: (sy[0], sy))))
    local = observable.local_state(autodiff(lambda sx: enter(sx[0], sx[1])), zero_cotangent_like)

    def prepare(unit_sx):
        _, (state, x) = unit_sx
        return ((state, x), ()), (x, state)

    def finish(result):
        (final, y), original = result
        return leave(original, final), y

    return Mealy(ParaLens(autodiff(prepare) >> local.first(Proxy()).arrow >> autodiff(finish)))


def group_optimizer(algorithm: Algorithm) -> Optimizer[Tree, Tree, Tree]:
    make = algorithm.make
    return optimizer(
        algorithm.targets,
        algorithm.controls,
        optimizers.optimizer(make, optimizers.additive),
        init.Apply(lambda values: optimizers.init_optimizer(make, values[0], values[1])),
    )


def pack_groups(values: tuple[Tree, ...]) -> Tree:
    return reduce(lambda left, right: (left, right), values)


def unpack_groups(value: Tree, size: int) -> tuple[Tree, ...]:
    if size == 1:
        return (value,)
    left, right = value
    return unpack_groups(left, size - 1) + (right,)


@overload
def lower(
    term: s.Learn, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    algorithms = catalog.algorithms[term.optimizer]
    opt = reduce(lambda left, right: left @ right, tuple(group_optimizer(a) for a in algorithms))
    if overlap := written(term.body, catalog, stack) & set(opt.targets):
        raise ValueError(f"objective also writes its optimizer targets: {overlap}; initialize a separate inner copy")
    selected = select(request(exposed, opt.targets))
    partition, assemble = parse_selection(selected)
    body = lower(term.body, catalog, exposed, inputs, stack)

    def put_targets(p_rx):
        (targets, _), ((key, rest), x) = p_rx
        return (), ((key, assemble.get((targets, rest))), x)

    def keep_rest(sy):
        (key, values), y = sy
        _, rest = partition.get(values)
        return (key, rest), y

    objective = Mealy(ParaLens(autodiff(put_targets) >> body.step.arrow >> autodiff(keep_rest)))
    learning = combinators.learn(opt.rule, objective.post(learning_rate_log))
    controls = value_focus(exposed, opt.controls.refs)
    memories = tuple(value_focus(exposed, (Ref(f"{a.name}.memory"),)) for a in algorithms)
    targets = tuple(value_focus(exposed, a.targets) for a in algorithms)
    count = len(algorithms)

    def step(unit: Unit, state: Frame, x: Tree):
        key, values = state
        _, rest = partition.get(values)
        groups = tuple(
            (memory.get(state)[0], target.get(state)) for memory, target in zip(memories, targets, strict=True)
        )
        (updated, (key, rest)), y = learning.step.arrow.get(
            ((controls.get(state), ()), ((pack_groups(groups), (key, rest)), x))
        )
        groups = unpack_groups(updated, count)
        next_targets = tuple(value for memory, group_targets in groups for value in group_targets)
        state = key, assemble.get((next_targets, rest))
        for at, (memory, group_targets) in zip(memories, groups, strict=True):
            _, replace = at.run(state)
            state = replace((memory,))
        return state, y

    return mealy_autodiff(step)


@overload
def lower(
    term: s.Term, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]:
    raise TypeError(f"no compiler for {type(term).__name__}")


@dispatch
def lower(
    term: s.Term, catalog: Catalog, exposed: Layout, inputs: Shape, stack: tuple[str, ...]
) -> Mealy[Unit, Unit, Frame, Frame, Tree, Tree, Tree, Tree]: ...


@dataclass(frozen=True)
class Executable[X, Y]:
    initial: Callable[[PRNG, X], Frame]
    body: Cell[X, Y]
    exposed: Layout
    aliases: tuple[tuple[Ref, Ref], ...]

    def start(self, key: PRNG, inputs: X) -> Frame:
        """Enter the root scope once, without executing its body."""
        return self.initial(key, inputs)

    def step(self, state: Frame, inputs: X) -> tuple[Frame, Y]:
        """Execute the root body, carrying explicitly supplied state."""
        return self.body.step.arrow.get(((), (state, inputs)))

    def run(self, key: PRNG, inputs: X) -> Y:
        """One fresh invocation, including root initialization."""
        _, result = self.step(self.start(key, inputs), inputs)
        return result

    def binding(self, name: str) -> Focus[Frame, Tree]:
        """Resolve once outside a jitted function; use the returned optic inside."""
        ref = Ref(name)
        ref = next((target for alias, target in self.aliases if alias == ref), ref)
        at = value_focus(self.exposed, (ref,))
        return at >> focus(lambda values: values[0], lambda values, value: (value,))


def compile[X](config: s.Config, inputs: X) -> Executable[X, Tree]:
    catalog = Catalog(config)
    shape = abstract(inputs)
    if missing := required(config.run, catalog, (), ()):
        raise ValueError(f"bindings have no enclosing initialization scope: {missing}")
    infer(config.run, catalog, shape, ())
    empty = layout(())
    root, stack = config.run, ()
    while isinstance(root, s.Call):
        root, stack = called(root, catalog, stack)
    if isinstance(root, s.Scope):
        exposed, enter, _ = enter_scope(root, catalog, empty, shape)
        body = lower(root.body, catalog, exposed, shape, stack)

        def initial(key: PRNG, inputs: Tree) -> Frame:
            return enter((key, ()), inputs)
    else:
        exposed = empty
        body = lower(root, catalog, empty, shape, stack)

        def initial(key: PRNG, inputs: Tree) -> Frame:
            return key, ()

    return Executable(initial, body, exposed, tuple((ref, catalog.canonical(ref)) for ref in catalog.aliases))
