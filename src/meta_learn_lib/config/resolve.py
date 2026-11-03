"""Compile-time interpretation of declarations, references, effects, and shapes.

Catalog is discarded after compilation. Runtime values are addressed by lenses.
Each interpretation is open to additional terms through Plum.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from plum import dispatch

from ..algorithms import optimizers
from ..lib_types import PRNG
from . import initialization as init
from . import terms as s
from .bindings import Ref, Refs

type Shape = PyTree[jax.ShapeDtypeStruct]
type Values = tuple[PyTree[jax.Array], ...]


@dataclass(frozen=True)
class Algorithm:
    name: str
    targets: Refs
    controls: Refs
    make: Callable[[Values], optax.GradientTransformation]


class Catalog:
    def __init__(self, config: s.Config):
        self.config = config
        self.aliases: dict[Ref, Ref] = {}
        self.recipes: dict[Ref, s.Recipe] = {}
        self.shapes: dict[Ref, Shape] = {}
        self.components: dict[int, Refs] = {}
        self.changes: dict[int, Refs] = {}
        self.groups: dict[Ref, Refs] = {}
        self.constraints: list[tuple[Ref, Shape]] = []
        self.algorithms: dict[str, tuple[Algorithm, ...]] = {}
        self.memories: dict[Ref, Algorithm] = {}
        self.resolving_shapes: set[Ref] = set()
        for name, term in config.computations.items():
            collect(term, name, self)
        collect(config.run, "$run", self)
        for name, groups in config.optimizers.items():
            for group in groups:
                for field, value in controls(group.rule):
                    if not isinstance(value, (float, int)):
                        self.declare(f"{name}.{group.name}.{field}", value, jax.ShapeDtypeStruct((), jnp.float32))
        collected = set()
        while pending := tuple(ref for ref in self.recipes if ref not in collected):
            for ref in pending:
                collected.add(ref)
                collect_recipe(self.recipes[ref], f"{ref.name}.initial", self)
        for ref, shape in self.constraints:
            self.constrain(ref, shape)
        for name, groups in config.optimizers.items():
            if not groups or len({g.name for g in groups}) != len(groups):
                raise ValueError(f"optimizer {name} needs nonempty, uniquely named groups")
            algorithms = tuple(self.algorithm(f"{name}.{g.name}", g) for g in groups)
            targets = tuple(ref for algorithm in algorithms for ref in algorithm.targets)
            if len(set(targets)) != len(targets):
                raise ValueError(f"optimizer {name} selects overlapping targets (including aliases)")
            self.algorithms[name] = algorithms
            for algorithm in algorithms:
                self.memories[Ref(f"{algorithm.name}.memory")] = algorithm

    def declare(self, name: str, value: s.Slot, shape: Shape | None) -> Ref:
        ref = Ref(name)
        if ref in self.recipes or ref in self.aliases:
            raise ValueError(f"duplicate binding: {name}")
        if isinstance(value, Ref):
            self.aliases[ref] = value
        else:
            self.recipes[ref] = value
        if shape is not None:
            self.constraints.append((ref, shape))
        return ref

    def canonical(self, ref: Ref) -> Ref:
        seen = set()
        while ref in self.aliases:
            if ref in seen:
                raise ValueError(f"cyclic aliases at {ref.name}")
            seen.add(ref)
            ref = self.aliases[ref]
        return ref

    def constrain(self, ref: Ref, shape: Shape) -> None:
        ref = self.canonical(ref)
        if ref not in self.recipes:
            raise ValueError(f"unknown binding: {ref.name}")
        if ref in self.shapes and not eqx.tree_equal(self.shapes[ref], shape):
            raise ValueError(f"incompatible shapes for shared binding {ref.name}")
        self.shapes[ref] = shape

    def expand(self, refs: Refs) -> Refs:
        result = []
        for ref in refs:
            for member in self.groups.get(ref, (ref,)):
                canonical = self.canonical(member)
                if canonical not in self.recipes and canonical not in self.memories:
                    raise ValueError(f"unknown binding: {member.name}")
                result.append(canonical)
        return tuple(result)

    def algorithm(self, name: str, group: s.Group) -> Algorithm:
        refs, getters = [], []
        for field, value in controls(group.rule):
            if isinstance(value, (float, int)):
                getters.append(constant_getter(jnp.asarray(value)))
            else:
                ref = self.canonical(Ref(f"{name}.{field}"))
                if ref not in refs:
                    refs.append(ref)
                getters.append(position_getter(refs.index(ref)))
        selections, rule = tuple(getters), group.rule

        def make(values: Values):
            return transformation(rule, tuple(get(values) for get in selections))

        return Algorithm(name, self.expand(group.targets), tuple(refs), make)


def position_getter[T](position: int) -> Callable[[tuple[T, ...]], T]:
    return lambda values: values[position]


def constant_getter[T](value: T) -> Callable[[Values], T]:
    return lambda values: value


@overload
def controls(rule: s.Sgd | s.Adam) -> tuple[tuple[str, s.Control], ...]:
    return (
        ("learning_rate", rule.learning_rate),
        ("weight_decay", rule.weight_decay),
        ("momentum", rule.momentum),
    )


@overload
def controls(rule: s.Frozen) -> tuple[tuple[str, s.Control], ...]:
    return ()


@overload
def controls(rule: s.Rule) -> tuple[tuple[str, s.Control], ...]:
    raise TypeError(f"no controls for {type(rule).__name__}")


@dispatch
def controls(rule: s.Rule) -> tuple[tuple[str, s.Control], ...]: ...


@overload
def transformation(rule: s.Sgd, values: Values) -> optax.GradientTransformation:
    rate, decay, momentum = values
    return optimizers.sgd(rate, decay, momentum)


@overload
def transformation(rule: s.SgdNormalized, values: Values) -> optax.GradientTransformation:
    rate, decay, momentum = values
    return optimizers.sgd_normalized(rate, decay, momentum)


@overload
def transformation(rule: s.Adam, values: Values) -> optax.GradientTransformation:
    rate, decay, momentum = values
    return optimizers.adam(rate, decay, momentum, rule.b2, rule.eps, rule.eps_root)


@overload
def transformation(rule: s.Frozen, values: Values) -> optax.GradientTransformation:
    return optimizers.frozen()


@overload
def transformation(rule: s.Rule, values: Values) -> optax.GradientTransformation:
    raise TypeError(f"no optimizer for {type(rule).__name__}")


@dispatch
def transformation(rule: s.Rule, values: Values) -> optax.GradientTransformation: ...


@overload
def children(term: s.Leaf) -> tuple[tuple[str, s.Term], ...]:
    return ()


@overload
def children(term: s.Unary) -> tuple[tuple[str, s.Term], ...]:
    return (("body", term.body),)


@overload
def children(term: s.Sequence) -> tuple[tuple[str, s.Term], ...]:
    return tuple((str(i), step.body) for i, step in enumerate(term.steps))


@overload
def children(term: s.Term) -> tuple[tuple[str, s.Term], ...]:
    raise TypeError(f"no children interpretation for {type(term).__name__}")


@dispatch
def children(term: s.Term) -> tuple[tuple[str, s.Term], ...]: ...


@overload
def collect(term: s.Value, name: str, catalog: Catalog) -> None:
    if id(term) not in catalog.components:
        catalog.components[id(term)] = (catalog.declare(name, term.initial, None),)


@overload
def collect(term: s.Linear | s.Bias, name: str, catalog: Catalog) -> None:
    if id(term) not in catalog.components:
        catalog.components[id(term)] = (catalog.declare(f"{name}.parameters", term.parameters, None),)


@overload
def collect(term: s.Rnn, name: str, catalog: Catalog) -> None:
    if id(term) in catalog.components:
        return
    recurrent = catalog.declare(f"{name}.recurrent", term.recurrent, None)
    incoming = catalog.declare(f"{name}.incoming", term.incoming, None)
    state = catalog.declare(f"{name}.state", term.state, jax.ShapeDtypeStruct((term.n,), jnp.float32))
    refs = (recurrent, incoming, state)
    if not isinstance(term.time_constant, (float, int)):
        refs += (catalog.declare(f"{name}.time_constant", term.time_constant, jax.ShapeDtypeStruct((), jnp.float32)),)
    catalog.components[id(term)] = refs
    catalog.changes[id(term)] = (state,)
    catalog.groups[Ref(f"{name}.parameters")] = (recurrent, incoming)


@overload
def collect(term: s.Term, name: str, catalog: Catalog) -> None:
    for field, child in children(term):
        collect(child, f"{name}.{field}", catalog)


@dispatch
def collect(term: s.Term, name: str, catalog: Catalog) -> None: ...


@overload
def collect_recipe(recipe: s.Computed, name: str, catalog: Catalog) -> None:
    collect(recipe.body, name, catalog)


@overload
def collect_recipe(recipe: init.Init | init.Initialization, name: str, catalog: Catalog) -> None:
    pass


@dispatch
def collect_recipe(recipe: init.Init | init.Initialization, name: str, catalog: Catalog) -> None: ...


def called(term: s.Call, catalog: Catalog, stack: tuple[str, ...]) -> tuple[s.Term, tuple[str, ...]]:
    if term.name in stack:
        raise ValueError(f"recursive computation references: {stack + (term.name,)}")
    if term.name not in catalog.config.computations:
        raise ValueError(f"unknown computation: {term.name}")
    return catalog.config.computations[term.name], stack + (term.name,)


def component_refs(term: s.Primitive, catalog: Catalog) -> Refs:
    return tuple(catalog.canonical(ref) for ref in catalog.components[id(term)])


def dependencies(ref: Ref, catalog: Catalog, stack: tuple[Ref, ...]) -> set[Ref]:
    if ref in stack:
        raise ValueError(f"cyclic initialization dependencies: {stack + (ref,)}")
    if ref in catalog.memories:
        algorithm = catalog.memories[ref]
        return set(algorithm.targets + algorithm.controls)
    return recipe_dependencies(catalog.recipes[ref], catalog, stack + (ref,))


@overload
def recipe_dependencies(recipe: s.Computed, catalog: Catalog, stack: tuple[Ref, ...]) -> set[Ref]:
    return required(recipe.body, catalog, (), stack)


@overload
def recipe_dependencies(recipe: init.Init | init.Initialization, catalog: Catalog, stack: tuple[Ref, ...]) -> set[Ref]:
    return set()


@dispatch
def recipe_dependencies(
    recipe: init.Init | init.Initialization, catalog: Catalog, stack: tuple[Ref, ...]
) -> set[Ref]: ...


@overload
def required(term: s.Call, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    body, stack = called(term, catalog, stack)
    return required(body, catalog, stack, initializing)


@overload
def required(term: s.Read, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    return set(catalog.expand((term.ref,)))


@overload
def required(term: s.Primitive, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    return set(component_refs(term, catalog))


@overload
def required(term: s.Scope, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    local = set(catalog.expand(term.initialize))
    needed = required(term.body, catalog, stack, initializing)
    for ref in local:
        needed |= dependencies(ref, catalog, initializing)
    return needed - local


@overload
def required(term: s.Learn, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    if term.optimizer not in catalog.algorithms:
        raise ValueError(f"unknown optimizer: {term.optimizer}")
    needed = required(term.body, catalog, stack, initializing)
    for algorithm in catalog.algorithms[term.optimizer]:
        needed |= set(algorithm.targets + algorithm.controls + (Ref(f"{algorithm.name}.memory"),))
    return needed


@overload
def required(term: s.Term, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]:
    result = set()
    for _, child in children(term):
        result |= required(child, catalog, stack, initializing)
    return result


@dispatch
def required(term: s.Term, catalog: Catalog, stack: tuple[str, ...], initializing: tuple[Ref, ...]) -> set[Ref]: ...


@overload
def written(term: s.Call, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    body, stack = called(term, catalog, stack)
    return written(body, catalog, stack)


@overload
def written(term: s.Primitive, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    return {catalog.canonical(ref) for ref in catalog.changes.get(id(term), ())}


@overload
def written(term: s.Scope, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    result = written(term.body, catalog, stack)
    for ref in catalog.expand(term.initialize):
        if ref in catalog.recipes:
            result |= recipe_written(catalog.recipes[ref], catalog, stack)
    return result - set(catalog.expand(term.initialize))


@overload
def written(term: s.Learn, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    result = written(term.body, catalog, stack)
    for algorithm in catalog.algorithms[term.optimizer]:
        result |= set(algorithm.targets + (Ref(f"{algorithm.name}.memory"),))
    return result


@overload
def written(term: s.Term, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    result = set()
    for _, child in children(term):
        result |= written(child, catalog, stack)
    return result


@dispatch
def written(term: s.Term, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]: ...


@overload
def recipe_written(recipe: s.Computed, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    return written(recipe.body, catalog, stack)


@overload
def recipe_written(recipe: init.Init | init.Initialization, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]:
    return set()


@dispatch
def recipe_written(recipe: init.Init | init.Initialization, catalog: Catalog, stack: tuple[str, ...]) -> set[Ref]: ...


def field[T](name: str) -> Callable[[dict[str, T]], T]:
    return lambda data: data[name]


def source(ref: Ref | None, original: Shape, previous: Shape, names: tuple[str, ...], results: tuple[Shape, ...]):
    """Resolve a data reference to a getter on (input, previous, named results)."""
    if ref is None:
        return (lambda context: context[1]), previous
    root, *path = ref.name.split(".")
    if root == "input":
        get = lambda context: context[0]
        shape = original
    elif root in names:
        index = names.index(root)
        get = lambda context: context[2][index]
        shape = results[index]
    else:
        raise ValueError(f"unknown data reference: {ref.name}")
    getters = []
    for name in path:
        if not isinstance(shape, dict) or name not in shape:
            raise ValueError(f"unknown data field: {ref.name}")
        getters.append(field(name))
        shape = shape[name]
    accessors = tuple(getters)

    def select(context):
        value = get(context)
        for accessor in accessors:
            value = accessor(value)
        return value

    return select, shape


def unstack(shape: Shape, size: int) -> Shape:
    def remove_axis(array: jax.ShapeDtypeStruct):
        if not array.shape or array.shape[0] != size:
            raise ValueError(f"expected a leading axis of length {size}, received {array.shape}")
        return jax.ShapeDtypeStruct(array.shape[1:], array.dtype)

    return jax.tree.map(remove_axis, shape)


@overload
def activation(term: s.Tanh) -> Callable[[jax.Array], jax.Array]:
    return jnp.tanh


@overload
def activation(term: s.Relu) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.relu


@overload
def activation(term: s.Sigmoid) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.sigmoid


@overload
def activation(term: s.Softmax) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.softmax


@overload
def activation(term: s.Identity) -> Callable[[jax.Array], jax.Array]:
    return lambda x: x


@overload
def activation(term: s.Activation) -> Callable[[jax.Array], jax.Array]:
    raise TypeError(f"no activation for {type(term).__name__}")


@dispatch
def activation(term: s.Activation) -> Callable[[jax.Array], jax.Array]: ...


@overload
def reduction(term: s.Mean) -> Callable[[jax.Array], jax.Array]:
    return jnp.mean


@overload
def reduction(term: s.Sum) -> Callable[[jax.Array], jax.Array]:
    return jnp.sum


@overload
def reduction(term: s.Reduction) -> Callable[[jax.Array], jax.Array]:
    raise TypeError(f"no reduction for {type(term).__name__}")


@dispatch
def reduction(term: s.Reduction) -> Callable[[jax.Array], jax.Array]: ...


@overload
def loss(term: s.L2) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return optax.l2_loss


@overload
def loss(term: s.CrossEntropy) -> Callable[[jax.Array, jax.Array], jax.Array]:
    return optax.softmax_cross_entropy_with_integer_labels


@overload
def loss(term: s.Loss) -> Callable[[jax.Array, jax.Array], jax.Array]:
    raise TypeError(f"no loss for {type(term).__name__}")


@dispatch
def loss(term: s.Loss) -> Callable[[jax.Array, jax.Array], jax.Array]: ...


@overload
def check_target(term: s.L2, prediction: Shape, target: Shape) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"target shape {target.shape} does not match prediction {prediction.shape}")


@overload
def check_target(term: s.CrossEntropy, prediction: Shape, target: Shape) -> None:
    if target.shape != prediction.shape[:-1] or not jnp.issubdtype(target.dtype, jnp.integer):
        raise ValueError("cross entropy needs integer labels with the prediction's class axis removed")


@overload
def check_target(term: s.Loss, prediction: Shape, target: Shape) -> None:
    # A new loss can rely on its own abstract execution or add a stricter rule.
    pass


@dispatch
def check_target(term: s.Loss, prediction: Shape, target: Shape) -> None:
    # A new loss can rely on its own abstract execution or add a stricter rule.
    ...


def initialized_shape(description: init.Initialization) -> Shape:
    return eqx.filter_eval_shape(init.compile_initial(description), (), PRNG(jax.random.key(0)))


def module_shape(ref: Ref, inputs: int, outputs: int, dtype: str, catalog: Catalog) -> None:
    recipe = catalog.recipes[ref]
    shape = initialized_shape(init.LinearModule(inputs, outputs, dtype, init.Zeros()))
    catalog.constrain(ref, shape)
    if isinstance(recipe, init.Init):
        catalog.recipes[ref] = init.LinearModule(inputs, outputs, dtype, recipe)


def binding_shape(ref: Ref, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    if ref not in catalog.shapes:
        if ref in catalog.resolving_shapes:
            raise ValueError(f"cyclic initialization dependencies at {ref.name}")
        catalog.resolving_shapes.add(ref)
        catalog.constrain(ref, recipe_shape(catalog.recipes[ref], catalog, inputs, stack))
        catalog.resolving_shapes.remove(ref)
    return catalog.shapes[ref]


@overload
def recipe_shape(recipe: s.Computed, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return infer(recipe.body, catalog, inputs, stack)


@overload
def recipe_shape(recipe: init.Initialization, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return initialized_shape(recipe)


@overload
def recipe_shape(recipe: init.Constant, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return jax.ShapeDtypeStruct((), jnp.float32)


@overload
def recipe_shape(recipe: init.Init, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    raise ValueError("initializer needs a shape from its component or an explicit Array description")


@dispatch
def recipe_shape(recipe: init.Init, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape: ...


@overload
def infer(term: s.Call, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    body, stack = called(term, catalog, stack)
    return infer(body, catalog, inputs, stack)


@overload
def infer(term: s.Read, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    (ref,) = catalog.expand((term.ref,))
    return binding_shape(ref, catalog, inputs, stack)


@overload
def infer(term: s.Value, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    (ref,) = component_refs(term, catalog)
    return binding_shape(ref, catalog, inputs, stack)


def vector(inputs: Shape) -> jax.ShapeDtypeStruct:
    if not isinstance(inputs, jax.ShapeDtypeStruct) or len(inputs.shape) != 1:
        raise ValueError("component takes one vector; enclose it in scan or batch for extra axes")
    return inputs


@overload
def infer(term: s.Linear, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    inputs = vector(inputs)
    (ref,) = component_refs(term, catalog)
    module_shape(ref, inputs.shape[0], term.n, str(inputs.dtype), catalog)
    return jax.ShapeDtypeStruct((term.n,), inputs.dtype)


@overload
def infer(term: s.Bias, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    (ref,) = component_refs(term, catalog)
    catalog.constrain(ref, vector(inputs))
    return inputs


@overload
def infer(term: s.Rnn, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    inputs = vector(inputs)
    recurrent, incoming, state, *_ = component_refs(term, catalog)
    module_shape(recurrent, term.n, term.n, str(inputs.dtype), catalog)
    module_shape(incoming, inputs.shape[0], term.n, str(inputs.dtype), catalog)
    shape = jax.ShapeDtypeStruct((term.n,), inputs.dtype)
    catalog.constrain(state, shape)
    return shape


@overload
def infer(term: s.Activation, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return eqx.filter_eval_shape(activation(term), inputs)


@overload
def infer(term: s.Reduce, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return eqx.filter_eval_shape(reduction(term.operation), infer(term.body, catalog, inputs, stack))


@overload
def infer(term: s.Sequence, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    names, results, previous = (), (), inputs
    for step in term.steps:
        _, shape = source(step.input, inputs, previous, names, results)
        previous = infer(step.body, catalog, shape, stack)
        if step.name is not None:
            if step.name == "input" or step.name in names or "." in step.name:
                raise ValueError(f"duplicate or reserved result name: {step.name}")
            names, results = names + (step.name,), results + (previous,)
    return previous


def infer_mapped(term: s.Unary, size: int, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    if size <= 0:
        raise ValueError("scan/batch size must be positive")
    result = infer(term.body, catalog, unstack(inputs, size), stack)
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct((size,) + a.shape, a.dtype), result)


@overload
def infer(term: s.Scan, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return infer_mapped(term, term.steps, catalog, inputs, stack)


@overload
def infer(term: s.Batch, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return infer_mapped(term, term.size, catalog, inputs, stack)


@overload
def infer(term: s.Supervised, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    if not isinstance(inputs, dict) or set(inputs) != {"features", "target"}:
        raise ValueError("supervised takes {features: ..., target: ...}")
    prediction = infer(term.body, catalog, inputs["features"], stack)
    check_target(term.loss, prediction, inputs["target"])
    losses = eqx.filter_eval_shape(loss(term.loss), prediction, inputs["target"])
    return eqx.filter_eval_shape(reduction(term.reduction), losses)


@overload
def infer(term: s.Scope, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    return infer(term.body, catalog, inputs, stack)


@overload
def infer(term: s.Learn, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    result = infer(term.body, catalog, inputs, stack)
    if not isinstance(result, jax.ShapeDtypeStruct) or result.shape != ():
        raise ValueError("learn needs a scalar objective; reduce its losses explicitly")
    return result


@overload
def infer(term: s.Term, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape:
    raise TypeError(f"no shape inference for {type(term).__name__}")


@dispatch
def infer(term: s.Term, catalog: Catalog, inputs: Shape, stack: tuple[str, ...]) -> Shape: ...
