from meta_learn_lib.algorithms.combinators import learning_rate_log
from meta_learn_lib.algorithms.optimizers import additive, frozen, optimizer
from meta_learn_lib.category.lens import Lens
from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.named import Named, Running, leaf
from meta_learn_lib.category.paralens import ParaLens, para_autodiff
from meta_learn_lib.construct import config
from meta_learn_lib.construct.creating import Creating, OptState, SideBySide, Size, Starts, Unknown, Unsaid, Value
from meta_learn_lib.construct.leaves import activation, objective, transformation
from meta_learn_lib.construct.ports import Offset, Rates, RnnPort, Target, Weights
from meta_learn_lib.lib_types import LOSS, ArrayTree

from typing import Callable, overload
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from plum import dispatch

Loose = Named[PyTree, PyTree, PyTree, PyTree, PyTree, PyTree]

# ── a piece, running: the numbers ─────────────────────────────────────────────


@overload
def piece(
    r: Running, p: config.Linear, names: tuple[str, ...]
) -> Named[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def run(port: Weights[jax.Array], x: jax.Array) -> jax.Array:
        return port.weights @ x

    return leaf(r, Weights, names, para_autodiff(run))


@overload
def piece(
    r: Running, p: config.Bias, names: tuple[str, ...]
) -> Named[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def run(port: Offset[jax.Array], x: jax.Array) -> jax.Array:
        return x + port.bias

    return leaf(r, Offset, names, para_autodiff(run))


@overload
def piece(
    r: Running,
    p: config.Tanh | config.Relu | config.Sigmoid | config.Softmax | config.Identity,
    names: tuple[str, ...],
) -> Named[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    f = activation(p)

    def run(port: Unit, x: jax.Array) -> jax.Array:
        return f(x)

    return leaf(r, Unit, names, para_autodiff(run))


@overload
def piece(
    r: Running, p: config.Rnn, names: tuple[str, ...]
) -> Named[
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]:
    f = activation(p.act)

    def run(port: RnnPort[jax.Array], xh: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x, h = xh
        h1 = (1 - port.time_constant) * h + port.time_constant * f(port.weights @ jnp.concatenate([h, x]))
        return h1, h1

    return leaf(r, RnnPort, names, para_autodiff(run))


@overload
def piece(
    r: Running, p: config.L2 | config.CrossEntropy, names: tuple[str, ...]
) -> Named[jax.Array, jax.Array, jax.Array, jax.Array, LOSS, LOSS]:
    f = objective(p)

    def run(port: Target[jax.Array], guess: jax.Array) -> LOSS:
        return LOSS(f(guess, port.target))

    return leaf(r, Target, names, para_autodiff(run))


@overload
def piece[T](
    r: Running, p: config.Sgd | config.SgdNormalized | config.Adam, names: tuple[str, ...]
) -> Named[jax.Array, jax.Array, tuple[ArrayTree, T], tuple[ArrayTree, T], T, T]:
    return leaf(r, Rates, names, optimizer(transformation(p), additive))


@overload
def piece[T](
    r: Running, p: config.Frozen, names: tuple[str, ...]
) -> Named[jax.Array, jax.Array, tuple[ArrayTree, T], tuple[ArrayTree, T], T, T]:
    def make(u: Unit) -> optax.GradientTransformation:
        return frozen()

    return leaf(r, Unit, names, optimizer(make, additive))


# ── a piece, creating: sizes forwards, start values back ──────────────────────


@overload
def piece(r: Creating, p: config.Linear, names: tuple[str, ...]) -> Loose:
    def run(port_fed: tuple[Weights[Unknown], Size]) -> tuple[Size, Callable[[PyTree], tuple[Weights[Starts], Unsaid]]]:
        _, fed = port_fed
        return Size((p.n,)), lambda _: (Weights(Starts(p.init, (p.n, fed.dims[-1]))), Unsaid())

    return leaf(r, Weights, names, ParaLens(Lens(run)))


@overload
def piece(r: Creating, p: config.Bias, names: tuple[str, ...]) -> Loose:
    def run(port_fed: tuple[Offset[Unknown], Size]) -> tuple[Size, Callable[[PyTree], tuple[Offset[Starts], Unsaid]]]:
        _, fed = port_fed
        return fed, lambda _: (Offset(Starts(p.init, fed.dims)), Unsaid())

    return leaf(r, Offset, names, ParaLens(Lens(run)))


@overload
def piece(
    r: Creating,
    p: config.Tanh | config.Relu | config.Sigmoid | config.Softmax | config.Identity,
    names: tuple[str, ...],
) -> Loose:
    def run(port_fed: tuple[Unit, Size]) -> tuple[Size, Callable[[PyTree], tuple[Unit, Unsaid]]]:
        _, fed = port_fed
        return fed, lambda _: (Unit(), Unsaid())

    return leaf(r, Unit, names, ParaLens(Lens(run)))


@overload
def piece(r: Creating, p: config.Rnn, names: tuple[str, ...]) -> Loose:
    def run(
        port_fed: tuple[RnnPort[Unknown], tuple[Size, PyTree]],
    ) -> tuple[tuple[Size, Size], Callable[[PyTree], tuple[RnnPort[SideBySide | Value], tuple[Unsaid, Starts]]]]:
        _, (x, _) = port_fed
        weights = SideBySide(Starts(config.Orthogonal(), (p.n, p.n)), Starts(p.init, (p.n, x.dims[-1])))
        return (Size((p.n,)), Size((p.n,))), lambda _: (
            RnnPort(weights, Value(p.time_constant)),
            (Unsaid(), Starts(p.h0, (p.n,))),
        )

    return leaf(r, RnnPort, names, ParaLens(Lens(run)))


@overload
def piece(r: Creating, p: config.L2 | config.CrossEntropy, names: tuple[str, ...]) -> Loose:
    def run(port_fed: tuple[Target[PyTree], PyTree]) -> tuple[Size, Callable[[PyTree], tuple[Target[Unsaid], Unsaid]]]:
        return Size(()), lambda _: (Target(Unsaid()), Unsaid())

    return leaf(r, Target, names, ParaLens(Lens(run)))


@overload
def piece(
    r: Creating, p: config.Sgd | config.SgdNormalized | config.Adam, names: tuple[str, ...]
) -> Loose:
    make = transformation(p)
    rates = Rates(Value(p.learning_rate), Value(p.weight_decay), Value(p.momentum))

    def run(
        port_fed: tuple[Rates[PyTree], PyTree],
    ) -> tuple[PyTree, Callable[[PyTree], tuple[Rates[Value], tuple[OptState, PyTree]]]]:
        _, (_, theta) = port_fed
        return theta, lambda theta_start: (rates, (OptState(make, rates, theta_start), theta_start))

    return leaf(r, Rates, names, ParaLens(Lens(run)))


@overload
def piece(r: Creating, p: config.Frozen, names: tuple[str, ...]) -> Loose:
    def run(port_fed: tuple[Unit, PyTree]) -> tuple[PyTree, Callable[[PyTree], tuple[Unit, tuple[OptState, PyTree]]]]:
        _, (_, theta) = port_fed
        return theta, lambda theta_start: (Unit(), (OptState(lambda u: frozen(), Unit(), theta_start), theta_start))

    return leaf(r, Unit, names, ParaLens(Lens(run)))


@dispatch
def piece(r: Running | Creating, p: config.Piece, names: tuple[str, ...]) -> Loose:
    raise NotImplementedError


# ── the seed after a loss: the loss goes on, and the backward pass starts here ─


@overload
def seed(r: Running) -> Named[jax.Array, jax.Array, LOSS, LOSS, LOSS, LOSS]:
    return leaf(r, Unit, (), learning_rate_log)


@overload
def seed(r: Creating) -> Loose:
    def run(port_fed: tuple[Unit, PyTree]) -> tuple[PyTree, Callable[[PyTree], tuple[Unit, Unsaid]]]:
        _, fed = port_fed
        return fed, lambda _: (Unit(), Unsaid())

    return leaf(r, Unit, (), ParaLens(Lens(run)))


@dispatch
def seed(r: Running | Creating) -> Loose:
    raise NotImplementedError
