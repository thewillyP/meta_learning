from meta_learn_lib.algorithms import combinators, learning
from meta_learn_lib.category.lens import Proxy, swap
from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.named import (
    Running,
    copies,
    first,
    fresh,
    join,
    learned,
    online,
    online_with,
    post,
    pre,
    second,
)
from meta_learn_lib.category.paralens import ParaLens
from meta_learn_lib.construct import config, creating
from meta_learn_lib.construct.creating import Creating, Key, Value, ZeroRows, ZerosFlat
from meta_learn_lib.construct.leaves import sampler
from meta_learn_lib.construct.pieces import Loose, piece, seed
from meta_learn_lib.construct.wires import rewired

from typing import overload
from jaxtyping import PyTree
from plum import dispatch

Positional = ParaLens[PyTree, PyTree, PyTree, PyTree, PyTree, PyTree]

# ── the operations that differ between the readings ───────────────────────────


@overload
def wrapped(r: Running, c: config.MapAccum, below: Positional) -> Positional:
    return combinators.mapAccum(below)


@overload
def wrapped(r: Creating, c: config.MapAccum, below: Positional) -> Positional:
    return creating.map_accum(below)


@overload
def wrapped(r: Running, c: config.Batch, below: Positional) -> Positional:
    return combinators.batch(below)


@overload
def wrapped(r: Creating, c: config.Batch, below: Positional) -> Positional:
    return creating.batch(c.n, below)


@overload
def wrapped(r: Running, c: config.Rtrl, below: Positional) -> Positional:
    return learning.rtrl(below)


@overload
def wrapped(r: Creating, c: config.Rtrl, below: Positional) -> Positional:
    return creating.online(below, Unit(), ZeroRows)


@overload
def wrapped(r: Running, c: config.Rflo, below: Positional) -> Positional:
    return learning.rflo(below, lambda decay: decay)


@overload
def wrapped(r: Creating, c: config.Rflo, below: Positional) -> Positional:
    return creating.online(below, Value(c.decay_start), ZeroRows)


@overload
def wrapped(r: Running, c: config.Uoro, below: Positional) -> Positional:
    return learning.uoro(below, sampler(c.noise))


@overload
def wrapped(r: Creating, c: config.Uoro, below: Positional) -> Positional:
    return creating.online(below, Unit(), lambda s, theta: (ZerosFlat(s), ZerosFlat(theta), Key()))


@dispatch
def wrapped(
    r: Running | Creating,
    c: config.MapAccum | config.Batch | config.Rtrl | config.Rflo | config.Uoro,
    below: Positional,
) -> Positional:
    raise NotImplementedError


@overload
def taught(r: Running, opt: Positional, body: Positional) -> Positional:
    return combinators.learn(opt, body)


@overload
def taught(r: Creating, opt: Positional, body: Positional) -> Positional:
    return creating.put(combinators.optimised(opt, body))


@dispatch
def taught(r: Running | Creating, opt: Positional, body: Positional) -> Positional:
    raise NotImplementedError


# ── the conversion: one clause per node of config.py ──────────────────────────


@overload
def convert(r: Running | Creating, c: config.Leaf) -> Loose:
    return piece(r, c.piece, c.names)


@overload
def convert(r: Running | Creating, c: config.LearningRateLog) -> Loose:
    return seed(r)


@overload
def convert(r: Running | Creating, c: config.Then) -> Loose:
    return convert(r, c.first) >> convert(r, c.second)


@overload
def convert(r: Running | Creating, c: config.Beside) -> Loose:
    return convert(r, c.first) @ convert(r, c.second)


@overload
def convert(r: Running | Creating, c: config.Join) -> Loose:
    return join(c.name, convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.MapAccum | config.Batch) -> Loose:
    return copies(lambda cell: wrapped(r, c, cell), c.over, convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Learn) -> Loose:
    return learned(lambda opt, body: taught(r, opt, body), convert(r, c.opt), c.names, convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Rtrl | config.Uoro) -> Loose:
    return online(lambda model: wrapped(r, c, model), c.names, convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Rflo) -> Loose:
    return online_with(lambda model: wrapped(r, c, model), c.decay, c.names, convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Fresh) -> Loose:
    return fresh(c.name, convert(r, c.below))


# ── the shorter spellings ─────────────────────────────────────────────────────


@overload
def convert(r: Running | Creating, c: config.First) -> Loose:
    return first(convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Second) -> Loose:
    return second(convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Pre) -> Loose:
    return pre(rewired(c.before, c.after), convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Post) -> Loose:
    return post(rewired(c.before, c.after), convert(r, c.below))


@overload
def convert(r: Running | Creating, c: config.Input) -> Loose:
    return post(swap(Proxy[tuple[PyTree, PyTree, PyTree, PyTree]]()), join(c.name, convert(r, c.below)))


@overload
def convert(r: Running | Creating, c: config.Scored) -> Loose:
    return convert(r, config.Then(c.model, config.Second(config.Then(c.loss, config.LearningRateLog()))))


@dispatch
def convert(r: Running | Creating, c: config.Config) -> Loose:
    raise NotImplementedError
