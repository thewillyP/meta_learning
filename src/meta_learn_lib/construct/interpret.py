from meta_learn_lib.algorithms.combinators import reparametrize
from meta_learn_lib.category.lens import autodiff
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.construct.build import build
from meta_learn_lib.construct.init import port, state
from meta_learn_lib.construct.reparam import cook, store
from meta_learn_lib.construct.term import Term
from meta_learn_lib.lib_types import PRNG


def machine[S, X, Y, HP, P](t: Term[S, X, Y, HP, P]) -> Mealy[S, S, X, X, Y, Y, HP, HP, P, P]:
    return reparametrize(build(t), autodiff(cook(t)))


def setup[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    hp_p = port(t, ctx, key)
    return (store(t, hp_p), state(t, hp_p, ctx, key))
