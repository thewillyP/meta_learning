from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.leaves import initializer, knob
from meta_learn_lib.construct.shape import out
from meta_learn_lib.construct.term import (
    Activation,
    BatchData,
    BatchPop,
    Bias,
    Linear,
    Loss,
    Meta,
    Orthogonal,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Sgd,
    Sup,
    Term,
    Validator,
)
from meta_learn_lib.lib_types import ArrayTree, PRNG

from typing import cast, overload
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from plum import dispatch


@overload
def val_init[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], s: S, ctx: int, key: PRNG) -> tuple[Unit, S]:
    return (Unit(), s)


@overload
def val_init[S, X, Y, HP, P, SV, XV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, PV], s: S, ctx: int, key: PRNG
) -> tuple[PV, SV]:
    raise NotImplementedError


@dispatch
def val_init[S, X, Y, HP, P, SV, XV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, PV], s: S, ctx: int, key: PRNG
) -> tuple[PV, SV]:
    raise NotImplementedError


@overload
def init(t: Linear, ctx: int, key: PRNG) -> tuple[tuple[Unit, eqx.nn.Linear], Unit]:
    k_layer, k_w = jax.random.split(key)
    layer = eqx.nn.Linear(ctx, t.n, use_bias=False, key=k_layer)
    w = initializer(t.init)(k_w, (t.n, ctx))
    return ((Unit(), eqx.tree_at(lambda l: l.weight, layer, w)), Unit())


@overload
def init(t: Bias, ctx: int, key: PRNG) -> tuple[tuple[Unit, jax.Array], Unit]:
    return ((Unit(), initializer(t.init)(key, (ctx,))), Unit())


@overload
def init[HPA, PA](t: Rnn[HPA, PA], ctx: int, key: PRNG) -> tuple[tuple[HPA, tuple[PA, eqx.nn.Linear]], jax.Array]:
    k_layer, k_rec, k_in, k_h = jax.random.split(key, 4)
    layer = eqx.nn.Linear(t.n + ctx, t.n, use_bias=False, key=k_layer)
    w_rec = initializer(Orthogonal())(k_rec, (t.n, t.n))
    w_in = initializer(t.init)(k_in, (t.n, ctx))
    w = jnp.hstack([w_rec, w_in])
    hp, p = knob(t.alpha)
    return ((hp, (p, eqx.tree_at(lambda l: l.weight, layer, w))), initializer(t.h0)(k_h, (t.n,)))


@overload
def init(t: Activation, ctx: int, key: PRNG) -> tuple[tuple[Unit, Unit], Unit]:
    return ((Unit(), Unit()), Unit())


@overload
def init(t: Loss, ctx: int, key: PRNG) -> tuple[tuple[Unit, Unit], Unit]:
    return ((Unit(), Unit()), Unit())


@overload
def init[P](t: Sgd[P], ctx: P, key: PRNG) -> tuple[tuple[Unit, jax.Array], ArrayTree]:
    return ((Unit(), jnp.asarray(t.lr)), optax.sgd(t.lr).init(cast(optax.Params, ctx)))


@overload
def init[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2], ctx: int, key: PRNG
) -> tuple[tuple[tuple[HP1, HP2], tuple[P1, P2]], tuple[S1, S2]]:
    k1, k2 = jax.random.split(key)
    (hp1, p1), s1 = init(t.first, ctx, PRNG(k1))
    (hp2, p2), s2 = init(t.second, out(t.first, ctx), PRNG(k2))
    return (((hp1, hp2), (p1, p2)), (s1, s2))


@overload
def init[S, X, HP, P](
    t: Sup[S, X, HP, P], ctx: int, key: PRNG
) -> tuple[
    tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]],
    tuple[tuple[S, Unit], Unit],
]:
    k1, k2 = jax.random.split(key)
    (hp_a, p_a), s_a = init(t.arch, ctx, PRNG(k1))
    (hp_l, p_l), s_l = init(t.loss, out(t.arch, ctx), PRNG(k2))
    return ((((hp_a, Unit()), hp_l), ((p_a, Unit()), p_l)), ((s_a, Unit()), s_l))


@overload
def init[S, X, HP, P, SV, XV, PV](
    t: Meta[S, X, HP, P, SV, XV, PV], ctx: int, key: PRNG
) -> tuple[
    tuple[tuple[tuple[HP, Unit], HP], tuple[tuple[jax.Array, Unit], PV]],
    tuple[tuple[tuple[S, tuple[ArrayTree, P]], Unit], SV],
]:
    k1, k2, k3 = jax.random.split(key, 3)
    (hp, p), s = init(t.below, ctx, PRNG(k1))
    (_, lr), opt_st = init(t.opt, p, PRNG(k2))
    p_v, s_v = val_init(t.val, s, ctx, PRNG(k3))
    return ((((hp, Unit()), hp), ((lr, Unit()), p_v)), (((s, (opt_st, p)), Unit()), s_v))


@overload
def init[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    return init(t.below, ctx, key)


@overload
def init[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    k0, kb = jax.random.split(key)
    (hp, p), _ = init(t.below, ctx, PRNG(k0))
    _, s = eqx.filter_vmap(lambda k: init(t.below, ctx, PRNG(k)))(jax.random.split(kb, t.n))
    return ((hp, p), s)


@overload
def init[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, P], S]:
    return eqx.filter_vmap(lambda k: init(t.below, ctx, PRNG(k)))(jax.random.split(key, t.n))


@overload
def init[S, X, Y, HP, P, C](t: Term[S, X, Y, HP, P], ctx: C, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError


@dispatch
def init[S, X, Y, HP, P, C](t: Term[S, X, Y, HP, P], ctx: C, key: PRNG) -> tuple[tuple[HP, P], S]:
    raise NotImplementedError
