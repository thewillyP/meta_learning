from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.leaves import initializer, knob, sampler
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
    RFLO,
    RTRL,
    Rnn,
    SameModel,
    Scan,
    Seq,
    Sgd,
    Sup,
    Term,
    UORO,
    Validator,
)
import meta_learn_lib.lib_types
from meta_learn_lib.lib_types import ArrayTree, JACOBIAN, PRNG
from meta_learn_lib.utility.util import to_vector

from typing import Callable, cast, overload
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from plum import dispatch


@overload
def val_port[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def val_port[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], ctx: int, key: PRNG
) -> tuple[HPV, PV]:
    raise NotImplementedError


@dispatch
def val_port[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], ctx: int, key: PRNG
) -> tuple[HPV, PV]:
    raise NotImplementedError


@overload
def port(t: Linear, ctx: int, key: PRNG) -> tuple[Unit, eqx.nn.Linear]:
    k_layer, k_w = jax.random.split(key)
    layer = eqx.nn.Linear(ctx, t.n, use_bias=False, key=k_layer)
    return (Unit(), eqx.tree_at(lambda l: l.weight, layer, initializer(t.init)(k_w, (t.n, ctx))))


@overload
def port(t: Bias, ctx: int, key: PRNG) -> tuple[Unit, jax.Array]:
    return (Unit(), initializer(t.init)(key, (ctx,)))


@overload
def port[HPA, PA](t: Rnn[HPA, PA], ctx: int, key: PRNG) -> tuple[HPA, tuple[PA, eqx.nn.Linear]]:
    k_layer, k_rec, k_in = jax.random.split(key, 3)
    layer = eqx.nn.Linear(t.n + ctx, t.n, use_bias=False, key=k_layer)
    w_rec = initializer(Orthogonal())(k_rec, (t.n, t.n))
    w_in = initializer(t.init)(k_in, (t.n, ctx))
    hp, p = knob(t.alpha)
    return (hp, (p, eqx.tree_at(lambda l: l.weight, layer, jnp.hstack([w_rec, w_in]))))


@overload
def port(t: Activation, ctx: int, key: PRNG) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def port(t: Loss, ctx: int, key: PRNG) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def port[P](t: Sgd[P], ctx: int, key: PRNG) -> tuple[Unit, jax.Array]:
    return (Unit(), jnp.asarray(t.lr))


@overload
def port[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2], ctx: int, key: PRNG
) -> tuple[tuple[HP1, HP2], tuple[P1, P2]]:
    k1, k2 = jax.random.split(key)
    hp1, p1 = port(t.first, ctx, PRNG(k1))
    hp2, p2 = port(t.second, out(t.first, ctx), PRNG(k2))
    return ((hp1, hp2), (p1, p2))


@overload
def port[S, X, HP, P](
    t: Sup[S, X, HP, P], ctx: int, key: PRNG
) -> tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]]:
    hp_a, p_a = port(t.arch, ctx, key)
    return (((hp_a, Unit()), Unit()), ((p_a, Unit()), Unit()))


@overload
def port[S, X, HP, P, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, H, HPO, HPV, SV, XV, PV], ctx: int, key: PRNG
) -> tuple[tuple[tuple[HPO, Unit], HPV], tuple[tuple[tuple[HP, H], Unit], PV]]:
    k1, k2, k3 = jax.random.split(key, 3)
    hp, _ = port(t.below, ctx, PRNG(k1))
    hp_o, h = port(t.opt, ctx, PRNG(k2))
    hpv, pv = val_port(t.val, ctx, PRNG(k3))
    return (((hp_o, Unit()), hpv), (((hp, h), Unit()), pv))


@overload
def port[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[HP, P]:
    return port(t.below, ctx, key)


@overload
def port[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[HP, P]:
    return port(t.below, ctx, key)


@overload
def port[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[HP, P]:
    return eqx.filter_vmap(lambda k: port(t.below, ctx, PRNG(k)))(jax.random.split(key, t.n))


@overload
def port[S, X, Y, HP, P](t: RTRL[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, Unit], P]:
    hp, p = port(t.below, ctx, key)
    return ((hp, Unit()), p)


@overload
def port[S, X, Y, HP, P](t: RFLO[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, jax.Array], P]:
    hp, p = port(t.below, ctx, key)
    d, _ = knob(t.decay)
    return ((hp, d), p)


@overload
def port[S, X, Y, HP, P](t: UORO[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[tuple[HP, Unit], P]:
    hp, p = port(t.below, ctx, key)
    return ((hp, Unit()), p)


@overload
def port[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[HP, P]:
    raise NotImplementedError


@dispatch
def port[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], ctx: int, key: PRNG) -> tuple[HP, P]:
    raise NotImplementedError


@overload
def val_state[S, X, Y, HP, P](v: SameModel[S, X, Y, HP, P], s: S, ctx: int, key: PRNG) -> S:
    return s


@overload
def val_state[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], s: S, ctx: int, key: PRNG
) -> SV:
    raise NotImplementedError


@dispatch
def val_state[S, X, Y, HP, P, SV, XV, HPV, PV](
    v: Validator[S, X, Y, HP, P, SV, XV, HPV, PV], s: S, ctx: int, key: PRNG
) -> SV:
    raise NotImplementedError


@overload
def state(t: Linear, hp_p: tuple[Unit, eqx.nn.Linear], ctx: int, key: PRNG) -> Unit:
    return Unit()


@overload
def state(t: Bias, hp_p: tuple[Unit, jax.Array], ctx: int, key: PRNG) -> Unit:
    return Unit()


@overload
def state(t: Activation, hp_p: tuple[Unit, Unit], ctx: int, key: PRNG) -> Unit:
    return Unit()


@overload
def state(t: Loss, hp_p: tuple[Unit, Unit], ctx: int, key: PRNG) -> Unit:
    return Unit()


@overload
def state[HPA, PA](t: Rnn[HPA, PA], hp_p: tuple[HPA, tuple[PA, eqx.nn.Linear]], ctx: int, key: PRNG) -> jax.Array:
    return initializer(t.h0)(key, (t.n,))


@overload
def state[P](t: Sgd[P], hp_p: tuple[Unit, jax.Array], ctx: P, key: PRNG) -> ArrayTree:
    _, lr = hp_p
    return optax.sgd(lr).init(cast(optax.Params, ctx))


@overload
def state[S1, S2, X, Y, Z, HP1, HP2, P1, P2](
    t: Seq[S1, S2, X, Y, Z, HP1, HP2, P1, P2],
    hp_p: tuple[tuple[HP1, HP2], tuple[P1, P2]],
    ctx: int,
    key: PRNG,
) -> tuple[S1, S2]:
    (hp1, hp2), (p1, p2) = hp_p
    k1, k2 = jax.random.split(key)
    return (state(t.first, (hp1, p1), ctx, PRNG(k1)), state(t.second, (hp2, p2), out(t.first, ctx), PRNG(k2)))


@overload
def state[S, X, HP, P](
    t: Sup[S, X, HP, P],
    hp_p: tuple[tuple[tuple[HP, Unit], Unit], tuple[tuple[P, Unit], Unit]],
    ctx: int,
    key: PRNG,
) -> tuple[tuple[S, Unit], Unit]:
    ((hp_a, _), _), ((p_a, _), _) = hp_p
    return ((state(t.arch, (hp_a, p_a), ctx, key), Unit()), Unit())


@overload
def state[S, X, HP, P, H, HPO, HPV, SV, XV, PV](
    t: Meta[S, X, HP, P, H, HPO, HPV, SV, XV, PV],
    hp_p: tuple[tuple[tuple[HPO, Unit], HPV], tuple[tuple[tuple[HP, H], Unit], PV]],
    ctx: int,
    key: PRNG,
) -> tuple[tuple[tuple[S, tuple[ArrayTree, P]], Unit], SV]:
    _, (((hp, h), _), _) = hp_p
    k1, k2, k3 = jax.random.split(key, 3)
    _, p_b = port(t.below, ctx, PRNG(k1))
    s = state(t.below, (hp, p_b), ctx, PRNG(k2))
    opt_st = state(t.opt, (Unit(), h), p_b, PRNG(k3))
    return (((s, (opt_st, p_b)), Unit()), val_state(t.val, s, ctx, key))


@overload
def state[S, X, Y, HP, P](t: Scan[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG) -> S:
    return state(t.below, hp_p, ctx, key)


@overload
def state[S, X, Y, HP, P](t: BatchData[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG) -> S:
    return eqx.filter_vmap(lambda k: state(t.below, hp_p, ctx, PRNG(k)))(jax.random.split(key, t.n))


@overload
def state[S, X, Y, HP, P](t: BatchPop[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG) -> S:
    hp, p = hp_p
    return eqx.filter_vmap(lambda h, q, k: state(t.below, (h, q), ctx, PRNG(k)))(hp, p, jax.random.split(key, t.n))


@overload
def state[S, X, Y, HP, P](
    t: RTRL[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P], ctx: int, key: PRNG
) -> tuple[S, JACOBIAN]:
    (hp, _), p = hp_p
    return (state(t.below, (hp, p), ctx, key), influence(t.below, (hp, p), ctx, key))


@overload
def state[S, X, Y, HP, P](
    t: RFLO[S, X, Y, HP, P], hp_p: tuple[tuple[HP, jax.Array], P], ctx: int, key: PRNG
) -> tuple[S, JACOBIAN]:
    (hp, _), p = hp_p
    return (state(t.below, (hp, p), ctx, key), influence(t.below, (hp, p), ctx, key))


@overload
def state[S, X, Y, HP, P](
    t: UORO[S, X, Y, HP, P], hp_p: tuple[tuple[HP, Unit], P], ctx: int, key: PRNG
) -> tuple[S, tuple[jax.Array, jax.Array, PRNG]]:
    (hp, _), p = hp_p
    k_nu, k_next = jax.random.split(key)
    s = state(t.below, (hp, p), ctx, key)
    nu = sampler(t.noise)(PRNG(k_nu), to_vector(s).vector.shape)
    return (s, (nu, rank_one(t.below, (hp, p), ctx, key, nu), PRNG(k_next)))


@overload
def state[S, X, Y, HP, P, C](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: C, key: PRNG) -> S:
    raise NotImplementedError


@dispatch
def state[S, X, Y, HP, P, C](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: C, key: PRNG) -> S:
    raise NotImplementedError


def flat_state[S, X, Y, HP, P](
    t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG
) -> Callable[[jax.Array], jax.Array]:
    hp, p = hp_p
    v = to_vector(p)
    return lambda p_flat: to_vector(state(t, (hp, v.to_param(p_flat)), ctx, key)).vector


def influence[S, X, Y, HP, P](t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG) -> JACOBIAN:
    _, p = hp_p
    p_flat = to_vector(p).vector
    f = flat_state(t, hp_p, ctx, key)
    (n_s,) = jax.eval_shape(f, p_flat).shape
    return JACOBIAN(jax.jacfwd(f)(p_flat) if n_s > p_flat.size else jax.jacrev(f)(p_flat))


def rank_one[S, X, Y, HP, P](
    t: Term[S, X, Y, HP, P], hp_p: tuple[HP, P], ctx: int, key: PRNG, nu: jax.Array
) -> jax.Array:
    _, p = hp_p
    _, vjp = jax.vjp(flat_state(t, hp_p, ctx, key), to_vector(p).vector)
    (d_p,) = vjp(nu)
    return d_p
