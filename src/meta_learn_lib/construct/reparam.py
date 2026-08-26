from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.construct.term import (
    Alongside,
    RCompose,
    Below,
    Demoted,
    PortOf,
    LowRank,
    RMap,
    RSplit,
    Rectify,
    Reparametrization,
    SiluPositive,
    SoftClip,
    SoftRelu,
    Softplus,
    Squared,
    Unconstrained,
)
from meta_learn_lib.utility.reparam import (
    silu_positive,
    silu_positive_inverse,
    soft_clip,
    soft_relu,
    softplus_inverse,
)

from typing import Callable, overload
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from plum import dispatch


@overload
def reparametrizer[P](r: Unconstrained[P]) -> Callable[[P], P]:
    return lambda x: x


@overload
def reparametrizer(r: Softplus) -> Callable[[jax.Array], jax.Array]:
    return jax.nn.softplus


@overload
def reparametrizer(r: Rectify) -> Callable[[jax.Array], jax.Array]:
    return lambda x: jnp.maximum(x, 0.0)


@overload
def reparametrizer(r: SoftRelu) -> Callable[[jax.Array], jax.Array]:
    return lambda x: soft_relu(x, r.sharpness)


@overload
def reparametrizer(r: SiluPositive) -> Callable[[jax.Array], jax.Array]:
    return lambda x: silu_positive(x, r.scale)


@overload
def reparametrizer(r: Squared) -> Callable[[jax.Array], jax.Array]:
    return lambda x: (r.scale * x) ** 2


@overload
def reparametrizer(r: SoftClip) -> Callable[[jax.Array], jax.Array]:
    return lambda x: soft_clip(x, r.a, r.b, r.sharpness)


@overload
def reparametrizer[P](r: RMap[P]) -> Callable[[P], P]:
    forward = reparametrizer(r.by)
    return lambda x: jax.tree.map(forward, x)


@overload
def reparametrizer[A2, A, B2, B](r: RSplit[A2, A, B2, B]) -> Callable[[tuple[A2, B2]], tuple[A, B]]:
    first = reparametrizer(r.first)
    second = reparametrizer(r.second)

    def apply(ab: tuple[A2, B2]) -> tuple[A, B]:
        a, b = ab
        return (first(a), second(b))

    return apply


@overload
def reparametrizer(r: LowRank) -> Callable[[tuple[jax.Array, jax.Array]], eqx.nn.Linear]:
    def expand(ab: tuple[jax.Array, jax.Array]) -> eqx.nn.Linear:
        a, b = ab
        n_out, _ = a.shape
        _, n_in = b.shape
        shell = eqx.nn.Linear(n_in, n_out, use_bias=False, key=jax.random.key(0))
        return eqx.tree_at(lambda l: l.weight, shell, a @ b)

    return expand


@overload
def reparametrizer[HP, P, S](
    r: PortOf[HP, P, S],
) -> Callable[[tuple[tuple[tuple[HP, P], S], tuple[Unit, Unit]]], tuple[HP, P]]:
    def project(p: tuple[tuple[tuple[HP, P], S], tuple[Unit, Unit]]) -> tuple[HP, P]:
        (hp_p, _), _ = p
        return hp_p

    return project


@overload
def reparametrizer[HP1, HP0, H, PV, S0, SO, P0, SV](
    r: Demoted[HP1, HP0, H, PV, S0, SO, P0, SV],
) -> Callable[
    [
        tuple[
            tuple[tuple[HP1, tuple[tuple[tuple[HP0, H], Unit], PV]], tuple[tuple[tuple[S0, tuple[SO, P0]], Unit], SV]],
            tuple[Unit, Unit],
        ]
    ],
    tuple[HP0, P0],
]:
    def project(
        p: tuple[
            tuple[tuple[HP1, tuple[tuple[tuple[HP0, H], Unit], PV]], tuple[tuple[tuple[S0, tuple[SO, P0]], Unit], SV]],
            tuple[Unit, Unit],
        ],
    ) -> tuple[HP0, P0]:
        ((_, (((hp0, _), _), _)), (((_, (_, theta)), _), _)), _ = p
        return (hp0, theta)

    return project


@overload
def reparametrizer[A, B, C](r: RCompose[A, B, C]) -> Callable[[A], C]:
    first, second = reparametrizer(r.first), reparametrizer(r.second)
    return lambda a: second(first(a))


@overload
def reparametrizer[HP2, HP1, H2, PV2, S1, SO2, P1, SV2](
    r: Below[HP2, HP1, H2, PV2, S1, SO2, P1, SV2],
) -> Callable[
    [
        tuple[
            tuple[
                tuple[HP2, tuple[tuple[tuple[HP1, H2], Unit], PV2]], tuple[tuple[tuple[S1, tuple[SO2, P1]], Unit], SV2]
            ],
            tuple[Unit, Unit],
        ]
    ],
    tuple[tuple[tuple[HP1, P1], S1], tuple[Unit, Unit]],
]:
    def project(
        p: tuple[
            tuple[
                tuple[HP2, tuple[tuple[tuple[HP1, H2], Unit], PV2]], tuple[tuple[tuple[S1, tuple[SO2, P1]], Unit], SV2]
            ],
            tuple[Unit, Unit],
        ],
    ) -> tuple[tuple[tuple[HP1, P1], S1], tuple[Unit, Unit]]:
        ((_, (((hp1, _), _), _)), (((s1, (_, p1)), _), _)), own = p
        return (((hp1, p1), s1), own)

    return project


@overload
def reparametrizer[W, HPV, PV, HQ, Q](
    r: Alongside[W, HPV, PV, HQ, Q],
) -> Callable[[tuple[W, tuple[HPV, PV]]], tuple[tuple[HQ, HPV], tuple[Q, PV]]]:
    project = reparametrizer(r.wire)

    def merge(p: tuple[W, tuple[HPV, PV]]) -> tuple[tuple[HQ, HPV], tuple[Q, PV]]:
        w, (hpv, pv) = p
        hq, q = project((w, (Unit(), Unit())))
        return ((hq, hpv), (q, pv))

    return merge


@overload
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
    raise NotImplementedError


@dispatch
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
    raise NotImplementedError


@overload
def own[W, HPV, PV, HQ, Q](
    r: Alongside[W, HPV, PV, HQ, Q], natural: tuple[tuple[HQ, HPV], tuple[Q, PV]]
) -> tuple[HPV, PV]:
    (_, hpv), (_, pv) = natural
    return (hpv, pv)


@overload
def own[HP, P, S](r: PortOf[HP, P, S], natural: tuple[HP, P]) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def own[HP1, HP0, H, PV, S0, SO, P0, SV](
    r: Demoted[HP1, HP0, H, PV, S0, SO, P0, SV], natural: tuple[HP0, P0]
) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def own[HP2, HP1, H2, PV2, S1, SO2, P1, SV2](
    r: Below[HP2, HP1, H2, PV2, S1, SO2, P1, SV2], natural: tuple[tuple[tuple[HP1, P1], S1], tuple[Unit, Unit]]
) -> tuple[Unit, Unit]:
    return (Unit(), Unit())


@overload
def own[A, B, C](r: RCompose[A, B, C], natural: C) -> PyTree:
    return own(r.second, natural)


@overload
def own[P2, P](r: Reparametrization[P2, P], natural: P) -> PyTree:
    raise NotImplementedError


@dispatch
def own[P2, P](r: Reparametrization[P2, P], natural: P) -> PyTree:
    raise NotImplementedError


@overload
def seed[P](r: Unconstrained[P], p: P) -> P:
    return p


@overload
def seed(r: Softplus, p: jax.Array) -> jax.Array:
    return softplus_inverse(p)


@overload
def seed(r: Rectify, p: jax.Array) -> jax.Array:
    return p


@overload
def seed(r: SoftRelu, p: jax.Array) -> jax.Array:
    return p


@overload
def seed(r: SiluPositive, p: jax.Array) -> jax.Array:
    return silu_positive_inverse(p, r.scale)


@overload
def seed(r: Squared, p: jax.Array) -> jax.Array:
    return jnp.sqrt(p) / r.scale


@overload
def seed(r: SoftClip, p: jax.Array) -> jax.Array:
    return p


@overload
def seed[P](r: RMap[P], p: P) -> P:
    return jax.tree.map(lambda a: seed(r.by, a), p)


@overload
def seed[A2, A, B2, B](r: RSplit[A2, A, B2, B], p: tuple[A, B]) -> tuple[A2, B2]:
    a, b = p
    return (seed(r.first, a), seed(r.second, b))


@overload
def seed(r: LowRank, p: eqx.nn.Linear) -> tuple[jax.Array, jax.Array]:
    u, sv, vt = jnp.linalg.svd(p.weight, full_matrices=False)
    root = jnp.sqrt(sv[: r.rank])
    return (u[:, : r.rank] * root, vt[: r.rank, :] * root[:, None])


@overload
def seed[A, B, C](r: RCompose[A, B, C], p: C) -> A:
    return seed(r.first, seed(r.second, p))


@overload
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError


@dispatch
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError
