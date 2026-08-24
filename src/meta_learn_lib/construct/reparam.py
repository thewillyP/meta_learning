from meta_learn_lib.construct.term import (
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
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
    raise NotImplementedError


@dispatch
def reparametrizer[P2, P](r: Reparametrization[P2, P]) -> Callable[[P2], P]:
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
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError


@dispatch
def seed[P2, P](r: Reparametrization[P2, P], p: P) -> P2:
    raise NotImplementedError
