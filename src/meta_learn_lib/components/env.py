import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap
from pyrsistent.typing import PMap

from ..backend.pytree import register_pclass


@register_pclass
class Env(PClass):
    arrays: PMap[str, jax.Array] = field(initial=pmap())
    norms: PMap[str, eqx.nn.LayerNorm | eqx.nn.GroupNorm] = field(initial=pmap())
    optstates: PMap[str, optax.OptState] = field(initial=pmap())