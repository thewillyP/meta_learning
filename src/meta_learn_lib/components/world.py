import equinox as eqx
import jax
from pyrsistent import PClass, field
from pyrsistent.typing import PMap

from ..pytree.pyrsistent import register_pclass


@register_pclass
class World(PClass):
    arrays: PMap[str, jax.Array] = field()
    norms: PMap[str, eqx.nn.LayerNorm | eqx.nn.GroupNorm] = field()
