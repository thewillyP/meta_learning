import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap
from pyrsistent.typing import PMap

from ..backend.pytree import register_pclass
from ..lib_types import JACOBIAN, PRNG

type UORO_AUX = tuple[jax.Array, jax.Array]


@register_pclass
class Env(PClass):
    arrays: PMap[str, jax.Array] = field(initial=pmap())
    norms: PMap[str, eqx.nn.LayerNorm | eqx.nn.GroupNorm] = field(initial=pmap())
    optstates: PMap[str, optax.OptState] = field(initial=pmap())
    prngs: PMap[str, PRNG] = field(initial=pmap())
    influences: PMap[str, JACOBIAN] = field(initial=pmap())
    uoros: PMap[str, UORO_AUX] = field(initial=pmap())


type Log = Env
type HP = Env
type Par = Env
type St = Env
