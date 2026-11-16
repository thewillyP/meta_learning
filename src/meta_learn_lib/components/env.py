import equinox as eqx
import jax
import optax
from pyrsistent import pmap
from pyrsistent.typing import PMap

from ..backend.pytree import Field, Record, register_record
from ..lib_types import JACOBIAN, PRNG

type UORO_AUX = tuple[jax.Array, jax.Array]


@register_record
class Env(Record):
    arrays: Field[PMap[str, jax.Array]] = Field(default=pmap())
    norms: Field[PMap[str, eqx.nn.LayerNorm | eqx.nn.GroupNorm]] = Field(default=pmap())
    optstates: Field[PMap[str, optax.OptState]] = Field(default=pmap())
    prngs: Field[PMap[str, PRNG]] = Field(default=pmap())
    influences: Field[PMap[str, JACOBIAN]] = Field(default=pmap())
    uoros: Field[PMap[str, UORO_AUX]] = Field(default=pmap())


type Log = Env
type HP = Env
type Par = Env
type St = Env
