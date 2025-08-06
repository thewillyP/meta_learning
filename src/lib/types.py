from typing import NewType
import jax

PRNG = NewType("PRNG", jax.Array)
FractionalList = NewType("FractionalList", list[float])
