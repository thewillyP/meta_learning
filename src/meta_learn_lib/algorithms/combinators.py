from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.lib_types import LOSS

import jax
import jax.numpy as jnp


learning_rate: ParaLens[Unit, Unit, LOSS, LOSS, Unit, Unit] = unit(Lens(lambda l: (Unit(), lambda _: jnp.ones_like(l))))


learning_rate_log: ParaLens[Unit, Unit, LOSS, LOSS, LOSS, LOSS] = post(
    snd(Proxy[tuple[Unit, Unit, LOSS, LOSS]]()),
    pre(copy(Proxy[tuple[LOSS, LOSS]]()), first(learning_rate)),
)
