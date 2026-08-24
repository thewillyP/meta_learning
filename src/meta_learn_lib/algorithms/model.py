from meta_learn_lib.category.lib_types import Unit
from meta_learn_lib.category.mealy import Mealy
from meta_learn_lib.category.paralens import para_autodiff

from typing import Callable
import jax


def leaf[S, P, X, Y](f: Callable[[P, S, X], tuple[S, Y]]) -> Mealy[S, S, X, X, Y, Y, Unit, Unit, P, P]:
    def h(hp_p: tuple[Unit, P], sx: tuple[S, X]) -> tuple[S, Y]:
        _, p = hp_p
        s, x = sx
        return f(p, s, x)

    return Mealy(para_autodiff(h))


def read_value[HP, P](
    m: Mealy[Unit, Unit, Unit, Unit, jax.Array, jax.Array, HP, HP, P, P],
) -> Callable[[tuple[HP, P]], jax.Array]:
    def go(hp_p: tuple[HP, P]) -> jax.Array:
        _, y = m.arrow.arrow.get((hp_p, (Unit(), Unit())))
        return y

    return go
