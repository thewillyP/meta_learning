from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.utility.util import zero_cotangent_like


def validation[S, XV, Y, HP, P, HPV](
    machine: Mealy[S, S, XV, XV, Y, Y, HP, HP, P, P],
) -> Mealy[S, S, tuple[tuple[Y, tuple[HP, P]], XV], tuple[tuple[Y, tuple[HP, P]], XV], Y, Y, HPV, HPV, Unit, Unit]:
    def run(
        p_sx: tuple[tuple[HPV, Unit], tuple[S, tuple[tuple[Y, tuple[HP, P]], XV]]],
    ) -> tuple[
        tuple[S, Y],
        Callable[[tuple[S, Y]], tuple[tuple[HPV, Unit], tuple[S, tuple[tuple[Y, tuple[HP, P]], XV]]]],
    ]:
        (hpv, u), (s, ((y_tr, (hp, theta)), x)) = p_sx
        (s1, y), put = machine.arrow.arrow.run(((hp, theta), (s, x)))

        def rev(ct: tuple[S, Y]) -> tuple[tuple[HPV, Unit], tuple[S, tuple[tuple[Y, tuple[HP, P]], XV]]]:
            d_s1, d_y = ct
            (d_hp, d_theta), (d_s, d_x) = put((d_s1, d_y))
            return (zero_cotangent_like(hpv), u), (d_s, ((zero_cotangent_like(y_tr), (d_hp, d_theta)), d_x))

        return (s1, y), rev

    return Mealy(ParaLens(Lens(run)))


def level[S, X, E, XV, YV, SV, HP, H, HPV, PV](
    trainer: Mealy[S, S, X, X, E, E, HP, HP, H, H],
    val: Mealy[SV, SV, tuple[E, XV], tuple[E, XV], YV, YV, HPV, HPV, PV, PV],
) -> Mealy[
    tuple[tuple[S, Unit], SV],
    tuple[tuple[S, Unit], SV],
    tuple[X, XV],
    tuple[X, XV],
    YV,
    YV,
    tuple[tuple[HP, Unit], HPV],
    tuple[tuple[HP, Unit], HPV],
    tuple[tuple[H, Unit], PV],
    tuple[tuple[H, Unit], PV],
]:
    id_v = to_mealy(to_paralens(identity(Proxy[tuple[XV, XV]]())))
    return (trainer @ id_v) >> val
