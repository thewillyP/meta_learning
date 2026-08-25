from meta_learn_lib.category.lens import *
from meta_learn_lib.category.paralens import *
from meta_learn_lib.category.mealy import Mealy, to_mealy
from meta_learn_lib.utility.util import zero_cotangent_like

import equinox as eqx


def validation[S, SV, XV, Y, YV, HP, P, HPV, PV, HQ, Q](
    machine: Mealy[SV, SV, XV, XV, YV, YV, HQ, HQ, Q, Q],
    r: Callable[[tuple[tuple[HP, P], S], tuple[HPV, PV]], tuple[HQ, Q]],
) -> Mealy[
    SV,
    SV,
    tuple[tuple[Y, tuple[tuple[HP, P], S]], XV],
    tuple[tuple[Y, tuple[tuple[HP, P], S]], XV],
    YV,
    YV,
    HPV,
    HPV,
    PV,
    PV,
]:
    def run(
        p_sx: tuple[tuple[HPV, PV], tuple[SV, tuple[tuple[Y, tuple[tuple[HP, P], S]], XV]]],
    ) -> tuple[
        tuple[SV, YV],
        Callable[[tuple[SV, YV]], tuple[tuple[HPV, PV], tuple[SV, tuple[tuple[Y, tuple[tuple[HP, P], S]], XV]]]],
    ]:
        own, (sv, ((y_tr, wire), x)) = p_sx
        port, r_rev = eqx.filter_vjp(r, wire, own)
        (sv1, y), put = machine.arrow.arrow.run((port, (sv, x)))

        def rev(ct: tuple[SV, YV]) -> tuple[tuple[HPV, PV], tuple[SV, tuple[tuple[Y, tuple[tuple[HP, P], S]], XV]]]:
            d_sv1, d_y = ct
            d_port, (d_sv, d_x) = put((d_sv1, d_y))
            d_wire, d_own = r_rev(d_port)
            return d_own, (d_sv, ((zero_cotangent_like(y_tr), d_wire), d_x))

        return (sv1, y), rev

    return Mealy(ParaLens(Lens(run)))


def same_model[HP, P, S](wire: tuple[tuple[HP, P], S], own: tuple[Unit, Unit]) -> tuple[HP, P]:
    hp_p, _ = wire
    return hp_p


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
