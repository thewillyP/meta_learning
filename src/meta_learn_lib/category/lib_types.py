import equinox as eqx


class Proxy[A](eqx.Module): ...


class Unit(eqx.Module): ...


class Batched[A](eqx.Module):
    value: A
