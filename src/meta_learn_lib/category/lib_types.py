import equinox as eqx


class Proxy[A](eqx.Module): ...


class Unit(eqx.Module): ...


class Batched[A](eqx.Module):
    value: A


class Seq[A](eqx.Module):
    value: A


type Axes[A] = Seq[Axes[A]] | Batched[Axes[A]] | A
