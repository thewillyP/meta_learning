import equinox as eqx


class Weights[A](eqx.Module):
    weights: A


class Offset[A](eqx.Module):
    bias: A


class RnnPort[A](eqx.Module):
    weights: A
    time_constant: A


class Target[A](eqx.Module):
    target: A


class Rates[A](eqx.Module):
    learning_rate: A
    weight_decay: A
    momentum: A
