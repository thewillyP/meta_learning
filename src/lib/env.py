from typing import Callable, Literal, Optional, Union
import equinox as eqx
import jax
import optax

from lib.lib_types import *


class Logs(eqx.Module):
    gradient: Optional[jax.Array] = eqx.field(default=None)
    influence_tensor: Optional[jax.Array] = eqx.field(default=None)
    immediate_influence_tensor: Optional[jax.Array] = eqx.field(default=None)
    jac_eigenvalue: Optional[jax.Array] = eqx.field(default=None)
    hessian: Optional[jax.Array] = eqx.field(default=None)


class CustomSequential(eqx.Module):
    model: eqx.nn.Sequential

    def __init__(
        self, layer_defs: list[tuple[int, bool, Callable[[jax.Array], jax.Array]]], input_size: int, key: PRNG
    ):
        layers = []
        in_size = input_size
        layer_keys = jax.random.split(key, len(layer_defs))

        for (out_size, use_bias, activation), k in zip(layer_defs, layer_keys):
            layers.append(eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=k))
            layers.append(eqx.nn.Lambda(activation))
            in_size = out_size

        self.model = eqx.nn.Sequential(layers)

    def __call__(self, x):
        return self.model(x)


class RNN(eqx.Module):
    activation: jax.Array
    w_rec: jax.Array
    n_h: int = eqx.field(static=True)
    n_in: int = eqx.field(static=True)
    n_out: int = eqx.field(static=True)
    activationFn: Literal["tanh", "relu"] = eqx.field(static=True)


class UORO(eqx.Module):
    A: jax.Array
    B: jax.Array


class General(eqx.Module):
    prng: PRNG
    current_virtual_minibatch: int
    logs: Optional[Logs] = eqx.field(default=None)


class Inference(eqx.Module):
    rnn: Optional[RNN] = eqx.field(default=None)
    readout_fn: Optional[CustomSequential] = eqx.field(default=None)


class Learning(eqx.Module):
    rflo_timeconstant: float = eqx.field(static=True)
    influence_tensor: Optional[JACOBIAN] = eqx.field(default=None)
    uoro: Optional[UORO] = eqx.field(default=None)
    opt_state: Optional[optax.OptState] = eqx.field(default=None)
    learning_rate: Optional[jax.Array] = eqx.field(default=None)


class GodState(eqx.Module):
    states: dict[int, tuple[General, tuple[dict[int, Inference], Inference], Learning]]
    base_inference: tuple[dict[int, Inference], Inference]
    start_epoch: int
