from collections.abc import Callable
from dataclasses import dataclass
from typing import overload
import equinox as eqx
import jax
from jaxtyping import PyTree
from plum import dispatch
from ..lib_types import PRNG
from ..algorithms.lib_types import UORO_AUX
from ..utility.distributions import rademacher, uniform_unit
from ..utility.util import to_vector

type Initialize[P, A] = Callable[[P, jax.Array], A]


@dataclass(frozen=True)
class Init: ...


@dataclass(frozen=True)
class Zeros(Init): ...


@dataclass(frozen=True)
class Constant(Init):
    value: float


@dataclass(frozen=True)
class Normal(Init):
    std: float


@dataclass(frozen=True)
class LecunNormal(Init): ...


@dataclass(frozen=True)
class PytorchUniform(Init): ...


@dataclass(frozen=True)
class Orthogonal(Init): ...


@dataclass(frozen=True)
class Noise: ...


@dataclass(frozen=True)
class Gaussian(Noise): ...


@dataclass(frozen=True)
class Rademacher(Noise): ...


@dataclass(frozen=True)
class UniformUnit(Noise): ...


@overload
def sampler(noise: Gaussian) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return jax.random.normal


@overload
def sampler(noise: Rademacher) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return rademacher


@overload
def sampler(noise: UniformUnit) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    return uniform_unit


@overload
def sampler(noise: Noise) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    raise NotImplementedError


@dispatch
def sampler(noise: Noise) -> Callable[[PRNG, tuple[int, ...]], jax.Array]:
    raise NotImplementedError


@overload
def initializer(how: Zeros) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.zeros


@overload
def initializer(how: Constant) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.constant(how.value)


@overload
def initializer(how: Normal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.normal(how.std)


@overload
def initializer(how: LecunNormal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", in_axis=-1, out_axis=-2)


@overload
def initializer(how: PytorchUniform) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=-1, out_axis=-2)


@overload
def initializer(how: Orthogonal) -> jax.nn.initializers.Initializer:
    return jax.nn.initializers.orthogonal()


@overload
def initializer(how: Init) -> jax.nn.initializers.Initializer:
    raise TypeError(f"no initializer for {type(how).__name__}")


@dispatch
def initializer(how: Init) -> jax.nn.initializers.Initializer:
    raise TypeError(f"no initializer for {type(how).__name__}")


@dataclass(frozen=True)
class Initialization[P, A]:
    """Description of (available values: P, key) -> initialized value: A."""


@dataclass(frozen=True)
class Array[P](Initialization[P, jax.Array]):
    shape: tuple[int, ...]
    dtype: str
    init: Init


@dataclass(frozen=True)
class Input[P](Initialization[P, P]):
    """Use the supplied values without allocating them again."""


@dataclass(frozen=True)
class Apply[P, A](Initialization[P, A]):
    """Compute a value from the supplied context without drawing randomness."""

    function: Callable[[P], A]


@dataclass(frozen=True)
class Pair[P, A, B](Initialization[P, tuple[A, B]]):
    """A heterogeneous pair whose two output types remain distinct."""

    first: Initialization[P, A]
    second: Initialization[P, B]


@dataclass(frozen=True)
class Product[P, A](Initialization[P, tuple[A, ...]]):
    """A variable-length tuple of outputs with a common element type."""

    parts: tuple[Initialization[P, A], ...]


@dataclass(frozen=True)
class Then[P, A, B](Initialization[P, B]):
    first: Initialization[P, A]
    second: Initialization[A, B]


@dataclass(frozen=True)
class Copies[P, A](Initialization[P, A]):
    body: Initialization[P, A]
    size: int
    in_axes: PyTree[int | None]


@dataclass(frozen=True)
class WithSensitivity[P, S](Initialization[P, tuple[S, jax.Array]]):
    """Initial state and its full Jacobian, as used by RTRL and RFLO.

    Rows and columns follow to_vector's state and parameter ordering.
    """

    state: Initialization[P, S]


@dataclass(frozen=True)
class WithRankOne[P, S](Initialization[P, tuple[S, UORO_AUX]]):
    """Initial state and UORO's (noise, Jacobian-transpose times noise, key)."""

    state: Initialization[P, S]
    noise: Noise


@overload
def compile_initial[P](description: Array[P]) -> Callable[[P, jax.Array], jax.Array]:
    return lambda p, key: initializer(description.init)(key, description.shape, description.dtype)


@overload
def compile_initial[P](description: Input[P]) -> Callable[[P, jax.Array], P]:
    return lambda parameters, key: parameters


@overload
def compile_initial[P, A](description: Apply[P, A]) -> Callable[[P, jax.Array], A]:
    return lambda context, key: description.function(context)


@overload
def compile_initial[P, A, B](description: Pair[P, A, B]) -> Callable[[P, jax.Array], tuple[A, B]]:

    def run(parameters: P, key: jax.Array) -> tuple[A, B]:
        first = compile_initial(description.first)
        second = compile_initial(description.second)
        first_key, second_key = jax.random.split(key)
        return (first(parameters, first_key), second(parameters, second_key))

    return run


@overload
def compile_initial[P, A](description: Product[P, A]) -> Callable[[P, jax.Array], tuple[A, ...]]:
    def run(parameters: P, key: jax.Array) -> tuple[A, ...]:
        parts = tuple(compile_initial(part) for part in description.parts)
        return tuple(make(parameters, k) for make, k in zip(parts, jax.random.split(key, len(parts)), strict=True))

    return run


@overload
def compile_initial[P, A, B](description: Then[P, A, B]) -> Callable[[P, jax.Array], B]:
    def run(parameters: P, key: jax.Array) -> B:
        first = compile_initial(description.first)
        second = compile_initial(description.second)
        first_key, second_key = jax.random.split(key)
        return second(first(parameters, first_key), second_key)

    return run


@overload
def compile_initial[P, A](description: Copies[P, A]) -> Callable[[P, jax.Array], A]:
    def run(parameters: P, key: jax.Array) -> A:
        body = compile_initial(description.body)
        mapped = eqx.filter_vmap(body, in_axes=(description.in_axes, 0))
        return mapped(parameters, jax.random.split(key, description.size))

    return run


@overload
def compile_initial[P, S](description: WithSensitivity[P, S]) -> Callable[[P, jax.Array], tuple[S, jax.Array]]:
    def run(parameters: P, key: jax.Array) -> tuple[S, jax.Array]:
        make = compile_initial(description.state)
        state = make(parameters, key)
        parameters_flat = to_vector(parameters)

        def flat_state(vector: jax.Array) -> jax.Array:
            return to_vector(make(parameters_flat.to_param(vector), key)).vector

        n_state = to_vector(state).vector.size
        n_parameters = parameters_flat.vector.size
        differentiate = jax.jacfwd if n_state > n_parameters else jax.jacrev
        return (state, differentiate(flat_state)(parameters_flat.vector))

    return run


@overload
def compile_initial[P, S](description: WithRankOne[P, S]) -> Callable[[P, jax.Array], tuple[S, UORO_AUX]]:
    def run(parameters: P, key: jax.Array) -> tuple[S, UORO_AUX]:
        make = compile_initial(description.state)
        draw = sampler(description.noise)
        state_key, noise_key, next_key = jax.random.split(key, 3)
        state = make(parameters, state_key)
        parameters_flat = to_vector(parameters)

        def flat_state(vector: jax.Array) -> jax.Array:
            return to_vector(make(parameters_flat.to_param(vector), state_key)).vector

        state_flat, backward = jax.vjp(flat_state, parameters_flat.vector)
        nu = draw(PRNG(noise_key), state_flat.shape).astype(state_flat.dtype)
        (projected,) = backward(nu)
        return state, (nu, projected, PRNG(next_key))

    return run


@overload
def compile_initial[P, A](description: Initialization[P, A]) -> Callable[[P, jax.Array], A]:
    raise NotImplementedError


@dispatch
def compile_initial[P, A](description: Initialization[P, A]) -> Callable[[P, jax.Array], A]:
    raise NotImplementedError
