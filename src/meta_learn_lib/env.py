from typing import Optional
import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap, pvector, thaw
from pyrsistent.typing import PVector
from pyrsistent._pmap import PMap as PMapClass

from meta_learn_lib.lib_types import *


def deep_serialize(_, obj):
    """Recursively serialize pyrsistent objects to Python built-ins"""
    if isinstance(obj, PClass):
        serialized = obj.serialize()
        return {k: deep_serialize(_, v) for k, v in serialized.items()}
    elif isinstance(obj, PMapClass):
        thawed = thaw(obj)
        return {k: deep_serialize(_, v) for k, v in thawed.items()}
    elif isinstance(obj, PVector):
        return [deep_serialize(_, v) for v in obj]
    elif isinstance(obj, dict):
        return {k: deep_serialize(_, v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_serialize(_, v) for v in obj)
    else:
        return obj


# Register PMap as PyTree
def _pmap_tree_flatten(pm):
    keys = tuple(sorted(pm.keys()))
    values = [pm[k] for k in keys]
    return values, keys


def _pmap_tree_unflatten(keys, values):
    return pmap(zip(keys, values))


jax.tree_util.register_pytree_node(PMapClass, _pmap_tree_flatten, _pmap_tree_unflatten)


# Register PVector as PyTree
def _pvector_tree_flatten(pv):
    return list(pv), None


def _pvector_tree_unflatten(_, values):
    return pvector(values)


jax.tree_util.register_pytree_node(PVector, _pvector_tree_flatten, _pvector_tree_unflatten)


# PyTree registration helpers
def register_pytree(cls, static_fields):
    """Register a class as a PyTree"""

    def tree_flatten(obj):
        all_fields = set(cls._pclass_fields.keys())
        dynamic_fields = all_fields - static_fields
        static_field_values = {name: getattr(obj, name) for name in static_fields if hasattr(obj, name)}
        dynamic_values = [getattr(obj, name) for name in sorted(dynamic_fields) if hasattr(obj, name)]
        return dynamic_values, (sorted(dynamic_fields), static_field_values)

    def tree_unflatten(aux_data, values):
        dynamic_fields, static_field_values = aux_data
        kwargs = dict(zip(dynamic_fields, values))
        kwargs.update(static_field_values)
        return cls(**kwargs)

    jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================


class Parameter[T](PClass):
    value: T = field(serializer=deep_serialize)
    is_learnable: bool = field()
    min_value: float = field()
    max_value: float = field()


class State[T](PClass):
    value: T = field(serializer=deep_serialize)
    is_batched: bool = field()
    is_stateful: bool = field()


# rather, put in configs a boolean for each field whether to allocate or not
class Logs(PClass):
    gradient: Optional[jax.Array] = field(initial=None)
    hessian_contains_nans: Optional[bool] = field(initial=None)
    immediate_influence_contains_nans: Optional[bool] = field(initial=None)
    largest_eigenvalue: Optional[jax.Array] = field(initial=None)
    influence_tensor: Optional[jax.Array] = field(initial=None)
    immediate_influence_tensor: Optional[jax.Array] = field(initial=None)
    largest_jac_eigenvalue: Optional[jax.Array] = field(initial=None)
    jacobian: Optional[jax.Array] = field(initial=None)


class MLP(PClass):
    model: Parameter[eqx.nn.Sequential] = field(serializer=deep_serialize)


class RNN(PClass):
    w_rec: Parameter[jax.Array] = field(serializer=deep_serialize)
    b_rec: Optional[Parameter[jax.Array]] = field(serializer=deep_serialize)
    layer_norm: Optional[Parameter[eqx.nn.LayerNorm]] = field(serializer=deep_serialize)


type Model = MLP | RNN | eqx.nn.GRUCell | eqx.nn.LSTMCell


class RecurrentState(PClass):
    activation: State[jax.Array] = field(serializer=deep_serialize)


class VanillaRecurrentState(RecurrentState):
    activation_fn: ACTIVATION_FN = field(serializer=deep_serialize)


class LSTMState(PClass):
    h: State[jax.Array] = field(serializer=deep_serialize)
    c: State[jax.Array] = field(serializer=deep_serialize)


class UOROState(PClass):
    A: State[jax.Array] = field(serializer=deep_serialize)
    B: State[jax.Array] = field(serializer=deep_serialize)


class Parameters(PClass):
    models: PVector[Model] = field(serializer=deep_serialize)
    learning_rates: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)
    weight_decays: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)
    rflo_timeconstants: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)
    kl_regularizer_betas: PVector[Parameter[jax.Array]] = field(serializer=deep_serialize)


class States(PClass):
    influence_tensors: PVector[JACOBIAN] = field()
    uoros: PVector[UOROState] = field(serializer=deep_serialize)
    opt_states: PVector[optax.OptState] = field(serializer=deep_serialize)
    ticks: PVector[jax.Array] = field()
    logs: PVector[Logs] = field(serializer=deep_serialize)
    recurrent_states: PVector[RecurrentState] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PVector[VanillaRecurrentState] = field(serializer=deep_serialize)
    lstm_states: PVector[LSTMState] = field(serializer=deep_serialize)
    prngs: PVector[State[PRNG]] = field(serializer=deep_serialize)


class GodState(PClass):
    meta_states: PVector[States] = field(serializer=deep_serialize)
    meta_parameters: PVector[Parameters] = field(serializer=deep_serialize)


# Idea: we let the interface take care of the index mappings. Just make it so that each level has access to all the info it needs

# ============================================================================
# PYTREE REGISTRATIONS
# ============================================================================

Parameter.is_learnable
# Register leaf types first
register_pytree(Parameter, {"is_learnable", "min_value", "max_value"})
register_pytree(State, {"is_batched", "is_stateful"})
register_pytree(Logs, set())
register_pytree(RecurrentState, set())
register_pytree(VanillaRecurrentState, {"activation_fn"})
register_pytree(LSTMState, set())
register_pytree(RNN, set())
register_pytree(MLP, set())
register_pytree(UOROState, set())

# Register container types that depend on leaf types
register_pytree(Parameters, set())
register_pytree(States, set())

# Register top-level container last
register_pytree(GodState, set())


"""
1. I need to adopt equinox inference mode
2. checking virtual minibatch is a symptom of potential infinite streaming data. no way to jit infinitely composed functions
4. there should typically be len size 3 (state, param) pairs where last corresponds to test loss. 
typically user can put the identity optimizer and all is well but potentially they could also just choose whatever optimizer.
5. I can add an is_stateful and a config flag to set all to true or some subset to true
i.e. validation inference state minibatch=1 basically means it gets reset every time which means its not stateful so of 
course we dont have to include it in our influence tensor construction. so set my validation states to false. 
i.e. if actually my virtual minibatch size is 1 for all levels, then technically my rnn activations are not stateful since they get
reset at the end of every turn. then I can set those to false as well and code works the same. 
message: virtual minibatch implies we do gradient updates within a time series, not when time series is finished
and this is stateful persistence is overriden at end of example when some things are reset no matter what. 
6. batch norm in online is allowed but technically counts as a mistake on user's part. 
7. for recurrent steps, always do a scan on the incoming input assuming it is some series,
but then we do a proper scan for the virtual minibatches too. that way it is agnostic to 
whether the input proper is scannable in first place. need to do some massaging for base case 1 dim. 
This let's us obtain stateless rnns for decoders and what nots. 
8. also add logic to do unfoldr as well
8.5 okay so for the inner thing we can do
- unfold (generation)
- fold (used for recurrent classsification where only last element matters. the f ignores the previous prediction so only most recent)
- scan (classic rnn)
I should create these primitives,
make unfold integer dependent,
and make them a basic unit.

Then in rnn classification tasks, its natural to not be online.
If could be online but it would just be nonsensical. 

The goal of the outer scan is that its a scan over losses. i.e. theres only ever on sepcific type of
(DATA -> ENV -> (ENV, LOSS)) -> ENV -> [DATA] -> (ENV, [LOSS])


This helps me avoid having to deal with masking losses for nonsensical intermediate outputs.
So if reading out an intermediate state into a loss doesn't make sense, it won't do that.
So no need to check the sequence number or antyhing. 
Masking minibatches is still needed.


9. is_batched is to let me know how to build axes. theres no batched parameter as of yet bc that doesnt make sense to me
10.
If the is_stateful for rnn is false, then its offline and we can expect the structure to be
(time, batch, time) where the last time is taken care of the the inner scan. i.e. the data itself is structured with batch outermost
If the is_stateful for rnn is true, then its online and we can expect the structure to be
(time, batch, time) where the first time is across gradient updates, and the last time is the offline part, usually 1. 
11. This data structure should be delivered by the dataloader system, not massaged by the inference system
12. if everything is set to is_batched false, then I can save on space while still running the same code.
i.e. for rnn if its offline I can just store one copy, say its not stateful, then I wont vmap over it so the same code
uses that same one copy instead of a batched duplicate, massively saving on space. 
This is in special case when I want to share the same initial state across all batches. But usually this isn't
the case since I probs want rando init for each element in the batch. 
I will need at least one dummy batched axis for this technique to work though. 
13. It doesnt make sense even in offline case to put is_stateful false state as a separate input 
since it is isomorphic to just putting everything in env which is THE separate input to begin with. 
14. The loss function should be part of the interface now as well.
The dataset layer should not matter at all. The interface should query whether I get this specific type of 
input data or this output data or None. THe model outputs will now be stored in the env as well. 
Then the loss functions should be task specific i.e.
- classification
- regression
- 

15. There will be two learning algorithms for each level. One for transition and one for readout.
THe pattern is very simple. 

it doesnt make sense to do an exact ecs style systems things because
I never do the same operation on a list of components.

"""
