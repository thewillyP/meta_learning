from typing import Optional
import equinox as eqx
import jax
import optax
from pyrsistent import PClass, field, pmap, pvector, thaw
from pyrsistent.typing import PMap, PVector
from pyrsistent._pmap import PMap as PMapClass
from pyrsistent._pvector import PythonPVector

from meta_learn_lib.lib_types import *


def deep_serialize(_, obj):
    """Recursively serialize pyrsistent objects to Python built-ins"""
    if isinstance(obj, PClass):
        serialized = obj.serialize()
        return {k: deep_serialize(_, v) for k, v in serialized.items()}
    elif isinstance(obj, PMapClass):
        thawed = thaw(obj)
        return {k: deep_serialize(_, v) for k, v in thawed.items()}
    elif isinstance(obj, PythonPVector):
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


jax.tree_util.register_pytree_node(PythonPVector, _pvector_tree_flatten, _pvector_tree_unflatten)


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
    is_batched: bool = field()
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
    mlps: PMap[int, MLP] = field(serializer=deep_serialize)
    rnns: PMap[int, RNN] = field(serializer=deep_serialize)
    grus: PMap[int, Parameter[eqx.nn.GRUCell]] = field(serializer=deep_serialize)
    lstms: PMap[int, Parameter[eqx.nn.LSTMCell]] = field(serializer=deep_serialize)
    learning_rates: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    weight_decays: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    time_constants: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    momentums: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)
    kl_regularizer_betas: PMap[int, Parameter[jax.Array]] = field(serializer=deep_serialize)


class States(PClass):
    influence_tensors: PMap[int, State[JACOBIAN]] = field(serializer=deep_serialize)
    uoros: PMap[int, UOROState] = field(serializer=deep_serialize)
    opt_states: PMap[int, State[optax.OptState]] = field(serializer=deep_serialize)
    ticks: PMap[int, jax.Array] = field(serializer=deep_serialize)
    log: Logs = field(serializer=deep_serialize)
    recurrent_states: PMap[int, RecurrentState] = field(serializer=deep_serialize)
    vanilla_recurrent_states: PMap[int, VanillaRecurrentState] = field(serializer=deep_serialize)
    lstm_states: PMap[int, LSTMState] = field(serializer=deep_serialize)
    prngs: PMap[int, State[PRNG]] = field(serializer=deep_serialize)
    autoregressive_predictions: PMap[int, State[jax.Array]] = field(serializer=deep_serialize)


class GodState(PClass):
    meta_states: PVector[States] = field(serializer=deep_serialize)
    meta_parameters: PVector[Parameters] = field(serializer=deep_serialize)


class Outputs(PClass):
    prediction: PREDICTION = field(serializer=deep_serialize)
    logit: LOGITS = field(serializer=deep_serialize)


# Idea: we let the interface take care of the index mappings. Just make it so that each level has access to all the info it needs

# ============================================================================
# PYTREE REGISTRATIONS
# ============================================================================

# Register leaf types first
register_pytree(Parameter, {"is_learnable", "is_batched", "min_value", "max_value"})
register_pytree(State, {"is_stateful", "is_batched"})
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
register_pytree(Outputs, set())

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
input data or this output data or None. The model outputs will now be stored in the env as well. 
Then the loss functions should be task specific i.e.
- classification
- regression
- vae objective

I will actually have another interface to another object that is soley generated during readout.
That way there won't be confusion with getting pred from state and not being able to grad over it. 

15. There will be two learning algorithms for each level. One for the model and one for the optimizer.
Each level has a model (base, validation1, validation2, ...) and an optimizer (normal learner, meta learner1, meta learner2, ...)
and each will get a learning algorithm. 

Nvm, learning algorithm only makes sense at each optimizer level. 
Every learning algorithm needs a transition function and a readout function.
Then I get to specify what learning algorithm is used to compute the gradient of the readout function.
But that learning algorithm requires a transition function and a readout...

For the model level, the user can specify whatever learning algorithm for the readout,
but the transition fn will always be DATA -> ENV -> ENV where it is the identity and ignores DATA.
In fact, the data being passed in 

16. Batching will have to change to support meta learning. 
Given a set of examples, these need to be batch fed. 
Given a set of tasks, these need to be batch fed.
Each level should create a new batched level. 
Also the batched timestep becomes nonsensical at the meta level since it doesn't
make sense to me what it means to intermediate update within a task. Like partially complete a task
and then update hyperparemter? Like what is partial of a task even mean?

(T3, B3, T2, B2, T1, B1, T0, ...)

Here T0 should avtually be part of the features ...
B comes first before T since we want to aggregate on batches before doing an update with T.
T1 refers to the virtual minibatches coming from chunking the data into sublists
B2 refers to batch of tasks.
T2 refers to number of gradient steps to take.

Basically the pattern of T is how many update steps should actually be done with BPTT
minibatches controls update steps of model state
gradient steps controls steps of parameter state

THen B3 for test is trivially 1 although it could in theory be arbitrary. 
And T3 controls the number of total steps done in one RAM cycle. So if
I'm running online, this controls how many online steps I can actually do all in one RAM cycle
before needing to load in more data. It becomes a nice proxy for managing data streaming. 

17. basically I will need two DAGs now, one for transition and one for readout.
Both will have ENV has input but one is static (except for prng which it will get from transition ENV)
so the API is the same. 

18. The readout_gr will compute both dL/dht+1 and dL/dtheta instead of me computing separately
so that I dont' have to pass in a readout as well. This makes sense since learning
functions typically only care about the loss and gradient so getting the gr directly is just better. 

19. I need to specify a task family for each model level,
and then the base family gets familly of familied. 
Task family for validation because technically after I "train" on a task, I can evaluate on multiple tasks.
So each line in a batch of tasks sprouts into another branches of tasks. Batch within batch.

20. How is_persistent works is it determines if your stateful and or your learnable and if you need to be batched.
If true and a state, then yes stateful
If true and a parameter, then yes learn it
If true then regardless YOU MUST be batched. Otherwise not make sense to be batchless. 
If false then we can save space by not batching so only keep one copy.
This copy wont be updated because it gets reset because its not stateful.
Nor will it get updated due to learning since its not learnable. 

Pattern is doing a vmal turns a learnable batch into
sending it to be a single learnable parameter for that branch in the batch. Then it gets updated
and the update fn assumes only one copy of it. 

hyperhyerparameter: [B3, ...]
hyperparameter: [B3, B2, ...]
parameter: [B3, B2, B1, ...]

it doesnt make sense to do an exact ecs style systems things because
I never do the same operation on a list of components.

What does online mean?
Means for level 1 that everything is stateful, T=1 timesteps.
Means for level 2 that everything is stateful, T=1 timesteps.
If inner loop offline then T>1 timesteps.
If outer loop online then T=1 timesteps.
If outer loop offline then T>1 timesteps.
Offline means after T timesteps, new examples are drawn.
Online means after T timesteps, either next step is drawn or new step is begun. 

What if I want to achieve meta learning?
That is just OHO but offline version and batched across both task and examples where at each level there is a batch of tasks and a
batch of examples. 
MAML is just batched on level 1 and level 2 and offline both levels.


I still need to mask losses if I want to do online learning with scans that have periodic losses
I will put some token or label to tell it where to do masking instead of tracking sequence number or something

I still need to do readouts from env for output


I dont need a fold because my scan takes care of that. I can just output nothings. 

do I need to change my archiecture based on offline vs online. 
no because technically if rnn can be offline it can be put as scan on the readout section. 
but then if I want to reformat data to be online and run an online algorithm I would need to change it.

but if I put rnn in transition, then it can work for online since its in the transition.
but also it can work for offline since transition can literally be one step as well before reset.
but thats only true for folds like sequential mnist task.
what about output at every timestep tasks? 

like the readout changes. like if its offline should there be a readout??? 
"""
