import math
from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from itertools import islice
from pyrsistent import pmap
from pyrsistent.typing import PMap
from graphlib import TopologicalSorter
import functools
from pyrsistent import pmap, pvector


from meta_learn_lib.config import *
from meta_learn_lib.create_axes import create_axes, diff_axes
from meta_learn_lib.env import *
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import *
from meta_learn_lib.optimizer import get_opt_state
from meta_learn_lib.util import get_activation_fn, hyperparameter_reparametrization


"""
crazy idea new way to generate parameters/states that reflect the underlying generative process

I start with an initial function that knows how to take model parameters and output state. 
Then I wrap this function with a higher order function that knows how to take in hyperparameter 
and output optimizer states and influence tensors and the parameters themselves. 

But then I can wrap this around that same function that knows how to take in hyperhyperparameters
and output optimizer state, influence tensor, and hyperparameters. 

then I can just feed in hyperhyperparameters and get the whole create env for free.

Additionally I can use each intermediate I know how to create X function as reinitialization
type functions so if meet certain number of ticks then I will trigger that function at that level. 


basically in dS/dP, everything in S = f(P) 
needs to be my generative function.

_ = f0(h)
level 0: h = f1(theta)  # vmap over minibatch, num tasks
level 1: h, theta, opt_state_1, influence_tensor_1 = f2(alpha)  # vmap over num opt in level 1
level 2: h, theta, opt_state_1, influence_tensor_1, alpha, opt_state_2, influence_tensor_2 = f3(beta)  # vmap over num opt in level 2
level 3: h, theta, opt_state_1, influence_tensor_1, alpha, opt_state_2, influence_tensor_2, beta, opt_state_3, influence_tensor_3 = f4(_) # you get it
which cannot be a function of anything since its at the top

g(f-1, h) = f0(h)
g(f0, theta) = f1(theta)
doesnt make sense to apply g to f0 since there are no learning parameters being made here. so f1 must be the base case

f1(theta)
g(id, theta, i1) = f1(theta)
g(f1, alpha, i2) = f2(alpha)
g(f2, beta, i3) = f3(beta)
g(f3, _, i4) = f4(_)

g(g(g(g(id, theta, i1), alpha, i2), beta, i3), _, i4)

I also need someway to actually create theta to pass into f1. g since its glue 
doesn't know how to create the actual theta. 

ID as the base case makes sense because even if I create opt and learning state from it, 
because the interface ignores any inputs and just outputs the env, the env remains unchanged so im gucci. 
interface by default should return jnp 0 on get param and state so it should still work. 

and now because these f1..f4 are so cleanly separable, I can call vmap 
at each level directly to easily get my batched structure without interference. 

and using interfaces, I can actually get f0..f4 using the same function with different interfaces

I must attach resets to the beginning of the function so I can get zero jacobians.

I need 4 levels actually for persistence. 

Final form:

foldl (\acc i -> g acc i) id [i1, i2, i3, i4] :: ENV -> ENV

this can eventually become probabilistic as well!!!!
like if I fold over a sublist then I can use the previous meta parameters (these could be NNs!)
to initialize this level's parameters instead of random init.
Then I can just use reparametrization trick!! 

How do hp like rnn time constant or kl regularizaer that needs to be optimizized at level 2 but are created actually at level 1 work?
well I have a function of beta that ouputs an env. 
inside this function I create the alphas but I dont create the time constant yet.
but I am given this f :: env -> env and I can plug in my alpha into this f to get out a more complete
env. so this env now contains my alpha, my time constant (at level 2), and everything else.
THEN to create the larning states, I can then take this same function
with some interface I compute the jacobian by plugging in the state wrt to parameters.
but now because the env I created contains all the hyperparmeters, I can properly use the interface to get the wrt to. 

so it doesnt matter at what point I created the hyperparemter, the order of evaluation and interface helps me
get the correct values. 

IMPORTANT: this is just to say it works as an initialization. 
if I pass in a fully initialized env it should still work properly reinitializing everything. 
like if I want to take derivative wrt to the initialization fn, taking in the complete env is needed. 

Just realized, if I initialize hyperparemeters at lower levels, it poses a question,
should my hyperparemters be batched or shared across batches? idk so I have to specify in the config. 


problem 1:
For vmap logic to work I need to construct bottom up
But by cosntruction of env I need to cosntruct from top down

problem 2:
what to do with hyperparemters created at lower levels in terms of batching?

"""


def get_output_shapes(
    node_features: dict[str, tuple[int, ...]],
    node_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    data_shape: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    x_shape, y_shape = data_shape
    for node_name in TopologicalSorter(node_graph).static_order():
        match nodes[node_name]:
            case NNLayer(n, activation_fn, use_bias, layer_norm):
                node_features[node_name] = (n,)
            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                node_features[node_name] = (nn_layer.n,)
            case GRULayer(n, use_bias, use_random_init):
                node_features[node_name] = (n,)
            case LSTMLayer(n, use_bias, use_random_init):
                node_features[node_name] = (n,)
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                # Because I dont know how to concatenate your multiple dependencies, im just going to take the last one
                scan_x_shape = [node_features[n] for n in node_graph[node_name] if n != pred_source][-1]
                scan_y_shape = node_features[pred_source]
                last_sub_node = list(TopologicalSorter(graph).static_order())[-1]
                node_features = get_output_shapes(node_features, graph, nodes, (scan_x_shape[1:], scan_y_shape[1:]))
                n_out = (scan_x_shape[0],) + node_features[last_sub_node]
                node_features[node_name] = n_out
            case Repeat(n):
                # same issue as scan, just take the last one
                n_in = [node_features[n] for n in node_graph[node_name]][-1]
                node_features[node_name] = (n,) + n_in
            case Concat():
                n_in = sum([math.prod(node_features[n]) for n in node_graph[node_name]])
                node_features[node_name] = (n_in,)
            case ToEmpty():
                node_features[node_name] = ()
            case UnlabeledSource():
                node_features[node_name] = x_shape
            case LabeledSource():
                node_features[node_name] = y_shape
            case _:
                node_features[node_name] = ()

    return node_features


def create_inference_state[ENV](
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
    node_features: dict[str, tuple[int, ...]],
    track_influence: bool,
    is_init: bool,
    env: ENV,
    prng: PRNG,
) -> ENV:

    for node_name, node in nodes.items():
        interface = meta_interface[node_name]
        match node:
            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (nn_layer.n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((nn_layer.n,)))

                rnn_state = VanillaRecurrentState(
                    activation=State(
                        value=activation,
                        is_stateful=track_influence,
                    ),
                    activation_fn=nn_layer.activation_fn,
                )
                env = interface.put_vanilla_rnn_state(env, rnn_state)
                env = interface.put_prng(env, k2)
            case GRULayer(n, use_bias, use_random_init):
                k1, k2, prng = jax.random.split(prng, 3)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))

                gru_state = RecurrentState(activation=State(value=activation, is_stateful=track_influence))
                env = interface.put_gru_state(env, gru_state)
                env = interface.put_prng(env, k2)
            case LSTMLayer(n, use_bias, use_random_init):
                k1, k2, k3, prng = jax.random.split(prng, 4)
                if use_random_init:
                    activation = ACTIVATION(jax.random.normal(k1, (n,)))
                    cell = ACTIVATION(jax.random.normal(k2, (n,)))
                else:
                    activation = ACTIVATION(jnp.zeros((n,)))
                    cell = ACTIVATION(jnp.zeros((n,)))

                lstm_state = LSTMState(
                    h=State(value=activation, is_stateful=track_influence),
                    c=State(value=cell, is_stateful=track_influence),
                )
                env = interface.put_lstm_state(env, lstm_state)
                env = interface.put_prng(env, k3)
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                k1, prng = jax.random.split(prng, 2)
                shape = node_features[node_name][1:]  # remove time dimension
                match start_token:
                    case "zeros":
                        token = jnp.zeros(shape)
                env = interface.put_autoregressive_predictions(env, State(value=token, is_stateful=track_influence))
                env = interface.put_prng(env, k1)
            case _:
                # could save space here on deterministic nodes
                k1, prng = jax.random.split(prng, 2)
                env = interface.put_prng(env, k1)

    return env


def create_inference_parameters[ENV](
    nodes: dict[str, Node],
    node_graph: dict[str, list[str]],
    meta_interface: dict[str, GodInterface[ENV]],
    node_features: dict[str, tuple[int, ...]],
    learnables: frozenset[str],
    env: ENV,
    prng: PRNG,
) -> ENV:

    for node_name, node in nodes.items():
        interface = meta_interface[node_name]
        is_learnable = node_name in learnables
        match node:
            case NNLayer(n, activation_fn, use_bias, layer_norm):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                linear = eqx.nn.Linear(n_in, n, use_bias=use_bias, key=k1)
                new_weight = jax.random.normal(k1, (n, n_in)) * jnp.sqrt(1 / n_in)
                new_bias = jnp.zeros((n,)) if use_bias else None
                where = lambda l: (l.weight, l.bias)
                new_linear: eqx.Module = eqx.tree_at(where, linear, (new_weight, new_bias))

                match layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
                    case None:
                        layer = eqx.nn.Identity()

                nn_layer = eqx.nn.Sequential([new_linear, layer, eqx.nn.Lambda(get_activation_fn(activation_fn))])
                env = interface.put_mlp_param(
                    env,
                    MLP(
                        model=Parameter[eqx.nn.Sequential](
                            value=nn_layer,
                            is_learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                        )
                    ),
                )
                env = interface.put_prng(env, k2)

            case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, k3, prng = jax.random.split(prng, 4)

                W_in = jax.random.normal(k1, (nn_layer.n, n_in)) * jnp.sqrt(1 / n_in)
                W_rec = jnp.linalg.qr(jax.random.normal(k2, (nn_layer.n, nn_layer.n)))[0]
                w_rec = Parameter(
                    value=jnp.hstack([W_rec, W_in]),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                )
                b_rec: jax.Array = Parameter(
                    value=jnp.zeros((nn_layer.n,)),
                    is_learnable=nn_layer.use_bias,
                    min_value=-math.inf,
                    max_value=math.inf,
                )

                match nn_layer.layer_norm:
                    case LayerNorm(epsilon, use_weight, use_bias):
                        layer = Parameter(
                            value=eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias),
                            is_learnable=is_learnable,
                            min_value=-math.inf,
                            max_value=math.inf,
                        )
                    case None:
                        layer = Parameter(
                            value=eqx.nn.Identity(),
                            is_learnable=False,
                            min_value=-math.inf,
                            max_value=math.inf,
                        )

                rnn_param = RNN(w_rec=w_rec, b_rec=b_rec, layer_norm=layer)
                env = interface.put_vanilla_rnn_param(env, rnn_param)
                env = interface.put_prng(env, k3)
            case GRULayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                gru = Parameter(
                    value=eqx.nn.GRUCell(n_in, n, use_bias=use_bias, key=k1),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                )
                env = interface.put_gru_param(env, gru)
                env = interface.put_prng(env, k2)

            case LSTMLayer(n, use_bias, use_random_init):
                n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])
                k1, k2, prng = jax.random.split(prng, 3)

                lstm = Parameter(
                    value=eqx.nn.LSTMCell(n_in, n, use_bias=use_bias, key=k1),
                    is_learnable=is_learnable,
                    min_value=-math.inf,
                    max_value=math.inf,
                )
                env = interface.put_lstm_param(env, lstm)
                env = interface.put_prng(env, k2)
            case _:
                # could save space here on deterministic nodes
                k1, prng = jax.random.split(prng, 2)
                env = interface.put_prng(env, k1)

    return env


def create_hyperparameters[ENV](
    hps: dict[HP, HyperparameterConfig],
    meta_interface: dict[str, GodInterface[ENV]],
    learnables: frozenset[str],
    env: ENV,
    prng: PRNG,
) -> ENV:
    # make sure to loop through current meta interface's hyperparameters only. not ALL hyperparameters
    for hp_name, interface in meta_interface.items():
        hp_config = hps[hp_name]
        is_learnable = hp_name in learnables
        _, invert = hyperparameter_reparametrization(hp_config.hyperparameter_parametrization)
        match hp_config.kind:
            case "learning_rate":
                k1, prng = jax.random.split(prng, 2)
                lrs = Parameter(
                    value=jnp.full((hp_config.count,), invert(hp_config.value)),
                    is_learnable=is_learnable,
                    min_value=hp_config.min_value,
                    max_value=hp_config.max_value,
                )
                env = interface.put_learning_rate(env, lrs)
                env = interface.put_prng(env, k1)
            case "weight_decay":
                k1, prng = jax.random.split(prng, 2)
                wds = Parameter(
                    value=jnp.full((hp_config.count,), invert(hp_config.value)),
                    is_learnable=is_learnable,
                    min_value=hp_config.min_value,
                    max_value=hp_config.max_value,
                )
                env = interface.put_weight_decay(env, wds)
                env = interface.put_prng(env, k1)
            case "momentum":
                k1, prng = jax.random.split(prng, 2)
                ms = Parameter(
                    value=jnp.full((hp_config.count,), invert(hp_config.value)),
                    is_learnable=is_learnable,
                    min_value=hp_config.min_value,
                    max_value=hp_config.max_value,
                )
                env = interface.put_momentum(env, ms)
                env = interface.put_prng(env, k1)
            case "time_constant":
                k1, prng = jax.random.split(prng, 2)
                tcs = Parameter(
                    value=jnp.full((hp_config.count,), invert(hp_config.value)),
                    is_learnable=is_learnable,
                    min_value=hp_config.min_value,
                    max_value=hp_config.max_value,
                )
                env = interface.put_time_constant(env, tcs)
                env = interface.put_prng(env, k1)
            case "kl_regularizer_beta":
                k1, prng = jax.random.split(prng, 2)
                kl_betas = Parameter(
                    value=jnp.full((hp_config.count,), invert(hp_config.value)),
                    is_learnable=is_learnable,
                    min_value=hp_config.min_value,
                    max_value=hp_config.max_value,
                )
                env = interface.put_kl_regularizer_beta(env, kl_betas)
                env = interface.put_prng(env, k1)

    return env


def create_learner_states[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    method: GradientMethod,
    interface: GodInterface[ENV],
    readout_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
    track_influence: bool,
    env: ENV,
    prng: PRNG,
) -> ENV:
    match method:
        case RTRLConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig() | RFLOConfig():
            k1, k2, prng = jax.random.split(prng, 3)
            param = interface.get_param(env)

            def infl_fn(param: jax.Array) -> jax.Array:
                _env = interface.put_param(env, param)
                _env = factory(_env, k1)
                state = interface.get_state(_env)
                return state

            new_env = factory(env, k1)
            new_state = interface.get_state(new_env)
            if new_state.shape[0] > param.shape[0]:
                dhdp = eqx.filter_jacfwd(infl_fn)(param)
            else:
                dhdp = eqx.filter_jacrev(infl_fn)(param)
            env = interface.put_forward_mode_jacobian(new_env, dhdp)
            env = interface.put_prng(env, k2)
        case UOROConfig():
            k0, k1, k2, k3, prng = jax.random.split(prng, 5)
            env = factory(env, k0)

            param = interface.get_param(env)
            state = interface.get_state(env)

            # A: random init, shape = (|h|,)
            a = jax.random.normal(k1, state.shape)

            # B: start fully random, then zero out readout node params
            b_env = interface.put_param(env, jax.random.normal(k2, param.shape))

            for node_name in readout_graph.keys():
                node = nodes[node_name]
                node_interface = meta_interface[node_name]
                match node:
                    case NNLayer():
                        mlp = node_interface.get_mlp_param(b_env)
                        zeroed = jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, mlp)
                        b_env = node_interface.put_mlp_param(b_env, zeroed)
                    case VanillaRNNLayer():
                        rnn = node_interface.get_vanilla_rnn_param(b_env)
                        zeroed = jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, rnn)
                        b_env = node_interface.put_vanilla_rnn_param(b_env, zeroed)
                    case GRULayer():
                        gru = node_interface.get_gru_param(b_env)
                        zeroed = jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, gru)
                        b_env = node_interface.put_gru_param(b_env, zeroed)
                    case LSTMLayer():
                        lstm = node_interface.get_lstm_param(b_env)
                        zeroed = jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, lstm)
                        b_env = node_interface.put_lstm_param(b_env, zeroed)

            b = interface.get_param(b_env)

            uoro = UOROState(
                A=State(value=a, is_stateful=track_influence),
                B=State(value=b, is_stateful=track_influence),
            )
            env = interface.put_uoro_state(env, uoro)
            env = interface.put_prng(env, k3)
        case _:
            k1, k2, prng = jax.random.split(prng, 3)
            env = factory(env, k1)
            env = interface.put_prng(env, k2)

    return env


def vmap_factory[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    batch: list[int],  # vmap this many times
) -> Callable[[ENV, PRNG], ENV]:
    def vmap_env(env: ENV, prng: PRNG) -> ENV:
        k1, k2, prng = jax.random.split(prng, 3)
        old_axes = create_axes(env)
        new_axes = create_axes(factory(env, k1))
        axes = diff_axes(old_axes, new_axes)
        batched_keys = jax.random.split(k2, math.prod(batch)).reshape(*batch)
        b_fn = functools.reduce(
            lambda f, _: eqx.filter_vmap(f, in_axes=(None, 0), out_axes=axes), batch, lambda k: factory(env, k)
        )
        return b_fn(batched_keys)

    return vmap_env


def generate_states[ENV](
    factory: Callable[[ENV, PRNG], ENV],
    meta_interface: dict[str, GodInterface[ENV]],
    learn_interfaces: tuple[GodInterface[ENV], GodInterface[ENV]],
    meta_config: MetaConfig,
    hyperparameters: dict[HP, HyperparameterConfig],
    nodes: dict[str, Node],
    transition_graph: dict[str, list[str]],
    readout_graph: dict[str, list[str]],
    node_features: dict[str, tuple[int, ...]],
    create_inference_param: bool,  # its shared across levels. can be used to not gen params too to just gen hidden states
    is_init: bool,  # if need to reset things like RL ENV
) -> Callable[[ENV, PRNG], ENV]:

    learnables: frozenset[str] = frozenset().union(*[v.target for v in meta_config.learner.optimizer.values()])
    vl_learner, meta_learner = learn_interfaces
    node_graph = transition_graph | readout_graph

    def create_env(env: ENV, prng: PRNG) -> ENV:
        """
        okay so pass in the inference meta interface twice, one for state and one for parameter.
        then move on to optimizer parameters.
        because inference interfaces trivally create no optimizers I can always make them knowing I will get identity for first two.
        """

        """
        feel free here to make your parameters, hyperparameters, and learning states.
        if the meta interface at this level doesn't map to any hyperparameters, then
        this level wont make any hyerparameters. 
        if the meta interface at this level doesn't map to any parameters then it wont make any parameters. 
        if it doesnt map to any states, it wont make any states.
        this way I can also handle making validation states at level 2.  

        so it should be a loop over meta_interface, checking for each key if
        is state
        is parameter 
        is hyperparameter 
        and will create them one by one. the order doesn't even have to matter as long as everything
        gets created. then my other code doesn't have to loop through nodes or whatever

        then using the fully created env, I can create the validation learning and meta learning states from them.
        I can also create optimizer states from here

        New idea here.

        First of all I actually need to create a batched state on every level
        for training, validation and test.
        so this function will be passed in the num tasks and minibatch size for each.
        so actually I need to manually call vmap here.
        the other vmaps for the opt batching can still be composed outside in the scan. 

        the exception to creating something at every level is the parameters since its shared across levels. 
        everything else
        state
        hyperparameter
        learning state
        opt state

        they are attempted to be created at every level. that way I actually only need list of size 3, not 4. 
        the separation of the meta_interface between levels is whats important here.
        that lets me loop over it can do checks and make stuff. 

        final issue is how to make optimizers optimize only a portion?
        
        """

        # 1. Create inference state
        k1, prng = jax.random.split(prng, 2)
        f1 = lambda e, k: create_inference_state(
            nodes, meta_interface, node_features, meta_config.track_influence, is_init, e, k
        )
        batch_size = [
            meta_config.dataset_validation.task_batch_size,
            meta_config.dataset_validation.num_examples_in_minibatch,
        ]
        env = vmap_factory(f1, batch_size)(env, k1)

        # 2. Create parameters
        if create_inference_param:
            k2, prng = jax.random.split(prng, 2)
            env = create_inference_parameters(nodes, node_graph, meta_interface, node_features, learnables, env, k2)

        # 3. Create hyperparameters
        k3, prng = jax.random.split(prng, 2)
        env = create_hyperparameters(hyperparameters, meta_interface, learnables, env, k3)

        # 3. use it for learning states
        k1, k2, prng = jax.random.split(prng, 3)

        # order important, meta learner init must come first before vl learner since vl learner needs to know the shape of full env after factory applied, but meta doesnt.
        env = create_learner_states(
            factory,
            meta_config.learner.optimizer_learner.method,
            meta_learner,
            readout_graph,
            nodes,
            meta_interface,
            meta_config.track_influence,
            env,
            k1,
        )
        env = create_learner_states(
            lambda e, k: e,
            meta_config.learner.model_learner.method,
            vl_learner,
            readout_graph,
            nodes,
            meta_interface,
            meta_config.track_influence,
            env,
            k2,
        )

        # 4. use it for optimizer states
        env = get_opt_state(
            meta_config.learner.optimizer,
            meta_interface,
            env,
            hyperparameters,
            meta_config.track_influence,
        )

        return env

    return create_env


def create_env(
    config: GodConfig,
    feature_shapes: list[tuple[int, ...]],
    node_interfaces: list[dict[str, GodInterface[GodState]]],
    learn_interfaces: list[tuple[GodInterface[GodState], GodInterface[GodState]]],
    task_interfaces: list[GodInterface[GodState]],
    prng: PRNG,
) -> GodState:

    env = GodState(
        model_states=pvector(
            [
                ModelStates(
                    recurrent_states=pmap({}),
                    vanilla_recurrent_states=pmap({}),
                    lstm_states=pmap({}),
                    autoregressive_predictions=pmap({}),
                )
                for _ in range(len(config.levels) + 1)
            ]
        ),
        learning_states=pvector(
            [
                LearningStates(
                    influence_tensors=pmap({}),
                    uoros=pmap({}),
                    opt_states=pmap({}),
                )
                for _ in range(len(config.levels) + 1)
            ]
        ),
        meta_parameters=pvector(
            [
                Parameters(
                    mlps=pmap({}),
                    rnns=pmap({}),
                    grus=pmap({}),
                    lstms=pmap({}),
                    learning_rates=pmap({}),
                    weight_decays=pmap({}),
                    time_constants=pmap({}),
                    momentums=pmap({}),
                    kl_regularizer_betas=pmap({}),
                )
                for _ in range(len(config.levels) + 1)
            ]
        ),
        level_meta=pvector(
            [
                LevelMeta(
                    tick=jnp.array(0),
                    log=Logs(),
                    prngs=pmap({}),
                )
                for _ in range(len(config.levels) + 1)
            ]
        ),
        prng=env_prng,
    )

    prng1, prng2, prng3, prng = jax.random.split(prng, 4)
    env = GodState(
        learning_states=pmap({}),
        inference_states=pmap({}),
        validation_learning_states=pmap({}),
        parameters=pmap({}),
        general=pmap({}),
        prng=pmap(
            {
                i: batched(jax.random.split(key, v.num_examples_in_minibatch))
                for i, (v, key) in enumerate(zip(config.data.values(), jax.random.split(prng1, len(config.data))))
            }
        ),
        prng_learning=pmap({i: key for i, key in enumerate(jax.random.split(prng2, len(config.learners)))}),
        start_epoch=jnp.array(0),
    )
    general: PMap[int, General] = pmap({})

    # Create inference states
    load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
    for _, data_config in sorted(config.data.items()):
        prng1, prng = jax.random.split(prng, 2)
        vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
        transition_state_vl: PMap[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
        env: GodState = env.transform(["inference_states", len(env.inference_states)], lambda _: transition_state_vl)

    parameter = create_inference_parameter(config, n_in_shape, prng3)
    env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
    for i, (_, learn_config) in enumerate(sorted(config.learners.items())):
        prng1, prng = jax.random.split(prng, 2)

        # Create learning state and new parameter
        prev_parameter = parameter
        learning_parameter = create_learning_parameter(learn_config)
        parameter = Parameter(transition_parameter=None, readout_fn=None, learning_parameter=learning_parameter)
        env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
        learning_state_vl = create_learning_state(learn_config, env, prev_parameter, learn_interfaces[i], prng1)

        if learn_config.track_logs:
            logs = Logs(
                gradient=jnp.zeros_like(filter_hyperparam(prev_parameter)),
                hessian_contains_nans=jnp.array(False),
                immediate_influence_contains_nans=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                largest_eigenvalue=jnp.array(0.0),
            )
        else:
            logs = None

        if learn_config.track_special_logs:
            special_logs = SpecialLogs(
                influence_tensor=(
                    jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
                ),
                immediate_influence_tensor=(
                    jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
                ),
                largest_jac_eigenvalue=jnp.array(0.0),
                jacobian=jnp.zeros((filter_hyperparam(env).size, filter_hyperparam(prev_parameter).size)),
            )
        else:
            special_logs = None

        general = general.set(
            len(general),
            General(
                current_virtual_minibatch=jnp.array(0),
                logs=logs,
                special_logs=special_logs,
            ),
        )

        env = env.transform(["learning_states", len(env.learning_states)], lambda _: learning_state_vl)

    env = env.set(general=general)

    # create learning states for validation
    validation_learning_states: PMap[int, LearningState] = pmap({})
    for i, _ in enumerate(islice(config.learners.items(), 1, None), 1):
        prng1, prng = jax.random.split(prng, 2)
        learning_state_vl = create_learning_state(
            config.learners[min(config.learners.keys())],
            env,
            learn_interfaces[i].get_state_pytree(env),
            validation_learn_interfaces[i],
            prng1,
        )
        validation_learning_states = validation_learning_states.set(i, learning_state_vl)
    env = env.set(validation_learning_states=validation_learning_states)

    return env
