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

from meta_learn_lib.config import *
from meta_learn_lib.create_axes import create_axes
from meta_learn_lib.env import *
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.lib_types import *
from meta_learn_lib.util import get_activation_fn


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
                n_in = [node_features[n] for n in node_graph[node_name] if n != pred_source][-1]
                last_sub_node = list(TopologicalSorter(node_graph).static_order())[-1]
                sub_features = get_output_shapes(graph, nodes, n_in)
                n_out = (n_in[0],) + sub_features[last_sub_node]
                node_features[node_name] = n_out
                node_features = {**node_features, **sub_features}
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


"""
I need utput shapes
so that I can know what the output shape of a scan is


1. in a subgraph, the output shapes of all nodes on the bottom layer of a dag are concatenated to produce
one big output shape.
2. I need to know what this output shape is in order to initialize the size of my autoregressive start token. 
3. I also need to know the output shape is in order to 

"""


def create_inference_state[ENV](
    nodes: dict[str, Node],
    meta_interface: dict[str, GodInterface[ENV]],
    node_features: dict[str, tuple[int, ...]],
    is_persistent: Persistent,
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
                        is_stateful=is_persistent.reset_states,
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

                gru_state = RecurrentState(activation=State(value=activation, is_stateful=is_persistent.reset_states))
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
                    h=State(value=activation, is_stateful=is_persistent.reset_states),
                    c=State(value=cell, is_stateful=is_persistent.reset_states),
                )
                env = interface.put_lstm_state(env, lstm_state)
                env = interface.put_prng(env, k3)
            case Scan(graph, autoregressive_mask, pred_source, start_token):
                k1, prng = jax.random.split(prng, 2)
                shape = node_features[node_name][1:]  # remove time dimension
                match start_token:
                    case "zeros":
                        token = jnp.zeros(shape)
                env = interface.put_autoregressive_predictions(
                    env, State(value=token, is_stateful=is_persistent.reset_states)
                )
                env = interface.put_prng(env, k1)
            case _:
                # could save space here on deterministic nodes
                k1, prng = jax.random.split(prng, 2)
                env = interface.put_prng(env, k1)

    return env


# def create_inference_parameters[ENV](
#     nodes: dict[str, Node],
#     meta_interface: dict[str, GodInterface[ENV]],
#     node_features: dict[str, tuple[int, ...]],
#     is_persistent: Persistent,
#     is_init: bool,
#     env: ENV,
#     prng: PRNG,
# ) -> ENV:

#     for node_name, node in nodes.items():
#         interface = meta_interface[node_name]
#         match node:
#             case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
#                 k1, k2, prng = jax.random.split(prng, 3)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(k1, (nn_layer.n,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((nn_layer.n,)))

#                 rnn_state = VanillaRecurrentState(
#                     activation=State(
#                         value=activation,
#                         is_stateful=is_persistent.reset_states,
#                     ),
#                     activation_fn=nn_layer.activation_fn,
#                 )
#                 env = interface.put_vanilla_rnn_state(env, rnn_state)
#                 env = interface.put_prng(env, k2)
#             case GRULayer(n, use_bias, use_random_init):
#                 k1, k2, prng = jax.random.split(prng, 3)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(k1, (n,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((n,)))

#                 gru_state = RecurrentState(activation=State(value=activation, is_stateful=is_persistent.reset_states))
#                 env = interface.put_gru_state(env, gru_state)
#                 env = interface.put_prng(env, k2)
#             case LSTMLayer(n, use_bias, use_random_init):
#                 k1, k2, k3, prng = jax.random.split(prng, 4)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(k1, (n,)))
#                     cell = ACTIVATION(jax.random.normal(k2, (n,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((n,)))
#                     cell = ACTIVATION(jnp.zeros((n,)))

#                 lstm_state = LSTMState(
#                     h=State(value=activation, is_stateful=is_persistent.reset_states),
#                     c=State(value=cell, is_stateful=is_persistent.reset_states),
#                 )
#                 env = interface.put_lstm_state(env, lstm_state)
#                 env = interface.put_prng(env, k3)
#             case Scan(graph, autoregressive_mask, pred_source, start_token):
#                 k1, prng = jax.random.split(prng, 2)
#                 shape = node_features[node_name][1:]  # remove time dimension
#                 match start_token:
#                     case "zeros":
#                         token = jnp.zeros(shape)
#                 env = interface.put_autoregressive_predictions(
#                     env, State(value=token, is_stateful=is_persistent.reset_states)
#                 )
#                 env = interface.put_prng(env, k1)
#             case _:
#                 # could save space here on deterministic nodes
#                 k1, prng = jax.random.split(prng, 2)
#                 env = interface.put_prng(env, k1)

#     return env


def generate_states[ENV](
    meta_interface: dict[str, GodInterface[ENV]],
    data_shape: tuple[tuple[int, ...], tuple[int, ...]],
    transition_graph: dict[str, list[str]],
    readout_graph: dict[str, list[str]],
    nodes: dict[str, Node],
    is_persistent: Persistent,
    is_init: bool,
    batch: list[int],  # vmap this many times
    prng: PRNG,
) -> Callable[[ENV], ENV]:
    node_features = get_output_shapes({}, transition_graph, nodes, data_shape)
    node_features = get_output_shapes(node_features, readout_graph, nodes, data_shape)

    def create_env(env: ENV) -> ENV:
        key = prng

        # 1. Do it for state if true
        f1 = lambda k: create_inference_state(nodes, meta_interface, node_features, is_persistent, is_init, env, k)
        k1, key = jax.random.split(key, 2)
        axes = create_axes(f1(k1))
        print("asdfasdf")
        eqx.tree_pprint(axes.serialize())
        batched_keys = jax.random.split(k1, math.prod(batch)).reshape(*batch)
        b_f1 = functools.reduce(lambda f, _: eqx.filter_vmap(f, in_axes=0, out_axes=axes), batch, f1)
        env = b_f1(batched_keys)

        # 2. use it for node parameter
        # for node_name in TopologicalSorter(node_graph).static_order():
        #     ...

        # 3. use it for learning states

        # 4. use it for optimizer states and parameters

        # 5. use it for objective hyperparameters

        return env

    return create_env


# def create_inference_parameters(
#     node_interface: dict[str, GodInterface[GodState]],
#     feature_shape: tuple[int, ...],
#     node_graph: dict[str, list[str]],
#     nodes: dict[str, Node],
#     is_persistent: Persistent,
#     static_prng: PRNG,
#     batch_prng: PRNG,
#     env: GodState,
# ) -> GodState:
#     node_features: dict[str, tuple[int, ...]] = {}
#     for node_name in TopologicalSorter(node_graph).static_order():
#         interface = node_interface[node_name]
#         match nodes[node_name]:
#             case NNLayer(n, activation_fn, use_bias, layer_norm):
#                 k1, batch_prng, jax.random.split(batch_prng, 2)
#                 node_features[node_name] = (n,)
#                 if len(node_graph[node_name]) == 0:
#                     n_in = math.prod(feature_shape)
#                 else:
#                     n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])

#                 linear = eqx.nn.Linear(n_in, n, use_bias=use_bias, key=k1)
#                 new_weight = jax.random.normal(k, (n, n_in)) * jnp.sqrt(1 / n_in)
#                 new_bias = jnp.zeros((n,)) if use_bias else None
#                 where = lambda l: (l.weight, l.bias)
#                 new_linear: list[eqx.Module] = [eqx.tree_at(where, linear, (new_weight, new_bias))]

#                 match layer_norm:
#                     case LayerNorm(epsilon, use_weight, use_bias):
#                         layer = [eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)]
#                     case None:
#                         layer: list[eqx.Module] = []

#                 nn_layer = eqx.nn.Sequential(new_linear + layer + [eqx.nn.Lambda(get_activation_fn(activation_fn))])
#                 env = interface.put_mlp_param(
#                     env,
#                     MLP(
#                         model=Parameter[eqx.nn.Sequential](
#                             value=nn_layer,
#                             is_batched=is_persistent.parameter_t != 1,
#                             is_learnable=True,
#                             min_value=-math.inf,
#                             max_value=math.inf,
#                         )
#                     ),
#                 )
#                 k2, batch_prng = jax.random.split(batch_prng, 2)
#                 env = interface.put_prng(env, k2)
#                 if is_persistent.parameter_t is not None and is_persistent.parameter_t > 1:
#                     env = interface.put_tick(env, 0)
#             case VanillaRNNLayer(nn_layer, use_random_init, time_constant):
#                 k1, prng = jax.random.split(prng, 2)
#                 node_features[node_name] = (nn_layer.n,)
#                 if len(node_graph[node_name]) == 0:
#                     n_in = math.prod(feature_shape)
#                 else:
#                     n_in = sum(math.prod(node_features[c]) for c in node_graph[node_name])

#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(k1, (nn_layer.n,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((nn_layer.n,)))

#                 rnn_state = VanillaRecurrentState(
#                     activation=State(
#                         value=activation,
#                         is_batched=...,
#                         is_stateful=is_persistent.state_t != 1,
#                     ),
#                     activation_fn=nn_layer.activation_fn,
#                 )


# def create_env(
#     config: GodConfig,
#     feature_shapes: list[tuple[int, ...]],
#     node_interfaces: list[dict[str, GodInterface[GodState]]],
#     learn_interfaces: list[tuple[GodInterface[GodState], GodInterface[GodState]]],
#     task_interfaces: list[GodInterface[GodState]],
#     prng: PRNG,
# ) -> GodState:
#     prng1, prng2, prng3, prng = jax.random.split(prng, 4)
#     env = GodState(
#         learning_states=pmap({}),
#         inference_states=pmap({}),
#         validation_learning_states=pmap({}),
#         parameters=pmap({}),
#         general=pmap({}),
#         prng=pmap(
#             {
#                 i: batched(jax.random.split(key, v.num_examples_in_minibatch))
#                 for i, (v, key) in enumerate(zip(config.data.values(), jax.random.split(prng1, len(config.data))))
#             }
#         ),
#         prng_learning=pmap({i: key for i, key in enumerate(jax.random.split(prng2, len(config.learners)))}),
#         start_epoch=jnp.array(0),
#     )
#     general: PMap[int, General] = pmap({})

#     # Create inference states
#     load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
#     for _, data_config in sorted(config.data.items()):
#         prng1, prng = jax.random.split(prng, 2)
#         vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
#         transition_state_vl: PMap[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
#         env: GodState = env.transform(["inference_states", len(env.inference_states)], lambda _: transition_state_vl)

#     parameter = create_inference_parameter(config, n_in_shape, prng3)
#     env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
#     for i, (_, learn_config) in enumerate(sorted(config.learners.items())):
#         prng1, prng = jax.random.split(prng, 2)

#         # Create learning state and new parameter
#         prev_parameter = parameter
#         learning_parameter = create_learning_parameter(learn_config)
#         parameter = Parameter(transition_parameter=None, readout_fn=None, learning_parameter=learning_parameter)
#         env = env.transform(["parameters", len(env.parameters)], lambda _: parameter)
#         learning_state_vl = create_learning_state(learn_config, env, prev_parameter, learn_interfaces[i], prng1)

#         if learn_config.track_logs:
#             logs = Logs(
#                 gradient=jnp.zeros_like(filter_hyperparam(prev_parameter)),
#                 hessian_contains_nans=jnp.array(False),
#                 immediate_influence_contains_nans=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#                 largest_eigenvalue=jnp.array(0.0),
#             )
#         else:
#             logs = None

#         if learn_config.track_special_logs:
#             special_logs = SpecialLogs(
#                 influence_tensor=(
#                     jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
#                 ),
#                 immediate_influence_tensor=(
#                     jnp.zeros_like(it) if (it := learning_state_vl.influence_tensor) is not None else None
#                 ),
#                 largest_jac_eigenvalue=jnp.array(0.0),
#                 jacobian=jnp.zeros((filter_hyperparam(env).size, filter_hyperparam(prev_parameter).size)),
#             )
#         else:
#             special_logs = None

#         general = general.set(
#             len(general),
#             General(
#                 current_virtual_minibatch=jnp.array(0),
#                 logs=logs,
#                 special_logs=special_logs,
#             ),
#         )

#         env = env.transform(["learning_states", len(env.learning_states)], lambda _: learning_state_vl)

#     env = env.set(general=general)

#     # create learning states for validation
#     validation_learning_states: PMap[int, LearningState] = pmap({})
#     for i, _ in enumerate(islice(config.learners.items(), 1, None), 1):
#         prng1, prng = jax.random.split(prng, 2)
#         learning_state_vl = create_learning_state(
#             config.learners[min(config.learners.keys())],
#             env,
#             learn_interfaces[i].get_state_pytree(env),
#             validation_learn_interfaces[i],
#             prng1,
#         )
#         validation_learning_states = validation_learning_states.set(i, learning_state_vl)
#     env = env.set(validation_learning_states=validation_learning_states)

#     return env


# def create_state(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> PMap[int, InferenceState]:
#     transition_state: PMap[int, InferenceState] = pmap({})
#     n_in, *_ = n_in_shape
#     for i, transition_fn in config.transition_function.items():
#         match transition_fn:
#             case NNLayer(n_h, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
#                 prng1, prng = jax.random.split(prng, 2)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((n_h,)))
#                 transition_state = transition_state.set(
#                     i,
#                     InferenceState(
#                         rnn=RNNState(
#                             activation=activation,
#                             n_h=n_h,
#                             n_in=n_in,
#                             activation_fn=activation_fn,
#                         ),
#                         lstm=None,
#                     ),
#                 )
#                 n_in = n_h  # Update n_in for the next layer
#             case GRULayer(n_h, use_bias, use_in_readout, use_random_init):
#                 prng1, prng = jax.random.split(prng, 2)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((n_h,)))

#                 transition_state = transition_state.set(
#                     i,
#                     InferenceState(
#                         rnn=RNNState(
#                             activation=activation,
#                             n_h=n_h,
#                             n_in=n_in,
#                             activation_fn="",
#                         ),
#                         lstm=None,
#                     ),
#                 )
#                 n_in = n_h
#             case LSTMLayer(n_h, use_bias, use_in_readout, use_random_init):
#                 prng1, prng2, prng = jax.random.split(prng, 3)
#                 if use_random_init:
#                     activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))
#                     cell = ACTIVATION(jax.random.normal(prng2, (n_h,)))
#                 else:
#                     activation = ACTIVATION(jnp.zeros((n_h,)))
#                     cell = ACTIVATION(jnp.zeros((n_h,)))

#                 transition_state = transition_state.set(
#                     i,
#                     InferenceState(
#                         lstm=LSTMState(h=activation, c=cell, n_h=n_h, n_in=n_in),
#                         rnn=None,
#                     ),
#                 )
#                 n_in = n_h
#             case IdentityLayer():
#                 transition_state = transition_state.set(
#                     i,
#                     InferenceState(
#                         rnn=None,
#                         lstm=None,
#                     ),
#                 )
#             case _:
#                 raise ValueError("Unsupported transition function")
#     return transition_state


# def create_inference_parameter(config: GodConfig, n_in_shape: tuple[int, ...], prng: PRNG) -> Parameter:
#     transition_parameter: PMap[int, InferenceParameter] = pmap({})
#     n_in_size = 0
#     n_in, *_ = n_in_shape
#     for i, transition_fn in config.transition_function.items():
#         match transition_fn:
#             case NNLayer(n_h, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
#                 prgn1, prng2, prng = jax.random.split(prng, 3)
#                 W_in = jax.random.normal(prgn1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
#                 W_rec = jnp.linalg.qr(jax.random.normal(prng2, (n_h, n_h)))[0]
#                 w_rec = jnp.hstack([W_rec, W_in])
#                 b_rec: jax.Array | None = jnp.zeros((n_h,)) if use_bias else None
#                 match layer_norm:
#                     case LayerNorm(epsilon, use_weight, use_bias):
#                         layer = eqx.nn.LayerNorm(n_h, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
#                     case None:
#                         layer = None

#                 transition_parameter = transition_parameter.set(
#                     i,
#                     InferenceParameter(
#                         rnn=RNN(
#                             w_rec=w_rec,
#                             b_rec=b_rec,
#                             layer_norm=layer,
#                         ),
#                         gru=None,
#                         lstm=None,
#                     ),
#                 )
#                 if use_in_readout:
#                     n_in_size += n_h
#                 n_in = n_h  # Update n_in for the next layer
#             case GRULayer(n_h, use_bias, use_in_readout, use_random_init):
#                 prng1, prng = jax.random.split(prng, 2)
#                 gru = eqx.nn.GRUCell(n_in, n_h, use_bias=use_bias, key=prng1)
#                 transition_parameter = transition_parameter.set(i, InferenceParameter(gru=gru, rnn=None, lstm=None))
#                 if use_in_readout:
#                     n_in_size += n_h
#                 n_in = n_h
#             case LSTMLayer(n_h, use_bias, use_in_readout, use_random_init):
#                 prng1, prng = jax.random.split(prng, 2)
#                 lstm = eqx.nn.LSTMCell(n_in, n_h, use_bias=use_bias, key=prng1)
#                 transition_parameter = transition_parameter.set(i, InferenceParameter(lstm=lstm, rnn=None, gru=None))
#                 if use_in_readout:
#                     n_in_size += n_h
#                 n_in = n_h
#             case IdentityLayer():
#                 pass
#             case _:
#                 raise ValueError("Unsupported transition function")

#     match config.readout_function:
#         case FeedForwardConfig(ffw_layers):
#             layers = []
#             data_in, *_ = n_in_shape
#             n_in_size += data_in
#             for i, layer in ffw_layers.items():
#                 match layer:  # purely for pattern matching, no other case should actually exist
#                     case NNLayer(n, activation_fn, use_bias, use_in_readout, layer_norm, use_random_init):
#                         match layer_norm:
#                             case LayerNorm(epsilon, use_weight, use_bias):
#                                 layer = eqx.nn.LayerNorm(n, eps=epsilon, use_weight=use_weight, use_bias=use_bias)
#                             case None:
#                                 layer = None

#                         layers.append((n, use_bias, get_activation_fn(activation_fn), layer))
#             prng1, prng = jax.random.split(prng, 2)
#             readout_fn = CustomSequential(layers, n_in_size, prng1)
#         case _:
#             raise ValueError("Unsupported readout function")

#     return Parameter(
#         transition_parameter=TransitionParameter(param=transition_parameter),
#         readout_fn=readout_fn,
#         learning_parameter=None,
#     )


# def create_optimizer_parameter(optimizer: Optimizer) -> LearningParameter:
#     parameter = LearningParameter(
#         learning_rate=None, weight_decay=None, rflo_timeconstant=None, multiple_parameters=None
#     )
#     match optimizer:
#         case (
#             SGDConfig()
#             | SGDNormalizedConfig()
#             | AdamConfig()
#             | ExponentiatedGradientConfig()
#             | ExponentiatedGradientAdamConfig() as opt
#         ):
#             _, lr_backward = hyperparameter_reparametrization(opt.learning_rate.hyperparameter_parametrization)
#             _, wd_backward = hyperparameter_reparametrization(opt.weight_decay.hyperparameter_parametrization)
#             parameter = parameter.set(
#                 learning_rate=Hyperparameter(
#                     value=lr_backward(jnp.array([opt.learning_rate.value])),
#                     learnable=opt.learning_rate.learnable,
#                     min_value=opt.learning_rate.min_value,
#                     max_value=opt.learning_rate.max_value,
#                 ),
#                 weight_decay=Hyperparameter(
#                     value=wd_backward(jnp.array([opt.weight_decay.value])),
#                     learnable=opt.weight_decay.learnable,
#                     min_value=opt.weight_decay.min_value,
#                     max_value=opt.weight_decay.max_value,
#                 ),
#             )
#         case RecurrenceConfig(recurrence_optimizer, out_optimizer):
#             recurrence_param = create_optimizer_parameter(recurrence_optimizer)
#             out_param = create_optimizer_parameter(out_optimizer)
#             parameter = parameter.set(multiple_parameters=(recurrence_param, out_param))

#     return parameter


# def create_learning_parameter(learn_config: LearnConfig) -> LearningParameter:
#     parameter = create_optimizer_parameter(learn_config.optimizer)

#     match learn_config.learner:
#         case RFLOConfig(time_constant, use_reverse_mode):
#             parameter = parameter.set(rflo_timeconstant=time_constant)

#     return parameter


# def create_learning_state(
#     learn_config: LearnConfig,
#     env: GodState,
#     parameter: Parameter,
#     learn_interface: LearnInterface[GodState],
#     prng: PRNG,
# ) -> LearningState:
#     state = LearningState(
#         influence_tensor=None, influence_tensor_squared=None, uoro=None, opt_state=None, rflo_t=None, rtrl_t=None
#     )
#     flat_state = learn_interface.get_state(env)
#     flat_param = filter_hyperparam(parameter)
#     match learn_config.learner:
#         case RTRLConfig() | RFLOConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig():
#             # prng1, prng = jax.random.split(prng, 2)
#             influence_tensor = jnp.zeros((flat_state.size, flat_param.size))
#             influence_tensor_squared = jnp.zeros((flat_state.size, flat_param.size))
#             state = state.set(
#                 influence_tensor=JACOBIAN(influence_tensor), influence_tensor_squared=JACOBIAN(influence_tensor_squared)
#             )
#         case UOROConfig():
#             prng1, prng2, prng = jax.random.split(prng, 3)

#             # Step 1: Get the tree structure and leaves
#             leaves = jax.tree_util.tree_leaves(parameter, is_leaf=lambda x: isinstance(x, CustomSequential))
#             treedef = jax.tree_util.tree_structure(parameter, is_leaf=lambda x: isinstance(x, CustomSequential))

#             # Step 2: Split keys for each leaf
#             keys = jax.random.split(prng1, len(leaves))
#             keys_tree = jax.tree_util.tree_unflatten(treedef, keys)

#             # Step 3: Function to edit each leaf
#             def edit_fn(leaf, key):
#                 if isinstance(leaf, CustomSequential):
#                     return jax.tree.map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, leaf)
#                 else:
#                     return jax.random.normal(key, leaf.shape) if eqx.is_array(leaf) else leaf

#             _parameter = jax.tree.map(edit_fn, parameter, keys_tree, is_leaf=lambda x: isinstance(x, CustomSequential))

#             a = jax.random.normal(prng2, (flat_state.size,))
#             b = filter_hyperparam(_parameter)
#             uoro = UOROState(A=a, B=b)
#             state = state.set(uoro=uoro)
#         case BPTTConfig():
#             ...
#         case IdentityConfig():
#             ...

#     match learn_config.learner:
#         case RFLOConfig():
#             state = state.set(rflo_t=jax.lax.stop_gradient(jnp.array(1)))
#         case RTRLConfig() | RTRLHessianDecompConfig() | RTRLFiniteHvpConfig():
#             state = state.set(rtrl_t=jax.lax.stop_gradient(jnp.array(1)))

#     _opt = learn_interface.get_optimizer(env)
#     if _opt is not None:
#         opt_state = _opt.init(flat_param)
#         state = state.set(opt_state=opt_state)

#     return state


# def reinitialize_env(
#     env: GodState,
#     config: GodConfig,
#     n_in_shape: tuple[int, ...],
#     prng: PRNG,
# ) -> GodState:
#     # Create inference states
#     inference_states: PMap[int, PMap[int, InferenceState]] = pmap({})
#     load_state = eqx.filter_vmap(create_state, in_axes=(None, None, 0))
#     for i, (_, data_config) in enumerate(sorted(config.data.items())):
#         prng1, prng = jax.random.split(prng, 2)
#         vl_prngs = jax.random.split(prng1, data_config.num_examples_in_minibatch)
#         transition_state_vl: PMap[int, InferenceState] = load_state(config, n_in_shape, vl_prngs)
#         inference_states = inference_states.set(i, transition_state_vl)
#     env = env.set(inference_states=inference_states)

#     return env
