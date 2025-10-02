import jax

from meta_learn_lib.config import GodConfig
from meta_learn_lib.env import GodState
from meta_learn_lib.util import hyperparameter_reparametrization


def get_logs(config: GodConfig, env: GodState) -> tuple[jax.Array, ...]:
    lr_forward, _ = hyperparameter_reparametrization(
        config.learners[0].optimizer.learning_rate.hyperparameter_parametrization
    )
    wd_forward, _ = hyperparameter_reparametrization(
        config.learners[0].optimizer.weight_decay.hyperparameter_parametrization
    )
    tr_gr = env.general[0].logs.gradient
    tr_gr_norm = jax.numpy.linalg.norm(tr_gr) if tr_gr is not None else jax.numpy.array(0.0)
    meta_gr = env.general[1].logs.gradient
    meta_gr_norm = jax.numpy.linalg.norm(meta_gr) if meta_gr is not None else jax.numpy.array(0.0)
    return (
        lr_forward(env.parameters[1].learning_parameter.learning_rate.value),
        wd_forward(env.parameters[1].learning_parameter.weight_decay.value),
        tr_gr_norm,
        meta_gr_norm,
    )
