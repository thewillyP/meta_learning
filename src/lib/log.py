import jax

from lib.config import GodConfig
from lib.env import GodState
from lib.util import hyperparameter_reparametrization


def get_logs(config: GodConfig, env: GodState) -> tuple[jax.Array, ...]:
    forward, _ = hyperparameter_reparametrization(config.learners[0].hyperparameter_parametrization)
    tr_gr = env.general[0].logs.gradient
    tr_gr_norm = jax.numpy.linalg.norm(tr_gr) if tr_gr is not None else jax.numpy.array(0.0)
    meta_gr = env.general[1].logs.gradient
    meta_gr_norm = jax.numpy.linalg.norm(meta_gr) if meta_gr is not None else jax.numpy.array(0.0)
    return (
        forward(env.parameters[1].learning_parameter.learning_rate),
        tr_gr_norm,
        meta_gr_norm,
    )
