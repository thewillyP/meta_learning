"""Math sanity tests for the VAE loss components.

Verifies KL closed-form formula and masked-recon behavior against hand-computed
expected values, including the pathological padded-input scenario that previously
caused inf/NaN gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from meta_learn_lib.config import (
    ELBOObjective,
    NoopObjective,
    RegressionObjective,
)
from meta_learn_lib.env import Outputs
from meta_learn_lib.loss_function import kl


def gaussian_kl_closed_form(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """KL(N(mu, exp(log_var)) || N(0, 1)) summed over latent dims, per example."""
    return 0.5 * np.sum(np.exp(log_var) + mu**2 - 1.0 - log_var, axis=-1)


def test_kl_gaussian_closed_form_all_valid():
    # mask shape matches "target": (batch, minibatch, *features). Mask collapsed inside kl().
    # Use shape (1, 4, 1, 28, 28) for a typical VAE setup — 4 examples in minibatch.
    mu = jnp.asarray([[[0.5, -0.3], [0.0, 1.2], [-1.0, 0.7], [2.0, -2.0]]])  # (1, 4, 2)
    log_var = jnp.asarray([[[-0.2, 0.1], [0.0, -0.5], [0.3, -0.3], [-1.0, 1.0]]])
    outputs = Outputs(mu=mu, log_var=log_var)
    mask = jnp.ones((1, 4, 1, 28, 28), dtype=bool)

    got = kl(ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(mu=0.0, log_var=0.0), outputs, mask)

    expected_per = gaussian_kl_closed_form(np.asarray(mu), np.asarray(log_var))
    expected = float(np.mean(expected_per))
    assert np.isclose(float(got), expected, rtol=1e-5), (float(got), expected)


def test_kl_masked_examples_excluded():
    # 4 examples; mask the middle two. Result should be the mean of KL over the unmasked pair.
    mu = jnp.asarray([[[0.5, -0.3], [10.0, 10.0], [99.0, 99.0], [-1.0, 0.7]]])
    log_var = jnp.asarray([[[-0.2, 0.1], [5.0, 5.0], [5.0, 5.0], [0.3, -0.3]]])
    outputs = Outputs(mu=mu, log_var=log_var)
    mask = jnp.ones((1, 4, 1, 28, 28), dtype=bool)
    mask = mask.at[:, 1:3].set(False)

    got = kl(ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(mu=0.0, log_var=0.0), outputs, mask)

    valid_per = gaussian_kl_closed_form(np.asarray(mu)[:, [0, 3]], np.asarray(log_var)[:, [0, 3]])
    expected = float(np.mean(valid_per))
    assert np.isclose(float(got), expected, rtol=1e-5), (float(got), expected)


def test_kl_safe_against_pathological_padded_log_var():
    # Padded examples carry log_var = 5e9 (would overflow exp). The safe-substitute fix
    # inside kl() must keep both the forward value finite AND the gradient w.r.t. log_var
    # free of NaN at padded positions.
    mu = jnp.asarray([[[0.5, -0.3], [1e9, 1e9]]])
    log_var = jnp.asarray([[[-0.2, 0.1], [5e9, 5e9]]])
    mask = jnp.ones((1, 2, 1, 28, 28), dtype=bool).at[:, 1].set(False)

    def loss(mu_, lv_):
        outputs = Outputs(mu=mu_, log_var=lv_)
        return kl(ELBOObjective.GaussianPosterior(), ELBOObjective.GaussianPrior(mu=0.0, log_var=0.0), outputs, mask)

    forward = float(loss(mu, log_var))
    assert np.isfinite(forward), forward
    valid = gaussian_kl_closed_form(np.asarray(mu)[:, [0]], np.asarray(log_var)[:, [0]])
    assert np.isclose(forward, float(np.mean(valid)), rtol=1e-5)

    g_mu, g_lv = jax.grad(loss, argnums=(0, 1))(mu, log_var)
    assert np.all(np.isfinite(np.asarray(g_mu))), "gradient w.r.t. mu has NaN/Inf"
    assert np.all(np.isfinite(np.asarray(g_lv))), "gradient w.r.t. log_var has NaN/Inf"
    # Gradients at masked positions must be exactly 0 (no leakage from padded entries).
    assert np.allclose(np.asarray(g_mu)[:, 1], 0.0)
    assert np.allclose(np.asarray(g_lv)[:, 1], 0.0)


def test_reparameterize_is_stochastic():
    """The reparameterize formula z = mu + exp(0.5*log_var) * eps must produce different z
    across different PRNG keys (and differ from mu) — otherwise the VAE has collapsed to an
    autoencoder and the KL term is doing nothing."""
    mu = jnp.asarray([0.1, -0.3])
    log_var = jnp.asarray([-0.5, 0.2])  # std = exp(-0.25), exp(0.1) — both nonzero
    std = jnp.exp(0.5 * log_var)
    z1 = mu + std * jax.random.normal(jax.random.key(1), mu.shape)
    z2 = mu + std * jax.random.normal(jax.random.key(2), mu.shape)
    assert not np.allclose(np.asarray(z1), np.asarray(z2)), "z must differ across PRNG keys"
    assert not np.allclose(np.asarray(z1), np.asarray(mu)), "z must differ from mu"


def test_reparameterize_collapses_when_log_var_is_very_negative():
    """Sanity check on the *posterior collapse* signature: if log_var → -∞, std → 0, and
    z → mu regardless of eps. Confirms that watching log_var_mean is a valid diagnostic."""
    mu = jnp.asarray([0.1, -0.3])
    log_var = jnp.asarray([-50.0, -50.0])  # std = exp(-25) ≈ 1.4e-11
    std = jnp.exp(0.5 * log_var)
    z = mu + std * jax.random.normal(jax.random.key(0), mu.shape)
    assert np.allclose(np.asarray(z), np.asarray(mu), atol=1e-9), (z, mu)


def test_recon_masked_mse_sum_excludes_padded():
    # Mirror what create_loss_fn(RegressionObjective(reduction="sum")) computes: per-example
    # sum over pixels, then mean over batch using example-level mask.
    label_mask_value = -1e10
    pred = jnp.asarray([[[[0.2, 0.8]], [[0.5, 0.5]]]])  # (1, 2, 1, 2): two examples, 1×2 pixel image
    target_real = jnp.asarray([[[[0.1, 0.9]], [[0.4, 0.6]]]])
    # Mark second example as padded: target = sentinel everywhere
    target = target_real.at[:, 1].set(label_mask_value)

    mask = target != label_mask_value
    raw = optax.losses.squared_error(pred, target)
    per_example = jnp.sum(jnp.where(mask, raw, 0.0), axis=(2, 3))  # sum over feature dims
    example_mask = jnp.any(mask, axis=(2, 3))
    got = float(jnp.sum(jnp.where(example_mask, per_example, 0.0)) / jnp.maximum(jnp.sum(example_mask), 1.0))

    expected = float(np.sum((np.asarray(pred)[:, 0] - np.asarray(target_real)[:, 0]) ** 2))
    assert np.isclose(got, expected, rtol=1e-5), (got, expected)
