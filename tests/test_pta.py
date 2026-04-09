"""Tests for PTA on known games."""

import jax.numpy as jnp
import pytest

from fptajax.pta import pta
from fptajax.utils import disc


def test_rps_pta():
    """PTA on Rock-Paper-Scissors should give circular embedding."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    result = pta(F)

    # Should have 1 disc game
    assert result.n_components == 1
    assert result.embeddings.shape == (3, 1, 2)

    # All agents should be equidistant from origin
    norms = jnp.linalg.norm(result.embeddings[:, 0, :], axis=-1)
    assert jnp.allclose(norms, norms[0], atol=1e-6), "Agents not equidistant"

    # Angles should be 120 degrees apart
    Y = result.embeddings[:, 0, :]
    angles = jnp.arctan2(Y[:, 1], Y[:, 0])
    angles_sorted = jnp.sort(angles)
    diffs = jnp.diff(angles_sorted)
    # Allow for wrap-around
    assert jnp.allclose(jnp.abs(diffs), 2 * jnp.pi / 3, atol=0.1)


def test_rps_reconstruction():
    """Reconstructed F should match original for full rank."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    result = pta(F)
    F_hat = result.reconstruct()

    assert jnp.allclose(F_hat, F, atol=1e-5), f"Reconstruction error:\n{F - F_hat}"


def test_pta_skew_symmetric():
    """PTA embeddings should produce skew-symmetric reconstruction."""
    F = jnp.array([
        [0.0, 2.0, -1.0, 0.5],
        [-2.0, 0.0, 1.5, -0.5],
        [1.0, -1.5, 0.0, 1.0],
        [-0.5, 0.5, -1.0, 0.0],
    ])
    result = pta(F)
    F_hat = result.reconstruct()

    # Should be skew-symmetric
    assert jnp.allclose(F_hat, -F_hat.T, atol=1e-5)


def test_pta_importance_sums():
    """Importance should sum to <= 1."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    result = pta(F)
    imp = result.get_importance()
    # For full rank, should sum to approximately 1
    assert float(jnp.sum(imp)) <= 1.0 + 1e-6


def test_disc_consistency():
    """disc(Y(x), Y(y)) should equal the reconstruction."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    result = pta(F)
    Y = result.embeddings  # (3, 1, 2)

    for i in range(3):
        for j in range(3):
            f_ij = 0.0
            for k in range(result.n_components):
                f_ij += disc(Y[i, k, :], Y[j, k, :])
            assert jnp.isclose(f_ij, F[i, j], atol=1e-5), \
                f"Mismatch at ({i},{j}): {f_ij} vs {F[i,j]}"
