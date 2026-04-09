"""Tests for Schur decomposition."""

import jax.numpy as jnp
import pytest

from fptajax.decomposition import skew_symmetric_schur


def test_rps_matrix():
    """Rock-Paper-Scissors payoff matrix has one disc game component."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    result = skew_symmetric_schur(F)

    # RPS has rank 2 -> 1 disc game
    assert result.n_components == 1
    assert result.eigenvalues.shape == (1,)

    # omega should be sqrt(3)
    assert jnp.isclose(result.eigenvalues[0], jnp.sqrt(3.0), atol=1e-6)

    # Verify Q is orthogonal
    Q = result.Q
    QTQ = Q.T @ Q
    assert jnp.allclose(QTQ, jnp.eye(3), atol=1e-6)

    # Verify reconstruction C = Q U Q^T
    C_reconstructed = Q @ result.U @ Q.T
    assert jnp.allclose(C_reconstructed, F, atol=1e-6)


def test_4x4_skew():
    """4x4 skew-symmetric matrix with two disc games."""
    C = jnp.array([
        [0.0, 3.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
    ])
    result = skew_symmetric_schur(C)

    assert result.n_components == 2
    # Eigenvalues should be [3, 1] (decreasing)
    assert jnp.isclose(result.eigenvalues[0], 3.0, atol=1e-6)
    assert jnp.isclose(result.eigenvalues[1], 1.0, atol=1e-6)

    # Verify reconstruction
    Q = result.Q
    C_reconstructed = Q @ result.U @ Q.T
    assert jnp.allclose(C_reconstructed, C, atol=1e-5)


def test_zero_matrix():
    """Zero matrix should have zero components."""
    C = jnp.zeros((4, 4))
    result = skew_symmetric_schur(C)
    assert result.n_components == 0


def test_orthogonality():
    """Schur vectors should be orthogonal."""
    import jax
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (6, 6))
    C = 0.5 * (A - A.T)  # random skew-symmetric

    result = skew_symmetric_schur(C)
    Q = result.Q
    QTQ = Q.T @ Q
    assert jnp.allclose(QTQ, jnp.eye(6), atol=1e-5)
