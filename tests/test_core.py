"""Tests for the core FPTA algorithm."""

import jax.numpy as jnp
import pytest

from fptajax.core import fpta
from fptajax.basis import FourierBasis, LegendreBasis


def test_fpta_simple_disc_game():
    """FPTA on f(x,y) = sin(x)cos(y) - cos(x)sin(y) = sin(x-y).

    This is a single disc game in the Fourier basis:
    f(x,y) = disc([sin(x), cos(x)], [sin(y), cos(y)])

    FPTA with Fourier basis should recover it exactly with one component.
    """
    def f(x, y):
        return jnp.sin(x - y)

    basis = FourierBasis()
    result = fpta(f, basis, n_basis=5, n_quad=100)

    # Should have at least 1 significant component
    assert result.n_components >= 1
    assert result.eigenvalues[0] > 0.1

    # Reconstruction should be accurate
    x = jnp.linspace(0, 2 * jnp.pi, 30)
    xx, yy = jnp.meshgrid(x, x, indexing='ij')
    f_true = f(xx, yy)
    f_hat = result.reconstruct(x, x, basis, n_components=1)

    rmse = float(jnp.sqrt(jnp.mean((f_true - f_hat) ** 2)))
    assert rmse < 0.1, f"RMSE = {rmse}"


def test_fpta_two_components():
    """FPTA on a sum of two disc games should find two components."""
    def f(x, y):
        # Two disc games: sin(x-y) + 0.5*sin(2x-2y)
        return jnp.sin(x - y) + 0.5 * jnp.sin(2 * (x - y))

    basis = FourierBasis()
    result = fpta(f, basis, n_basis=7, n_quad=100)

    # Should find at least 2 components
    significant = jnp.sum(result.eigenvalues > 0.01)
    assert significant >= 2, f"Found {significant} components, expected >= 2"


def test_fpta_skew_symmetric_coefficient_matrix():
    """The coefficient matrix C should be skew-symmetric."""
    def f(x, y):
        return jnp.sin(x - y)

    basis = FourierBasis()
    result = fpta(f, basis, n_basis=5, n_quad=100)

    C = result.coefficient_matrix
    assert jnp.allclose(C, -C.T, atol=1e-10), "C is not skew-symmetric"


def test_fpta_importance():
    """Importance values should be non-negative and sum to <= 1."""
    def f(x, y):
        return jnp.sin(x - y)

    basis = FourierBasis()
    result = fpta(f, basis, n_basis=5, n_quad=100)

    imp = result.get_importance()
    assert jnp.all(imp >= 0)
    assert float(jnp.sum(imp)) <= 1.0 + 1e-6


def test_fpta_embed_at_nodes():
    """Embeddings at quadrature nodes should have correct shape."""
    def f(x, y):
        return jnp.sin(x - y)

    basis = FourierBasis()
    result = fpta(f, basis, n_basis=5, n_quad=50)

    Y = result.embed_at_nodes()
    assert Y.shape[0] == 50  # n_quad nodes
    assert Y.shape[2] == 2  # 2D embeddings
