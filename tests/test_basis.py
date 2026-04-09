"""Tests for basis functions and Gram-Schmidt."""

import jax.numpy as jnp
import pytest

from fptajax.basis import (
    FourierBasis,
    ChebyshevBasis,
    LegendreBasis,
    MonomialBasis,
    gram_schmidt,
)
from fptajax.quad import inner_product_1d


def test_fourier_orthonormality():
    """Fourier basis should be orthonormal w.r.t. uniform measure on [0, 2pi]."""
    basis = FourierBasis()
    n = 7
    quad = basis.quad_rule(200)
    B = basis.evaluate(quad.nodes, n)  # (N, n)

    # Gram matrix should be identity
    G = inner_product_1d(B, B, quad.weights)
    assert jnp.allclose(G, jnp.eye(n), atol=1e-6), f"Gram matrix:\n{G}"


def test_legendre_orthonormality():
    """Legendre basis should be orthonormal w.r.t. dx/2 on [-1, 1]."""
    basis = LegendreBasis()
    n = 6
    quad = basis.quad_rule(100)
    B = basis.evaluate(quad.nodes, n)

    G = inner_product_1d(B, B, quad.weights)
    assert jnp.allclose(G, jnp.eye(n), atol=1e-5), f"Gram matrix:\n{G}"


def test_chebyshev_orthonormality():
    """Chebyshev basis orthonormality w.r.t. Chebyshev weight."""
    basis = ChebyshevBasis()
    n = 5
    quad = basis.quad_rule(100)
    B = basis.evaluate(quad.nodes, n)

    G = inner_product_1d(B, B, quad.weights)
    assert jnp.allclose(G, jnp.eye(n), atol=1e-4), f"Gram matrix:\n{G}"


def test_gram_schmidt_monomial():
    """Gram-Schmidt on monomials should produce orthonormal basis."""
    basis = MonomialBasis(domain=(-1.0, 1.0))
    n = 5
    quad = basis.quad_rule(100)
    B_ortho = gram_schmidt(basis, quad, n=n)

    # Check orthonormality (float32 precision: ~1e-5)
    G = inner_product_1d(B_ortho, B_ortho, quad.weights)
    assert jnp.allclose(G, jnp.eye(n), atol=1e-4), f"Gram matrix:\n{G}"


def test_fourier_evaluate_shape():
    """Check shape of fourier basis evaluation."""
    basis = FourierBasis()
    x = jnp.linspace(0, 2 * jnp.pi, 50)
    B = basis.evaluate(x, 7)
    assert B.shape == (50, 7)


def test_gram_schmidt_preserves_span():
    """Gram-Schmidt should produce basis spanning the same space."""
    basis = MonomialBasis(domain=(0.0, 1.0))
    quad = basis.quad_rule(100)
    n = 4

    B_raw = basis.evaluate(quad.nodes, n)
    B_ortho = gram_schmidt(basis, quad, n=n)

    # B_ortho should be in the span of B_raw
    # Project B_ortho onto B_raw and check residual is small
    # Since both span the same space (degree <= n-1 polynomials),
    # B_ortho = B_raw @ T for some invertible T
    T, _, _, _ = jnp.linalg.lstsq(B_raw, B_ortho)
    residual = B_raw @ T - B_ortho
    assert jnp.allclose(residual, 0.0, atol=1e-4)
