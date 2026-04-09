"""Numerical quadrature for computing inner products.

Provides nodes and weights for various quadrature rules used to approximate:
- <b_i, b_j>_pi = integral b_i(x) b_j(x) d pi_x
- <f, [b_i, b_j]>_{pi x pi} = double integral f(x,y) b_i(x) b_j(y) d pi_x d pi_y
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class QuadRule(NamedTuple):
    """Quadrature rule: nodes and weights."""
    nodes: Array   # (n,) or (n, T) for multi-dimensional
    weights: Array  # (n,)


def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0) -> QuadRule:
    """Gauss-Legendre quadrature on [a, b].

    Exact for polynomials of degree <= 2n-1.
    """
    # Use numpy for the roots computation, then convert to jax
    import numpy as np
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [a, b]
    nodes = jnp.array(0.5 * (b - a) * nodes_ref + 0.5 * (a + b))
    weights = jnp.array(0.5 * (b - a) * weights_ref)
    return QuadRule(nodes, weights)


def gauss_chebyshev(n: int, a: float = -1.0, b: float = 1.0) -> QuadRule:
    """Gauss-Chebyshev quadrature for weight 1/sqrt(1 - x^2) on [a, b].

    The weight function is built into the weights, so integrate as:
        sum_i w_i * f(x_i)  approximates  integral f(x) / sqrt(1-x^2) dx
    """
    k = jnp.arange(1, n + 1)
    nodes_ref = jnp.cos((2 * k - 1) * jnp.pi / (2 * n))
    weights_ref = jnp.full(n, jnp.pi / n)
    # Transform from [-1, 1] to [a, b]
    nodes = 0.5 * (b - a) * nodes_ref + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_ref
    return QuadRule(nodes, weights)


def gauss_hermite(n: int) -> QuadRule:
    """Gauss-Hermite quadrature for weight exp(-x^2) on (-inf, inf).

    The weight function is built into the weights.
    """
    import numpy as np
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n)
    return QuadRule(jnp.array(nodes_np), jnp.array(weights_np))


def gauss_laguerre(n: int) -> QuadRule:
    """Gauss-Laguerre quadrature for weight exp(-x) on [0, inf).

    The weight function is built into the weights.
    """
    import numpy as np
    nodes_np, weights_np = np.polynomial.laguerre.laggauss(n)
    return QuadRule(jnp.array(nodes_np), jnp.array(weights_np))


def trapezoidal(n: int, a: float = 0.0, b: float = 2 * jnp.pi) -> QuadRule:
    """Trapezoidal rule on [a, b).

    Exponentially convergent for periodic functions on [a, b].
    """
    h = (b - a) / n
    nodes = jnp.linspace(a, b, n, endpoint=False)
    weights = jnp.full(n, h)
    return QuadRule(nodes, weights)


def empirical(samples: Array) -> QuadRule:
    """Empirical quadrature from samples.

    Inner products become sample averages:
        <b_i, b_j>_pi = (1/N) sum_k b_i(x_k) b_j(x_k)

    Args:
        samples: (N,) or (N, T) array of trait samples.
    """
    n = samples.shape[0]
    weights = jnp.full(n, 1.0 / n)
    return QuadRule(samples, weights)


def gauss_jacobi(n: int, alpha: float, beta: float) -> QuadRule:
    """Gauss-Jacobi quadrature for weight (1-x)^alpha (1+x)^beta on (-1, 1).

    The weight function is built into the weights.
    """
    import numpy as np
    from numpy.polynomial import legendre

    # Compute Jacobi quadrature via the Golub-Welsch algorithm
    # using the three-term recurrence coefficients
    if n == 0:
        return QuadRule(jnp.array([]), jnp.array([]))

    i = jnp.arange(1, n, dtype=jnp.float64)
    ab = alpha + beta

    # Diagonal of the Jacobi matrix (a_n coefficients)
    a_diag = jnp.zeros(n)
    if n > 1:
        numerator = beta**2 - alpha**2
        denom = (ab + 2 * i) * (ab + 2 * i + 2)
        a_diag = a_diag.at[1:].set(numerator / denom)
    a_diag = a_diag.at[0].set((beta - alpha) / (ab + 2))

    # Off-diagonal of the Jacobi matrix (b_n coefficients)
    b_offdiag = jnp.zeros(n - 1) if n > 1 else jnp.array([])
    if n > 1:
        num = 4 * i * (i + alpha) * (i + beta) * (i + ab)
        den = (ab + 2 * i)**2 * ((ab + 2 * i)**2 - 1)
        b_offdiag = jnp.sqrt(num / den)

    # Eigendecomposition of the tridiagonal Jacobi matrix
    J = jnp.diag(a_diag) + jnp.diag(b_offdiag, 1) + jnp.diag(b_offdiag, -1)
    eigenvalues, eigenvectors = jnp.linalg.eigh(J)

    nodes = eigenvalues
    # Weight = mu_0 * v_0^2 where mu_0 = integral of weight function
    import scipy.special as sp
    mu_0 = (2.0**(ab + 1) * sp.gamma(alpha + 1) * sp.gamma(beta + 1)
            / sp.gamma(ab + 2))
    weights = mu_0 * eigenvectors[0, :]**2

    # Sort by node position
    idx = jnp.argsort(nodes)
    return QuadRule(nodes[idx], jnp.array(weights)[idx])


def inner_product_1d(
    f_vals: Array,
    g_vals: Array,
    weights: Array,
) -> Array:
    """Compute <f, g>_pi = sum_i w_i f(x_i) g(x_i).

    Args:
        f_vals: f evaluated at quadrature nodes, shape (n,) or (n, m).
        g_vals: g evaluated at quadrature nodes, shape (n,) or (n, m).
        weights: quadrature weights, shape (n,).
    """
    if f_vals.ndim == 1 and g_vals.ndim == 1:
        return jnp.sum(weights * f_vals * g_vals)
    elif f_vals.ndim == 2 and g_vals.ndim == 2:
        # (n, m1) and (n, m2) -> (m1, m2) Gram matrix
        return (f_vals * weights[:, None]).T @ g_vals
    elif f_vals.ndim == 1 and g_vals.ndim == 2:
        return (f_vals * weights) @ g_vals  # (m,)
    else:
        return g_vals.T @ (weights[:, None] * f_vals)  # (m,)


def inner_product_2d(
    f_vals: Array,
    bi_vals: Array,
    bj_vals: Array,
    weights_x: Array,
    weights_y: Array,
) -> Array:
    """Compute <f, [b_i, b_j]>_{pi x pi} via double quadrature.

    <f, [b_i, b_j]> = sum_{k,l} w_k w_l f(x_k, y_l) b_i(x_k) b_j(y_l)

    Args:
        f_vals: f evaluated at grid points, shape (n_x, n_y).
        bi_vals: basis i evaluated at x-nodes, shape (n_x,).
        bj_vals: basis j evaluated at y-nodes, shape (n_y,).
        weights_x: x-quadrature weights, shape (n_x,).
        weights_y: y-quadrature weights, shape (n_y,).

    Returns:
        Scalar inner product value.
    """
    # sum_k w_k b_i(x_k) [sum_l w_l f(x_k, y_l) b_j(y_l)]
    inner_y = f_vals @ (weights_y * bj_vals)  # (n_x,)
    return jnp.sum(weights_x * bi_vals * inner_y)


def coefficient_matrix(
    f_vals: Array,
    basis_vals: Array,
    weights_x: Array,
    weights_y: Array,
) -> Array:
    """Compute the full m x m coefficient matrix C.

    C_ij = <f, [b_i, b_j]>_{pi x pi}

    This is the vectorized version computing all entries at once.

    Args:
        f_vals: f evaluated on the quadrature grid, shape (n_x, n_y).
        basis_vals: all basis functions evaluated at nodes, shape (n, m).
            If using the same quadrature for x and y, n_x = n_y = n.
        weights_x: x-quadrature weights, shape (n_x,).
        weights_y: y-quadrature weights, shape (n_y,).

    Returns:
        C: skew-symmetric coefficient matrix, shape (m, m).
    """
    # B_w = diag(weights) @ basis_vals, shape (n, m)
    Bx_w = basis_vals * weights_x[:, None]  # (n_x, m)
    By_w = basis_vals * weights_y[:, None]  # (n_y, m)
    # C = Bx_w^T @ f_vals @ By_w
    C = Bx_w.T @ f_vals @ By_w
    return C
