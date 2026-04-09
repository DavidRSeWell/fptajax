"""Utility functions for FPTA/PTA.

Core operations: disc game cross product, skew-symmetry enforcement,
importance computation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def disc(x: Array, y: Array) -> Array:
    """Disc game: cross product in R^2.

    disc(x, y) = x_1 * y_2 - x_2 * y_1

    This is the simplest functional form game allowing cyclic advantage —
    a continuous analog of rock-paper-scissors.

    Args:
        x: shape (..., 2) — first player's embedding.
        y: shape (..., 2) — second player's embedding.

    Returns:
        Scalar or array of disc game values.
    """
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def disc_embedding(Y_x: Array, Y_y: Array) -> Array:
    """Compute the full disc game embedding: sum_k disc(Y^(k)(x), Y^(k)(y)).

    Reconstructs the performance function from embeddings.

    Args:
        Y_x: embeddings of x, shape (..., d, 2).
        Y_y: embeddings of y, shape (..., d, 2).

    Returns:
        Approximated f(x, y) = sum_k Y^(k)(x) x Y^(k)(y).
    """
    # disc along the last dimension, sum over components
    discs = Y_x[..., 0] * Y_y[..., 1] - Y_x[..., 1] * Y_y[..., 0]
    return jnp.sum(discs, axis=-1)


def make_skew_symmetric(F: Array) -> Array:
    """Enforce skew-symmetry: return (F - F^T) / 2.

    Args:
        F: matrix, shape (n, n).

    Returns:
        Skew-symmetric matrix.
    """
    return 0.5 * (F - F.T)


def importance(eigenvalues: Array, f_norm_sq: float | None = None) -> Array:
    """Compute relative importance of each disc game component.

    importance_k = 2 * omega_k^2 / ||f||^2_{pi x pi}

    If f_norm_sq is not provided, returns unnormalized 2*omega_k^2.

    Args:
        eigenvalues: omega_k values, shape (d,).
        f_norm_sq: ||f||^2 if known.

    Returns:
        Importance values, shape (d,). Sums to <= 1 when f_norm_sq is provided
        (equality when projection error is zero).
    """
    vals = 2.0 * eigenvalues ** 2
    if f_norm_sq is not None and f_norm_sq > 0:
        return vals / f_norm_sq
    return vals


def cumulative_importance(eigenvalues: Array, f_norm_sq: float | None = None) -> Array:
    """Cumulative explained variance from first k disc games.

    Args:
        eigenvalues: omega_k values, shape (d,).
        f_norm_sq: ||f||^2 if known.

    Returns:
        Cumulative importance, shape (d,).
    """
    imp = importance(eigenvalues, f_norm_sq)
    return jnp.cumsum(imp)
