"""GPU-friendly Schur decomposition for skew-symmetric and general matrices.

The real Schur decomposition C = Q U Q^T is the core linear algebra step in
both PTA and FPTA. For skew-symmetric C, eigenvalues are purely imaginary
(+/- i*omega_k) and U is block-diagonal with 2x2 rotation blocks.

JAX's jax.scipy.linalg.schur is CPU-only. We provide a GPU-compatible path
using jnp.linalg.eig and reconstructing the real Schur form from eigenpairs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SchurResult(NamedTuple):
    """Result of a real Schur decomposition C = Q U Q^T.

    Attributes:
        Q: orthogonal matrix, shape (m, m). Columns are Schur vectors.
        U: quasi-upper-triangular matrix, shape (m, m). For skew-symmetric
           input, this is block-diagonal with 2x2 blocks [[0, w_k], [-w_k, 0]].
        eigenvalues: magnitudes omega_k sorted in decreasing order, shape (d,)
            where d = floor(m/2).
        n_components: number of disc game components (= number of nonzero omega_k).
    """
    Q: Array
    U: Array
    eigenvalues: Array
    n_components: int


def skew_symmetric_schur(C: Array, tol: float = 1e-10) -> SchurResult:
    """Real Schur decomposition of a skew-symmetric matrix.

    For skew-symmetric C, eigenvalues are purely imaginary: lambda_k = +/- i*omega_k.
    We decompose C = Q U Q^T where Q is orthogonal and U is block-diagonal with
    2x2 blocks [[0, omega_k], [-omega_k, 0]].

    This uses jnp.linalg.eig which works on GPU, avoiding the CPU-only
    jax.scipy.linalg.schur.

    Args:
        C: skew-symmetric matrix, shape (m, m).
        tol: threshold below which eigenvalues are treated as zero.

    Returns:
        SchurResult with Q, U, eigenvalues, n_components.
    """
    m = C.shape[0]

    # Enforce skew-symmetry
    C = 0.5 * (C - C.T)

    # Eigendecomposition: eigenvalues of skew-symmetric are purely imaginary
    eigvals, eigvecs = jnp.linalg.eig(C)

    # Extract imaginary parts and sort by magnitude (decreasing)
    imag_parts = jnp.imag(eigvals)
    magnitudes = jnp.abs(imag_parts)
    sort_idx = jnp.argsort(-magnitudes)
    imag_parts = imag_parts[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    magnitudes = magnitudes[sort_idx]

    # Pair up conjugate eigenvalues: +i*omega and -i*omega
    # After sorting by magnitude, they come in adjacent pairs
    Q_cols = []
    omegas = []
    used = [False] * m

    i = 0
    while i < m:
        if magnitudes[i] < tol:
            # Zero eigenvalue — skip or add as a trivial column
            i += 1
            continue

        omega = magnitudes[i]

        # Find the conjugate partner (should be adjacent)
        v = eigvecs[:, i]

        # Real Schur vectors from complex eigenvector: q1 = Re(v), q2 = Im(v)
        # Normalize to get orthonormal pair
        q1 = jnp.real(v)
        q2 = jnp.imag(v)

        # Normalize
        n1 = jnp.linalg.norm(q1)
        n2 = jnp.linalg.norm(q2)
        q1 = q1 / jnp.maximum(n1, 1e-15)
        q2 = q2 / jnp.maximum(n2, 1e-15)

        # Ensure orthogonality via modified Gram-Schmidt against prior columns
        for prev_q in Q_cols:
            prev_q_arr = jnp.array(prev_q)
            q1 = q1 - jnp.dot(q1, prev_q_arr) * prev_q_arr
            q2 = q2 - jnp.dot(q2, prev_q_arr) * prev_q_arr
        q1 = q1 / jnp.maximum(jnp.linalg.norm(q1), 1e-15)
        q2 = q2 - jnp.dot(q2, q1) * q1
        q2 = q2 / jnp.maximum(jnp.linalg.norm(q2), 1e-15)

        Q_cols.append(q1)
        Q_cols.append(q2)
        omegas.append(float(omega))

        # Skip the conjugate partner
        i += 2

    n_components = len(omegas)

    # Handle any remaining zero-eigenvalue directions
    if len(Q_cols) < m:
        if len(Q_cols) == 0:
            # All eigenvalues are zero — Q is just the identity
            Q_cols = [jnp.eye(m)[:, j] for j in range(m)]
        else:
            # Complete the orthogonal basis using null space
            Q_partial = jnp.stack(Q_cols, axis=1)
            # Find orthogonal complement
            null_dim = m - len(Q_cols)
            proj = jnp.eye(m) - Q_partial @ Q_partial.T
            _, vecs = jnp.linalg.eigh(proj)
            for j in range(null_dim):
                Q_cols.append(jnp.real(vecs[:, -(j + 1)]))

    Q = jnp.stack(Q_cols, axis=1)
    omegas_arr = jnp.array(omegas)

    # Build U: block-diagonal with [[0, omega_k], [-omega_k, 0]] blocks
    U = jnp.zeros((m, m))
    for k in range(n_components):
        U = U.at[2 * k, 2 * k + 1].set(omegas[k])
        U = U.at[2 * k + 1, 2 * k].set(-omegas[k])

    return SchurResult(
        Q=Q,
        U=U,
        eigenvalues=omegas_arr,
        n_components=n_components,
    )


def general_real_schur(A: Array, tol: float = 1e-10) -> SchurResult:
    """Real Schur decomposition for a general real matrix.

    Uses jnp.linalg.eig (GPU-compatible) and reconstructs the real Schur form.
    Complex eigenvalue pairs produce 2x2 blocks; real eigenvalues produce 1x1 blocks.

    For PTA on general (not necessarily skew-symmetric) payoff matrices.

    Args:
        A: real matrix, shape (m, m).
        tol: threshold for treating imaginary parts as zero.

    Returns:
        SchurResult. For a general matrix the 2x2 blocks in U correspond to
        complex conjugate eigenvalue pairs.
    """
    m = A.shape[0]
    eigvals, eigvecs = jnp.linalg.eig(A)

    # Sort by magnitude of imaginary part (complex pairs first), then by magnitude
    imag_mag = jnp.abs(jnp.imag(eigvals))
    real_mag = jnp.abs(jnp.real(eigvals))
    total_mag = jnp.abs(eigvals)
    # Sort: complex pairs first (descending by total magnitude), then real (descending)
    is_complex = imag_mag > tol
    sort_key = -total_mag - 1e6 * is_complex  # complex pairs sort first
    sort_idx = jnp.argsort(sort_key)

    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    Q_cols = []
    omegas = []  # For complex pairs: magnitude of imaginary part
    U_blocks = []

    i = 0
    while i < m:
        if jnp.abs(jnp.imag(eigvals[i])) > tol:
            # Complex conjugate pair
            v = eigvecs[:, i]
            omega = jnp.abs(jnp.imag(eigvals[i]))
            sigma = jnp.real(eigvals[i])

            q1 = jnp.real(v)
            q2 = jnp.imag(v)

            n1 = jnp.linalg.norm(q1)
            n2 = jnp.linalg.norm(q2)
            q1 = q1 / jnp.maximum(n1, 1e-15)
            q2 = q2 / jnp.maximum(n2, 1e-15)

            # Orthogonalize against prior
            for prev_q in Q_cols:
                prev_q_arr = jnp.array(prev_q)
                q1 = q1 - jnp.dot(q1, prev_q_arr) * prev_q_arr
                q2 = q2 - jnp.dot(q2, prev_q_arr) * prev_q_arr
            q1 = q1 / jnp.maximum(jnp.linalg.norm(q1), 1e-15)
            q2 = q2 - jnp.dot(q2, q1) * q1
            q2 = q2 / jnp.maximum(jnp.linalg.norm(q2), 1e-15)

            Q_cols.extend([q1, q2])
            omegas.append(float(omega))
            i += 2
        else:
            # Real eigenvalue
            v = jnp.real(eigvecs[:, i])
            # Orthogonalize
            for prev_q in Q_cols:
                prev_q_arr = jnp.array(prev_q)
                v = v - jnp.dot(v, prev_q_arr) * prev_q_arr
            v = v / jnp.maximum(jnp.linalg.norm(v), 1e-15)
            Q_cols.append(v)
            i += 1

    Q = jnp.stack(Q_cols, axis=1)
    omegas_arr = jnp.array(omegas)
    n_components = len(omegas)

    # Build U = Q^T A Q
    U = Q.T @ A @ Q

    return SchurResult(
        Q=Q,
        U=U,
        eigenvalues=omegas_arr,
        n_components=n_components,
    )
