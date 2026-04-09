"""Pointwise Principal Tradeoff Analysis (PTA).

PTA operates on a finite sample of agents with an observed payoff matrix F.
It embeds agents into 2D disc game planes via the real Schur decomposition
of F.

Also provides fpta_empirical: FPTA using the empirical measure (sample data +
basis evaluation matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from fptajax.decomposition import skew_symmetric_schur, general_real_schur
from fptajax.utils import importance, cumulative_importance, make_skew_symmetric


@dataclass
class PTAResult:
    """Result of Principal Tradeoff Analysis.

    Attributes:
        embeddings: pointwise embeddings, shape (N, d, 2).
            embeddings[i, k, :] = Y^(k)(x_i) is the k-th disc game
            embedding of agent i.
        eigenvalues: omega_k in decreasing order, shape (d,).
        Q: Schur vectors, shape (N, N).
        U: Schur form, shape (N, N).
        n_components: number of disc game components.
        f_norm_sq: ||F||^2_Fro / N^2.
    """
    embeddings: Array
    eigenvalues: Array
    Q: Array
    U: Array
    n_components: int
    f_norm_sq: float | None = None

    def get_importance(self) -> Array:
        """Relative importance of each disc game."""
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        """Cumulative explained variance."""
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)

    def reconstruct(self, n_components: int | None = None) -> Array:
        """Reconstruct the payoff matrix from the first k disc games.

        Args:
            n_components: number of disc games. If None, uses all.

        Returns:
            Reconstructed payoff matrix, shape (N, N).
        """
        nc = n_components or self.n_components
        Y = self.embeddings[:, :nc, :]  # (N, nc, 2)

        # F_hat[i,j] = sum_k disc(Y^(k)(x_i), Y^(k)(x_j))
        # = sum_k (Y[i,k,0]*Y[j,k,1] - Y[i,k,1]*Y[j,k,0])
        F_hat = jnp.einsum('ik,jk->ij', Y[:, :, 0], Y[:, :, 1]) \
              - jnp.einsum('ik,jk->ij', Y[:, :, 1], Y[:, :, 0])
        return F_hat


def pta(
    F: Array,
    n_components: int | None = None,
    enforce_skew: bool = True,
) -> PTAResult:
    """Principal Tradeoff Analysis on a payoff matrix.

    Given an N x N payoff matrix F where F_ij = f(x_i, x_j), decomposes
    F into a sequence of disc games via the real Schur decomposition.

    Args:
        F: payoff matrix, shape (N, N). Should be approximately skew-symmetric.
        n_components: maximum number of disc game components. If None, keeps all.
        enforce_skew: if True, enforces F = (F - F^T)/2 before decomposition.

    Returns:
        PTAResult with embeddings, eigenvalues, etc.
    """
    if enforce_skew:
        F = make_skew_symmetric(F)

    N = F.shape[0]

    # Schur decomposition
    schur = skew_symmetric_schur(F)

    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    Q = schur.Q
    omegas = schur.eigenvalues[:nc]

    # Embeddings: Y^(k)(x_i) = omega_k^{1/2} * [Q[i, 2k-1], Q[i, 2k]]
    # Following the manuscript: Y_X^PTA = Q Lambda^{1/2}
    embeddings = []
    for k in range(nc):
        y1 = jnp.sqrt(omegas[k]) * Q[:, 2 * k]      # (N,)
        y2 = jnp.sqrt(omegas[k]) * Q[:, 2 * k + 1]  # (N,)
        embeddings.append(jnp.stack([y1, y2], axis=-1))

    embeddings = jnp.stack(embeddings, axis=1)  # (N, d, 2)

    f_norm_sq = float(jnp.sum(F ** 2))

    return PTAResult(
        embeddings=embeddings,
        eigenvalues=omegas,
        Q=schur.Q,
        U=schur.U,
        n_components=nc,
        f_norm_sq=f_norm_sq,
    )


@dataclass
class FPTAEmpiricalResult:
    """Result of FPTA with empirical measure.

    Combines PTA's sample-based approach with FPTA's basis function
    representation, allowing interpolation to new agents.

    Attributes:
        eigenvalues: omega_k, shape (d,).
        schur_vectors: Q from C = QUQ^T, shape (m, m).
        coefficient_matrix: C, shape (m, m).
        weight_matrix: W = (1/N) Lambda^{1/2} Q^T B_X, shape (m, N).
        pointwise_embeddings: embeddings at sample points, shape (N, d, 2).
        n_components: number of disc game components.
    """
    eigenvalues: Array
    schur_vectors: Array
    coefficient_matrix: Array
    weight_matrix: Array
    pointwise_embeddings: Array
    n_components: int

    def embed(self, B_new: Array) -> Array:
        """Evaluate embeddings at new points using basis evaluation.

        Args:
            B_new: basis functions evaluated at new points, shape (M, m).

        Returns:
            Embeddings, shape (M, d, 2).
        """
        Q = self.schur_vectors
        d = self.n_components
        omegas = self.eigenvalues

        embeddings = []
        for k in range(d):
            q1 = Q[:, 2 * k]
            q2 = Q[:, 2 * k + 1]
            y1 = jnp.sqrt(omegas[k]) * (B_new @ q1)
            y2 = jnp.sqrt(omegas[k]) * (B_new @ q2)
            embeddings.append(jnp.stack([y1, y2], axis=-1))

        return jnp.stack(embeddings, axis=1)


def fpta_empirical(
    F_X: Array,
    B_X: Array,
    n_components: int | None = None,
    enforce_skew: bool = True,
) -> FPTAEmpiricalResult:
    """FPTA with empirical measure.

    Given sample payoff matrix F_X and basis evaluation matrix B_X,
    computes the coefficient matrix C and decomposes it.

    This gives functional embeddings that can be evaluated at new trait
    values, unlike pure PTA which is limited to sample agents.

    Args:
        F_X: payoff matrix, shape (N, N).
        B_X: basis evaluation matrix, shape (N, m). B_X[i, j] = b_j(x_i).
        n_components: max disc game components. If None, keeps all.
        enforce_skew: if True, enforces F_X skew-symmetric.

    Returns:
        FPTAEmpiricalResult with functional embeddings.
    """
    if enforce_skew:
        F_X = make_skew_symmetric(F_X)

    N = F_X.shape[0]
    m = B_X.shape[1]

    # Orthogonalize basis w.r.t. empirical measure
    # <b_i, b_j>_empirical = (1/N) B_X^T B_X
    # Gram-Schmidt: Q_B, R_B = qr(sqrt(W) * B_X) where W = (1/N)*I
    # Simpler: compute Gram matrix and Cholesky
    G = (1.0 / N) * B_X.T @ B_X  # (m, m)
    L = jnp.linalg.cholesky(G)
    # Transform: B_ortho = B_X @ inv(L)^T  so that (1/N) B_ortho^T B_ortho = I
    L_inv_T = jnp.linalg.inv(L).T
    B_ortho = B_X @ L_inv_T  # (N, m)

    # Coefficient matrix: C = (1/N^2) B_ortho^T F_X B_ortho
    C = (1.0 / N**2) * B_ortho.T @ F_X @ B_ortho  # (m, m)

    # Enforce skew-symmetry
    C = 0.5 * (C - C.T)

    # Schur decomposition
    schur = skew_symmetric_schur(C)

    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    Q = schur.Q
    omegas = schur.eigenvalues[:nc]

    # Pointwise embeddings at sample points
    embeddings = []
    for k in range(nc):
        q1 = Q[:, 2 * k]
        q2 = Q[:, 2 * k + 1]
        y1 = jnp.sqrt(omegas[k]) * (B_ortho @ q1)  # (N,)
        y2 = jnp.sqrt(omegas[k]) * (B_ortho @ q2)  # (N,)
        embeddings.append(jnp.stack([y1, y2], axis=-1))

    pointwise = jnp.stack(embeddings, axis=1) if embeddings else jnp.empty((N, 0, 2))

    # Weight matrix (for embed at new points)
    W = (1.0 / N) * schur.Q.T @ B_ortho.T  # Not stored for now

    return FPTAEmpiricalResult(
        eigenvalues=omegas,
        schur_vectors=Q,
        coefficient_matrix=C,
        weight_matrix=L_inv_T,  # transformation from raw to orthonormal
        pointwise_embeddings=pointwise,
        n_components=nc,
    )
