"""Functional Principal Tradeoff Analysis (FPTA).

Implements Algorithm 3.1 from the manuscript:
    1. Basis Formulation: orthogonalize basis via Gram-Schmidt
    2. Projection: compute coefficient matrix C_ij = <f, [b_i, b_j]>_{pi x pi}
    3. Decomposition: real Schur form of C
    4. Embedding: Y^(k)(x) = sqrt(omega_k) * b(x)^T [q_{2k-1}, q_{2k}]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array

from fptajax.basis import BasisFamily, gram_schmidt
from fptajax.decomposition import skew_symmetric_schur
from fptajax.quad import QuadRule, coefficient_matrix
from fptajax.utils import disc, importance, cumulative_importance


@dataclass
class FPTAResult:
    """Result of Functional Principal Tradeoff Analysis.

    Attributes:
        eigenvalues: omega_k in decreasing order, shape (d,).
            Magnitude of each disc game component.
        schur_vectors: Q from C = QUQ^T, shape (m, m).
        coefficient_matrix: C, the skew-symmetric projection matrix, shape (m, m).
        basis_values: orthonormalized basis evaluated at quad nodes, shape (N, m).
        quad: the quadrature rule used.
        n_components: number of disc game components.
        f_norm_sq: ||f||^2_{pi x pi} if computable, else None.
    """
    eigenvalues: Array
    schur_vectors: Array
    coefficient_matrix: Array
    basis_values: Array
    quad: QuadRule
    n_components: int
    f_norm_sq: float | None = None

    def embed(self, x: Array) -> Array:
        """Evaluate all embeddings at trait points x.

        Args:
            x: trait values, shape (N,) or (N, T).

        Returns:
            Embeddings, shape (N, d, 2) where d = n_components.
        """
        x = jnp.atleast_1d(x)
        # We need to evaluate basis at new points x
        # This requires knowing the basis family — store it or use interpolation
        # For now, raise if called without basis_family
        raise NotImplementedError(
            "embed() at new points requires the basis family. "
            "Use embed_from_basis() or embed_at_nodes() instead."
        )

    def embed_at_nodes(self) -> Array:
        """Evaluate embeddings at the quadrature nodes.

        Returns:
            Embeddings, shape (N, d, 2).
        """
        Q = self.schur_vectors
        B = self.basis_values  # (N, m)
        d = self.n_components
        omegas = self.eigenvalues

        embeddings = []
        for k in range(d):
            q_2k_minus_1 = Q[:, 2 * k]      # (m,)
            q_2k = Q[:, 2 * k + 1]            # (m,)
            # Y^(k)(x) = sqrt(omega_k) * [b(x)^T q_{2k-1}, b(x)^T q_{2k}]
            y1 = jnp.sqrt(omegas[k]) * (B @ q_2k_minus_1)  # (N,)
            y2 = jnp.sqrt(omegas[k]) * (B @ q_2k)          # (N,)
            embeddings.append(jnp.stack([y1, y2], axis=-1))  # (N, 2)

        return jnp.stack(embeddings, axis=1)  # (N, d, 2)

    def embed_from_basis(self, basis: BasisFamily, x: Array) -> Array:
        """Evaluate embeddings at arbitrary trait points using a basis family.

        Args:
            basis: the same basis family used to compute the FPTA.
            x: trait values, shape (N,).

        Returns:
            Embeddings, shape (N, d, 2).
        """
        x = jnp.atleast_1d(x)
        m = self.coefficient_matrix.shape[0]

        # Evaluate raw basis at x, then orthogonalize using stored info
        # For proper interpolation, we re-evaluate the orthonormal basis
        # via the Gram-Schmidt coefficients stored implicitly in basis_values
        # Simpler approach: re-run Gram-Schmidt at the new points
        # This is only exact if basis is analytically orthonormal (Fourier, etc.)
        # For general bases, we use the stored Q from Gram-Schmidt:
        # b_ortho(x) = B_raw(x) @ R^{-1} where B_ortho = B_raw @ R^{-1}

        # Evaluate raw basis at new points
        B_new_raw = basis.evaluate(x, m)

        # We need the transformation from raw to orthonormal
        # B_ortho_nodes = B_raw_nodes @ T for some T
        # So B_ortho_new = B_raw_new @ T
        B_raw_nodes = basis.evaluate(self.quad.nodes, m)

        # Solve for T: B_ortho = B_raw @ T in least-squares sense
        # T = (B_raw^T W B_raw)^{-1} B_raw^T W B_ortho
        w = self.quad.weights
        BwB = (B_raw_nodes * w[:, None]).T @ B_raw_nodes
        BwO = (B_raw_nodes * w[:, None]).T @ self.basis_values
        T = jnp.linalg.solve(BwB, BwO)

        B_new = B_new_raw @ T  # (N_new, m)

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

    def reconstruct(
        self,
        x: Array,
        y: Array,
        basis: BasisFamily,
        n_components: int | None = None,
    ) -> Array:
        """Reconstruct f_hat(x, y) from the first n disc game components.

        Args:
            x: first player traits, shape (N,).
            y: second player traits, shape (M,).
            basis: basis family for evaluating embeddings.
            n_components: number of disc games to use. If None, uses all.

        Returns:
            Reconstructed payoff matrix, shape (N, M).
        """
        Y_x = self.embed_from_basis(basis, x)  # (N, d, 2)
        Y_y = self.embed_from_basis(basis, y)  # (M, d, 2)

        nc = n_components or self.n_components
        Y_x = Y_x[:, :nc, :]
        Y_y = Y_y[:, :nc, :]

        # f_hat(x_i, y_j) = sum_k disc(Y^(k)(x_i), Y^(k)(y_j))
        # = sum_k (Y_x[i,k,0]*Y_y[j,k,1] - Y_x[i,k,1]*Y_y[j,k,0])
        F_hat = jnp.einsum('ik,jk->ij', Y_x[:, :, 0], Y_y[:, :, 1]) \
              - jnp.einsum('ik,jk->ij', Y_x[:, :, 1], Y_y[:, :, 0])
        return F_hat

    def get_importance(self) -> Array:
        """Relative importance of each disc game."""
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        """Cumulative explained variance."""
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


def fpta(
    f: Callable[[Array, Array], Array],
    basis: BasisFamily,
    n_basis: int,
    n_quad: int | None = None,
    quad: QuadRule | None = None,
    n_components: int | None = None,
) -> FPTAResult:
    """Functional Principal Tradeoff Analysis.

    Implements Algorithm 3.1: given a performance function f(x,y) and a basis,
    finds the optimal disc game embedding in the span of the basis.

    Args:
        f: performance function f(x, y) -> scalar. Must be skew-symmetric:
            f(x, y) = -f(y, x). Should accept arrays and broadcast.
        basis: basis function family (FourierBasis, ChebyshevBasis, etc.).
        n_basis: number of basis functions m.
        n_quad: number of quadrature nodes. If None, uses 2*n_basis.
        quad: explicit quadrature rule. Overrides n_quad if provided.
        n_components: max number of disc game components to keep. If None, keeps all.

    Returns:
        FPTAResult containing embeddings, eigenvalues, etc.
    """
    # Step 1: Set up quadrature
    if quad is None:
        n_quad = n_quad or 2 * n_basis
        quad = basis.quad_rule(n_quad)

    nodes = quad.nodes
    weights = quad.weights

    # Step 2: Evaluate and orthogonalize basis
    B_ortho = gram_schmidt(basis, quad, n=n_basis)  # (N, m)

    # Step 3: Compute coefficient matrix C
    # C_ij = <f, [b_i, b_j]>_{pi x pi}
    # = sum_{k,l} w_k w_l f(x_k, y_l) b_i(x_k) b_j(y_l)

    # Evaluate f on the quadrature grid
    nodes_x = nodes  # (N,)
    nodes_y = nodes  # (N,)
    # Create meshgrid: f_vals[i, j] = f(nodes_x[i], nodes_y[j])
    xx, yy = jnp.meshgrid(nodes_x, nodes_y, indexing='ij')
    f_vals = f(xx, yy)  # (N, N)

    C = coefficient_matrix(f_vals, B_ortho, weights, weights)

    # Enforce skew-symmetry (should be by construction, but numerical safety)
    C = 0.5 * (C - C.T)

    # Step 4: Schur decomposition
    schur = skew_symmetric_schur(C)

    # Optionally truncate
    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    # Compute ||f||^2 for importance
    f_norm_sq = float(jnp.sum(weights[:, None] * weights[None, :] * f_vals**2))

    return FPTAResult(
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        coefficient_matrix=C,
        basis_values=B_ortho,
        quad=quad,
        n_components=nc,
        f_norm_sq=f_norm_sq,
    )
