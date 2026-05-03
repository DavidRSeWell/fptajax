"""Shared protocol for the classical FPTA experiment suite.

Every dataset implements :class:`FPTADataset` and exposes:

  - ``traits``      — agent trait vectors, shape (N, T)
  - ``F``           — skew-symmetric payoff matrix, shape (N, N)
  - ``f_norm_sq``   — ‖F‖²_F / N² (matches the paper's normalisation)
  - ``available_bases`` — list of basis identifiers
  - ``basis(name, m)`` — basis evaluation matrix at the agents, shape (N, m)
  - ``train_pairs`` / ``test_pairs`` — np.ndarray (P, 2) of agent index pairs

The fit/eval helpers are dataset-agnostic and used by every hypothesis script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dataset interface
# ---------------------------------------------------------------------------


@dataclass
class FPTADataset:
    """Container for one dataset of the suite."""
    name: str
    traits: np.ndarray              # (N, T)
    F: np.ndarray                   # (N, N) skew-symmetric
    train_pairs: np.ndarray         # (P_train, 2)
    test_pairs: np.ndarray          # (P_test, 2)
    available_bases: tuple[str, ...]
    basis_fn: Callable[[str, int], np.ndarray]
    eval_f: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    metadata: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.F.shape[0]

    @property
    def f_norm_sq(self) -> float:
        # ‖f‖²_{π×π} = (1/N²) ‖F‖²_F under empirical measure.
        return float(np.sum(self.F ** 2) / (self.N ** 2))

    def basis(self, name: str, m: int) -> np.ndarray:
        return self.basis_fn(name, m)


# ---------------------------------------------------------------------------
# Train/test split helpers
# ---------------------------------------------------------------------------


def random_pair_split(
    N: int, frac_train: float = 0.8, seed: int = 0,
    symmetrise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Random split of the off-diagonal pair set into train / test.

    If ``symmetrise=True`` (default), pair (i, j) and (j, i) are kept on the
    same side of the split — needed because skew-symmetry means the two carry
    the same information.
    """
    rng = np.random.RandomState(seed)
    pairs_unordered = np.array([(i, j) for i in range(N) for j in range(i + 1, N)],
                               dtype=np.int64)
    rng.shuffle(pairs_unordered)
    n_train = int(frac_train * len(pairs_unordered))
    tr = pairs_unordered[:n_train]
    te = pairs_unordered[n_train:]
    if symmetrise:
        # Add the (j, i) versions
        tr = np.concatenate([tr, tr[:, [1, 0]]], axis=0)
        te = np.concatenate([te, te[:, [1, 0]]], axis=0)
    return tr, te


# ---------------------------------------------------------------------------
# Fit: skew-symmetric coefficient matrix on train pairs
# ---------------------------------------------------------------------------


def orthonormalise(B: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Whiten ``B`` so that (1/N) Bᵀ B ≈ I via Gram inverse-sqrt.

    Same convention as the engine's ``fpta_empirical`` Cholesky orthogonalisation
    but tolerant to rank-deficient ``B`` thanks to the ridge.
    """
    N = B.shape[0]
    G = B.T @ B / N
    eigvals, eigvecs = np.linalg.eigh(G + ridge * np.eye(G.shape[0]))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    return B @ inv_sqrt


MAX_M_FOR_FIT = 250     # sanity cap; MC projection has no m² Gram so this can be generous


def fit_skew_C_train(
    B: np.ndarray, F: np.ndarray, train_pairs: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Monte-Carlo projection of f onto span(B) over the train pair set.

    Implements §4.2 / §4.3.1 of the FPTA paper exactly:
    once ``B`` is orthonormalised under the empirical agent measure
    ((1/N) Bᵀ B ≈ I, done by ``orthonormalise``), the moment estimator
    of the variational coefficient matrix on a tournament edge set
    :math:`\\mathcal{E}` is

        C_ij = (1/|E|) ∑_{(k,l) ∈ E} F_{kl} B_{ki} B_{lj}
             = (1/|E|) Bᵀ F_E B,    F_E[k,l] = F[k,l] · 𝟙[(k,l) ∈ E].

    No matrix inversion, no regularisation parameter. Skew-symmetry is
    enforced after projection (it would only be violated by sampling noise).

    The ``ridge`` parameter is retained for backward compatibility but is
    ignored — kept so existing callers don't break. The MC projection is
    naturally well-defined even on rank-deficient bases (the
    orthonormalisation step zeros the null directions).

    Args:
        B:           (N, m) basis matrix, already orthonormalised under
                     the empirical agent measure.
        F:           (N, N) skew-symmetric payoff matrix.
        train_pairs: (P, 2) ordered pair indices in the edge set.
        ridge:       ignored (kept for API compatibility).

    Returns:
        C: (m, m) skew-symmetric coefficient matrix.
    """
    del ridge   # MC projection has no regulariser
    m = B.shape[1]
    if m > MAX_M_FOR_FIT:
        raise ValueError(
            f"Basis dim m={m} exceeds MAX_M_FOR_FIT={MAX_M_FOR_FIT}; "
            f"truncate the basis before calling fit_skew_C_train."
        )
    rows = train_pairs[:, 0]
    cols = train_pairs[:, 1]
    F_masked = np.zeros_like(F)
    F_masked[rows, cols] = F[rows, cols]
    C = (B.T @ F_masked @ B) / len(train_pairs)
    return 0.5 * (C - C.T)


# ---------------------------------------------------------------------------
# Predict + evaluate
# ---------------------------------------------------------------------------


def predict(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """f̂[i, j] = B[i] @ C @ B[j].  Returns (N, N)."""
    return B @ C @ B.T


def pair_mse(F: np.ndarray, F_hat: np.ndarray, pairs: np.ndarray) -> float:
    if len(pairs) == 0:
        return float("nan")
    diffs = F[pairs[:, 0], pairs[:, 1]] - F_hat[pairs[:, 0], pairs[:, 1]]
    return float(np.mean(diffs ** 2))


def truncate_C(C: np.ndarray, k_keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schur-decompose C, zero out all but the top-``k_keep`` 2×2 blocks.

    Returns ``(C_trunc, omegas_full, Q)`` where omegas_full is the full sorted
    spectrum (so caller can compute fractions over the whole thing).
    """
    from fptajax.decomposition import skew_symmetric_schur
    sch = skew_symmetric_schur(np.asarray(C, dtype=np.float64))
    Q = np.asarray(sch.Q)
    omegas = np.asarray(sch.eigenvalues)            # already sorted desc
    nc_full = int(sch.n_components)
    k = max(0, min(k_keep, nc_full))
    U = np.zeros_like(C)
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    for j in range(k):
        U[2 * j:2 * j + 2, 2 * j:2 * j + 2] = omegas[j] * R
    C_trunc = Q @ U @ Q.T
    return C_trunc, omegas, Q


def fit_eval(
    ds: FPTADataset, basis_name: str, m: int,
    k_trunc: Optional[int] = None,
    ridge: float = 1e-4,
) -> dict:
    """Single fit + evaluate. Returns a metrics dict."""
    B_raw = ds.basis(basis_name, m)
    B = orthonormalise(B_raw)                                        # whiten
    C = fit_skew_C_train(B, ds.F, ds.train_pairs, ridge=ridge)
    if k_trunc is not None:
        C_used, omegas, _ = truncate_C(C, k_trunc)
    else:
        from fptajax.decomposition import skew_symmetric_schur
        sch = skew_symmetric_schur(np.asarray(C, dtype=np.float64))
        omegas = np.asarray(sch.eigenvalues)
        C_used = C
    F_hat = predict(B, C_used)
    return {
        "basis": basis_name,
        "m": m,
        "k_trunc": k_trunc if k_trunc is not None else len(omegas),
        "train_mse": pair_mse(ds.F, F_hat, ds.train_pairs),
        "test_mse":  pair_mse(ds.F, F_hat, ds.test_pairs),
        "full_mse":  float(np.mean((ds.F - F_hat) ** 2)),
        "f_norm_sq": ds.f_norm_sq,
        "spectrum":  omegas,
        "C": C,
        "B": B,
        "F_hat": F_hat,
    }


def normalised_test_mse(metrics: dict) -> float:
    """Test MSE divided by ‖f‖² — fraction of variance unexplained."""
    fnsq = max(metrics["f_norm_sq"], 1e-12)
    return metrics["test_mse"] / fnsq
