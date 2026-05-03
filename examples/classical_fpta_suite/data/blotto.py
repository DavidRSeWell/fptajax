"""Blotto dataset for the classical FPTA suite.

Discrete Colonel Blotto with parameters ``(K fields, B budget)``:

    Each agent's trait = an integer allocation a ∈ ℤ^K_{≥0} with sum(a) = B.
    Payoff f(a, b) = (#fields a wins  −  #fields b wins) / K  ∈ [-1, 1].

The trait space is the discrete simplex; we expose normalised allocations
``a / B`` so it lives in the standard simplex Δ^{K-1} (each coord ∈ [0, 1]).
This matches the regularity hypotheses where bases over [0, 1]^T are natural.

Three setups:

  - "small"  : (K=3, B=10)   — 66 agents, fully enumerated
  - "medium" : (K=5, B=20)   — 10626 total, sub-sampled to ~300
  - "large"  : (K=10, B=30)  — ≈211M total, sub-sampled to ~200

Available bases on the simplex:

  - "monomial_d2"      : {1, a_k, a_j a_k} — degree ≤ 2, full multivariate
  - "monomial_d3"      : up to degree 3
  - "bernstein_d2"     : 2-d Bernstein polynomials over the simplex
  - "bernstein_d3"     : 3-d Bernstein polynomials
  - "symmetric_power"  : power-sum symmetric polynomials p_1, ..., p_d
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np

from examples.classical_fpta_suite.protocol import FPTADataset, random_pair_split


# ---------------------------------------------------------------------------
# Allocation enumeration / sampling
# ---------------------------------------------------------------------------


def enumerate_allocations(K: int, B: int) -> np.ndarray:
    """All non-negative integer K-tuples summing to B. Shape (M, K)."""
    out: list[list[int]] = []

    def rec(remaining: int, slots: int, prefix: list[int]):
        if slots == 1:
            out.append(prefix + [remaining]); return
        for v in range(remaining + 1):
            rec(remaining - v, slots - 1, prefix + [v])

    rec(B, K, [])
    return np.asarray(out, dtype=np.int64)


def sample_allocations(K: int, B: int, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample ``n`` distinct integer K-tuples summing to ``B`` uniformly.

    Uses the classical "stars and bars" bijection: sampling a sorted set of
    K-1 dividers from {1, ..., K + B - 1}. We rejection-sample for distinctness.
    """
    seen: set[tuple] = set()
    out: list[list[int]] = []
    max_attempts = max(n * 20, 1000)
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        dividers = sorted(rng.choice(K + B - 1, size=K - 1, replace=False))
        slots = [0] + [d + 1 for d in dividers] + [K + B]
        alloc = tuple(slots[i + 1] - slots[i] - 1 for i in range(K))
        if alloc in seen:
            continue
        seen.add(alloc)
        out.append(list(alloc))
    if len(out) < n:
        raise RuntimeError(
            f"Could only sample {len(out)} distinct allocations after "
            f"{attempts} attempts; consider lower n or larger (K, B)."
        )
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Payoff
# ---------------------------------------------------------------------------


def blotto_pair_payoff(a: np.ndarray, b: np.ndarray) -> float:
    """Single-pair payoff f(a, b) = (#a-wins - #b-wins) / K."""
    K = a.shape[0]
    diffs = a - b
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / K)


def blotto_F(allocations: np.ndarray) -> np.ndarray:
    """All-vs-all payoff matrix on a stack of allocations. Shape (N, N)."""
    N, K = allocations.shape
    A = allocations[:, None, :]
    B_ = allocations[None, :, :]
    diffs = A - B_                                  # (N, N, K)
    a_wins = np.sum(diffs > 0, axis=-1)
    b_wins = np.sum(diffs < 0, axis=-1)
    return ((a_wins - b_wins) / K).astype(np.float64)


# ---------------------------------------------------------------------------
# Bases on normalised allocations (simplex points)
# ---------------------------------------------------------------------------


def _multi_index_up_to(K: int, max_deg: int) -> list[tuple[int, ...]]:
    """All multi-indices α ∈ ℤ^K_{≥0} with sum(α) ≤ max_deg."""
    out: list[tuple[int, ...]] = []
    for d in range(max_deg + 1):
        for combo in combinations_with_replacement(range(K), d):
            counts = [0] * K
            for c in combo:
                counts[c] += 1
            out.append(tuple(counts))
    return out


def basis_monomials(traits: np.ndarray, max_deg: int) -> tuple[np.ndarray, list[str]]:
    """Multivariate monomials a_1^{α_1} ... a_K^{α_K} with |α| ≤ max_deg."""
    N, K = traits.shape
    indices = _multi_index_up_to(K, max_deg)
    cols, labels = [], []
    for alpha in indices:
        col = np.ones(N)
        for k, p in enumerate(alpha):
            if p > 0:
                col = col * (traits[:, k] ** p)
        cols.append(col)
        labels.append("a^" + "".join(map(str, alpha)))
    return np.stack(cols, axis=1), labels


def basis_bernstein(traits: np.ndarray, max_deg: int) -> tuple[np.ndarray, list[str]]:
    """Bernstein polynomials over the simplex: B_α(x) = (n!/Πα_i!) Πx_i^{α_i}.

    We use exactly the multi-indices α with |α| = max_deg (the standard
    Bernstein basis at degree ``max_deg``). This spans all degree-``max_deg``
    polynomials on the simplex (because Σx_i = 1).
    """
    from math import factorial
    N, K = traits.shape
    indices = [a for a in _multi_index_up_to(K, max_deg) if sum(a) == max_deg]
    cols, labels = [], []
    n_fac = factorial(max_deg)
    for alpha in indices:
        coeff = n_fac
        for p in alpha:
            coeff //= factorial(p)
        col = float(coeff) * np.ones(N)
        for k, p in enumerate(alpha):
            if p > 0:
                col = col * (traits[:, k] ** p)
        cols.append(col)
        labels.append("B_" + "".join(map(str, alpha)))
    return np.stack(cols, axis=1), labels


def basis_symmetric_power(traits: np.ndarray, max_deg: int) -> tuple[np.ndarray, list[str]]:
    """Power-sum symmetric polynomials p_d(x) = Σ x_k^d for d = 1, ..., max_deg.

    Spans the symmetric polynomial subspace of the monomial space — useful as
    a coarser basis (not all monomials, just the ones invariant under field
    permutation).
    """
    N, K = traits.shape
    cols = [np.ones(N)]
    labels = ["1"]
    for d in range(1, max_deg + 1):
        cols.append(np.sum(traits ** d, axis=1))
        labels.append(f"p_{d}")
    return np.stack(cols, axis=1), labels


# ---------------------------------------------------------------------------
# Public dataset constructor
# ---------------------------------------------------------------------------


_SETUPS = {
    "small":  dict(K=3,  B=10, max_agents=None,  seed=0),
    "medium": dict(K=5,  B=20, max_agents=300,   seed=0),
    "large":  dict(K=10, B=30, max_agents=200,   seed=0),
}


def build_blotto(setup: str = "small", frac_train: float = 0.8) -> FPTADataset:
    cfg = _SETUPS[setup]
    K, B, max_agents, seed = cfg["K"], cfg["B"], cfg["max_agents"], cfg["seed"]
    rng = np.random.RandomState(seed)

    if max_agents is None:
        allocs = enumerate_allocations(K, B)
    else:
        allocs = sample_allocations(K, B, n=max_agents, rng=rng)

    N = allocs.shape[0]
    traits = allocs.astype(np.float64) / B          # (N, K) on simplex
    F = blotto_F(allocs)
    F = 0.5 * (F - F.T)                             # exact skew (already)
    train_pairs, test_pairs = random_pair_split(N, frac_train=frac_train, seed=seed)

    available = (
        "monomial_d2", "monomial_d3", "monomial_d4",
        "bernstein_d2", "bernstein_d3", "bernstein_d4",
        "symmetric_power_d4", "symmetric_power_d6",
    )

    def _basis(name: str, m_target: int) -> np.ndarray:
        # m_target is advisory; many bases have a fixed dimensionality
        if name.startswith("monomial_d"):
            d = int(name.split("d")[-1])
            B_, _ = basis_monomials(traits, d)
        elif name.startswith("bernstein_d"):
            d = int(name.split("d")[-1])
            B_, _ = basis_bernstein(traits, d)
        elif name.startswith("symmetric_power_d"):
            d = int(name.split("d")[-1])
            B_, _ = basis_symmetric_power(traits, d)
        else:
            raise KeyError(f"Unknown basis: {name}")
        # If caller asked for a smaller m, truncate; never grow beyond the
        # natural size since growing would change the basis identity.
        if 0 < m_target < B_.shape[1]:
            B_ = B_[:, :m_target]
        return B_

    def _eval_f(a: np.ndarray, b: np.ndarray) -> float:
        return blotto_pair_payoff(a, b)

    return FPTADataset(
        name=f"blotto_{setup}",
        traits=traits,
        F=F,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        available_bases=available,
        basis_fn=_basis,
        eval_f=_eval_f,
        metadata=dict(K=K, B=B, allocations=allocs, setup=setup),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for setup in ("small", "medium", "large"):
        ds = build_blotto(setup)
        K = ds.metadata["K"]
        Bbud = ds.metadata["B"]
        nz = int(np.sum(np.abs(ds.F) > 1e-12))
        print(
            f"  {ds.name:18s}  N={ds.N:4d}  K={K:2d}  B={Bbud:2d}  "
            f"||F||²/N²={ds.f_norm_sq:.4f}  "
            f"diag-zero ok={np.all(np.diag(ds.F)==0)}  "
            f"skew err={np.max(np.abs(ds.F + ds.F.T)):.2e}  "
            f"train/test pairs={len(ds.train_pairs)}/{len(ds.test_pairs)}"
        )
        for bn in ds.available_bases[:3]:
            B_ = ds.basis(bn, 1000)
            print(f"      basis {bn:20s}: shape={B_.shape}")
