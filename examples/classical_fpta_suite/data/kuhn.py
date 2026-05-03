"""Kuhn poker dataset for the classical FPTA suite.

Standard Kuhn poker (Kuhn 1950): three-card deck {J, Q, K}, both players ante
1, then a bet/check protocol. Player 1 (P1) acts first; if P1 checks, P2 may
bet, then P1 may call or fold; if P1 bets, P2 may call or fold; pot is 2 or 4.

Each agent's policy = 12 numbers in [0, 1]:

    α_c = P(bet  | P1 first decision, card c)        c ∈ {J, Q, K}
    β_c = P(bet  | P2 acts after P1 checked, card c) c ∈ {J, Q, K}
    γ_c = P(call | P2 acts after P1 bet, card c)     c ∈ {J, Q, K}
    δ_c = P(call | P1 second decision (P1 checked,
                P2 bet), card c)                     c ∈ {J, Q, K}

Concatenation order: (α_J, α_Q, α_K, β_J, β_Q, β_K, γ_J, γ_Q, γ_K, δ_J, δ_Q, δ_K).

Performance function: each pair plays both roles; payoff to P1 averaged over
the 6 equally-likely deals; symmetric ``f(x, y) = ½ (EU(x as P1, y as P2)
− EU(y as P1, x as P2))``.

Two registered populations:

    "random"        : N agents drawn uniformly from [0, 1]^12.
    "nash_family"   : a 1-parameter NE family with bounded perturbations.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from examples.classical_fpta_suite.protocol import FPTADataset, random_pair_split


# ---------------------------------------------------------------------------
# Vectorised expected payoff
# ---------------------------------------------------------------------------

# Six deals (c1, c2) with c1 != c2, c1 = P1's card, c2 = P2's card.
_DEALS: list[tuple[int, int]] = list(permutations(range(3), 2))   # 6 deals
_CARDS = ("J", "Q", "K")


def _expand(policy: np.ndarray) -> dict:
    """Split a (..., 12) policy tensor into named blocks (..., 3) each."""
    a = np.asarray(policy)
    return {
        "alpha": a[..., 0:3],
        "beta":  a[..., 3:6],
        "gamma": a[..., 6:9],
        "delta": a[..., 9:12],
    }


def _eu_p1_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Expected utility for P1 when x plays P1, y plays P2. Scalar."""
    px = _expand(x); py = _expand(y)
    eu = 0.0
    for c1, c2 in _DEALS:
        h = 1.0 if c1 > c2 else -1.0
        a = px["alpha"][c1]
        d = px["delta"][c1]
        b = py["beta"][c2]
        g = py["gamma"][c2]
        # P1 bets:
        bet_branch = a * (g * 2.0 * h + (1.0 - g) * 1.0)
        # P1 checks:
        chk_p2_bet  = b * (d * 2.0 * h + (1.0 - d) * (-1.0))
        chk_p2_chk  = (1.0 - b) * 1.0 * h
        check_branch = (1.0 - a) * (chk_p2_bet + chk_p2_chk)
        eu += (bet_branch + check_branch) / len(_DEALS)
    return float(eu)


def kuhn_F(policies: np.ndarray) -> np.ndarray:
    """All-vs-all symmetric payoff matrix. Vectorised over deals.

    Returns ``F`` with ``F[i, j] = ½ (EU(i as P1, j as P2) - EU(j as P1, i as P2))``.
    """
    P = np.asarray(policies, dtype=np.float64)
    if P.shape[1] != 12:
        raise ValueError(f"Expected 12-dim policy, got {P.shape[1]}.")
    N = P.shape[0]
    splits = _expand(P)
    alpha, beta = splits["alpha"], splits["beta"]
    gamma, delta = splits["gamma"], splits["delta"]

    EU = np.zeros((N, N), dtype=np.float64)
    for c1, c2 in _DEALS:
        h = 1.0 if c1 > c2 else -1.0
        a = alpha[:, c1]                   # (N,) — P1 bet rate, role=P1
        d = delta[:, c1]                   # (N,)
        b = beta[:, c2]                    # (N,) — P2 bet rate, role=P2
        g = gamma[:, c2]                   # (N,)
        # row i is P1 (alpha_i, delta_i) and col j is P2 (beta_j, gamma_j).
        a_i = a[:, None]; d_i = d[:, None]
        b_j = b[None, :]; g_j = g[None, :]
        bet  = a_i * (g_j * 2.0 * h + (1.0 - g_j) * 1.0)
        chk  = (1.0 - a_i) * (
            b_j * (d_i * 2.0 * h + (1.0 - d_i) * (-1.0))
          + (1.0 - b_j) * 1.0 * h
        )
        EU += (bet + chk) / len(_DEALS)

    F = 0.5 * (EU - EU.T)
    return F


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------


def sample_random_policies(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, size=(N, 12)).astype(np.float64)


def nash_family_policy(alpha: float) -> np.ndarray:
    """The textbook 1-parameter Kuhn NE family at level ``alpha ∈ [0, 1/3]``.

    P1: bet J at α, never bet Q, bet K at 3α.
        2nd decision: never call J, call Q at α + 1/3, always call K.
    P2: when P1 checks, bet J at 1/3, never bet Q, always bet K.
        when P1 bets,   never call J, call Q at 1/3, always call K.

    Returns a single 12-vector.
    """
    alpha = float(alpha)
    p = np.array([
        alpha, 0.0, 3.0 * alpha,            # α_J, α_Q, α_K
        1.0 / 3.0, 0.0, 1.0,                # β_J, β_Q, β_K
        0.0, 1.0 / 3.0, 1.0,                # γ_J, γ_Q, γ_K
        0.0, alpha + 1.0 / 3.0, 1.0,        # δ_J, δ_Q, δ_K
    ], dtype=np.float64)
    return np.clip(p, 0.0, 1.0)


def sample_nash_family(
    N: int = 120, noise_std: float = 0.05, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample policies near the Nash family.

    For each agent: pick α ~ U[0, 1/3], evaluate the NE policy, add Gaussian
    noise of stdev ``noise_std``, clip back to [0, 1]. Returns ``(policies, alphas)``.
    """
    rng = np.random.RandomState(seed)
    alphas = rng.uniform(0.0, 1.0 / 3.0, size=N)
    policies = np.stack([nash_family_policy(a) for a in alphas], axis=0)
    policies = policies + rng.normal(0.0, noise_std, size=policies.shape)
    return np.clip(policies, 0.0, 1.0), alphas


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


def basis_linear(traits: np.ndarray) -> np.ndarray:
    """[1, x_1, ..., x_12]   -> shape (N, 13)."""
    N = traits.shape[0]
    return np.concatenate([np.ones((N, 1)), traits], axis=1)


def basis_pairwise(traits: np.ndarray) -> np.ndarray:
    """{1} ∪ {x_i} ∪ {x_i x_j : i ≤ j}   -> shape (N, 1 + 12 + 78) = (N, 91)."""
    N, T = traits.shape
    cols = [np.ones(N)]
    cols.extend(traits[:, k] for k in range(T))
    for i in range(T):
        for j in range(i, T):
            cols.append(traits[:, i] * traits[:, j])
    return np.stack(cols, axis=1)


def basis_card_grouped(traits: np.ndarray) -> np.ndarray:
    """A natural Kuhn-aware basis: per-card linear and pairwise within blocks.

    Splits the 12-dim policy into 4 blocks of 3 (α, β, γ, δ), and includes:
      - constant
      - all 12 raw coords
      - within-block pairs (3 blocks of 3 pairs each = 12 pairs after dedup)
      - block × block interactions: only those that actually couple in f
        (P1's α × P2's γ, P1's α × P2's β, P1's δ × P2's β,
         and the symmetric counterparts) but we ignore role here so just take
         all cross-block linear products. Shape (N, 91).
    """
    N, T = traits.shape
    alpha = traits[:, 0:3]; beta = traits[:, 3:6]
    gamma = traits[:, 6:9]; delta = traits[:, 9:12]

    cols = [np.ones(N)]
    cols.extend(traits[:, k] for k in range(T))
    blocks = (alpha, beta, gamma, delta)
    for k1 in range(4):
        for k2 in range(k1, 4):
            for i in range(3):
                for j in range(3):
                    if k1 == k2 and j < i:
                        continue
                    cols.append(blocks[k1][:, i] * blocks[k2][:, j])
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Public dataset constructor
# ---------------------------------------------------------------------------


def build_kuhn(
    population: str = "random",
    N: int = 150,
    frac_train: float = 0.8,
    seed: int = 0,
    noise_std: float = 0.05,
) -> FPTADataset:
    """Build a Kuhn poker FPTADataset.

    Args:
        population: "random" or "nash_family".
        N:          number of agents.
        noise_std:  perturbation around Nash family (only for "nash_family").
    """
    if population == "random":
        traits = sample_random_policies(N, seed=seed)
        meta = dict(population="random")
    elif population == "nash_family":
        traits, alphas = sample_nash_family(N, noise_std=noise_std, seed=seed)
        meta = dict(population="nash_family", nash_alphas=alphas, noise_std=noise_std)
    else:
        raise ValueError(f"Unknown population: {population!r}")

    F = kuhn_F(traits)
    train_pairs, test_pairs = random_pair_split(N, frac_train=frac_train, seed=seed)

    available = ("linear", "pairwise", "card_grouped")

    def _basis(name: str, m_target: int) -> np.ndarray:
        if name == "linear":
            B_ = basis_linear(traits)
        elif name == "pairwise":
            B_ = basis_pairwise(traits)
        elif name == "card_grouped":
            B_ = basis_card_grouped(traits)
        else:
            raise KeyError(f"Unknown basis: {name}")
        if 0 < m_target < B_.shape[1]:
            B_ = B_[:, :m_target]
        return B_

    def _eval_f(x: np.ndarray, y: np.ndarray) -> float:
        return _eu_p1_pair(x, y) * 0.5 - _eu_p1_pair(y, x) * 0.5

    return FPTADataset(
        name=f"kuhn_{population}",
        traits=traits,
        F=F,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        available_bases=available,
        basis_fn=_basis,
        eval_f=_eval_f,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for pop in ("random", "nash_family"):
        ds = build_kuhn(population=pop, N=120)
        print(
            f"  {ds.name:24s}  N={ds.N}  ||F||²/N²={ds.f_norm_sq:.4f}  "
            f"diag={np.max(np.abs(np.diag(ds.F))):.2e}  "
            f"skew err={np.max(np.abs(ds.F + ds.F.T)):.2e}"
        )
        for bn in ds.available_bases:
            B_ = ds.basis(bn, 1000)
            print(f"      basis {bn:14s}: shape={B_.shape}")
    # Cross-check single-pair payoff against the matrix entry
    ds = build_kuhn("random", N=20)
    p, q = ds.traits[0], ds.traits[1]
    direct = 0.5 * (_eu_p1_pair(p, q) - _eu_p1_pair(q, p))
    matrix = ds.F[0, 1]
    print(f"  single-pair sanity: direct={direct:+.5f}  matrix={matrix:+.5f}  "
          f"err={abs(direct-matrix):.2e}")
