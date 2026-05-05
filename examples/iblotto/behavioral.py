"""Build behavioural-data tensors from iterated-Blotto trajectories.

For each agent we collect (state, action) tokens from many simulated games
against many opponents. The token format is documented in
``token_feature_names`` below; in short, a token is a vector
``(state || action)`` where the state is what the agent saw at decision
time (round, budget, running differential, last-round outcomes,
opponent's bid distribution) and the action is the agent's normalised
bid this round.

Shapes match the hierarchical-FPTA pipeline:

    agent_data       (N, G_max, L_max, sa_dim)  float32
    agent_token_mask (N, G_max, L_max)          bool
    agent_game_mask  (N, G_max)                 bool
    F                (N, N)                     float32, skew-symmetric
    F_std            (N, N)                     float32

The "game" dimension corresponds to one full iterated-Blotto trajectory
(an entire ``simulate_iblotto`` run); per agent we keep up to ``G_max``
randomly-selected such trajectories from its play history. Tokens within
a game are chronological (round 1 first).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from examples.iblotto.game import GameOptions
from examples.iblotto.simulate import simulate_iblotto


# ---------------------------------------------------------------------------
# Token layout
# ---------------------------------------------------------------------------


def token_dim(n_zones: int) -> int:
    return 3 + 3 * n_zones


def token_feature_names(n_zones: int) -> list[str]:
    """Human-readable feature names per token slot."""
    names = [
        "round_frac",                 # t / n_rounds
        "budget_frac",                # own_budget / initial_budget
        "score_diff_avg",             # (cum_self - cum_opp) / (n_zones * t)
    ]
    names += [f"prev_outcome_z{k}" for k in range(n_zones)]
    names += [f"prev_opp_share_z{k}" for k in range(n_zones)]
    names += [f"action_share_z{k}"   for k in range(n_zones)]
    assert len(names) == token_dim(n_zones)
    return names


def history_to_tokens(
    history,                            # StepOutput pytree from simulate_iblotto
    perspective: int,                   # 0 (P1) or 1 (P2)
    n_rounds: int,
    n_zones: int,
    initial_budget: float,
) -> np.ndarray:
    """Convert one game's history to a (n_rounds - 1, sa_dim) token matrix.

    Round 0 is excluded because the round-0 action is sampled from the
    initial Dirichlet, not the autoregressive update. Token at index k
    corresponds to round k + 1 (in 1-indexed game-round notation).
    """
    eps = 1e-12
    allocations    = np.asarray(history.allocations)          # (T, 2, n)
    budgets        = np.asarray(history.budgets)              # (T, 2)
    payouts_round  = np.asarray(history.payouts_round)        # (T, 2)
    zone_outcomes  = np.asarray(history.zone_outcomes)        # (T, n) — P1's POV
    opp_alloc_seen = np.asarray(history.opp_alloc_seen)       # (T, 2, n)

    T = n_rounds
    if perspective == 0:
        sign = 1.0
    elif perspective == 1:
        sign = -1.0
    else:
        raise ValueError("perspective must be 0 (P1) or 1 (P2)")

    cum_self = np.cumsum(payouts_round[:, perspective])
    cum_opp  = np.cumsum(payouts_round[:, 1 - perspective])

    tokens = np.zeros((T - 1, token_dim(n_zones)), dtype=np.float32)
    for k in range(1, T):
        own_alloc = allocations[k, perspective]
        own_total = max(float(own_alloc.sum()), eps)
        own_share = own_alloc / own_total

        prev_outcome = sign * zone_outcomes[k - 1]            # (n,)
        prev_opp = opp_alloc_seen[k - 1, perspective]
        prev_opp_total = max(float(prev_opp.sum()), eps)
        prev_opp_share = prev_opp / prev_opp_total

        score_diff = (cum_self[k - 1] - cum_opp[k - 1]) / (n_zones * k)

        feat = np.concatenate([
            np.array([k / n_rounds,
                      budgets[k, perspective] / initial_budget,
                      score_diff], dtype=np.float32),
            prev_outcome.astype(np.float32),
            prev_opp_share.astype(np.float32),
            own_share.astype(np.float32),
        ])
        tokens[k - 1] = feat
    return tokens


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------


@dataclass
class BlottoBehavioralDataset:
    policies: np.ndarray          # (N, 6) ground-truth trait vectors
    agent_data: np.ndarray        # (N, G_max, L_max, sa_dim) float32
    agent_token_mask: np.ndarray  # (N, G_max, L_max) bool
    agent_game_mask: np.ndarray   # (N, G_max) bool
    F: np.ndarray                 # (N, N) skew-symmetric (zeros where unobserved)
    F_std: np.ndarray             # (N, N) per-entry stderr
    observed_mask: np.ndarray     # (N, N) bool — True iff F[i,j] was estimated
    sa_dim: int
    n_zones: int
    n_rounds: int
    G_max: int
    L_max: int
    feature_names: list[str]
    metadata: dict


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _simulate_pair_with_history(
    p1, p2, opts: GameOptions, key: Array,
    p1_budget: float, p2_budget: float,
    p1_inv_frac: float, p2_inv_frac: float,
):
    pay, hist = simulate_iblotto(
        p1, p2, opts, key,
        p1_budget=p1_budget, p2_budget=p2_budget,
        p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
        return_history=True,
    )
    return pay, hist


def sample_sparse_edges(N: int, k: int, seed: int = 0) -> np.ndarray:
    """``k``-out random edge set: every agent picks ``k`` opponents uniformly.

    The union (deduplicated) gives the observed pair set. Each agent is
    guaranteed to be in at least ``k`` pairs (its own draws); typical agents
    end up in roughly ``2k`` pairs because they are also drawn by others.
    Returns a ``(P, 2)`` int array of unordered pairs ``(i, j)`` with
    ``i < j``.
    """
    if k >= N:
        raise ValueError(f"k={k} must be < N={N}")
    rng = np.random.RandomState(seed)
    edges: set[tuple[int, int]] = set()
    pool = np.arange(N)
    for i in range(N):
        rest = np.delete(pool, i)
        opps = rng.choice(rest, size=k, replace=False)
        for j in opps:
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def build_behavioral_dataset(
    policies: np.ndarray,
    opts: GameOptions,
    n_real: int = 5,
    G_max: int = 4,
    pairs: np.ndarray | None = None,
    seed: int = 0,
    p1_budget: float = 1000.0,
    p2_budget: float = 1000.0,
    p1_inv_frac: float = 0.1,
    p2_inv_frac: float = 0.1,
    verbose: bool = True,
) -> BlottoBehavioralDataset:
    """Run a (possibly sparse) tournament collecting F and per-agent trajectories.

    Args:
        policies: (N, 6) policy parameters.
        opts:     :class:`GameOptions`.
        n_real:   realisations per observed pair.
        G_max:    cap on games stored per agent (sub-sampled from the full
                  trajectory bag).
        pairs:    optional ``(P, 2)`` int array of unordered pairs ``(i, j)``
                  with ``i < j``. If ``None``, all ``N(N-1)/2`` pairs are used
                  (full round-robin).
        seed:     master PRNG seed.

    Returns:
        :class:`BlottoBehavioralDataset` with ``observed_mask`` set on the
        rows/columns of ``F`` that were actually estimated (and zeros
        elsewhere).
    """
    policies = jnp.asarray(policies, dtype=jnp.float64)
    N = policies.shape[0]
    n = opts.n_zones
    L_max = opts.n_rounds - 1
    sa_dim = token_dim(n)
    rng = np.random.RandomState(seed)

    if pairs is None:
        pairs = np.array(
            [(i, j) for i in range(N) for j in range(i + 1, N)], dtype=np.int64,
        )
    else:
        pairs = np.asarray(pairs, dtype=np.int64)
        # Guard: every pair (i, j) has i < j and 0 <= i, j < N
        assert pairs.ndim == 2 and pairs.shape[1] == 2
        assert (pairs[:, 0] < pairs[:, 1]).all()
        assert (pairs[:, 0] >= 0).all() and (pairs[:, 1] < N).all()

    # Precompute jit'd vmap'd realised diff helper
    @jax.jit
    def _vmap_pair(p1_params, p2_params, keys):
        return jax.vmap(
            lambda k: _simulate_pair_with_history(
                p1_params, p2_params, opts, k,
                p1_budget, p2_budget, p1_inv_frac, p2_inv_frac,
            )
        )(keys)

    games_per_agent: list[list[np.ndarray]] = [[] for _ in range(N)]
    F = np.zeros((N, N))
    F_std = np.zeros((N, N))
    observed_mask = np.zeros((N, N), dtype=bool)

    master_key = jax.random.PRNGKey(seed)
    P = pairs.shape[0]
    for p_idx, (i, j) in enumerate(pairs):
        i = int(i); j = int(j)
        key_pair = jax.random.fold_in(jax.random.fold_in(master_key, i), j)
        keys = jax.random.split(key_pair, n_real)
        pays, hist = _vmap_pair(policies[i], policies[j], keys)
        pays_np = np.asarray(pays)
        diffs = pays_np[:, 0] - pays_np[:, 1]
        F[i, j] = float(np.mean(diffs))
        F[j, i] = -F[i, j]
        stderr = float(np.std(diffs) / np.sqrt(n_real))
        F_std[i, j] = stderr; F_std[j, i] = stderr
        observed_mask[i, j] = True; observed_mask[j, i] = True

        for r in range(n_real):
            realisation = type(hist)(
                allocations    = hist.allocations[r],
                payouts_round  = hist.payouts_round[r],
                budgets        = hist.budgets[r],
                zone_outcomes  = hist.zone_outcomes[r],
                opp_alloc_seen = hist.opp_alloc_seen[r],
            )
            tok_i = history_to_tokens(realisation, perspective=0,
                                      n_rounds=opts.n_rounds, n_zones=n,
                                      initial_budget=p1_budget)
            tok_j = history_to_tokens(realisation, perspective=1,
                                      n_rounds=opts.n_rounds, n_zones=n,
                                      initial_budget=p2_budget)
            games_per_agent[i].append(tok_i)
            games_per_agent[j].append(tok_j)
        if verbose and (p_idx % 50 == 0 or p_idx == P - 1):
            print(f"  pair {p_idx + 1:5d}/{P} ({i:3d},{j:3d}): "
                  f"F = {F[i,j]:+8.3f} ± {stderr:.3f}")

    F = 0.5 * (F - F.T).astype(np.float32)
    F_std = F_std.astype(np.float32)

    agent_data = np.zeros((N, G_max, L_max, sa_dim), dtype=np.float32)
    agent_token_mask = np.zeros((N, G_max, L_max), dtype=bool)
    agent_game_mask  = np.zeros((N, G_max), dtype=bool)
    for i in range(N):
        games = games_per_agent[i]
        if len(games) == 0:
            continue
        sel = rng.choice(len(games), size=min(G_max, len(games)), replace=False)
        for g, idx in enumerate(sel):
            tok = games[idx]
            row_finite = np.all(np.isfinite(tok), axis=1)
            # Zero-out non-finite slots so they don't poison transformer
            # attention (mask=False isn't enough — JAX's softmax + matmul still
            # propagates NaN through masked positions).
            tok_safe = np.where(np.isfinite(tok), tok, 0.0).astype(np.float32)
            agent_data[i, g] = tok_safe
            agent_token_mask[i, g] = row_finite
            agent_game_mask[i, g] = bool(row_finite.any())

    return BlottoBehavioralDataset(
        policies=np.asarray(policies),
        agent_data=agent_data,
        agent_token_mask=agent_token_mask,
        agent_game_mask=agent_game_mask,
        F=F,
        F_std=F_std,
        observed_mask=observed_mask,
        sa_dim=sa_dim,
        n_zones=n,
        n_rounds=opts.n_rounds,
        G_max=G_max,
        L_max=L_max,
        feature_names=token_feature_names(n),
        metadata=dict(
            n_real=n_real, seed=seed,
            p1_budget=p1_budget, p2_budget=p2_budget,
            p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
            n_pairs_observed=int(pairs.shape[0]),
            n_pairs_full_round_robin=N * (N - 1) // 2,
        ),
    )
