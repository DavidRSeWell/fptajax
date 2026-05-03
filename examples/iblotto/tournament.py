"""All-pairs tournament — JAX port of Iterated_Blotto_Tournament.m.

Builds the skew-symmetric performance matrix ``F`` where
``F[i, j] = E[ payouts_i − payouts_j ]`` averaged over ``n_real``
independent realizations of the iterated game between agent ``i`` (as
P1) and agent ``j`` (as P2).

Vectorisation strategy: the inner-most loop over realisations is
``vmap``-ed; the outer loop over agent pairs is a Python loop because
each pair has its own ``(p1_params, p2_params)`` and we don't want to
broadcast the full ``(N, N)`` agent-pair tensor at once. For ``N=40,
n_real=800, n_rounds=100`` the total cost is dominated by the
``n_real * n_rounds`` work per pair, which JIT handles well.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from examples.iblotto.game import GameOptions
from examples.iblotto.simulate import simulate_iblotto


def _payout_diff(
    p1_params: Array, p2_params: Array, opts: GameOptions, key: Array,
    p1_budget: float, p2_budget: float,
    p1_inv_frac: float, p2_inv_frac: float,
) -> Array:
    """Run a single iterated game; return payout_p1 − payout_p2 (scalar)."""
    payouts = simulate_iblotto(
        p1_params, p2_params, opts, key,
        p1_budget=p1_budget, p2_budget=p2_budget,
        p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
    )
    return payouts[0] - payouts[1]


def run_tournament(
    policies: np.ndarray | Array,
    opts: GameOptions,
    n_real: int = 800,
    seed: int = 0,
    p1_budget: float = 1000.0,
    p2_budget: float = 1000.0,
    p1_inv_frac: float = 0.1,
    p2_inv_frac: float = 0.1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs iterated-Blotto tournament.

    Args:
        policies:  ``(N, 6)`` matrix of agent policy parameters.
        opts:      :class:`GameOptions`.
        n_real:    realisations per ordered pair.
        seed:      master PRNG seed.
        verbose:   print per-pair progress.

    Returns:
        ``(F, F_std)`` numpy arrays of shape ``(N, N)``.
        ``F`` is exactly skew-symmetric (we average i-vs-j and j-vs-i with
        opposite signs and split). ``F_std`` is the per-entry standard
        error.
    """
    policies = jnp.asarray(policies, dtype=jnp.float64)
    N = policies.shape[0]

    # vmap over realizations
    @jax.jit
    def realised_diffs(p1_params, p2_params, keys):
        return jax.vmap(
            lambda k: _payout_diff(
                p1_params, p2_params, opts, k,
                p1_budget, p2_budget, p1_inv_frac, p2_inv_frac,
            )
        )(keys)

    F = np.zeros((N, N))
    F_std = np.zeros((N, N))
    master_key = jax.random.PRNGKey(seed)

    for i in range(N):
        for j in range(i + 1, N):
            key_pair = jax.random.fold_in(jax.random.fold_in(master_key, i), j)
            keys = jax.random.split(key_pair, n_real)
            diffs = np.asarray(realised_diffs(policies[i], policies[j], keys))
            mean = float(np.mean(diffs))
            sem  = float(np.std(diffs) / np.sqrt(n_real))
            F[i, j] = mean
            F[j, i] = -mean
            F_std[i, j] = sem
            F_std[j, i] = sem
            if verbose and (j == i + 1 or (i * N + j) % 50 == 0):
                print(f"  pair ({i:3d}, {j:3d}): F = {mean:+.4f} ± {sem:.4f}")

    F = 0.5 * (F - F.T)    # belt & braces — already exactly skew
    return F, F_std
