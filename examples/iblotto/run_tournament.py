"""Driver script — JAX port of Iterated_Blotto_Tournament.m.

Generates a 40-agent population with random policy parameters, simulates
all pairs of iterated 100-round Blotto games (n_real realisations per
pair), assembles the skew-symmetric performance matrix F, and runs PTA
+ stability analysis.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto.run_tournament
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_NONE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.tournament import run_tournament
from examples.iblotto.pta_compat import perform_pta


_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)


def sample_random_policies(n_agents: int, seed: int = 0) -> np.ndarray:
    """Match the MATLAB tournament's policy-parameter sampling."""
    rng = np.random.RandomState(seed)
    learning_rates       = np.sort(np.linspace(0.0, 1.0, n_agents))[::-1]
    win_reinvestments    = rng.randn(n_agents)
    loss_disinvestments  = rng.randn(n_agents)
    opponent_allocations = rng.randn(n_agents)
    innovation_noises    = 0.05 * rng.exponential(1.0, size=n_agents)
    concentrations       = 1000.0 * np.ones(n_agents)
    return np.stack([
        learning_rates, win_reinvestments, loss_disinvestments,
        opponent_allocations, innovation_noises, concentrations,
    ], axis=1)


def main(n_agents: int = 40, n_real: int = 800, n_rounds: int = 100, seed: int = 0):
    print(f"=== Iterated Blotto tournament (JAX) ===")
    print(f"  n_agents = {n_agents}, n_real = {n_real}, n_rounds = {n_rounds}")

    policies = sample_random_policies(n_agents, seed=seed)
    print(f"  policies sampled, shape = {policies.shape}")

    n_zones = 5
    opts = GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )

    t0 = time.time()
    F, F_std = run_tournament(
        policies, opts, n_real=n_real, seed=seed,
        p1_budget=1000.0, p2_budget=1000.0,
        p1_inv_frac=0.1, p2_inv_frac=0.1, verbose=True,
    )
    print(f"\n  Tournament done in {time.time() - t0:.1f}s")
    print(f"  ‖F‖_F = {np.linalg.norm(F):.3f}, range = [{F.min():.2f}, {F.max():.2f}]")
    print(f"  mean |F_std| = {np.mean(F_std[np.triu_indices(n_agents, 1)]):.4f}")

    print("\n=== PTA + stability analysis ===")
    rep = perform_pta(F, F_std, variance_target=0.95)
    print(f"  Effective rank (95% var): {rep.effective_rank}")
    print(f"  Top 6 ω: {rep.omegas[:12:2].round(2)}")
    print(f"  Importances: {rep.importances[:6].round(4)}")
    print(f"  Cumulative: {rep.cumulative_importances[:6].round(4)}")
    print(f"  Embedding error bounds (top {rep.effective_rank}): "
          f"{rep.embedding_error_bounds[:rep.effective_rank].round(4)}")

    out_pkl = _OUT / f"tournament_{n_agents}agents_{n_real}real.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(
            policies=policies, F=F, F_std=F_std,
            n_zones=n_zones, n_rounds=n_rounds, n_real=n_real,
            pta=rep,
        ), f)
    print(f"\nSaved {out_pkl}")


if __name__ == "__main__":
    main()
