"""Build a full behavioural-data bundle with sparse pair sampling.

Mirrors the small-test generator but uses ``k``-out random pair sampling
instead of round-robin. Use this for runs at the scale of N=100-300.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto.generate_behavioral_data \
        --N 200 --k 20 --n_real 50 --n_rounds 100 --G_max 8 --tag main_v1
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from examples.iblotto.behavioral import (
    BlottoBehavioralDataset, build_behavioral_dataset, sample_sparse_edges,
)
from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_NONE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.simulate import simulate_iblotto


_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)


TRAIT_RANGES = dict(
    learning_rate       =(0.1, 0.7),
    win_reinvestment    =(-2.0, 2.0),
    loss_disinvestment  =(-2.0, 2.0),
    opponent_allocation =(-2.0, 2.0),
    innovation_noise    =(0.01, 0.30),
)
FIXED_CONCENTRATION = 1000.0


def sample_traits_lhs(n_agents: int, seed: int = 0) -> np.ndarray:
    """Latin-hypercube sample of (n_agents, 6). Concentration column is fixed."""
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    raw = sampler.random(n=n_agents)
    out = np.zeros((n_agents, 6), dtype=np.float64)
    keys = ("learning_rate", "win_reinvestment", "loss_disinvestment",
            "opponent_allocation", "innovation_noise")
    for j, k in enumerate(keys):
        lo, hi = TRAIT_RANGES[k]
        out[:, j] = lo + (hi - lo) * raw[:, j]
    out[:, 5] = FIXED_CONCENTRATION
    return out


def stability_filter(
    candidates: np.ndarray, opts: GameOptions, n_probes: int = 5,
    nan_threshold: float = 0.20, seed: int = 0,
) -> np.ndarray:
    """Probe each candidate against random opponents; drop if too many NaN games."""
    rng = np.random.RandomState(seed)
    N = candidates.shape[0]
    keep = np.zeros(N, dtype=bool)
    nan_frac = np.zeros(N)

    @jax.jit
    def one_game(p1, p2, key):
        return simulate_iblotto(p1, p2, opts, key,
                                p1_budget=1000.0, p2_budget=1000.0,
                                p1_inv_frac=0.1, p2_inv_frac=0.1)

    for i in range(N):
        n_nan = 0
        for r in range(n_probes):
            j = rng.randint(N)
            while j == i: j = rng.randint(N)
            k = jax.random.PRNGKey(seed + 1000 * i + r)
            pay = np.asarray(one_game(jnp.asarray(candidates[i]),
                                      jnp.asarray(candidates[j]), k))
            if not np.all(np.isfinite(pay)):
                n_nan += 1
        nan_frac[i] = n_nan / n_probes
        keep[i] = nan_frac[i] <= nan_threshold
    return keep, nan_frac


def estimate_compute(N: int, k: int, n_real: int, n_rounds: int) -> dict:
    n_pairs = max(N * k - N * (k - 1) // 2, 1)   # rough upper bound after dedup
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    total_games = n_pairs * n_real
    total_rounds = total_games * n_rounds
    return dict(
        n_pairs=n_pairs, total_games=total_games, total_rounds=total_rounds,
        full_rr_pairs=N * (N - 1) // 2,
        speedup_vs_round_robin=(N * (N - 1) // 2) / max(n_pairs, 1),
    )


def main(
    N: int, k: int, n_real: int, n_rounds: int, G_max: int,
    tag: str = "main_v1", seed: int = 0,
):
    print("=" * 72)
    print(f"  Iterated Blotto behavioural-data generator (sparse, k-out edges)")
    print("=" * 72)
    print(f"  N={N}  k={k}  n_real={n_real}  n_rounds={n_rounds}  G_max={G_max}  tag={tag}")

    est = estimate_compute(N, k, n_real, n_rounds)
    print(f"  estimated pairs: {est['n_pairs']} / {est['full_rr_pairs']}  "
          f"({est['speedup_vs_round_robin']:.2f}× speedup vs round-robin)")
    print(f"  total round-simulations: ~{est['total_rounds']:,}")

    n_zones = 5
    opts = GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )

    # Step 1: sample + stability filter
    n_oversample = int(np.ceil(N * 1.4))   # ~40% overhead so the filter has room
    cand = sample_traits_lhs(n_oversample, seed=seed)
    keep, nan_frac = stability_filter(cand, opts, n_probes=5, seed=seed)
    survivors = cand[keep]
    if survivors.shape[0] < N:
        print(f"  WARNING: only {survivors.shape[0]} stable candidates "
              f"(need {N}); using all of them.")
        policies = survivors
    else:
        policies = survivors[:N]
    N_actual = policies.shape[0]
    print(f"  Stability filter: {len(cand)} → {keep.sum()} stable; "
          f"using N_actual = {N_actual}")

    # Step 2: sample sparse edges
    pairs = sample_sparse_edges(N_actual, k, seed=seed)
    print(f"  Sampled {pairs.shape[0]} unique edges (target ≈ {N_actual * k - (N_actual * k * k) // (2 * N_actual)})")

    # Step 3: simulate + collect trajectories
    t0 = time.time()
    ds = build_behavioral_dataset(
        policies, opts, n_real=n_real, G_max=G_max, pairs=pairs,
        seed=seed, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Tournament + trajectory collection done in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # Step 4: dump bundle
    out_path = _OUT / f"behavioral_{tag}_N{N_actual}_k{k}_nr{n_real}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(ds, f)
    print(f"  Saved → {out_path}")

    # Step 5: summary
    print("\n  === Summary ===")
    print(f"    policies.shape         = {ds.policies.shape}")
    print(f"    agent_data.shape       = {ds.agent_data.shape}")
    print(f"    F.shape                = {ds.F.shape}")
    print(f"    observed pairs         = {int(ds.observed_mask.sum() // 2)} of "
          f"{N_actual * (N_actual - 1) // 2}")
    print(f"    games / agent (mean)   = {ds.agent_game_mask.sum(axis=1).mean():.2f} "
          f"of G_max={G_max}")
    print(f"    valid tokens / agent   = {ds.agent_token_mask.sum(axis=(1,2)).mean():.0f} "
          f"(median={np.median(ds.agent_token_mask.sum(axis=(1,2))):.0f})")
    print(f"    F std (off-diag, observed) = "
          f"{ds.F_std[ds.observed_mask].mean():.4f}")
    print(f"    ‖F‖_F                  = {np.linalg.norm(ds.F):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",       type=int, default=120)
    parser.add_argument("--k",       type=int, default=20)
    parser.add_argument("--n_real",  type=int, default=50)
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--G_max",   type=int, default=8)
    parser.add_argument("--tag",     type=str, default="main_v1")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()
    main(N=args.N, k=args.k, n_real=args.n_real, n_rounds=args.n_rounds,
         G_max=args.G_max, tag=args.tag, seed=args.seed)
