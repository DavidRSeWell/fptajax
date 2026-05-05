"""Build a small behavioural-data sample for inspection.

Uses Latin-hypercube sampling on the 5 trait axes (concentration is fixed),
filters out unstable agents that collapse during their probe games, and
runs the tournament/trajectory collection on the survivors. Dumps a
pickle and prints a verbose breakdown of the resulting tensor so the
shape and per-feature semantics are easy to inspect.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from examples.iblotto.behavioral import (
    BlottoBehavioralDataset, build_behavioral_dataset, token_feature_names,
)
from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_NONE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.simulate import simulate_iblotto


_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)


# Per-axis ranges for trait sampling. Avoid the very high lr corners
# (collapse-prone) and keep noise on a useful scale.
TRAIT_RANGES = dict(
    learning_rate       =(0.1, 0.7),
    win_reinvestment    =(-2.0, 2.0),
    loss_disinvestment  =(-2.0, 2.0),
    opponent_allocation =(-2.0, 2.0),
    innovation_noise    =(0.01, 0.30),
)
FIXED_CONCENTRATION = 1000.0


def sample_traits_lhs(n_agents: int, seed: int = 0) -> np.ndarray:
    """Latin-hypercube sample of (n_agents, 6) trait vectors.

    Each axis is uniformly stratified into ``n_agents`` strata; a random
    permutation per axis enforces independence. The 6th column is fixed
    at ``FIXED_CONCENTRATION``.
    """
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    raw = sampler.random(n=n_agents)             # (n_agents, 5) in [0, 1)^5

    out = np.zeros((n_agents, 6), dtype=np.float64)
    keys = ("learning_rate", "win_reinvestment", "loss_disinvestment",
            "opponent_allocation", "innovation_noise")
    for j, k in enumerate(keys):
        lo, hi = TRAIT_RANGES[k]
        out[:, j] = lo + (hi - lo) * raw[:, j]
    out[:, 5] = FIXED_CONCENTRATION
    return out


def stability_filter(
    candidates: np.ndarray, opts: GameOptions, n_probes: int = 4,
    nan_threshold: float = 0.20, seed: int = 0,
) -> np.ndarray:
    """Drop candidates that collapse to NaN allocations in too many probe games.

    For each candidate, simulate a few games against random opponents and
    measure the fraction of games where the agent ends up with a NaN
    cumulative payout (the marker we use for collapsed policies). Keep
    candidates whose fraction is below ``nan_threshold``.
    """
    rng = np.random.RandomState(seed)
    N = candidates.shape[0]
    keep_mask = np.zeros(N, dtype=bool)
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
        keep_mask[i] = nan_frac[i] <= nan_threshold
    return keep_mask, nan_frac


def main(
    n_agents_target: int = 8,
    n_real: int = 4,
    n_rounds: int = 30,
    G_max: int = 5,
    seed: int = 0,
):
    print("=" * 68)
    print("  Iterated Blotto — small behavioural-data sample")
    print("=" * 68)

    n_zones = 5
    opts = GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )

    # Over-sample, filter unstable, keep target count
    n_oversample = max(n_agents_target * 2, n_agents_target + 8)
    cand = sample_traits_lhs(n_oversample, seed=seed)
    keep_mask, nan_frac = stability_filter(cand, opts, n_probes=4, seed=seed)
    survivors = cand[keep_mask]
    print(f"  Sampled {n_oversample} candidates → {keep_mask.sum()} survivors after stability filter")
    if survivors.shape[0] < n_agents_target:
        print(f"  WARNING: only {survivors.shape[0]} stable; relaxing to keep all of them.")
        policies = survivors
    else:
        policies = survivors[:n_agents_target]
    N = policies.shape[0]

    print(f"\nTrait values for the {N} agents kept:")
    print(f"  {'agent':>5}  {'lr':>6}  {'win':>7}  {'loss':>7}  {'opp':>7}  "
          f"{'noise':>7}")
    for i in range(N):
        print(f"  {i:5d}  {policies[i, 0]:6.3f}  {policies[i, 1]:+7.3f}  "
              f"{policies[i, 2]:+7.3f}  {policies[i, 3]:+7.3f}  {policies[i, 4]:7.4f}")

    print(f"\nBuilding behavioural dataset (n_real={n_real}, n_rounds={n_rounds}, "
          f"G_max={G_max}) ...\n")
    ds = build_behavioral_dataset(
        policies, opts, n_real=n_real, G_max=G_max, seed=seed, verbose=True,
    )

    # ----------------------------------------------------------------------
    # Dump pickle + verbose summary of what's inside
    # ----------------------------------------------------------------------
    out_path = _OUT / f"behavioral_small_N{N}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(ds, f)
    print(f"\nSaved → {out_path}")

    print("\n" + "=" * 68)
    print("  Summary of the saved dataset")
    print("=" * 68)
    print(f"  policies.shape          = {ds.policies.shape}     (N, 6)")
    print(f"  agent_data.shape        = {ds.agent_data.shape}     (N, G_max, L_max, sa_dim)")
    print(f"  agent_token_mask.shape  = {ds.agent_token_mask.shape}")
    print(f"  agent_game_mask.shape   = {ds.agent_game_mask.shape}")
    print(f"  F.shape                 = {ds.F.shape},  ‖F‖_F = {np.linalg.norm(ds.F):.3f}")
    print(f"  F_std mean (off-diag)   = {ds.F_std[np.triu_indices(N, 1)].mean():.4f}")
    print(f"  sa_dim                  = {ds.sa_dim}        (= 3 + 3·n_zones for n_zones={n_zones})")
    print(f"  feature names           = {ds.feature_names}")
    print(f"  games per agent         = {ds.agent_game_mask.sum(axis=1).tolist()}")
    print(f"  valid tokens per agent  = {ds.agent_token_mask.sum(axis=(1,2)).tolist()}")

    print("\n  F matrix (rounded to 1 dp):")
    print(np.array_str(ds.F.round(1), max_line_width=120))

    # Show one example token broken into named slices
    i, g, t = 0, 0, 0
    if ds.agent_game_mask[i, g] and ds.agent_token_mask[i, g, t]:
        print(f"\n  Example token  agent={i}, game_slot={g}, t={t}:")
        for name, value in zip(ds.feature_names, ds.agent_data[i, g, t]):
            print(f"    {name:>22s}  {float(value):+.4f}")
    else:
        print(f"\n  agent {i} game {g} token {t} is masked-out")


if __name__ == "__main__":
    main()
