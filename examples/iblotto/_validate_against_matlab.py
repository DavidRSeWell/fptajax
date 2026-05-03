"""Validate the JAX port against IB_test.mat.

Loads the saved 40-agent policies + F matrix from the MATLAB run, picks a
small subset of pairs, runs the JAX tournament with matched config, and
checks that mean payouts agree to within a few standard errors.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_NONE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.tournament import run_tournament


_MAT = Path("/Users/davidsewell/Projects/fptajax/Alex_IBlotto/IB_test.mat")


def main(n_agents_subset: int = 6, n_real: int = 400):
    d = sio.loadmat(_MAT, struct_as_record=False, squeeze_me=True)
    r = d["results"]
    policies_all = np.asarray(r.attributes, dtype=np.float64)
    F_matlab = np.asarray(r.performance.F, dtype=np.float64)
    F_std_matlab = np.asarray(r.performance.F_stdev, dtype=np.float64)
    n_zones = int(r.game_rules.n_zones)
    n_rounds = int(r.game_rules.options.stopping.parameters)

    print(f"Loaded MATLAB tournament: N={policies_all.shape[0]} agents, "
          f"n_zones={n_zones}, n_rounds={n_rounds}")
    print(f"  Subsetting to first N={n_agents_subset} agents, n_real={n_real}")

    sub_policies = policies_all[:n_agents_subset]
    F_matlab_sub = F_matlab[:n_agents_subset, :n_agents_subset]
    F_std_matlab_sub = F_std_matlab[:n_agents_subset, :n_agents_subset]

    opts = GameOptions(
        n_zones=n_zones,
        zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION,
        resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE,
        depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS,
        info_noise=0.0,
        n_rounds=n_rounds,
    )

    F_jax, F_std_jax = run_tournament(
        sub_policies, opts, n_real=n_real, seed=0,
        p1_budget=1000.0, p2_budget=1000.0,
        p1_inv_frac=0.1, p2_inv_frac=0.1, verbose=False,
    )

    # Compare upper-triangular entries
    iu = np.triu_indices(n_agents_subset, k=1)
    print("\nPer-pair comparison:")
    print(f"  {'(i, j)':>8}  {'MATLAB F':>10}  {'JAX F':>10}  "
          f"{'matlab std':>11}  {'jax std':>10}  {'|Δ|':>8}  {'σ-norm |Δ|':>11}")
    diffs = []
    z_scores = []
    for i, j in zip(*iu):
        m_F = F_matlab_sub[i, j]; j_F = F_jax[i, j]
        m_s = F_std_matlab_sub[i, j]; j_s = F_std_jax[i, j]
        # Combined std for the difference of means (treat as independent samples)
        comb = float(np.sqrt(m_s * m_s + j_s * j_s))
        delta = float(j_F - m_F)
        z = float(delta / max(comb, 1e-12))
        diffs.append(delta); z_scores.append(z)
        print(f"  ({int(i):2d},{int(j):2d})  {m_F:+10.3f}  {j_F:+10.3f}  "
              f"{m_s:11.3f}  {j_s:10.3f}  {abs(delta):8.3f}  {z:+11.3f}")
    z_arr = np.asarray(z_scores)
    print(f"\nMean |z|: {np.mean(np.abs(z_arr)):.3f}, max |z|: {np.max(np.abs(z_arr)):.3f}")
    print(f"Fraction of pairs with |z| < 2: {(np.abs(z_arr) < 2).mean():.2f} (expected ≈ 0.95 if matching)")


if __name__ == "__main__":
    main()
