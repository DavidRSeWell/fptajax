"""Smoke test mirroring Test_Iterative_Blotto_With_Autoregressive_Policies.m.

Simulates one 100-round game between two specifically-configured agents
and prints the cumulative payouts. The plot block from the MATLAB driver
is omitted (use ``return_history=True`` and matplotlib if needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_KEEP_INV_DIFFERENCE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.simulate import simulate_iblotto


def main():
    n = 3
    opts = GameOptions(
        n_zones=n,
        zone_values=jnp.ones(n),
        csf_mode=CSF_AUCTION,
        resource_return_mode=RR_KEEP_INV_DIFFERENCE,
        reallocation_mode=REALLOC_STAY_IN_ZONE,
        depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS,
        info_noise=0.0,
        n_rounds=100,
    )

    # Player 1 doubles down on wins, avoids opponent zones; Player 2 just chases.
    p1 = jnp.array([0.2,  2.0,  2.0, -1.0, 0.0, 100.0])
    p2 = jnp.array([0.2,  0.0,  0.0,  2.0, 0.0, 100.0])

    payouts, history = simulate_iblotto(
        p1, p2, opts, key=jax.random.PRNGKey(0),
        p1_budget=1000.0, p2_budget=1000.0,
        p1_inv_frac=0.1, p2_inv_frac=0.1,
        return_history=True,
    )
    print(f"Cumulative payouts (P1, P2): {np.array(payouts)}")
    print(f"Per-round payout differential history (first/last 5):")
    diffs = np.array(history.payouts_round[:, 0] - history.payouts_round[:, 1])
    print(f"  first 5: {diffs[:5].round(3)}")
    print(f"  last  5: {diffs[-5:].round(3)}")
    print(f"  trajectory shape: allocations {history.allocations.shape}, "
          f"budgets {history.budgets.shape}")


if __name__ == "__main__":
    main()
