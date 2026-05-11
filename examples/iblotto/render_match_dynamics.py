"""Multi-game adaptive BR vs the same opponent — cumulative payoff plot.

For each test opponent, simulate ``K`` consecutive games. After each
game, the *adaptive* strategy ingests the opponent's per-round tokens
(from that game's history) into its particle filter and re-picks a
population BR for the next game. The non-adaptive baselines keep the
same BR throughout.

Strategies:

  * ``filter``  — particle posterior over trait, BR picked from the
                  posterior mean each game (purely on disc score; skill
                  excluded so the cyclic structure is the only source
                  of uplift, matching the result note).
  * ``uniform`` — BR fixed once based on the uniform prior (centroid
                  trait); never updates.
  * ``oracle``  — BR fixed using the true opponent's encoded trait;
                  upper bound on what disc-space ID can buy.
  * ``random``  — fresh random population BR every game; null baseline.

Output: ``figures/fig_online_id_match.pdf`` — two panels, cumulative
realised F (left) and per-game F (right), SE bands across opponents.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m \
        examples.iblotto.render_match_dynamics \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --n_opponents 20 --n_games 10 --n_repeats 4
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import (
    drop_dead_agents, history_to_tokens,
)
from examples.iblotto.game import (
    CSF_AUCTION, GameOptions, INFO_ALL_INVESTMENTS,
    REALLOC_STAY_IN_ZONE, RR_NONE,
)
from examples.iblotto.online_id import (
    OpponentPosterior, load_disc_direct_bc, population_best_response,
    split_token,
)
from examples.iblotto.simulate import simulate_iblotto


plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.format": "pdf",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "mathtext.fontset": "cm",
})

C = {
    "blue":   "#4477AA",
    "orange": "#EE7733",
    "green":  "#228833",
    "grey":   "#BBBBBB",
}


def _build_opts(n_zones, n_rounds):
    return GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )


def _make_simulate_fn(opts, p1_budget, p2_budget, p1_inv_frac, p2_inv_frac):
    """Returns a JIT'd simulator that gives BOTH payouts and history.

    History is needed to extract per-round tokens from the opponent's
    perspective for the filter update.
    """
    @jax.jit
    def sim(p1, p2, key):
        return simulate_iblotto(
            p1, p2, opts, key,
            p1_budget=p1_budget, p2_budget=p2_budget,
            p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
            return_history=True,
        )
    return sim


def run_match(ckpt, ds, true_idx: int, n_games: int, simulate_fn,
              n_zones: int, n_rounds: int, p2_budget: float,
              base_seed: int) -> dict:
    """Simulate ``n_games`` consecutive games for each strategy.

    Returns a dict with arrays of shape (n_games,) per strategy:
    realised F per game.
    """
    N = ds.policies.shape[0]
    rng = np.random.RandomState(base_seed)

    # --- pre-compute the static BR picks (uniform, oracle) ---
    post_uniform = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)
    z_uniform = post_uniform.posterior_mean()
    br_uniform, _ = population_best_response(
        ckpt, z_uniform, use_skill=False)

    z_oracle = ckpt.population_traits[true_idx]
    br_oracle, _ = population_best_response(
        ckpt, z_oracle, use_skill=False)

    # --- adaptive (filter) state ---
    post_filter = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)
    rng_filter = np.random.RandomState(base_seed + 7)

    F_filter  = np.zeros(n_games)
    F_uniform = np.zeros(n_games)
    F_oracle  = np.zeros(n_games)
    F_random  = np.zeros(n_games)

    p2_pol = jnp.asarray(ds.policies[true_idx])
    for g in range(n_games):
        # Each game gets its own PRNG key per strategy. We use the SAME
        # key across strategies so the opponent's randomness is matched
        # — paired comparison.
        k_game = jax.random.PRNGKey(base_seed + 1000 * (true_idx + 1) + g)

        # Filter's choice for this game from current posterior
        z_filt = post_filter.posterior_mean()
        br_filt, _ = population_best_response(
            ckpt, z_filt, use_skill=False)

        br_random = int(rng.randint(N))

        # Simulate each strategy
        def play(br):
            pay, hist = simulate_fn(jnp.asarray(ds.policies[br]),
                                    p2_pol, k_game)
            return float(pay[0] - pay[1]), hist

        F_filter[g],  hist_filt  = play(br_filt)
        F_uniform[g], _          = play(br_uniform)
        F_oracle[g],  _          = play(br_oracle)
        F_random[g],  _          = play(br_random)

        # Update the filter from the opponent's tokens *in the game we
        # just played as filter*. Realistic: you only see opponent
        # behaviour in the actual matches you play.
        tokens = history_to_tokens(
            hist_filt, perspective=1,
            n_rounds=n_rounds, n_zones=n_zones,
            initial_budget=p2_budget,
        )
        for tok in tokens:
            if not np.all(np.isfinite(tok)):
                continue
            s, a = split_token(np.asarray(tok), ckpt.n_zones)
            post_filter.update(s, a, rng=rng_filter)

    return dict(
        filter=F_filter, uniform=F_uniform,
        oracle=F_oracle, random=F_random,
        br_uniform=br_uniform, br_oracle=br_oracle,
    )


def main(bundle_path: Path, ckpt_dir: Path, out_path: Path,
         n_opponents: int, n_games: int, n_repeats: int,
         seed: int) -> None:
    print(f"  loading {ckpt_dir.name}")
    ckpt = load_disc_direct_bc(ckpt_dir, bundle_path, seed=seed)
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, _ = drop_dead_agents(ds, verbose=False)
    N = ds.policies.shape[0]
    md = ds.metadata
    p2_budget = float(md.get("p2_budget", 1000.0))

    opts = _build_opts(ds.n_zones, ds.n_rounds)
    simulate_fn = _make_simulate_fn(
        opts,
        p1_budget=float(md.get("p1_budget", 1000.0)),
        p2_budget=p2_budget,
        p1_inv_frac=float(md.get("p1_inv_frac", 0.1)),
        p2_inv_frac=float(md.get("p2_inv_frac", 0.1)),
    )

    rng = np.random.RandomState(seed)
    opp_indices = rng.choice(N, size=n_opponents, replace=False)
    print(f"  {n_opponents} opponents × {n_games} games × "
          f"{n_repeats} repeats")

    # (n_opp * n_repeats, n_games) for each strategy.
    F_arrays = {k: [] for k in ("filter", "uniform", "oracle", "random")}

    for k_opp, true_idx in enumerate(opp_indices):
        for r in range(n_repeats):
            res = run_match(
                ckpt, ds, int(true_idx), n_games, simulate_fn,
                ds.n_zones, ds.n_rounds, p2_budget,
                base_seed=seed + 1000 * (k_opp + 1) + r,
            )
            for k in F_arrays:
                F_arrays[k].append(res[k])
        if (k_opp + 1) % 5 == 0 or k_opp == n_opponents - 1:
            print(f"    {k_opp+1}/{n_opponents} opponents done")

    F = {k: np.stack(v) for k, v in F_arrays.items()}     # each (n_opp*r, K)
    cum = {k: np.cumsum(v, axis=1) for k, v in F.items()}

    # --- Render ---
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    games_x = np.arange(1, n_games + 1)

    spec = [
        ("filter",  C["blue"],   "-"),
        ("uniform", C["orange"], "--"),
        ("oracle",  C["green"],  "-"),
        ("random",  C["grey"],   ":"),
    ]
    n_traces = next(iter(F.values())).shape[0]
    se = lambda m: m.std(axis=0) / np.sqrt(m.shape[0])

    ax = axes[0]
    for label, color, ls in spec:
        m = cum[label].mean(axis=0)
        s = se(cum[label])
        ax.plot(games_x, m, color=color, ls=ls, label=label,
                marker="o", markersize=3)
        ax.fill_between(games_x, m - s, m + s, color=color, alpha=0.18,
                        linewidth=0)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.set_xlabel("games played vs. opponent")
    ax.set_ylabel(r"cumulative realised $F$")
    ax.set_title("cumulative payoff")
    ax.set_xticks(games_x)
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1]
    for label, color, ls in spec:
        m = F[label].mean(axis=0)
        s = se(F[label])
        ax.plot(games_x, m, color=color, ls=ls, label=label,
                marker="o", markersize=3)
        ax.fill_between(games_x, m - s, m + s, color=color, alpha=0.18,
                        linewidth=0)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.set_xlabel("games played vs. opponent")
    ax.set_ylabel(r"per-game $F$")
    ax.set_title("instantaneous payoff")
    ax.set_xticks(games_x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")

    # --- Print headline numbers ---
    print(f"\n  Headline (mean ± SE over {n_traces} (opp, repeat) pairs):")
    print(f"  {'strategy':>9s} | "
          + " ".join(f"g{g+1:>2d}" for g in range(n_games)))
    for label in ("filter", "uniform", "oracle", "random"):
        m = F[label].mean(axis=0)
        print(f"  {label:>9s} | "
              + " ".join(f"{x:>+5.0f}" for x in m))
    cum_final = {k: v[:, -1] for k, v in cum.items()}
    print(f"\n  Cumulative after {n_games} games:")
    for label in ("filter", "uniform", "oracle", "random"):
        m = cum_final[label].mean()
        s = cum_final[label].std() / np.sqrt(n_traces)
        print(f"    {label:>9s}: {m:>+8.1f} ± {s:.1f}")
    delta = cum_final["filter"] - cum_final["uniform"]
    print(f"\n  Δ filter − uniform (cumulative): "
          f"{delta.mean():+.1f} ± {delta.std()/np.sqrt(n_traces):.1f}, "
          f"positive on {(delta > 0).sum()}/{n_traces} matches")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("figures/fig_online_id_match.pdf"))
    p.add_argument("--n_opponents", type=int, default=20)
    p.add_argument("--n_games", type=int, default=10)
    p.add_argument("--n_repeats", type=int, default=4,
                   help="independent matches per opponent for variance "
                        "reduction in the average curves")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.out,
         args.n_opponents, args.n_games, args.n_repeats, args.seed)
