"""Render figures for the online opponent identification + BR result note.

Three panels (saved as PDFs for LaTeX inclusion):

  fig_online_id_hitrate.pdf  — top-1/5/10 ID hit rate vs tokens observed
  fig_online_id_br.pdf       — realised mean F by BR strategy vs tokens
  fig_online_id_disc.pdf     — disc-1 plane: posterior-mean trait moves
                               toward the true opponent as evidence accrues

Caches the underlying sweep data to ``--cache_npz`` so figure-only edits
don't trigger a recomputation. Re-run with ``--force_recompute`` to
refresh the cache after retraining.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m \
        examples.iblotto.render_online_id_figures \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --out_dir figures \
        --cache_npz figures/online_id_sweep_cache.npz
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.game import (
    CSF_AUCTION, GameOptions, INFO_ALL_INVESTMENTS,
    REALLOC_STAY_IN_ZONE, RR_NONE,
)
from examples.iblotto.online_id import (
    OpponentPosterior, disc_embedding_from_trait,
    load_disc_direct_bc, population_best_response, split_token,
)
from examples.iblotto.simulate import simulate_iblotto


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
    "lines.markersize": 5,
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
    "red":    "#CC3311",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
    "cyan":   "#66CCEE",
}


# ---------------------------------------------------------------------------
# Data sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepData:
    warmup_levels: np.ndarray            # (T,) int
    rank_at: np.ndarray                  # (N, T) int   — rank of true particle
    w_true_at: np.ndarray                # (N, T) float — posterior weight on true
    posterior_disc1: np.ndarray          # (N_opp, T, 2) — posterior-mean disc-1
    population_disc1: np.ndarray         # (N, 2) — every population disc-1 coord
    true_disc1: np.ndarray               # (N_opp, 2) — true opponent disc-1
    br_F: np.ndarray                     # (4, N_opp, T) — strategies × opp × warmup
    br_strategy_names: list              # length 4 list of strategy labels
    br_opponents: np.ndarray             # (N_opp,) — indices of opponents in BR sweep


def _build_opts(n_zones: int, n_rounds: int) -> GameOptions:
    return GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )


def _make_simulate_fn(opts, p1_budget, p2_budget, p1_inv_frac, p2_inv_frac):
    @jax.jit
    def vmap_pair(p1, p2, keys):
        return jax.vmap(
            lambda k: simulate_iblotto(
                p1, p2, opts, k,
                p1_budget=p1_budget, p2_budget=p2_budget,
                p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
            )
        )(keys)
    return vmap_pair


def _all_valid_tokens(ds, agent_idx: int) -> np.ndarray:
    """Return concatenated valid tokens across all of agent_idx's games."""
    games = np.asarray(ds.agent_data[agent_idx])         # (G, L, sa)
    masks = np.asarray(ds.agent_token_mask[agent_idx])   # (G, L)
    gmask = np.asarray(ds.agent_game_mask[agent_idx])    # (G,)
    out = []
    for g in range(games.shape[0]):
        if not gmask[g]:
            continue
        for t in range(games.shape[1]):
            if masks[g, t]:
                out.append(games[g, t])
    return np.stack(out) if out else np.empty((0, games.shape[-1]))


def _id_filter_trace(ckpt, tokens: np.ndarray, true_idx: int,
                     warmup_levels: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the no-resample filter through ``tokens``; snapshot at each level.

    Returns ``(rank_at, w_true_at, posterior_mean_disc1)``:
      * ``rank_at`` (T,)         — rank of true particle at each warmup level
      * ``w_true_at`` (T,)       — posterior weight on true particle
      * ``posterior_mean_disc1`` (T, 2) — disc-1 of posterior mean trait
    """
    posterior = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)
    T = len(warmup_levels)
    rank_at = np.zeros(T, dtype=int)
    w_true_at = np.zeros(T, dtype=float)
    pmean_disc1 = np.zeros((T, 2), dtype=float)

    next_snapshot = 0
    consumed = 0
    rng = np.random.RandomState(0)

    def take_snapshot(idx):
        w = posterior.weights()
        rank_at[idx] = int((w > w[true_idx]).sum())
        w_true_at[idx] = float(w[true_idx])
        zmean = posterior.posterior_mean()
        pmean_disc1[idx] = disc_embedding_from_trait(ckpt, zmean)[0]

    # Snapshot at warmup=0 (uniform prior) before any update.
    if warmup_levels[0] == 0:
        take_snapshot(0)
        next_snapshot = 1

    for tok in tokens:
        if consumed >= warmup_levels[-1]:
            break
        s, a = split_token(tok, ckpt.n_zones)
        posterior.update(s, a, rng=rng)
        consumed += 1
        while (next_snapshot < T
               and consumed == warmup_levels[next_snapshot]):
            take_snapshot(next_snapshot)
            next_snapshot += 1

    # If tokens ran out before the last requested level, fill remaining
    # snapshots with the final state.
    while next_snapshot < T:
        take_snapshot(next_snapshot)
        next_snapshot += 1

    return rank_at, w_true_at, pmean_disc1


def run_sweep(ckpt, ds, warmup_levels: np.ndarray,
              br_opponents: np.ndarray, simulate_vmap,
              n_eval_games: int, seed: int) -> SweepData:
    N = ds.policies.shape[0]
    T = len(warmup_levels)

    # ---- 1) Per-agent ID trace (all 100 agents) ----
    rank_at = np.zeros((N, T), dtype=int)
    w_true_at = np.zeros((N, T), dtype=float)
    pop_disc1 = np.zeros((N, 2), dtype=float)
    for i in range(N):
        pop_disc1[i] = ckpt.population_disc[i, 0]   # disc-1 plane

    print(f"  ID sweep over {N} agents × {T} warmup levels:")
    for i in range(N):
        toks = _all_valid_tokens(ds, i)
        if toks.shape[0] == 0:
            continue
        r, w, _pm = _id_filter_trace(ckpt, toks, i, warmup_levels)
        rank_at[i] = r
        w_true_at[i] = w
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N} done")

    # ---- 2) BR sweep + posterior-disc1 trace for the BR opponents ----
    n_opp = len(br_opponents)
    br_F = np.zeros((4, n_opp, T), dtype=float)   # filter, uniform, oracle, random
    posterior_disc1 = np.zeros((n_opp, T, 2), dtype=float)
    true_disc1 = np.zeros((n_opp, 2), dtype=float)
    rng = np.random.RandomState(seed)

    print(f"\n  BR sweep over {n_opp} opponents × {T} warmup levels:")
    for k_opp, true_idx in enumerate(br_opponents):
        true_idx = int(true_idx)
        true_disc1[k_opp] = ckpt.population_disc[true_idx, 0]
        toks = _all_valid_tokens(ds, true_idx)
        if toks.shape[0] == 0:
            continue

        # Snapshot posterior-mean disc1 at each warmup level.
        _r, _w, pm_disc1 = _id_filter_trace(
            ckpt, toks, true_idx, warmup_levels)
        posterior_disc1[k_opp] = pm_disc1

        # For each warmup level, build a fresh filter, advance it that
        # many tokens, pick the BR, and simulate.
        z_oracle = ckpt.population_traits[true_idx]
        br_oracle, _ = population_best_response(ckpt, z_oracle, use_skill=False)
        br_random = int(rng.choice(N))

        # Pre-encode posterior_disc1 → trait by reusing posterior_mean
        # (we need full traits, not just disc-1 — recompute via filter run).
        post = OpponentPosterior.from_population(
            ckpt, ess_resample_threshold=0.0)
        next_snapshot = 0
        consumed = 0
        rng_filter = np.random.RandomState(0)

        def eval_strategies(idx):
            z_filter = post.posterior_mean()
            br_filter, _ = population_best_response(
                ckpt, z_filter, use_skill=False)
            # uniform = posterior over uniform prior is uniform mean = first snapshot
            # but we recompute fresh:
            post_unif = OpponentPosterior.from_population(
                ckpt, ess_resample_threshold=0.0)
            z_uniform = post_unif.posterior_mean()
            br_uniform, _ = population_best_response(
                ckpt, z_uniform, use_skill=False)

            keys_f = jax.random.split(jax.random.PRNGKey(seed + 10_000 + k_opp * T + idx),
                                      n_eval_games)
            keys_u = jax.random.split(jax.random.PRNGKey(seed + 20_000 + k_opp * T + idx),
                                      n_eval_games)
            keys_o = jax.random.split(jax.random.PRNGKey(seed + 30_000 + k_opp * T + idx),
                                      n_eval_games)
            keys_r = jax.random.split(jax.random.PRNGKey(seed + 40_000 + k_opp * T + idx),
                                      n_eval_games)

            def mean_diff(br_idx, keys):
                pays = np.asarray(simulate_vmap(
                    jnp.asarray(ds.policies[br_idx]),
                    jnp.asarray(ds.policies[true_idx]),
                    keys,
                ))
                return float(np.mean(pays[:, 0] - pays[:, 1]))

            br_F[0, k_opp, idx] = mean_diff(br_filter, keys_f)
            br_F[1, k_opp, idx] = mean_diff(br_uniform, keys_u)
            br_F[2, k_opp, idx] = mean_diff(br_oracle, keys_o)
            br_F[3, k_opp, idx] = mean_diff(br_random, keys_r)

        if warmup_levels[0] == 0:
            eval_strategies(0)
            next_snapshot = 1

        for tok in toks:
            if consumed >= warmup_levels[-1]:
                break
            s, a = split_token(tok, ckpt.n_zones)
            post.update(s, a, rng=rng_filter)
            consumed += 1
            while (next_snapshot < T
                   and consumed == warmup_levels[next_snapshot]):
                eval_strategies(next_snapshot)
                next_snapshot += 1
        while next_snapshot < T:
            eval_strategies(next_snapshot)
            next_snapshot += 1

        if (k_opp + 1) % 5 == 0 or k_opp == n_opp - 1:
            print(f"    {k_opp+1}/{n_opp} BR opponents done")

    return SweepData(
        warmup_levels=warmup_levels,
        rank_at=rank_at, w_true_at=w_true_at,
        posterior_disc1=posterior_disc1,
        population_disc1=pop_disc1,
        true_disc1=true_disc1,
        br_F=br_F,
        br_strategy_names=["filter", "uniform", "oracle", "random"],
        br_opponents=np.asarray(br_opponents),
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _save_cache(path: Path, d: SweepData) -> None:
    np.savez(path,
             warmup_levels=d.warmup_levels,
             rank_at=d.rank_at, w_true_at=d.w_true_at,
             posterior_disc1=d.posterior_disc1,
             population_disc1=d.population_disc1,
             true_disc1=d.true_disc1,
             br_F=d.br_F,
             br_opponents=d.br_opponents,
             br_strategy_names=np.asarray(d.br_strategy_names))


def _load_cache(path: Path) -> SweepData:
    z = np.load(path, allow_pickle=True)
    return SweepData(
        warmup_levels=z["warmup_levels"],
        rank_at=z["rank_at"], w_true_at=z["w_true_at"],
        posterior_disc1=z["posterior_disc1"],
        population_disc1=z["population_disc1"],
        true_disc1=z["true_disc1"],
        br_F=z["br_F"],
        br_strategy_names=list(z["br_strategy_names"]),
        br_opponents=z["br_opponents"],
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_panel_a(d: SweepData, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    # Drop warmup=0 — at the uniform prior every weight equals 1/M
    # exactly, so the rank metric is degenerate (every agent reports
    # "rank 0" by tie-breaking, which would falsely show 100% top-1).
    keep = d.warmup_levels > 0
    x = d.warmup_levels[keep]
    rank = d.rank_at[:, keep]
    top1 = (rank == 0).mean(axis=0)
    top5 = (rank < 5).mean(axis=0)
    top10 = (rank < 10).mean(axis=0)

    ax.plot(x, top1,  marker="o", color=C["blue"],   label="top-1")
    ax.plot(x, top5,  marker="s", color=C["orange"], label="top-5")
    ax.plot(x, top10, marker="^", color=C["green"],  label="top-10")

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("warmup tokens observed")
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("opponent identification quality")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def render_panel_b(d: SweepData, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = d.warmup_levels
    n_opp = d.br_F.shape[1]
    se = lambda m: m.std(axis=0) / np.sqrt(n_opp)

    spec = [
        ("filter",  C["blue"],   "-"),
        ("uniform", C["orange"], "--"),
        ("oracle",  C["green"],  "-"),
        ("random",  C["grey"],   ":"),
    ]
    for label, color, ls in spec:
        i = d.br_strategy_names.index(label)
        m = d.br_F[i].mean(axis=0)
        s = se(d.br_F[i])
        ax.plot(x, m, color=color, ls=ls, label=label, marker="o", markersize=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("warmup tokens observed")
    ax.set_ylabel(r"realised $F$ (BR vs. opponent)")
    ax.set_title("best-response payoff vs. warmup")
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def render_panel_c(d: SweepData, out_path: Path,
                   focus_idx: int | None = None) -> None:
    """Disc-1 plane: posterior-mean disc-1 trajectory toward true opponent."""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    pop = d.population_disc1
    ax.scatter(pop[:, 0], pop[:, 1], s=12, color=C["grey"], alpha=0.45,
               edgecolor="none", label="population")

    # Pick the BR opponent whose final-warmup posterior-mean is closest to
    # the truth — illustrative "success" trajectory.
    if focus_idx is None:
        final_dev = np.linalg.norm(
            d.posterior_disc1[:, -1] - d.true_disc1, axis=-1)
        focus_idx = int(np.argmin(final_dev))
    traj = d.posterior_disc1[focus_idx]
    true_pt = d.true_disc1[focus_idx]
    opp_label = int(d.br_opponents[focus_idx])

    ax.plot(traj[:, 0], traj[:, 1], color=C["blue"], linewidth=1.4,
            alpha=0.85, zorder=2)
    # Each warmup level as a dot, sized by warmup count so endpoints stand out.
    sizes = 18 + 6 * np.arange(len(d.warmup_levels))
    sc = ax.scatter(traj[:, 0], traj[:, 1], s=sizes, c=d.warmup_levels,
                    cmap="viridis", edgecolor="white", linewidth=0.6,
                    zorder=3)
    # Annotate endpoints to make the trajectory direction unambiguous.
    ax.annotate(f"  n={int(d.warmup_levels[0])}",
                xy=(traj[0, 0], traj[0, 1]), fontsize=8, color="0.25",
                xytext=(4, 4), textcoords="offset points")
    ax.annotate(f"  n={int(d.warmup_levels[-1])}",
                xy=(traj[-1, 0], traj[-1, 1]), fontsize=8, color="0.25",
                xytext=(4, -10), textcoords="offset points")
    ax.scatter([true_pt[0]], [true_pt[1]], s=160, marker="*",
               color=C["red"], edgecolor="black", linewidth=0.6,
               zorder=4, label=f"true opponent (#{opp_label})")

    ax.set_xlabel(r"disc-1 $u_1$")
    ax.set_ylabel(r"disc-1 $v_1$")
    ax.set_title("posterior concentrates onto opponent")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("warmup tokens (posterior mean)")
    cb.ax.tick_params(labelsize=8)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=True,
                    facecolor="white", edgecolor="0.7", framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(bundle_path: Path, ckpt_dir: Path, out_dir: Path,
         cache_npz: Path | None, force_recompute: bool,
         n_br_opponents: int, n_eval_games: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if (cache_npz is not None and cache_npz.exists() and not force_recompute):
        print(f"  loading cached sweep from {cache_npz}")
        data = _load_cache(cache_npz)
    else:
        print("  computing sweep (this may take a few minutes)...")
        ckpt = load_disc_direct_bc(ckpt_dir, bundle_path, seed=seed)
        with open(bundle_path, "rb") as f:
            ds = pickle.load(f)
        if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
            ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                     ds.agent_data, 0.0).astype(np.float32)
        ds, _ = drop_dead_agents(ds, verbose=False)
        N = ds.policies.shape[0]

        warmup_levels = np.array([0, 5, 10, 25, 50, 100, 200, 392])
        rng = np.random.RandomState(seed)
        br_opponents = rng.choice(N, size=n_br_opponents, replace=False)

        opts = _build_opts(ds.n_zones, ds.n_rounds)
        md = ds.metadata
        simulate_vmap = _make_simulate_fn(
            opts,
            p1_budget=float(md.get("p1_budget", 1000.0)),
            p2_budget=float(md.get("p2_budget", 1000.0)),
            p1_inv_frac=float(md.get("p1_inv_frac", 0.1)),
            p2_inv_frac=float(md.get("p2_inv_frac", 0.1)),
        )

        data = run_sweep(ckpt, ds, warmup_levels, br_opponents,
                         simulate_vmap, n_eval_games, seed)
        if cache_npz is not None:
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            _save_cache(cache_npz, data)
            print(f"  cached sweep → {cache_npz}")

    print("\n  rendering figures...")
    render_panel_a(data, out_dir / "fig_online_id_hitrate.pdf")
    render_panel_b(data, out_dir / "fig_online_id_br.pdf")
    render_panel_c(data, out_dir / "fig_online_id_disc.pdf")
    print("  done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    p.add_argument("--cache_npz", type=Path,
                   default=Path("figures/online_id_sweep_cache.npz"))
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--n_br_opponents", type=int, default=20)
    p.add_argument("--n_eval_games", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.out_dir,
         args.cache_npz, args.force_recompute,
         args.n_br_opponents, args.n_eval_games, args.seed)
