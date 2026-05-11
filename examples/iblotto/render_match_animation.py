"""Animated multi-game match dynamics: posterior + cumulative payoff.

Renders one match against a chosen opponent as a GIF showing both:

  * how the particle posterior tightens around the true opponent as
    successive games are played and observed (top row);
  * how cumulative realised F accrues for the four strategies, with the
    filter line bending upward as the posterior concentrates (bottom).

Frames are per-game; each game's state is held for ``--hold_per_game``
frames so the eye can land on each panel before the next game appears.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m \
        examples.iblotto.render_match_animation \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --true_idx 55 --n_games 10 --hold_per_game 8 --fps 10 \
        --out figures/online_id_match.gif
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as manim
import matplotlib.gridspec as gridspec
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
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "cm",
})

C = {
    "blue":   "#4477AA",
    "orange": "#EE7733",
    "green":  "#228833",
    "red":    "#CC3311",
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
    @jax.jit
    def sim(p1, p2, key):
        return simulate_iblotto(
            p1, p2, opts, key,
            p1_budget=p1_budget, p2_budget=p2_budget,
            p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
            return_history=True,
        )
    return sim


def collect_match(ckpt, ds, true_idx: int, n_games: int, simulate_fn,
                  n_zones: int, n_rounds: int, p2_budget: float,
                  base_seed: int) -> dict:
    """Play ``n_games`` against opponent.

    Captures per-TOKEN posterior weights and per-ROUND cumulative payoff
    so the animation can visualise the within-game dynamics, not just
    game boundaries.
    """
    N = ds.policies.shape[0]
    rng_random = np.random.RandomState(base_seed)

    post_uniform = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)
    z_uniform = post_uniform.posterior_mean()
    br_uniform, _ = population_best_response(
        ckpt, z_uniform, use_skill=False)

    z_oracle = ckpt.population_traits[true_idx]
    br_oracle, _ = population_best_response(
        ckpt, z_oracle, use_skill=False)

    post_filter = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)
    rng_filter = np.random.RandomState(base_seed + 7)

    # Per-token posterior snapshots (uniform prior at index 0).
    weights_tok = [post_filter.weights().copy()]
    # Per-round F per strategy, concatenated across games.
    rF = {k: [] for k in ("filter", "uniform", "oracle", "random")}
    # x-coordinate (tokens-observed) at which each per-round F lives.
    # Round r in game g produces a payoff *and* a token (for r >= 1; r=0
    # has a payoff but no token). We align cumulative-F's x to tokens
    # observed AFTER that round.
    rF_x = []          # length n_rounds * n_games — strictly monotonic round-idx
    game_boundaries = [0]    # round-idx boundaries where new games start
    game_boundary_tokens = [0]  # corresponding tokens-observed at boundaries
    F_per_game = {k: np.zeros(n_games) for k in rF}
    br_choices = {k: np.zeros(n_games, dtype=int) for k in rF}
    n_tokens_total = 0

    p2_pol = jnp.asarray(ds.policies[true_idx])
    for g in range(n_games):
        k_game = jax.random.PRNGKey(base_seed + 1000 * (true_idx + 1) + g)
        z_filt = post_filter.posterior_mean()
        br_filt, _ = population_best_response(
            ckpt, z_filt, use_skill=False)
        br_random = int(rng_random.randint(N))
        br_choices["filter"][g]  = br_filt
        br_choices["uniform"][g] = br_uniform
        br_choices["oracle"][g]  = br_oracle
        br_choices["random"][g]  = br_random

        def play(br):
            pay, hist = simulate_fn(jnp.asarray(ds.policies[br]),
                                    p2_pol, k_game)
            return pay, hist

        pay_filt, hist_filt = play(br_filt)
        pay_unif, hist_unif = play(br_uniform)
        pay_orc,  hist_orc  = play(br_oracle)
        pay_rnd,  hist_rnd  = play(br_random)

        F_per_game["filter"][g]  = float(pay_filt[0] - pay_filt[1])
        F_per_game["uniform"][g] = float(pay_unif[0] - pay_unif[1])
        F_per_game["oracle"][g]  = float(pay_orc[0]  - pay_orc[1])
        F_per_game["random"][g]  = float(pay_rnd[0]  - pay_rnd[1])

        # Per-round F traces (payouts_round shape (n_rounds, 2))
        per_round = {
            "filter":  np.asarray(hist_filt.payouts_round),
            "uniform": np.asarray(hist_unif.payouts_round),
            "oracle":  np.asarray(hist_orc.payouts_round),
            "random":  np.asarray(hist_rnd.payouts_round),
        }
        for r in range(n_rounds):
            for k, pr in per_round.items():
                rF[k].append(float(pr[r, 0] - pr[r, 1]))
            # Strictly monotonic x in round-index across all games.
            # Game boundaries land at multiples of n_rounds; we relabel
            # ticks to "tokens observed" via the boundary annotation.
            rF_x.append(g * n_rounds + r)

        # Now feed the filter game's tokens into the posterior, recording
        # per-token snapshots.
        tokens = history_to_tokens(
            hist_filt, perspective=1,
            n_rounds=n_rounds, n_zones=n_zones,
            initial_budget=p2_budget,
        )
        for tok in tokens:
            t = np.asarray(tok)
            if not np.all(np.isfinite(t)):
                continue
            s, a = split_token(t, ckpt.n_zones)
            post_filter.update(s, a, rng=rng_filter)
            weights_tok.append(post_filter.weights().copy())
            n_tokens_total += 1

        game_boundaries.append((g + 1) * n_rounds)
        game_boundary_tokens.append(n_tokens_total)

    return dict(
        weights_tok=np.asarray(weights_tok),     # (T+1, M)
        rF={k: np.asarray(v) for k, v in rF.items()},
        rF_x=np.asarray(rF_x),                   # round-idx (monotonic)
        F_per_game=F_per_game,
        br_choices=br_choices,
        br_uniform=br_uniform,
        br_oracle=br_oracle,
        game_boundaries=np.asarray(game_boundaries),         # round-idx
        game_boundary_tokens=np.asarray(game_boundary_tokens), # tokens at boundaries
        n_tokens_total=n_tokens_total,
        n_rounds=n_rounds,
    )


def animate(ckpt, ds, true_idx: int, match: dict, out_path: Path,
            n_games: int, token_stride: int, fps: int) -> None:
    weights_tok = match["weights_tok"]
    M = weights_tok.shape[1]
    T_total = match["n_tokens_total"]
    n_rounds = match["n_rounds"]
    pop_disc1 = ckpt.population_disc[:, 0, :]
    true_pt = pop_disc1[true_idx]
    boundaries = match["game_boundaries"]            # in round-idx space
    boundary_toks = match["game_boundary_tokens"]    # tokens at each boundary

    rF = match["rF"]
    rF_x = match["rF_x"]                              # round-idx
    cum_rF = {k: np.cumsum(v) for k, v in rF.items()}
    R_total = n_games * n_rounds                      # total round-idx span

    # Animation indexed by round-idx so the cumulative plot is monotonic.
    # We derive tokens-observed from round-idx (used for the posterior
    # snapshot lookup and the header).
    frame_rounds = list(range(0, R_total, token_stride))
    if frame_rounds[-1] != R_total - 1:
        frame_rounds.append(R_total - 1)
    for b in boundaries:
        if 0 < int(b) < R_total and int(b) not in frame_rounds:
            frame_rounds.append(int(b))
    frame_rounds = sorted(set(frame_rounds))
    intro = [0] * max(int(fps) // 2, 4)
    outro = [R_total - 1] * max(int(fps), 6)
    frames = intro + frame_rounds + outro

    fig = plt.figure(figsize=(8.5, 6.5), dpi=120)
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.5, 1.0],
        hspace=0.40, wspace=0.30,
    )
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    # ----- Top-left: disc-1 scatter -----
    ax_l.scatter(pop_disc1[:, 0], pop_disc1[:, 1], s=10,
                 color=C["grey"], alpha=0.35, edgecolor="none", zorder=1)
    ax_l.scatter([true_pt[0]], [true_pt[1]], s=200, marker="*",
                 color=C["red"], edgecolor="black", linewidth=0.6,
                 zorder=4, label=f"true opp. (#{true_idx})")
    weight_scatter = ax_l.scatter(pop_disc1[:, 0], pop_disc1[:, 1],
                                  s=np.zeros(M), color=C["blue"],
                                  alpha=0.6, edgecolor="white",
                                  linewidth=0.4, zorder=3)
    ax_l.set_xlabel(r"disc-1 $u_1$")
    ax_l.set_ylabel(r"disc-1 $v_1$")
    ax_l.set_aspect("equal", adjustable="datalim")
    leg = ax_l.legend(loc="upper left", fontsize=8, frameon=True,
                      facecolor="white", edgecolor="0.7", framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)
    ax_l.set_title("posterior over opponent")

    # ----- Top-right: top-5 bars -----
    bars = ax_r.barh(range(5), np.zeros(5),
                     color=[C["blue"]] * 5, edgecolor="none")
    bar_labels = [ax_r.text(0, i, "", va="center", ha="left",
                            fontsize=9, color="0.2") for i in range(5)]
    ax_r.set_xlim(0, 1.0)
    ax_r.set_ylim(-0.5, 4.5)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("posterior weight")
    ax_r.set_yticks([])
    ax_r.set_title("top-5 particles")

    # ----- Bottom: per-round cumulative payoff vs tokens observed -----
    spec = [
        ("filter",  C["blue"],   "-"),
        ("uniform", C["orange"], "--"),
        ("oracle",  C["green"],  "-"),
        ("random",  C["grey"],   ":"),
    ]
    lines = {}
    points = {}
    for label, color, ls in spec:
        (line,) = ax_b.plot([], [], color=color, ls=ls, label=label,
                            linewidth=1.6)
        (pt,)   = ax_b.plot([], [], color=color, marker="o", ls="",
                            markersize=4)
        lines[label] = line
        points[label] = pt

    # Tighter y-limits using the actual cumulative-F range, but keep the
    # zero line near the top so the early phase has visible vertical
    # space (avoids the "compressed at the top" issue).
    all_y = np.concatenate([cum_rF[k] for k in cum_rF])
    y_min = min(0.0, all_y.min()) - 20
    y_max = max(0.0, all_y.max()) + 20
    ax_b.set_xlim(-3, R_total + 3)
    ax_b.set_ylim(y_min, y_max)
    ax_b.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    # Light vertical lines at game boundaries (round-idx space).
    for b in boundaries[1:-1]:
        ax_b.axvline(float(b), color="0.85", linewidth=0.6,
                     ls="-", zorder=0)
    # Game labels in the upper margin, plus token-count under each one
    # so the "tokens observed" interpretation is concrete.
    for g in range(len(boundaries) - 1):
        mid = 0.5 * (boundaries[g] + boundaries[g + 1])
        ax_b.text(mid, y_max - 0.05 * (y_max - y_min),
                  f"game {g+1}", ha="center", va="top",
                  fontsize=8, color="0.45")
    # Custom xticks: round indices labelled with cumulative tokens at game ends.
    tick_positions = list(boundaries)
    tick_labels = [f"{int(t)}" for t in boundary_toks]
    ax_b.set_xticks(tick_positions)
    ax_b.set_xticklabels(tick_labels)
    ax_b.set_xlabel("opponent tokens observed")
    ax_b.set_ylabel(r"cumulative realised $F$")
    ax_b.set_title("cumulative payoff (per round)")
    ax_b.legend(loc="lower left", ncol=4)

    # ----- Header -----
    header = fig.suptitle("", fontsize=12, y=0.98)

    base = 8.0
    gain = 1500.0
    def w_to_size(w):
        return base + gain * np.sqrt(np.clip(w, 0.0, 1.0))

    def round_to_tokens(r_idx: int) -> int:
        """How many opponent tokens have been observed by the END of round r_idx.

        Round-idx ``r`` lives in game ``g = r // n_rounds``. Tokens are
        produced at rounds 1..n_rounds-1 within each game (round 0 has
        no autoregressive token). After completing round r:
          - if r % n_rounds == 0  →  no new token this round
          - else                  →  one new token this round
        Cumulative tokens up to and including round r:
          tokens = boundary_toks[g] + (r - g*n_rounds)   [ if r%n_rounds > 0 ]
                 = boundary_toks[g]                       [ if r%n_rounds == 0 ]
        Equivalently: tokens = g * (n_rounds - 1) + max(r % n_rounds, 0)
        """
        g = r_idx // n_rounds
        r_in = r_idx - g * n_rounds
        return g * (n_rounds - 1) + r_in

    def update(r_idx: int):
        tok = round_to_tokens(r_idx)
        tok = min(tok, T_total)  # safety
        w = weights_tok[tok]
        weight_scatter.set_sizes(w_to_size(w))
        face = np.tile(np.array([0x44, 0x77, 0xAA, 255]) / 255.0, (M, 1))
        if tok > 0 and int(np.argmax(w)) == true_idx:
            face[true_idx] = np.array([0xCC, 0x33, 0x11, 255]) / 255.0
        weight_scatter.set_facecolors(face)

        order = np.argsort(w)[::-1][:5]
        for k, idx in enumerate(order):
            bars[k].set_width(float(w[idx]))
            color = C["red"] if int(idx) == true_idx else C["blue"]
            bars[k].set_color(color)
            label = (f" #{int(idx)}{' (true)' if int(idx)==true_idx else ''}: "
                     f"{w[idx]:.3f}")
            bar_labels[k].set_x(float(w[idx]) + 0.01)
            bar_labels[k].set_y(k)
            bar_labels[k].set_text(label)

        # Cumulative-payoff: rounds with rF_x ≤ current round-idx.
        keep = rF_x <= r_idx
        for label in cum_rF:
            xs = rF_x[keep]
            ys = cum_rF[label][keep]
            lines[label].set_data(xs, ys)
            if len(xs) > 0:
                points[label].set_data([xs[-1]], [ys[-1]])

        # Which game are we in.
        g_idx = int(np.searchsorted(boundaries, r_idx, side="right") - 1)
        g_idx = max(0, min(g_idx, n_games - 1))
        if r_idx == 0 and tok == 0:
            sub = "uniform prior — no opponent tokens observed"
        else:
            br_filt = int(match["br_choices"]["filter"][g_idx])
            mark = "  [matches oracle]" if br_filt == match["br_oracle"] else ""
            sub = (f"in game {g_idx+1}/{n_games}, tokens observed: {tok}/{T_total}"
                   f"  —  filter BR this game: #{br_filt}{mark}")
        header.set_text(
            f"opponent #{true_idx} — multi-game adaptive BR\n{sub}"
        )
        return [weight_scatter, header, *bars, *bar_labels,
                *lines.values(), *points.values()]

    print(f"  rendering {len(frames)} frames at {fps} fps...")
    ani = manim.FuncAnimation(fig, update, frames=frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = manim.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    print(f"  saved {out_path}")


def main(bundle_path: Path, ckpt_dir: Path, true_idx: int | None,
         n_games: int, token_stride: int, fps: int,
         out_path: Path, seed: int) -> None:
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

    if true_idx is None:
        rng = np.random.RandomState(seed)
        true_idx = int(rng.randint(N))
    print(f"  impersonating opponent #{true_idx}")

    opts = _build_opts(ds.n_zones, ds.n_rounds)
    simulate_fn = _make_simulate_fn(
        opts,
        p1_budget=float(md.get("p1_budget", 1000.0)),
        p2_budget=p2_budget,
        p1_inv_frac=float(md.get("p1_inv_frac", 0.1)),
        p2_inv_frac=float(md.get("p2_inv_frac", 0.1)),
    )

    print(f"  playing {n_games}-game match...")
    match = collect_match(
        ckpt, ds, true_idx, n_games, simulate_fn,
        ds.n_zones, ds.n_rounds, p2_budget, base_seed=seed,
    )
    F = match["F_per_game"]
    print(f"    per-game F  filter : {F['filter'].astype(int)}")
    print(f"    per-game F  uniform: {F['uniform'].astype(int)}")
    print(f"    per-game F  oracle : {F['oracle'].astype(int)}")
    print(f"    cumulative final : "
          f"filter={F['filter'].sum():+.0f}, "
          f"uniform={F['uniform'].sum():+.0f}, "
          f"oracle={F['oracle'].sum():+.0f}")
    print(f"    total tokens observed: {match['n_tokens_total']}")

    animate(ckpt, ds, true_idx, match, out_path,
            n_games, token_stride, fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--true_idx", type=int, default=None)
    p.add_argument("--n_games", type=int, default=5,
                   help="number of consecutive games against the opponent")
    p.add_argument("--token_stride", type=int, default=3,
                   help="render a frame every N tokens (game boundaries "
                        "are always rendered regardless)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", type=Path,
                   default=Path("figures/online_id_match.gif"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.true_idx,
         args.n_games, args.token_stride, args.fps, args.out, args.seed)
