"""Animate the posterior over opponent traits as tokens are observed.

Two-panel GIF:

  * Left: every population agent's disc-1 ``(u_1, v_1)`` coordinates,
    with point area proportional to its current posterior weight. The
    true opponent is marked with a red star. As evidence accumulates,
    a few dots grow and the rest shrink.
  * Right: top-5 particles (by weight) as a horizontal bar chart, so
    you can read ordinal concentration even when the spatial points
    overlap.

Header annotates tokens observed and the effective sample size.

Usage:
    PYTHONPATH=src:. python -m examples.iblotto.render_online_id_animation \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --true_idx 55 \
        --out figures/online_id_posterior.gif \
        --max_tokens 200 --frame_stride 2 --fps 10
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.online_id import (
    OpponentPosterior, load_disc_direct_bc, split_token,
)


# Style — match the static figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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


def collect_weight_trace(ckpt, ds, true_idx: int, max_tokens: int,
                         seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the filter through up to ``max_tokens`` tokens.

    Returns:
      weights: (T+1, M) — posterior weights after each observation
                          (row 0 = uniform prior).
      ess:     (T+1,)   — effective sample size at each step.
    """
    posterior = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0,
    )
    rng = np.random.RandomState(seed)

    games = np.asarray(ds.agent_data[true_idx])
    masks = np.asarray(ds.agent_token_mask[true_idx])
    gmask = np.asarray(ds.agent_game_mask[true_idx])

    M = posterior.particles.shape[0]
    weights = [posterior.weights().copy()]
    ess     = [posterior.effective_sample_size()]
    consumed = 0
    for g in range(games.shape[0]):
        if not gmask[g]:
            continue
        for t in range(games.shape[1]):
            if not masks[g, t]:
                continue
            if consumed >= max_tokens:
                break
            s, a = split_token(games[g, t], ckpt.n_zones)
            posterior.update(s, a, rng=rng)
            consumed += 1
            weights.append(posterior.weights().copy())
            ess.append(posterior.effective_sample_size())
        if consumed >= max_tokens:
            break

    return np.asarray(weights), np.asarray(ess)


def animate(ckpt, ds, true_idx: int, weights: np.ndarray, ess: np.ndarray,
            out_path: Path, frame_stride: int, fps: int) -> None:
    M = weights.shape[1]
    pop_disc1 = ckpt.population_disc[:, 0, :]            # (M, 2)
    true_pt = pop_disc1[true_idx]
    n_steps = weights.shape[0]
    frames = list(range(0, n_steps, frame_stride))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    # Hold the last frame for ~1 s so the eye lands on the result.
    hold = max(int(fps), 1)
    frames = frames + [n_steps - 1] * hold

    fig = plt.figure(figsize=(8.5, 4.0), dpi=130)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.35)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # -- Left panel: persistent backdrop --
    ax_l.scatter(pop_disc1[:, 0], pop_disc1[:, 1], s=10,
                 color=C["grey"], alpha=0.35, edgecolor="none", zorder=1)
    star = ax_l.scatter([true_pt[0]], [true_pt[1]], s=200, marker="*",
                        color=C["red"], edgecolor="black", linewidth=0.6,
                        zorder=4, label=f"true opp. (#{true_idx})")
    # Foreground particle scatter — sizes updated each frame.
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

    # -- Right panel: top-5 bar chart --
    bars = ax_r.barh(range(5), np.zeros(5),
                     color=[C["blue"]] * 5, edgecolor="none")
    bar_labels = [ax_r.text(0, i, "", va="center", ha="left",
                            fontsize=9, color="0.2") for i in range(5)]
    ax_r.set_xlim(0, 1.0)
    ax_r.set_ylim(-0.5, 4.5)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("posterior weight")
    ax_r.set_yticks([])
    ax_r.set_title("top-5 particles", fontsize=10)

    # -- Header --
    header = fig.suptitle("", fontsize=11, y=0.97)

    # Map weights → marker areas. Use sqrt scaling so the dynamic range
    # is visible (linear blows up the winner; log over-compresses).
    base = 8.0       # area at uniform 1/M
    gain = 1500.0    # area at full weight 1.0
    def w_to_size(w):
        return base + gain * np.sqrt(np.clip(w, 0.0, 1.0))

    def update(frame_idx: int):
        w = weights[frame_idx]
        weight_scatter.set_sizes(w_to_size(w))
        # Highlight the true particle in red when it wins; otherwise blue.
        face = np.tile(np.array([0x44, 0x77, 0xAA, 255]) / 255.0, (M, 1))
        if int(np.argmax(w)) == true_idx:
            face[true_idx] = np.array([0xCC, 0x33, 0x11, 255]) / 255.0
        weight_scatter.set_facecolors(face)

        # Bars: top-5 by weight.
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

        header.set_text(
            f"online opponent ID — tokens: {frame_idx:>3d} / {n_steps-1}    "
            f"ESS: {ess[frame_idx]:>5.1f} / {M}    "
            f"true rank: {int((w > w[true_idx]).sum())}"
        )
        return [weight_scatter, header, *bars, *bar_labels]

    print(f"  rendering {len(frames)} frames at {fps} fps...")
    ani = manim.FuncAnimation(
        fig, update, frames=frames, blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = manim.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    print(f"  saved {out_path}")


def main(bundle_path: Path, ckpt_dir: Path, true_idx: int | None,
         max_tokens: int, frame_stride: int, fps: int,
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

    if true_idx is None:
        rng = np.random.RandomState(seed)
        true_idx = int(rng.randint(N))
    print(f"  impersonating opponent #{true_idx}")

    weights, ess = collect_weight_trace(ckpt, ds, true_idx, max_tokens, seed)
    print(f"  collected {weights.shape[0]} posterior snapshots")

    animate(ckpt, ds, true_idx, weights, ess, out_path, frame_stride, fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--true_idx", type=int, default=None,
                   help="opponent index to impersonate; random if omitted")
    p.add_argument("--max_tokens", type=int, default=200,
                   help="upper bound on warmup tokens to feed in")
    p.add_argument("--frame_stride", type=int, default=2,
                   help="render every Nth posterior snapshot as a frame")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("figures/online_id_posterior.gif"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.true_idx,
         args.max_tokens, args.frame_stride, args.fps, args.out, args.seed)
