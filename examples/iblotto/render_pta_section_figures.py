"""Render the two figures referenced by ``paper/iblotto_pta_section.tex``.

Loads a behavioural bundle (for ground-truth trait values) and a dense
``F_full.npz`` (the full round-robin payoff matrix on the same agents),
runs classical PTA via ``fptajax.pta.pta``, and emits:

  figures/iblotto_pta_spectrum.pdf  — eigenvalue bar chart (log-y), top 10 discs
  figures/iblotto_pta_disc1.pdf     — two-panel disc-1 scatter coloured by
                                      opponent_allocation (left) and
                                      win_reinvestment (right)

Both figures are sized for a single-column or full-page LaTeX include.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -m examples.iblotto.render_pta_section_figures \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --F_full results/pta_vs_bfpta_main_v1_seed0/F_full.npz \
        --out_dir figures
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from fptajax.pta import pta as classical_pta


_TRAIT_INDEX = {
    "learning_rate":       0,
    "win_reinvestment":    1,
    "loss_disinvestment":  2,
    "opponent_allocation": 3,
    "innovation_noise":    4,
}


def _render_spectrum(omegas: np.ndarray, out_path: Path, k_show: int = 10):
    omegas = np.asarray(omegas)
    k_show = min(k_show, len(omegas))
    ratios = omegas[:k_show] / max(omegas[0], 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    xs = np.arange(1, k_show + 1)
    ax.bar(xs, ratios, color="#3a7ca5", alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"disc index $k$")
    ax.set_ylabel(r"$\omega_k / \omega_1$  (log scale)")
    ax.set_title("PTA disc-game spectrum on iblotto")
    ax.set_xticks(xs)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # Annotate top three with cumulative-importance percent
    e2 = omegas ** 2
    cum = np.cumsum(e2) / max(e2.sum(), 1e-12)
    for i in range(min(3, k_show)):
        ax.text(xs[i], ratios[i] * 1.15, f"{100 * cum[i]:.1f}\\%",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _render_disc1(Y1: np.ndarray, traits: np.ndarray, out_path: Path):
    """Y1: (N, 2) disc-1 embedding; traits: (N, 5) ground-truth.

    Colours points by gamma (opponent_allocation) on the left and
    alpha (win_reinvestment) on the right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=120,
                             constrained_layout=True)

    cfg = [
        ("opponent_allocation", r"$\gamma$  (opponent-pursuit)", "RdBu_r"),
        ("win_reinvestment",    r"$\alpha$  (win-reinvestment)", "PiYG"),
    ]
    for ax, (trait_name, label, cmap) in zip(axes, cfg):
        ti = _TRAIT_INDEX[trait_name]
        c = traits[:, ti]
        sc = ax.scatter(Y1[:, 0], Y1[:, 1], c=c, cmap=cmap,
                        s=18, edgecolors="k", linewidths=0.3, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, fontsize=9)
        ax.set_xlabel(r"$Y^{(1)}_1$")
        ax.set_ylabel(r"$Y^{(1)}_2$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(f"coloured by {trait_name}")

    fig.suptitle(r"PTA top-disc embedding $\bm{Y}^{(1)}_{\mathrm{PTA}}$ "
                 r"on iblotto", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main(bundle_path: Path, F_full_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering PTA section figures ===")
    print(f"  bundle:   {bundle_path}")
    print(f"  F_full:   {F_full_path}")
    print(f"  out_dir:  {out_dir}")

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]

    F_full = np.load(F_full_path)["F"]
    if F_full.shape[0] != N:
        raise SystemExit(
            f"F_full has shape {F_full.shape} but bundle has N={N} after "
            f"dropping dead agents. Was F_full generated on the same "
            f"agent set?"
        )
    print(f"  N = {N}, F_full shape = {F_full.shape}")

    res = classical_pta(F_full)
    omegas = np.asarray(res.eigenvalues)            # (K,)
    embeddings = np.asarray(res.embeddings)         # (N, K, 2)
    K = len(omegas)
    print(f"  PTA: K = {K}, omega_1 = {omegas[0]:.3f}, "
          f"top-3 ratios = {(omegas[:3] / omegas[0]).round(3)}")

    # Disc-1 embedding for the scatter
    Y1 = embeddings[:, 0, :]                        # (N, 2)

    # Ground-truth traits live in ds.policies[:, :5] (concentration is col 5)
    traits = np.asarray(ds.policies[:, :5])

    _render_spectrum(omegas, out_dir / "iblotto_pta_spectrum.pdf")
    _render_disc1(Y1, traits, out_dir / "iblotto_pta_disc1.pdf")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True,
                   help="behavioural bundle .pkl (for ground-truth traits + N)")
    p.add_argument("--F_full", type=Path, required=True,
                   help="dense F_full.npz from the round-robin run")
    p.add_argument("--out_dir", type=Path, default=Path("figures"),
                   help="where to write iblotto_pta_*.pdf")
    args = p.parse_args()
    main(args.bundle, args.F_full, args.out_dir)
