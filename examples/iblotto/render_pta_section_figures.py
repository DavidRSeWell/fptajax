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

_TRAIT_SYMBOL = {
    "learning_rate":       r"$\lambda$",
    "win_reinvestment":    r"$\alpha$",
    "loss_disinvestment":  r"$\beta$",
    "opponent_allocation": r"$\gamma$",
    "innovation_noise":    r"$\sigma$",
}

_TRAIT_CMAP = {
    "learning_rate":       "viridis",
    "win_reinvestment":    "PiYG",
    "loss_disinvestment":  "PuOr",
    "opponent_allocation": "RdBu_r",
    "innovation_noise":    "magma",
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
        ax.text(xs[i], ratios[i] * 1.15, f"{100 * cum[i]:.1f}%",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _joint_r2(Y_disc: np.ndarray, trait_vals: np.ndarray) -> float:
    """OLS joint R^2 of ``trait_vals`` (N,) onto two-column ``Y_disc`` (N, 2)."""
    X = np.concatenate([Y_disc, np.ones((Y_disc.shape[0], 1))], axis=1)
    # Standardise the trait first so R^2 is unit-free
    y = (trait_vals - trait_vals.mean()) / max(trait_vals.std(), 1e-12)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _best_trait_per_disc(
    embeddings: np.ndarray, traits: np.ndarray, n_discs: int,
) -> list[tuple[str, float]]:
    """For each of the top ``n_discs`` discs, pick the trait with highest
    joint R^2 against that disc's 2D embedding.

    Returns a list of ``(trait_name, R^2)`` tuples, length ``n_discs``.
    """
    out = []
    for k in range(n_discs):
        Y = embeddings[:, k, :]                     # (N, 2)
        scores = {}
        for name, ti in _TRAIT_INDEX.items():
            scores[name] = _joint_r2(Y, traits[:, ti])
        best = max(scores, key=scores.get)
        out.append((best, scores[best]))
    return out


def _render_disc_panels(
    embeddings: np.ndarray, omegas: np.ndarray, traits: np.ndarray,
    out_path: Path, n_discs: int = 6,
):
    """Multi-disc panel figure: top-``n_discs`` disc embeddings, each panel
    coloured by the trait with the highest joint R^2 against that disc.

    Annotates each panel with disc index, dominant trait, R^2, and the
    eigenvalue ratio omega_k / omega_1.
    """
    n_discs = min(n_discs, embeddings.shape[1], len(omegas))
    if n_discs <= 0:
        raise SystemExit("No disc embeddings to plot.")

    # Auto-pick layout
    if n_discs <= 2:
        nrows, ncols = 1, n_discs
    elif n_discs <= 4:
        nrows, ncols = 2, 2
    elif n_discs <= 6:
        nrows, ncols = 2, 3
    else:
        nrows = (n_discs + 2) // 3
        ncols = 3

    best = _best_trait_per_disc(embeddings, traits, n_discs)
    panel_w, panel_h = 3.6, 3.4
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             dpi=120, constrained_layout=True)
    axes = np.atleast_2d(axes).flatten()

    for k in range(n_discs):
        ax = axes[k]
        Y = embeddings[:, k, :]
        trait_name, r2 = best[k]
        ti = _TRAIT_INDEX[trait_name]
        c = traits[:, ti]
        # z-score for a stable colour range across panels
        c_z = (c - c.mean()) / max(c.std(), 1e-12)
        cmap = _TRAIT_CMAP.get(trait_name, "viridis")
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=c_z, cmap=cmap,
                        s=14, edgecolors="k", linewidths=0.25, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(_TRAIT_SYMBOL.get(trait_name, trait_name) +
                     " (z-score)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
        ax.set_xlabel(rf"$Y^{{({k+1})}}_1$", fontsize=9)
        ax.set_ylabel(rf"$Y^{{({k+1})}}_2$", fontsize=9)
        ax.tick_params(labelsize=7)

        ratio = omegas[k] / max(omegas[0], 1e-12)
        ax.set_title(
            f"disc {k + 1}: {trait_name} "
            rf"($R^2 = {r2:.2f}$,  $\omega/\omega_1 = {ratio:.2f}$)",
            fontsize=9,
        )

    # Hide any unused axes
    for j in range(n_discs, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        rf"PTA top-{n_discs} disc embeddings $\mathbf{{Y}}^{{(k)}}_{{\mathrm{{PTA}}}}$ "
        r"on iblotto (colour = best-loading trait, z-scored)",
        fontsize=11,
    )
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

    fig.suptitle(r"PTA top-disc embedding $\mathbf{Y}^{(1)}_{\mathrm{PTA}}$ "
                 r"on iblotto", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main(bundle_path: Path, F_full_path: Path, out_dir: Path, n_discs: int = 6):
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
    _render_disc_panels(embeddings, omegas, traits,
                        out_dir / "iblotto_pta_discs.pdf",
                        n_discs=n_discs)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True,
                   help="behavioural bundle .pkl (for ground-truth traits + N)")
    p.add_argument("--F_full", type=Path, required=True,
                   help="dense F_full.npz from the round-robin run")
    p.add_argument("--out_dir", type=Path, default=Path("figures"),
                   help="where to write iblotto_pta_*.pdf")
    p.add_argument("--n_discs", type=int, default=6,
                   help="number of disc embeddings to characterise in the "
                        "multi-disc panel figure (4-6 recommended)")
    args = p.parse_args()
    main(args.bundle, args.F_full, args.out_dir, args.n_discs)
