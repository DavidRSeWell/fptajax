"""Render classical-FPTA figures for the iblotto section.

Loads a behavioural bundle (for ground-truth trait values + dense F via
``F_full``), runs classical FPTA with a total-degree polynomial basis on
the standardised traits, and emits two figures analogous to the PTA
renderer:

  figures/iblotto_classical_fpta_spectrum.pdf  eigenvalue bar chart (log-y),
                                               top N omega_k / omega_1
  figures/iblotto_classical_fpta_discs.pdf     multi-disc embedding panel,
                                               each panel coloured by the
                                               best-loading trait (raw values)

Default basis is ``monomial_d3`` (m=56) -- the strongest configuration in the
predictive-performance table. The basis is orthonormalised under the
empirical agent measure, ``C = (1/N) B^T F B`` is computed on the dense F,
and the Schur decomposition gives ``omega_k`` and Q. Per-agent embeddings
follow ``Y^(k)(x_i) = sqrt(omega_k) * b(x_i)^T Q[:, paired columns]``.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -u -m examples.iblotto.render_classical_fpta_section_figures \\
        --bundle /Users/davidsewell/Downloads/behavioral_main_v1_N200_k20_nr50.pkl \\
        --F_full /Users/davidsewell/Downloads/F_full.npz \\
        --out_dir figures --max_deg 3 --n_discs 6
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.protocol import (
    fit_skew_C_train, orthonormalise, truncate_C,
)
from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.benchmark import (
    standardise_traits, total_degree_monomials,
)


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

_TRAIT_RANGE = {
    "learning_rate":       (0.10, 0.70),
    "win_reinvestment":    (-2.0, 2.0),
    "loss_disinvestment":  (-2.0, 2.0),
    "opponent_allocation": (-2.0, 2.0),
    "innovation_noise":    (0.01, 0.30),
}

_PANEL_CMAP = "RdBu_r"


def _joint_r2(Y_disc: np.ndarray, trait_vals: np.ndarray) -> float:
    X = np.concatenate([Y_disc, np.ones((Y_disc.shape[0], 1))], axis=1)
    y = (trait_vals - trait_vals.mean()) / max(trait_vals.std(), 1e-12)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _best_trait_per_disc(
    embeddings: np.ndarray, traits: np.ndarray, n_discs: int,
) -> list[tuple[str, float]]:
    out = []
    for k in range(n_discs):
        Y = embeddings[:, k, :]
        scores = {name: _joint_r2(Y, traits[:, ti])
                  for name, ti in _TRAIT_INDEX.items()}
        best = max(scores, key=scores.get)
        out.append((best, scores[best]))
    return out


def _render_spectrum(omegas: np.ndarray, out_path: Path, k_show: int = 10,
                     title: str = "classical FPTA disc-game spectrum"):
    omegas = np.asarray(omegas)
    k_show = min(k_show, len(omegas))
    ratios = omegas[:k_show] / max(omegas[0], 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    xs = np.arange(1, k_show + 1)
    ax.bar(xs, ratios, color="#7e6b8f", alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"disc index $k$")
    ax.set_ylabel(r"$\omega_k / \omega_1$  (log scale)")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    e2 = omegas ** 2
    cum = np.cumsum(e2) / max(e2.sum(), 1e-12)
    for i in range(min(3, k_show)):
        ax.text(xs[i], ratios[i] * 1.15, f"{100 * cum[i]:.1f}%",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _render_disc_panels(
    embeddings: np.ndarray, omegas: np.ndarray, traits: np.ndarray,
    out_path: Path, n_discs: int = 6, suptitle_extra: str = "",
):
    n_discs = min(n_discs, embeddings.shape[1], len(omegas))
    if n_discs <= 0:
        raise SystemExit("No disc embeddings to plot.")

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
        vmin, vmax = _TRAIT_RANGE.get(trait_name, (c.min(), c.max()))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, cmap=_PANEL_CMAP,
                        vmin=vmin, vmax=vmax,
                        s=14, edgecolors="k", linewidths=0.25, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(_TRAIT_SYMBOL.get(trait_name, trait_name), fontsize=8)
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

    for j in range(n_discs, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        rf"Classical FPTA top-{n_discs} disc embeddings "
        rf"$\mathbf{{Y}}^{{(k)}}_{{\mathrm{{FPTA}}}}$ on iblotto"
        + (f"  ({suptitle_extra})" if suptitle_extra else ""),
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main(bundle_path: Path, F_full_path: Path, out_dir: Path,
         max_deg: int, n_discs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Rendering classical-FPTA section figures ===")
    print(f"  bundle:   {bundle_path}")
    print(f"  F_full:   {F_full_path}")
    print(f"  basis:    monomial_d{max_deg}")
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
            f"dropping dead agents."
        )
    print(f"  N = {N}, F_full shape = {F_full.shape}")

    # Standardised traits + polynomial basis
    traits_std = standardise_traits(ds.policies)
    B_raw, labels = total_degree_monomials(traits_std, max_deg=max_deg)
    m = B_raw.shape[1]
    B = orthonormalise(B_raw)
    print(f"  basis: monomial_d{max_deg}, m = {m} basis functions")

    # Closed-form C on the dense F (all pairs). The benchmark.py predictive
    # numbers use sparse training pairs; here we use dense F for the same
    # "best-case geometry" analysis we did for PTA.
    iu_all = np.argwhere(np.triu(np.ones((N, N), dtype=bool), 1))
    all_pairs = np.concatenate([iu_all, iu_all[:, [1, 0]]], axis=0)
    C = fit_skew_C_train(B, F_full, all_pairs, ridge=0.0)

    # Full Schur (no truncation) — we want to see the actual spectrum
    _, omegas, Q = truncate_C(C, k_keep=len(C))
    K = int(np.sum(omegas > 0))
    print(f"  Schur: K = {K} active discs, omega_1 = {omegas[0]:.3f}, "
          f"top-3 ratios = {(omegas[:3] / omegas[0]).round(3)}")

    # Per-agent embeddings: Y^(k)(x_i) = sqrt(omega_k) * b(x_i)^T Q[:, paired]
    # Z = B @ Q has shape (N, m); for disc k we take paired columns 2k, 2k+1.
    Z = B @ Q
    embeddings = np.zeros((N, K, 2))
    for k in range(K):
        embeddings[:, k, 0] = np.sqrt(omegas[k]) * Z[:, 2 * k]
        embeddings[:, k, 1] = np.sqrt(omegas[k]) * Z[:, 2 * k + 1]

    traits = np.asarray(ds.policies[:, :5])

    _render_spectrum(
        omegas, out_dir / "iblotto_classical_fpta_spectrum.pdf",
        title=rf"Classical FPTA disc-game spectrum  (monomial $d{{=}}{max_deg}$, $m{{=}}{m}$)",
    )
    _render_disc_panels(
        embeddings, omegas, traits,
        out_dir / "iblotto_classical_fpta_discs.pdf",
        n_discs=n_discs,
        suptitle_extra=rf"monomial $d{{=}}{max_deg}$, $m{{=}}{m}$",
    )
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True,
                   help="behavioural bundle .pkl (for traits + N)")
    p.add_argument("--F_full", type=Path, required=True,
                   help="dense F_full.npz from the round-robin run")
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    p.add_argument("--max_deg", type=int, default=3,
                   help="max total degree of monomial basis (2, 3, or 4)")
    p.add_argument("--n_discs", type=int, default=6)
    args = p.parse_args()
    main(args.bundle, args.F_full, args.out_dir,
         args.max_deg, args.n_discs)
