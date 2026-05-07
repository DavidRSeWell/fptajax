"""Compare classical PTA on full F vs trained BFPTA disc-game embeddings.

For the same N agents we compute two independent disc-game decompositions:

  1. Classical PTA: full round-robin tournament → dense F → real-skew Schur
     of F → Y_PTA[i, k] in R^2.
  2. Behavioural FPTA: load the trained encoder + basis + Schur factors
     from a saved checkpoint, encode every agent, derive Y_BFPTA[i, k].

Per-disc Procrustes alignment with rotation + scale (rotation-only, since
reflection would change the sign of the disc-game inner product).

Outputs in ``--out_dir``:

    F_full.npz           — saved dense F + F_std (cached for re-runs)
    overlay.png          — top-K disc 2D scatters, both methods, coloured
                           by a chosen ground-truth trait
    residuals.png        — per-disc Procrustes residual and per-axis Pearson
    eigenvalues.png      — normalised ω_k spectra for both methods
    comparison.json      — all numbers
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.game import (
    GameOptions, CSF_AUCTION, RR_NONE,
    REALLOC_STAY_IN_ZONE, INFO_ALL_INVESTMENTS,
)
from examples.iblotto.tournament import run_tournament
from examples.iblotto.trait_recovery import (
    _TRAIT_NAMES, compute_embeddings, load_bfpta,
)
from fptajax.pta import pta as classical_pta


# ---------------------------------------------------------------------------
# Procrustes (rotation + scale, rotation-only / no reflection)
# ---------------------------------------------------------------------------


def procrustes_with_scale(Y_a: np.ndarray, Y_b: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Best similarity transform mapping Y_a -> Y_b.

    Solves ``min_{s, R} ||s · Y_a R - Y_b||²_F`` with R restricted to
    SO(2) (rotation only — no reflection, because a reflection within a
    disc plane would flip the sign of the disc-game inner product).

    Returns ``(Y_aligned, R, s, residual)`` where ``residual`` is the
    relative Frobenius residual after alignment.
    """
    H = Y_a.T @ Y_b                                      # (2, 2)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:                             # restrict to SO(2)
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = U @ Vt
        S = np.array([S[0], -S[1]])
    s = float(S.sum() / max((Y_a ** 2).sum(), 1e-12))
    Y_aligned = s * (Y_a @ R)
    residual = float(np.linalg.norm(Y_aligned - Y_b)
                     / max(np.linalg.norm(Y_b), 1e-12))
    return Y_aligned, R, s, residual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(bundle_path: Path, bfpta_dir: Path, out_dir: Path,
         n_real: int = 50, n_rounds: int = 100,
         color_trait: str = "opponent_allocation",
         force_regen: bool = False, seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  PTA on full F  vs  BFPTA disc games  —  consistency check")
    print("=" * 72)

    # ----- Load bundle (drop dead agents, sanitise NaN tokens) -----
    print(f"\n[1] Loading bundle: {bundle_path}")
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"    N={N}  (dead agents dropped: {dropped})")

    # ----- Step 1: Full round-robin F (cached) -----
    F_path = out_dir / "F_full.npz"
    if F_path.exists() and not force_regen:
        print(f"\n[2] Loading cached full F from {F_path}")
        d = np.load(F_path)
        F_full, F_std = d["F"], d["F_std"]
    else:
        n_pairs = N * (N - 1) // 2
        print(f"\n[2] Running full round-robin (N={N}, pairs={n_pairs}, "
              f"n_real={n_real}, n_rounds={n_rounds}) ...")
        opts = GameOptions(
            n_zones=ds.n_zones, zone_values=jnp.ones(ds.n_zones),
            csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
            reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
            info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
        )
        t0 = time.time()
        F_full, F_std = run_tournament(
            ds.policies, opts, n_real=n_real, seed=seed,
            p1_budget=1000.0, p2_budget=1000.0,
            p1_inv_frac=0.1, p2_inv_frac=0.1, verbose=True,
        )
        print(f"    Tournament done in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / 60:.1f} min)")
        np.savez(F_path, F=F_full, F_std=F_std)
        print(f"    Saved {F_path}")

    print(f"    ‖F_full‖_F = {np.linalg.norm(F_full):.3f}, "
          f"mean F_std (off-diag) = "
          f"{F_std[np.triu_indices(N, 1)].mean():.4f}")

    # ----- Step 2: Classical PTA on F -----
    print("\n[3] Classical PTA on full F ...")
    pta_res = classical_pta(jnp.asarray(F_full))
    Y_PTA = np.asarray(pta_res.embeddings)
    omegas_PTA = np.asarray(pta_res.eigenvalues)
    K_PTA = int(pta_res.n_components)
    print(f"    K_PTA = {K_PTA}, top-6 ω = {omegas_PTA[:6].round(2)}")
    print(f"    importance fractions: "
          f"{(omegas_PTA[:6] ** 2 / max((omegas_PTA ** 2).sum(), 1e-12)).round(4)}")

    # ----- Step 3: BFPTA disc-game embeddings -----
    print("\n[4] Loading BFPTA checkpoint and computing disc embeddings ...")
    encoder, basis, meta = load_bfpta(bfpta_dir)
    _, Y_BFPTA, omegas_BFPTA = compute_embeddings(encoder, basis, meta, ds)
    K_BFPTA = Y_BFPTA.shape[1]
    print(f"    K_BFPTA = {K_BFPTA}, top-6 ω = {omegas_BFPTA[:6].round(2)}")
    print(f"    importance fractions: "
          f"{(omegas_BFPTA[:6] ** 2 / max((omegas_BFPTA ** 2).sum(), 1e-12)).round(4)}")

    # ----- Step 4: Per-disc Procrustes alignment -----
    K = min(K_PTA, K_BFPTA)
    print(f"\n[5] Per-disc Procrustes alignment (top {K} discs):")
    print(f"    {'k':>3s}  {'resid':>7s}  {'scale':>8s}  {'rot°':>7s}  "
          f"{'corr_x':>7s}  {'corr_y':>7s}  {'ω_PTA':>9s}  {'ω_BFPTA':>9s}")
    rows = []
    aligned = np.zeros((N, K, 2))
    for k in range(K):
        Y_aligned, R, s, residual = procrustes_with_scale(
            Y_BFPTA[:, k], Y_PTA[:, k],
        )
        aligned[:, k] = Y_aligned
        corr_x = float(np.corrcoef(Y_aligned[:, 0], Y_PTA[:, k, 0])[0, 1])
        corr_y = float(np.corrcoef(Y_aligned[:, 1], Y_PTA[:, k, 1])[0, 1])
        rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        rows.append(dict(
            k=k + 1, residual=residual, scale=s, rotation_deg=rot_deg,
            corr_x=corr_x, corr_y=corr_y,
            omega_PTA=float(omegas_PTA[k]),
            omega_BFPTA=float(omegas_BFPTA[k]),
            importance_PTA=float(omegas_PTA[k] ** 2
                                 / max((omegas_PTA ** 2).sum(), 1e-12)),
            importance_BFPTA=float(omegas_BFPTA[k] ** 2
                                   / max((omegas_BFPTA ** 2).sum(), 1e-12)),
        ))
        print(f"    {k+1:3d}  {residual:7.4f}  {s:8.4f}  {rot_deg:+7.1f}  "
              f"{corr_x:+7.3f}  {corr_y:+7.3f}  "
              f"{omegas_PTA[k]:9.2f}  {omegas_BFPTA[k]:9.2f}")

    # ----- Step 5: Plots -----
    color_idx = list(_TRAIT_NAMES).index(color_trait)
    color_vals = ds.policies[:, color_idx].astype(np.float64)
    color_z = (color_vals - color_vals.mean()) / max(color_vals.std(), 1e-12)

    K_show = min(K, 4)
    fig, axes = plt.subplots(2, K_show, figsize=(3.0 * K_show + 1, 6.0),
                             dpi=120, squeeze=False)
    for k in range(K_show):
        # Top: PTA
        ax_t = axes[0, k]
        sc = ax_t.scatter(Y_PTA[:, k, 0], Y_PTA[:, k, 1],
                          c=color_z, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                          s=14, edgecolor="none")
        ax_t.set_title(f"PTA disc k={k+1}\n"
                       f"ω={omegas_PTA[k]:.1f}, "
                       f"imp={rows[k]['importance_PTA']:.1%}", fontsize=9)
        ax_t.axhline(0, color="gray", lw=0.4, ls=":")
        ax_t.axvline(0, color="gray", lw=0.4, ls=":")
        ax_t.grid(True, alpha=0.2)
        if k == 0:
            ax_t.set_ylabel("PTA   $Y_2$", fontsize=10)

        # Bottom: BFPTA aligned
        ax_b = axes[1, k]
        ax_b.scatter(aligned[:, k, 0], aligned[:, k, 1],
                     c=color_z, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                     s=14, edgecolor="none")
        ax_b.set_title(
            f"BFPTA aligned\n"
            f"resid = {rows[k]['residual']:.3f},  "
            f"corr = ({rows[k]['corr_x']:+.2f}, {rows[k]['corr_y']:+.2f})",
            fontsize=9,
        )
        ax_b.axhline(0, color="gray", lw=0.4, ls=":")
        ax_b.axvline(0, color="gray", lw=0.4, ls=":")
        ax_b.grid(True, alpha=0.2)
        if k == 0:
            ax_b.set_ylabel("BFPTA aligned   $Y_2$", fontsize=10)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6,
                        fraction=0.025, pad=0.02)
    cbar.set_label(f"{color_trait}  (z-score)", fontsize=9)
    fig.suptitle(
        f"PTA on full F  vs  BFPTA disc games — coloured by {color_trait}\n"
        f"(rotation + scale Procrustes per disc)",
        fontsize=11, y=1.02,
    )
    fig.savefig(out_dir / "overlay.png", bbox_inches="tight")
    plt.close(fig)

    # Residuals + correlations bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ks = np.arange(1, K + 1)
    width = 0.27
    ax.bar(ks - width, [r["residual"] for r in rows], width=width,
           label="Procrustes residual (lower = better)", color="crimson")
    ax.bar(ks,         [r["corr_x"]   for r in rows], width=width,
           label=r"corr($Y_1$)", color="steelblue")
    ax.bar(ks + width, [r["corr_y"]   for r in rows], width=width,
           label=r"corr($Y_2$)", color="seagreen")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":", alpha=0.7)
    ax.axhline(0.0, color="black", lw=0.4)
    ax.set_xticks(ks)
    ax.set_xlabel("disc game k")
    ax.set_ylabel("residual / Pearson correlation")
    ax.set_title(
        f"Per-disc Procrustes residual + per-axis correlation\n"
        f"(perfect agreement = residual 0, both correlations = 1)"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig)

    # Eigenvalue comparison
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    K_eig = min(K, len(omegas_PTA), len(omegas_BFPTA))
    ks = np.arange(1, K_eig + 1)
    ax.plot(ks, omegas_PTA[:K_eig] / max(omegas_PTA[0], 1e-12), "-o",
            label="PTA  (normalised by ω₁)", color="darkblue")
    ax.plot(ks, omegas_BFPTA[:K_eig] / max(omegas_BFPTA[0], 1e-12), "-s",
            label="BFPTA (normalised by ω₁)", color="darkred")
    ax.set_xlabel("disc game k"); ax.set_ylabel(r"$\omega_k / \omega_1$ (log)")
    ax.set_yscale("log"); ax.set_xticks(ks)
    ax.set_title("Eigenvalue spectra: PTA vs BFPTA  "
                 "(each normalised by its own dominant ω₁)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "eigenvalues.png", bbox_inches="tight")
    plt.close(fig)

    # JSON
    out_json = out_dir / "comparison.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            N=int(N), K_compared=int(K),
            n_real=int(n_real), n_rounds=int(n_rounds),
            color_trait=color_trait,
            per_disc=rows,
            omegas_PTA=omegas_PTA[:30].tolist(),
            omegas_BFPTA=omegas_BFPTA.tolist(),
        ), f, indent=2)
    print(f"\n  Saved JSON       → {out_json}")
    print(f"  Saved overlay    → {out_dir / 'overlay.png'}")
    print(f"  Saved residuals  → {out_dir / 'residuals.png'}")
    print(f"  Saved eigvals    → {out_dir / 'eigenvalues.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--bfpta_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_real", type=int, default=50)
    p.add_argument("--n_rounds", type=int, default=100)
    p.add_argument("--color_trait", default="opponent_allocation",
                   choices=["learning_rate", "win_reinvestment",
                            "loss_disinvestment", "opponent_allocation",
                            "innovation_noise"])
    p.add_argument("--force_regen", action="store_true",
                   help="re-run the full tournament even if F_full.npz exists")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.bundle, args.bfpta_dir, args.out_dir,
         args.n_real, args.n_rounds, args.color_trait,
         args.force_regen, args.seed)
