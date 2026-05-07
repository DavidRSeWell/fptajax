"""Trait recovery on classical PTA disc games (mirror of trait_recovery.py).

Loads the dense F (cached as F_full.npz), runs ``fptajax.pta.pta``, and
applies the same per-disc / joint / cumulative regression analysis that
``trait_recovery.py`` runs on BFPTA. Optionally compares the two
recovery profiles side-by-side if a BFPTA trait-recovery JSON is given.

Usage:
    python -m examples.iblotto.pta_trait_recovery \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --F_full results/pta_vs_bfpta_main_v1_seed0/F_full.npz \
        --bfpta_json results/trait_recovery_main_v1_seed0/trait_recovery.json \
        --out_dir results/pta_trait_recovery_main_v1_seed0
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.trait_recovery import (
    _TRAIT_NAMES, cumulative_r2, lasso_joint, ols_joint_r2, ols_partial_r2,
    standardise_gt,
)
from fptajax.pta import pta as classical_pta


def main(bundle_path: Path, F_full_path: Path, out_dir: Path,
         bfpta_json: Path | None = None,
         k_keep: int = 6, lasso_alpha: float = 0.05):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== PTA trait-recovery analysis ===")
    print(f"  bundle:  {bundle_path}")
    print(f"  F_full:  {F_full_path}")
    print(f"  out_dir: {out_dir}")

    # Bundle: only need policies + observed_mask after drop
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"  N agents: {N}")

    # F_full
    d = np.load(F_full_path)
    F_full = d["F"]
    if F_full.shape[0] != N:
        raise RuntimeError(
            f"F_full has shape {F_full.shape} but bundle has N={N} after drop. "
            f"Did you generate F_full on the same agent set?"
        )

    # PTA
    print("\n  Running classical PTA on full F ...")
    pta_res = classical_pta(jnp.asarray(F_full))
    Y_PTA = np.asarray(pta_res.embeddings)               # (N, K_PTA, 2)
    omegas_PTA = np.asarray(pta_res.eigenvalues)
    K_PTA = int(pta_res.n_components)
    K = min(k_keep, K_PTA)
    print(f"  K_PTA = {K_PTA}, using top {K}; ω = {omegas_PTA[:K].round(2)}")

    Y = Y_PTA[:, :K]
    traits_z = standardise_gt(ds.policies)
    print(f"  traits_z shape: {traits_z.shape}, "
          f"feature names: {list(_TRAIT_NAMES)}")

    # Recovery analyses
    print("\n[1] Per-disc partial R² (PTA)")
    R2_pd, coefs_pd = ols_partial_r2(Y, traits_z)
    for j, name in enumerate(_TRAIT_NAMES):
        row = " ".join(f"k{k+1}={R2_pd[j, k]:.2f}" for k in range(K))
        print(f"    {name:>20s}: {row}")

    print("\n[2] Joint R² (OLS)")
    R2_joint, _ = ols_joint_r2(Y, traits_z)
    for j, name in enumerate(_TRAIT_NAMES):
        print(f"    {name:>20s}: R² = {R2_joint[j]:.4f}")

    print(f"\n[3] Joint R² (LASSO α={lasso_alpha})")
    R2_lasso, coefs_lasso = lasso_joint(Y, traits_z, alpha=lasso_alpha)
    for j, name in enumerate(_TRAIT_NAMES):
        print(f"    {name:>20s}: R² = {R2_lasso[j]:.4f}")

    print("\n[4] Cumulative R² (top-K disc games by |ω|)")
    cum_R2 = cumulative_r2(Y, traits_z, omegas_PTA[:K])

    # Side-by-side comparison if BFPTA recovery JSON is provided
    bfpta = None
    if bfpta_json is not None and bfpta_json.exists():
        with open(bfpta_json) as f:
            bfpta = json.load(f)
        print(f"\n[5] Side-by-side with BFPTA trait recovery: {bfpta_json}")
        print(f"  {'trait':>20s}  {'PTA OLS':>10s}  {'BFPTA OLS':>10s}  "
              f"{'PTA LASSO':>11s}  {'BFPTA LASSO':>13s}")
        for j, name in enumerate(_TRAIT_NAMES):
            print(f"    {name:>20s}  {R2_joint[j]:>10.4f}  "
                  f"{bfpta['R2_joint_ols'][j]:>10.4f}  "
                  f"{R2_lasso[j]:>11.4f}  "
                  f"{bfpta['R2_joint_lasso'][j]:>13.4f}")

    # Plots
    # --- Heatmap (PTA) ---
    fig, ax = plt.subplots(figsize=(min(13, 1.2 + K * 1.0), 4), dpi=120)
    vmax = max(0.001, R2_pd.max())
    im = ax.imshow(R2_pd, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"k={k+1}\nω={omegas_PTA[k]:.1f}" for k in range(K)],
                       fontsize=8)
    ax.set_yticks(range(len(_TRAIT_NAMES)))
    ax.set_yticklabels(_TRAIT_NAMES, fontsize=9)
    for j in range(R2_pd.shape[0]):
        for k in range(R2_pd.shape[1]):
            ax.text(k, j, f"{R2_pd[j, k]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if R2_pd[j, k] > vmax * 0.5 else "black")
    plt.colorbar(im, ax=ax, label="partial $R^2$")
    ax.set_title("PTA: per-disc-per-trait partial $R^2$")
    fig.tight_layout()
    fig.savefig(out_dir / "pta_r2_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # --- Joint R² + cumulative ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)
    T = len(_TRAIT_NAMES)
    xs = np.arange(T)
    if bfpta is None:
        axes[0].bar(xs - 0.2, R2_joint, width=0.4, label="OLS",
                    color="steelblue", alpha=0.85)
        axes[0].bar(xs + 0.2, R2_lasso, width=0.4, label="LASSO",
                    color="darkorange", alpha=0.85)
    else:
        axes[0].bar(xs - 0.30, R2_joint, width=0.20, label="PTA OLS",
                    color="navy", alpha=0.85)
        axes[0].bar(xs - 0.10, np.asarray(bfpta["R2_joint_ols"]),
                    width=0.20, label="BFPTA OLS", color="firebrick",
                    alpha=0.85)
        axes[0].bar(xs + 0.10, R2_lasso, width=0.20, label="PTA LASSO",
                    color="steelblue", alpha=0.7)
        axes[0].bar(xs + 0.30, np.asarray(bfpta["R2_joint_lasso"]),
                    width=0.20, label="BFPTA LASSO", color="lightcoral",
                    alpha=0.7)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(_TRAIT_NAMES, rotation=20, fontsize=8)
    axes[0].set_ylabel("joint $R^2$")
    axes[0].set_title("Joint regression: trait ~ disc-game coords")
    axes[0].set_ylim(0, max(1.02, max(R2_joint.max(),
        max((bfpta or {"R2_joint_ols": [0]})["R2_joint_ols"])) * 1.1))
    axes[0].grid(True, axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    for j in range(T):
        axes[1].plot(np.arange(1, K + 1), cum_R2[j], "-o",
                     label=_TRAIT_NAMES[j], markersize=4)
    axes[1].set_xlabel("top-K disc games kept (sorted by |ω|)")
    axes[1].set_ylabel("cumulative joint $R^2$")
    axes[1].set_title("PTA cumulative variance explained by top-K discs")
    axes[1].set_ylim(0, max(1.02, cum_R2.max() * 1.1))
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "pta_joint_R2_bar.png", bbox_inches="tight")
    plt.close(fig)

    # JSON
    out_json = out_dir / "pta_trait_recovery.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            trait_names=list(_TRAIT_NAMES),
            n_agents=int(N), n_disc_games=int(K),
            omegas=omegas_PTA[:K].tolist(),
            R2_per_disc=R2_pd.tolist(),
            coefs_per_disc=coefs_pd.tolist(),
            R2_joint_ols=R2_joint.tolist(),
            R2_joint_lasso=R2_lasso.tolist(),
            cumulative_R2=cum_R2.tolist(),
        ), f, indent=2)
    print(f"\nSaved JSON → {out_json}")
    print(f"Saved plots in {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--F_full", type=Path, required=True,
                   help="path to the cached F_full.npz produced by compare_pta_vs_bfpta.py")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--bfpta_json", type=Path, default=None,
                   help="optional: trait_recovery.json from the BFPTA run, "
                        "to render side-by-side R² bars")
    p.add_argument("--k_keep", type=int, default=6)
    p.add_argument("--lasso_alpha", type=float, default=0.05)
    args = p.parse_args()
    main(args.bundle, args.F_full, args.out_dir,
         args.bfpta_json, args.k_keep, args.lasso_alpha)
