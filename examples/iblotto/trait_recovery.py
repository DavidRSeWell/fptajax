"""Trait recovery analysis for a trained Behavioural FPTA model on iblotto.

Loads a saved BFPTA checkpoint (produced by ``benchmark.py --save_bfpta``)
and the underlying behavioural-data bundle. Encodes every agent through
the trained encoder + basis to recover its inferred trait vector and its
disc-game embedding ``Y^{(k)}(x_i) ∈ ℝ²``. Then does three things:

  1. **Per-disc partial R²**: for each disc game k and each ground-truth
     trait t, OLS regression t ~ Y^{(k)}(x). Gives a (T_traits × K_discs)
     heatmap.
  2. **Joint R²**: for each trait, OLS / LASSO regression onto the full
     (2K)-dim disc-game coordinate vector. Bar chart per trait. Cumulative
     curves: how much of trait t's variance is captured if we keep only
     the top-K' disc games (sorted by ω_k)?
  3. **Visualisation**: 2D scatter of disc-game embeddings, coloured by
     each trait. Each disc plane is rotated so the strongest-correlating
     trait's gradient points along +x — makes the colour gradient visible.

Outputs (alongside ``--out_dir``):

    trait_recovery.json          — all R² and coefficient tables
    r2_heatmap.png               — per-disc partial R² heatmap
    joint_R2_bar.png             — joint + cumulative R² per trait
    disc_panels.png              — 2D scatter grid (T_traits × K_discs')

Usage:
    python -m examples.iblotto.trait_recovery \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --bfpta_dir results/bfpta_main_v1_seed0 \
        --out_dir results/trait_recovery_main_v1_seed0
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents


_TRAIT_NAMES = ("learning_rate", "win_reinvestment", "loss_disinvestment",
                "opponent_allocation", "innovation_noise")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_bfpta(bfpta_dir: Path):
    """Reconstruct the trained encoder + basis from saved artefacts.

    Returns ``(encoder, basis, meta)``. ``meta`` is a dict containing
    ``coefficient_matrix``, ``eigenvalues``, ``schur_vectors``, ``n_components``.
    """
    from fptajax.hierarchical import HierarchicalSetEncoder
    from fptajax.neural import NeuralBasis

    with open(bfpta_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    key = jax.random.PRNGKey(0)
    k_enc, k_basis = jax.random.split(key)
    encoder_template = HierarchicalSetEncoder(
        token_dim=meta["sa_dim"], L_max=meta["L_max"],
        trait_dim=meta["trait_dim"],
        d_model=meta["d_model"], n_heads=meta["n_heads"],
        n_layers=meta["n_layers"], mlp_ratio=meta["mlp_ratio"],
        rho_hidden=meta["rho_hidden"], key=k_enc,
    )
    basis_template = NeuralBasis(
        trait_dim=meta["trait_dim"], d=meta["d"],
        hidden_dims=meta["basis_hidden"], key=k_basis,
    )
    encoder = eqx.tree_deserialise_leaves(str(bfpta_dir / "encoder.eqx"),
                                          encoder_template)
    basis = eqx.tree_deserialise_leaves(str(bfpta_dir / "basis.eqx"),
                                        basis_template)
    return encoder, basis, meta


def load_bundle(bundle_path: Path):
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    # Defensive sanitation
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    if dropped:
        print(f"  dropped agents: {dropped}")
    return ds


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def compute_embeddings(encoder, basis, meta, ds):
    """Run encoder + basis on every agent and return ``(traits, Y, omegas)``.

    ``traits``  : (N, trait_dim) inferred trait vectors
    ``Y``       : (N, K, 2) disc-game embeddings, scaled by sqrt(ω_k)
    ``omegas``  : (K,) ω_k spectrum
    """
    sa = jnp.asarray(ds.agent_data)
    tmask = jnp.asarray(ds.agent_token_mask)
    gmask = jnp.asarray(ds.agent_game_mask)
    traits = np.asarray(encoder.encode_batch(sa, tmask, gmask))   # (N, T)
    B = np.asarray(basis.evaluate_batch(jnp.asarray(traits)))     # (N, d)
    Q = np.asarray(meta["schur_vectors"])
    omegas = np.asarray(meta["eigenvalues"])
    K = int(meta["n_components"])
    N = ds.policies.shape[0]
    Y = np.zeros((N, K, 2), dtype=np.float64)
    for k in range(K):
        s = float(np.sqrt(max(omegas[k], 0.0)))
        Y[:, k, 0] = s * (B @ Q[:, 2 * k])
        Y[:, k, 1] = s * (B @ Q[:, 2 * k + 1])
    return traits, Y, omegas[:K]


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------


def standardise_gt(policies: np.ndarray) -> np.ndarray:
    """Return ground-truth traits (N, 5) z-score normalised.

    We drop the concentration column (always 1000) since it has zero variance.
    """
    raw = policies[:, :5].astype(np.float64)
    mu = raw.mean(axis=0, keepdims=True)
    sd = raw.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (raw - mu) / sd


def ols_partial_r2(Y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-disc-per-trait OLS.

    Args:
        Y: (N, K, 2)
        t: (N, T) traits.
    Returns:
        R2: (T, K) per-disc partial R² per trait.
        coefs: (T, K, 2) per-disc coefficients (excluding intercept).
    """
    N, K, _ = Y.shape
    T = t.shape[1]
    R2 = np.zeros((T, K))
    coefs = np.zeros((T, K, 2))
    for k in range(K):
        X = Y[:, k, :]                                   # (N, 2)
        for j in range(T):
            y = t[:, j]
            # OLS with intercept
            X1 = np.concatenate([np.ones((N, 1)), X], axis=1)
            beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            yhat = X1 @ beta
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            R2[j, k] = 1.0 - ss_res / max(ss_tot, 1e-12)
            coefs[j, k] = beta[1:]
    return R2, coefs


def ols_joint_r2(Y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint OLS regression of each trait onto all 2K disc-game coordinates.

    Returns:
        R2_full: (T,) joint R² per trait.
        coefs_full: (T, 2K) coefficient vector per trait (post-intercept).
    """
    N, K, _ = Y.shape
    T = t.shape[1]
    X = Y.reshape(N, 2 * K)
    X1 = np.concatenate([np.ones((N, 1)), X], axis=1)
    R2 = np.zeros(T)
    coefs = np.zeros((T, 2 * K))
    for j in range(T):
        y = t[:, j]
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        yhat = X1 @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        R2[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
        coefs[j] = beta[1:]
    return R2, coefs


def cumulative_r2(Y: np.ndarray, t: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    """OLS R² per trait if only the top-k disc games (by |ω|) are kept.

    Returns array of shape ``(T, K)``: ``cum[j, k]`` is the joint R² for
    trait ``j`` using disc games sorted by descending |ω|, top (k+1) of them.
    """
    N, K, _ = Y.shape
    T = t.shape[1]
    order = np.argsort(-np.abs(omegas))
    cum = np.zeros((T, K))
    for top_k in range(1, K + 1):
        idx = order[:top_k]
        X = Y[:, idx, :].reshape(N, 2 * top_k)
        X1 = np.concatenate([np.ones((N, 1)), X], axis=1)
        for j in range(T):
            y = t[:, j]
            beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            ss_res = float(np.sum((y - X1 @ beta) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            cum[j, top_k - 1] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return cum


def lasso_joint(Y: np.ndarray, t: np.ndarray, alpha: float = 0.05):
    """Optional sparse fit. Returns (R²_per_trait, coefs).

    Uses scikit-learn if available; falls back to OLS otherwise.
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        print("  sklearn not installed; skipping LASSO; falling back to OLS")
        return ols_joint_r2(Y, t)
    N, K, _ = Y.shape
    T = t.shape[1]
    X = Y.reshape(N, 2 * K)
    R2 = np.zeros(T); coefs = np.zeros((T, 2 * K))
    for j in range(T):
        y = t[:, j]
        # Standardise X-columns to give LASSO a fair chance per feature
        x_sd = X.std(axis=0); x_sd = np.where(x_sd > 1e-12, x_sd, 1.0)
        X_norm = X / x_sd
        m = Lasso(alpha=alpha, max_iter=10_000).fit(X_norm, y)
        yhat = m.predict(X_norm)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        R2[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
        coefs[j] = m.coef_ / x_sd
    return R2, coefs


# ---------------------------------------------------------------------------
# Rotation pinning (per disc-plane)
# ---------------------------------------------------------------------------


def rotate_disc_to_align(Y: np.ndarray, R2_per_disc: np.ndarray,
                        coefs_per_disc: np.ndarray) -> np.ndarray:
    """For each disc k, rotate so the largest-R² trait's gradient direction
    aligns with the +x axis. Useful for the visualisation.

    Returns rotated ``Y_rot`` with the same shape.
    """
    N, K, _ = Y.shape
    Y_rot = np.copy(Y)
    for k in range(K):
        j_best = int(np.argmax(R2_per_disc[:, k]))
        bx, by = coefs_per_disc[j_best, k]
        angle = float(np.arctan2(by, bx))   # gradient direction
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        Y_rot[:, k] = Y[:, k] @ rot.T
    return Y_rot


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_r2_heatmap(R2: np.ndarray, omegas: np.ndarray, out_path: Path):
    T, K = R2.shape
    fig, ax = plt.subplots(figsize=(min(13, 1.2 + K * 1.0), 4), dpi=120)
    vmax = max(0.001, R2.max())
    im = ax.imshow(R2, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"k={k+1}\nω={omegas[k]:.1f}" for k in range(K)],
                       fontsize=8)
    ax.set_yticks(range(T))
    ax.set_yticklabels(_TRAIT_NAMES, fontsize=9)
    for j in range(T):
        for k in range(K):
            ax.text(k, j, f"{R2[j, k]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if R2[j, k] > vmax * 0.5 else "black")
    plt.colorbar(im, ax=ax, label="partial $R^2$")
    ax.set_title("Per-disc-per-trait OLS partial $R^2$\n"
                 "(how much of trait variance does each disc's 2D coords explain)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_joint_and_cumulative(joint_R2_ols: np.ndarray, joint_R2_lasso: np.ndarray,
                              cum_R2: np.ndarray, omegas: np.ndarray,
                              out_path: Path):
    T = joint_R2_ols.shape[0]
    K = cum_R2.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)

    # Bar chart: joint R² per trait, OLS vs LASSO
    xs = np.arange(T)
    axes[0].bar(xs - 0.2, joint_R2_ols, width=0.4, label="OLS",
                color="steelblue", alpha=0.85)
    axes[0].bar(xs + 0.2, joint_R2_lasso, width=0.4, label="LASSO",
                color="darkorange", alpha=0.85)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(_TRAIT_NAMES, rotation=20, fontsize=8)
    axes[0].set_ylabel("joint $R^2$")
    axes[0].set_title("Joint regression: trait ~ all 2K disc coords")
    axes[0].set_ylim(0, max(1.02, joint_R2_ols.max() * 1.1))
    axes[0].grid(True, axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    # Cumulative curves: how many discs do we need?
    for j in range(T):
        axes[1].plot(np.arange(1, K + 1), cum_R2[j], "-o",
                     label=_TRAIT_NAMES[j], markersize=4)
    axes[1].set_xlabel("top-K disc games kept (sorted by |ω|)")
    axes[1].set_ylabel("cumulative joint $R^2$")
    axes[1].set_title("Cumulative variance explained by top-K disc games")
    axes[1].set_ylim(0, max(1.02, cum_R2.max() * 1.1))
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_disc_panels(Y_rot: np.ndarray, traits_gt_z: np.ndarray, R2: np.ndarray,
                     omegas: np.ndarray, K_show: int, out_path: Path):
    T = traits_gt_z.shape[1]
    fig, axes = plt.subplots(T, K_show,
                             figsize=(2.4 * K_show + 1, 2.0 * T), dpi=120,
                             sharex="col", sharey="row", squeeze=False)
    extent = float(np.max(np.abs(Y_rot[:, :K_show])) * 1.05)
    for j in range(T):
        for k in range(K_show):
            ax = axes[j, k]
            sc = ax.scatter(Y_rot[:, k, 0], Y_rot[:, k, 1],
                            c=traits_gt_z[:, j], cmap="RdBu_r",
                            vmin=-2.5, vmax=2.5, s=12, edgecolor="none")
            ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
            ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.set_title(f"disc k={k+1}, ω={omegas[k]:.1f}", fontsize=8)
            if k == 0:
                ax.set_ylabel(_TRAIT_NAMES[j], fontsize=8)
            ax.text(0.02, 0.96, f"$R^2$={R2[j, k]:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7,
                              ec="none"))
    fig.suptitle("Disc-game embeddings, coloured by ground-truth traits "
                 "(z-scored). Each disc is rotated so its strongest-R² trait "
                 "gradient points along +x.", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(bundle_path: Path, bfpta_dir: Path, out_dir: Path,
         lasso_alpha: float = 0.05):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Trait recovery analysis ===")
    print(f"  bundle:    {bundle_path}")
    print(f"  bfpta_dir: {bfpta_dir}")
    print(f"  out_dir:   {out_dir}")

    ds = load_bundle(bundle_path)
    encoder, basis, meta = load_bfpta(bfpta_dir)
    print(f"  N agents: {ds.policies.shape[0]}, "
          f"trait_dim={meta['trait_dim']}, d={meta['d']}, "
          f"K disc games={meta['n_components']}")

    traits_inf, Y, omegas = compute_embeddings(encoder, basis, meta, ds)
    K = Y.shape[1]
    traits_gt_z = standardise_gt(ds.policies)
    print(f"  Y shape: {Y.shape}; ω = {omegas.tolist()}")

    print("\n[1] Per-disc partial R²")
    R2_pd, coefs_pd = ols_partial_r2(Y, traits_gt_z)
    for j, name in enumerate(_TRAIT_NAMES):
        row = " ".join(f"k{k+1}={R2_pd[j,k]:.2f}" for k in range(K))
        print(f"    {name:>20s}: {row}")

    print("\n[2] Joint R² (OLS)")
    R2_joint, coefs_joint = ols_joint_r2(Y, traits_gt_z)
    for j, name in enumerate(_TRAIT_NAMES):
        print(f"    {name:>20s}: R² = {R2_joint[j]:.4f}")

    print(f"\n[3] Joint R² (LASSO α={lasso_alpha})")
    R2_lasso, coefs_lasso = lasso_joint(Y, traits_gt_z, alpha=lasso_alpha)
    for j, name in enumerate(_TRAIT_NAMES):
        n_active = int(np.sum(np.abs(coefs_lasso[j]) > 1e-6))
        print(f"    {name:>20s}: R² = {R2_lasso[j]:.4f}   "
              f"({n_active}/{2*K} non-zero coefs)")

    print("\n[4] Cumulative R² (top-K disc games by |ω|)")
    cum_R2 = cumulative_r2(Y, traits_gt_z, omegas)

    # Visualisation: rotate per-disc, then plot
    Y_rot = rotate_disc_to_align(Y, R2_pd, coefs_pd)
    K_show = min(K, max(2, int(np.sum((omegas ** 2) /
                                      max(np.sum(omegas ** 2), 1e-12) > 0.01))))
    print(f"\n[5] Plots (K_show = {K_show} discs over the ω²-fraction>1% threshold)")
    plot_r2_heatmap(R2_pd, omegas, out_dir / "r2_heatmap.png")
    plot_joint_and_cumulative(R2_joint, R2_lasso, cum_R2, omegas,
                              out_dir / "joint_R2_bar.png")
    plot_disc_panels(Y_rot, traits_gt_z, R2_pd, omegas, K_show,
                     out_dir / "disc_panels.png")

    out_json = out_dir / "trait_recovery.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            trait_names=list(_TRAIT_NAMES),
            n_agents=int(ds.policies.shape[0]), n_disc_games=K,
            omegas=omegas.tolist(),
            R2_per_disc=R2_pd.tolist(),
            coefs_per_disc=coefs_pd.tolist(),
            R2_joint_ols=R2_joint.tolist(),
            coefs_joint_ols=coefs_joint.tolist(),
            R2_joint_lasso=R2_lasso.tolist(),
            coefs_joint_lasso=coefs_lasso.tolist(),
            cumulative_R2=cum_R2.tolist(),
        ), f, indent=2)
    print(f"\nSaved JSON → {out_json}")
    print(f"Saved PNGs in {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle",    type=Path, required=True)
    p.add_argument("--bfpta_dir", type=Path, required=True)
    p.add_argument("--out_dir",   type=Path, required=True)
    p.add_argument("--lasso_alpha", type=float, default=0.05)
    args = p.parse_args()
    main(args.bundle, args.bfpta_dir, args.out_dir, lasso_alpha=args.lasso_alpha)
