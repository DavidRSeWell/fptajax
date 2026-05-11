"""Render BFPTA figures for the iblotto section.

Handles both BFPTA variants in one script (chosen by ``meta["basis_kind"]``):

  - **neural basis** (basis.eqx present, ``basis_kind`` absent or anything
    other than ``"rbf_kmeans"``): loads the trained ``NeuralBasis`` and
    evaluates it on encoded traits.
  - **k-means RBF basis** (``basis_kind == "rbf_kmeans"``): uses the
    stored ``rbf_centres`` + ``rbf_sigma`` from ``meta.pkl``; no separate
    ``basis.eqx`` is loaded.

The disc embeddings are reconstructed by:
  1. Load encoder, run bundle agents through it to get traits in R^trait_dim.
  2. Evaluate the basis at each agent's trait to get B in R^(N, d).
  3. Pull the stored Schur decomposition (eigenvalues, schur_vectors)
     from ``meta.pkl`` -- no re-decomposition needed.
  4. Compute Y^(k)(x_i) = sqrt(omega_k) * b(x_i)^T Q[:, paired columns]
     via Z = B @ Q and pair off into discs.

Outputs:
  figures/iblotto_bfpta_spectrum_<tag>.pdf  log-y omega_k / omega_1
  figures/iblotto_bfpta_discs_<tag>.pdf     2x3 multi-disc panel

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -u -m examples.iblotto.render_bfpta_section_figures \\
        --bundle /Users/davidsewell/Downloads/behavioral_main_v1_N200_k20_nr50.pkl \\
        --checkpoint checkpoints/main_v1_seed4 \\
        --tag neural --out_dir figures

    JAX_ENABLE_X64=1 PYTHONPATH=. python -u -m examples.iblotto.render_bfpta_section_figures \\
        --bundle /Users/davidsewell/Downloads/behavioral_main_v1_N200_k20_nr50.pkl \\
        --checkpoint checkpoints/main_v1_m50_long_seed4 \\
        --tag rbf_m50 --out_dir figures
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games
from fptajax.neural import NeuralBasis


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


def _joint_r2(Y_disc, trait_vals):
    X = np.concatenate([Y_disc, np.ones((Y_disc.shape[0], 1))], axis=1)
    y = (trait_vals - trait_vals.mean()) / max(trait_vals.std(), 1e-12)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _best_trait_per_disc(embeddings, traits, n_discs):
    out = []
    for k in range(n_discs):
        Y = embeddings[:, k, :]
        scores = {name: _joint_r2(Y, traits[:, ti])
                  for name, ti in _TRAIT_INDEX.items()}
        best = max(scores, key=scores.get)
        out.append((best, scores[best]))
    return out


def _render_spectrum(omegas, out_path, k_show, title, colour="#c0504d"):
    omegas = np.asarray(omegas)
    k_show = min(k_show, len(omegas))
    ratios = omegas[:k_show] / max(omegas[0], 1e-12)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    xs = np.arange(1, k_show + 1)
    ax.bar(xs, ratios, color=colour, alpha=0.85, edgecolor="k", linewidth=0.4)
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


def _render_disc_panels(embeddings, omegas, traits, out_path, n_discs,
                        suptitle_extra="", method_label="BFPTA"):
    n_discs = min(n_discs, embeddings.shape[1], len(omegas))
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
        rf"{method_label} top-{n_discs} disc embeddings "
        rf"$\mathbf{{Y}}^{{(k)}}$ on iblotto"
        + (f"  ({suptitle_extra})" if suptitle_extra else ""),
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _rbf_evaluate(traits: np.ndarray, centres: np.ndarray,
                  sigma: float) -> np.ndarray:
    """Same convention as ablate_basis_rbf.rbf_evaluate. Numpy version."""
    diff = traits[:, None, :] - centres[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    return np.exp(-sq_dist / (2.0 * sigma * sigma))


def main(bundle_path: Path, checkpoint_dir: Path, out_dir: Path,
         tag: str, n_discs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Rendering BFPTA section figures ===")
    print(f"  bundle:     {bundle_path}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  tag:        {tag}")

    with open(checkpoint_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    basis_kind = meta.get("basis_kind")
    is_rbf = (basis_kind == "rbf_kmeans")
    if is_rbf:
        print(f"  basis_kind: rbf_kmeans (m={meta['m_centres']}, "
              f"sigma={meta['rbf_sigma']:.4f})")
        method_label = "BFPTA + k-means RBF"
    else:
        print(f"  basis_kind: neural (d={meta['d']})")
        method_label = "BFPTA (neural basis)"

    d = int(meta["d"])
    trait_dim = int(meta["trait_dim"])
    sa_dim = int(meta["sa_dim"])
    L_max = int(meta["L_max"])
    d_model = int(meta["d_model"])
    n_heads = int(meta["n_heads"])
    n_layers = int(meta["n_layers"])
    rho_hidden = tuple(meta.get("rho_hidden", (64,)))

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"  N = {N}, d = {d}")

    # ---- Load encoder ----
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    encoder_template = HierarchicalSetEncoder(
        token_dim=sa_dim, L_max=L_max, trait_dim=trait_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        mlp_ratio=int(meta.get("mlp_ratio", 4)),
        rho_hidden=rho_hidden, key=k1,
    )
    encoder = eqx.tree_deserialise_leaves(
        str(checkpoint_dir / "encoder.eqx"), encoder_template)

    # ---- Encode bundle agents ----
    rng = np.random.RandomState(0)
    G_sample = min(8, ds.G_max)
    games, tmask, gmask = _sample_games(
        np.asarray(ds.agent_data), np.asarray(ds.agent_token_mask),
        np.asarray(ds.agent_game_mask), np.arange(N), G_sample, rng,
    )
    traits_enc = np.asarray(encoder.encode_batch(
        jnp.asarray(games), jnp.asarray(tmask), jnp.asarray(gmask)))

    # ---- Evaluate basis -> B in R^(N, d) ----
    if is_rbf:
        centres = np.asarray(meta["rbf_centres"])
        sigma = float(meta["rbf_sigma"])
        B = _rbf_evaluate(traits_enc.astype(np.float64), centres.astype(np.float64), sigma)
    else:
        basis_hidden = tuple(meta.get("basis_hidden", (128, 128)))
        basis_template = NeuralBasis(
            trait_dim=trait_dim, d=d, hidden_dims=basis_hidden, key=k2)
        basis = eqx.tree_deserialise_leaves(
            str(checkpoint_dir / "basis.eqx"), basis_template)
        B = np.asarray(basis.evaluate_batch(jnp.asarray(traits_enc)))

    # ---- Reconstruct embeddings using stored Schur ----
    omegas = np.asarray(meta["eigenvalues"])
    Q = np.asarray(meta["schur_vectors"])
    n_components = int(meta.get("n_components", len(omegas)))
    K = n_components
    print(f"  K = {K} active discs, omega_1 = {omegas[0]:.3f}, "
          f"top-3 ratios = {(omegas[:min(3, K)] / max(omegas[0], 1e-12)).round(3)}")

    Z = B @ Q                                   # (N, d)
    embeddings = np.zeros((N, K, 2))
    for k in range(K):
        embeddings[:, k, 0] = np.sqrt(omegas[k]) * Z[:, 2 * k]
        embeddings[:, k, 1] = np.sqrt(omegas[k]) * Z[:, 2 * k + 1]

    traits = np.asarray(ds.policies[:, :5])

    _render_spectrum(
        omegas, out_dir / f"iblotto_bfpta_spectrum_{tag}.pdf",
        k_show=min(K, 10),
        title=rf"{method_label} disc-game spectrum  ($d={d}$, $K={K}$)",
    )
    _render_disc_panels(
        embeddings, omegas, traits,
        out_dir / f"iblotto_bfpta_discs_{tag}.pdf",
        n_discs=n_discs,
        suptitle_extra=rf"$d={d}$",
        method_label=method_label,
    )
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="BFPTA checkpoint directory (neural or RBF; "
                        "branch picked from meta['basis_kind'])")
    p.add_argument("--tag", type=str, required=True,
                   help="suffix for output filenames, e.g. 'neural' or 'rbf_m50'")
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    p.add_argument("--n_discs", type=int, default=6)
    args = p.parse_args()
    main(args.bundle, args.checkpoint, args.out_dir, args.tag, args.n_discs)
