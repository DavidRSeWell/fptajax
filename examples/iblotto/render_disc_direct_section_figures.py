"""Render disc_direct figures for the iblotto section.

Loads a saved disc_direct checkpoint (``encoder.eqx + disc_head.eqx +
meta.pkl``), runs the bundle's agents through the encoder, applies the
disc head to get per-agent disc embeddings ``Y^{(k)}(x_i) = (u_k, v_k)``,
and emits two figures analogous to the PTA / classical-FPTA renderers:

  figures/iblotto_disc_direct_spectrum_<tag>.pdf  log-y bar chart of
                                                  the K disc magnitudes
                                                  (sqrt(F-energy) per disc)
  figures/iblotto_disc_direct_discs_<tag>.pdf     2x3 multi-disc panel,
                                                  coloured by best-loading
                                                  trait (raw values)

Unlike BFPTA and classical FPTA, no Schur decomposition is needed --- the
disc head's 2K outputs *are* the disc-embedding coordinates directly,
already paired into (u, v) per disc by the model's parameterisation.
Disc magnitudes are computed as
  omega_k_eff = sqrt(2 (||u_k||^2 ||v_k||^2 - <u_k, v_k>^2)) / N,
which is the disc-game analog of the Schur eigenvalue (the Frobenius
norm of the rank-2 skew matrix u_k v_k^T - v_k u_k^T divided by sqrt(2) N).

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -u -m examples.iblotto.render_disc_direct_section_figures \\
        --bundle /Users/davidsewell/Downloads/behavioral_main_v1_N200_k20_nr50.pkl \\
        --checkpoint checkpoints/main_v1_no_skill_seed4 \\
        --tag no_skill --out_dir figures
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
from fptajax.hierarchical import HierarchicalSetEncoder


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


def _best_trait_per_disc(embeddings, traits, n_discs):
    out = []
    for k in range(n_discs):
        Y = embeddings[:, k, :]
        scores = {name: _joint_r2(Y, traits[:, ti])
                  for name, ti in _TRAIT_INDEX.items()}
        best = max(scores, key=scores.get)
        out.append((best, scores[best]))
    return out


def _render_spectrum(omegas, out_path, k_show, title):
    omegas = np.asarray(omegas)
    k_show = min(k_show, len(omegas))
    ratios = omegas[:k_show] / max(omegas[0], 1e-12)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    xs = np.arange(1, k_show + 1)
    ax.bar(xs, ratios, color="#c0504d", alpha=0.85,
           edgecolor="k", linewidth=0.4)
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
                        suptitle_extra=""):
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
        rf"disc\_direct top-{n_discs} disc embeddings "
        rf"$\mathbf{{Y}}^{{(k)}}$ on iblotto"
        + (f"  ({suptitle_extra})" if suptitle_extra else ""),
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _disc_magnitudes_from_Z(Z: np.ndarray, N: int) -> np.ndarray:
    """Per-disc effective magnitude analogous to a Schur eigenvalue.

    Z has shape (N, K, 2) with axis -1 indexing (u, v). For each disc,
    compute ||u v^T - v u^T||_F / (sqrt(2) N) = sqrt(||u||^2 ||v||^2
    - <u, v>^2) / N, then sort descending.
    """
    u = Z[..., 0]; v = Z[..., 1]                # (N, K)
    u_norm_sq = np.sum(u ** 2, axis=0)          # (K,)
    v_norm_sq = np.sum(v ** 2, axis=0)
    uv = np.sum(u * v, axis=0)
    energy = u_norm_sq * v_norm_sq - uv ** 2
    return np.sqrt(np.maximum(energy, 0.0)) / max(N, 1)


def main(bundle_path: Path, checkpoint_dir: Path, out_dir: Path,
         tag: str, n_discs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Rendering disc_direct section figures ===")
    print(f"  bundle:     {bundle_path}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  tag:        {tag}")

    with open(checkpoint_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    if meta.get("basis_kind") != "disc_direct":
        raise SystemExit(f"meta.pkl has basis_kind={meta.get('basis_kind')!r}; "
                         f"expected 'disc_direct'.")

    K = int(meta["K"])
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
    print(f"  N = {N}, K = {K}")

    # ---- Load encoder + disc_head ----
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

    disc_head_template = eqx.nn.MLP(
        in_size=trait_dim, out_size=2 * K,
        width_size=64, depth=2,
        activation=jax.nn.gelu, key=k2,
    )
    disc_head = eqx.tree_deserialise_leaves(
        str(checkpoint_dir / "disc_head.eqx"), disc_head_template)

    # ---- Encode bundle agents on a fixed game sub-sample ----
    rng = np.random.RandomState(0)
    G_sample = min(8, ds.G_max)
    from fptajax.hierarchical import _sample_games
    games, tmask, gmask = _sample_games(
        np.asarray(ds.agent_data), np.asarray(ds.agent_token_mask),
        np.asarray(ds.agent_game_mask), np.arange(N), G_sample, rng,
    )
    traits_enc = np.asarray(encoder.encode_batch(
        jnp.asarray(games), jnp.asarray(tmask), jnp.asarray(gmask)))

    # ---- Disc-head output -> (N, K, 2) embeddings ----
    Z_flat = np.asarray(jax.vmap(disc_head)(jnp.asarray(traits_enc)))
    Z = Z_flat.reshape(N, K, 2)
    omegas = _disc_magnitudes_from_Z(Z, N)
    # Sort discs by magnitude (descending) so the panel order matches the
    # eigenvalue-spectrum convention.
    order = np.argsort(omegas)[::-1]
    Z = Z[:, order, :]
    omegas = omegas[order]

    embeddings = Z * np.sqrt(N)  # rescale so axis units match per-agent
                                  # coordinate magnitude rather than per-Z
                                  # column-norm; visually clearer.

    # Ground-truth traits
    traits = np.asarray(ds.policies[:, :5])

    use_skill = bool(meta.get("use_skill", False))
    ortho = float(meta.get("ortho_weight", 0.0))
    config = (("with skill" if use_skill else "no skill")
              + (", with ortho" if ortho > 0 else ", no ortho"))

    _render_spectrum(
        omegas, out_dir / f"iblotto_disc_direct_spectrum_{tag}.pdf",
        k_show=K,
        title=rf"disc\_direct disc-game spectrum  ({config})",
    )
    _render_disc_panels(
        embeddings, omegas, traits,
        out_dir / f"iblotto_disc_direct_discs_{tag}.pdf",
        n_discs=n_discs, suptitle_extra=config,
    )
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="disc_direct checkpoint directory")
    p.add_argument("--tag", type=str, required=True,
                   help="suffix for output filenames, e.g. 'no_skill' "
                        "or 'with_skill_ortho'")
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    p.add_argument("--n_discs", type=int, default=6)
    args = p.parse_args()
    main(args.bundle, args.checkpoint, args.out_dir, args.tag, args.n_discs)
