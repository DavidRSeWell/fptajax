#!/usr/bin/env python3
"""Post-hoc sanity check: do disc-game distances track empirical F-row distances?

Loads the trained Behavioral FPTA checkpoint produced by
``rps_behavioral_example.py`` and computes two pairwise distance matrices
on the 42 RoShamBo bots:

    RD_closed[i, j]    = || Y(x_i) - Y(x_j) ||_2
                         where Y(x) = (sqrt(omega_k) * (b(x) . q_{2k}, b(x) . q_{2k+1}))_k
                         is the concatenation of all disc-game 2D embeddings.

    RD_empirical[i, j] = || F[i, :] - F[j, :] ||_2
                         on the symmetrised payoff matrix F = 0.5 (F - F^T).

If the disc-game decomposition faithfully captures F, off-diagonal upper-
triangular entries of these two matrices should be highly correlated.

Outputs:
  - stdout: Pearson r, Spearman rho, eigenvalue spectrum, cumulative variance.
  - rd_sanity_check.png: scatter with y = a x best-fit line.
  - rd_truncation.png:   correlation as a function of #disc-games kept.

No model weights are touched.  Run from the repo root:

    .venv/bin/python examples/rd_sanity_check.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def upper_offdiag(M: np.ndarray) -> np.ndarray:
    """Return the strictly-upper-triangular entries of M as a flat array."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = Pearson on ranks; no scipy required."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_y_eq_ax(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of y = a x (no intercept).  Returns (a, R^2).

    R^2 is the standard coefficient of determination *with* an intercept term
    (i.e. squared Pearson r), which is the more commonly-quoted quantity when
    showing scatter fits.  The slope ``a`` itself comes from the y = a x fit.
    """
    a = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
    r = pearson_r(x, y)
    return a, r * r


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(ckpt_dir: Path) -> dict:
    """Load encoder/basis (eqx) + meta (pickle) from the checkpoint dir."""
    import equinox as eqx
    from fptajax.behavioral import SetEncoder
    from fptajax.neural import NeuralBasis

    with open(ckpt_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    encoder_template = SetEncoder(
        sa_dim=meta["sa_dim"], trait_dim=meta["trait_dim"],
        phi_hidden=meta["phi_hidden"], rho_hidden=meta["rho_hidden"],
        key=k1,
    )
    basis_template = NeuralBasis(
        trait_dim=meta["trait_dim"], d=meta["d"],
        hidden_dims=meta["basis_hidden"], key=k2,
    )
    encoder = eqx.tree_deserialise_leaves(
        str(ckpt_dir / "encoder.eqx"), encoder_template,
    )
    basis = eqx.tree_deserialise_leaves(
        str(ckpt_dir / "basis.eqx"), basis_template,
    )
    meta["encoder"] = encoder
    meta["basis"] = basis
    return meta


# ---------------------------------------------------------------------------
# Disc-game embeddings
# ---------------------------------------------------------------------------


def build_disc_game_embeddings(
    encoder, basis,
    agent_data: np.ndarray, agent_mask: np.ndarray,
    schur_vectors: np.ndarray, eigenvalues: np.ndarray, n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-encode each agent and project onto the disc-game basis.

    Returns:
        Y_full: (N, n_components, 2) array — Y[i, k] is the 2D coord of agent
            i in disc-game k, pre-multiplied by sqrt(omega_k).
        b_mat:  (N, d) array — raw basis matrix b(x_i).
    """
    sa = jnp.asarray(agent_data)
    mask = jnp.asarray(agent_mask)
    traits = encoder.encode_batch(sa, mask)             # (N, trait_dim)
    b_mat = np.asarray(basis.evaluate_batch(traits))    # (N, d)

    Q = np.asarray(schur_vectors)                       # (d, d)
    omegas = np.asarray(eigenvalues)                    # (n_components,)
    N, d = b_mat.shape
    Y_full = np.zeros((N, n_components, 2), dtype=np.float64)
    for k in range(n_components):
        q1 = Q[:, 2 * k]
        q2 = Q[:, 2 * k + 1]
        s = float(np.sqrt(max(omegas[k], 0.0)))
        Y_full[:, k, 0] = s * (b_mat @ q1)
        Y_full[:, k, 1] = s * (b_mat @ q2)
    return Y_full, b_mat


def rd_closed_from_embeddings(Y: np.ndarray, k_keep: int | None = None) -> np.ndarray:
    """Pairwise Euclidean distance using disc games 0..k_keep-1.

    Args:
        Y:      (N, K_total, 2) full disc-game embeddings.
        k_keep: number of leading disc games to use; None means all.
    """
    if k_keep is None:
        k_keep = Y.shape[1]
    flat = Y[:, :k_keep, :].reshape(Y.shape[0], -1)     # (N, 2 * k_keep)
    diff = flat[:, None, :] - flat[None, :, :]          # (N, N, 2 k_keep)
    return np.sqrt(np.sum(diff * diff, axis=-1))         # (N, N)


def rd_empirical(F: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between rows of F."""
    diff = F[:, None, :] - F[None, :, :]                 # (N, N, N)
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter(
    rd_emp: np.ndarray, rd_closed: np.ndarray,
    out_path: Path,
) -> tuple[float, float, float, float]:
    x = upper_offdiag(rd_emp)
    y = upper_offdiag(rd_closed)
    pear = pearson_r(x, y)
    spear = spearman_rho(x, y)
    a, r2 = fit_y_eq_ax(x, y)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    ax.scatter(x, y, s=12, alpha=0.45, color="steelblue", edgecolor="none")
    xs = np.linspace(0.0, x.max() * 1.02, 50)
    ax.plot(xs, a * xs, color="crimson", lw=2,
            label=f"y = {a:.3f} x   ($R^2$ = {r2:.3f})")
    ax.set_xlabel("RD$^{\\mathrm{empirical}}$  (||F[i,:] - F[j,:]||)")
    ax.set_ylabel("RD$^{\\mathrm{closed}}$  (disc-game embedding distance)")
    ax.set_title(f"RD sanity check  (Pearson r = {pear:.3f},  Spearman $\\rho$ = {spear:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return pear, spear, a, r2


def plot_truncation(
    rd_emp: np.ndarray, Y: np.ndarray, eigenvalues: np.ndarray,
    out_path: Path,
) -> list[tuple[int, float, float]]:
    n_components = Y.shape[1]
    K_max = max(1, n_components // 2)
    x_emp = upper_offdiag(rd_emp)

    rows: list[tuple[int, float, float]] = []
    for K in range(1, n_components + 1):
        rd_c = rd_closed_from_embeddings(Y, k_keep=K)
        y = upper_offdiag(rd_c)
        rows.append((K, pearson_r(x_emp, y), spearman_rho(x_emp, y)))

    Ks = np.array([r[0] for r in rows])
    pears = np.array([r[1] for r in rows])
    spears = np.array([r[2] for r in rows])
    omegas = np.asarray(eigenvalues)
    cum_var = np.cumsum(omegas[: len(Ks)] ** 2)
    cum_var = cum_var / max(cum_var[-1], 1e-12)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(Ks, pears,  marker="o", lw=2, color="crimson", label="Pearson r")
    ax.plot(Ks, spears, marker="s", lw=2, color="darkorange", label="Spearman $\\rho$")
    ax.set_xlabel("Top-K disc games kept")
    ax.set_ylabel("Correlation with RD$^{\\mathrm{empirical}}$")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(Ks)
    ax.grid(True, alpha=0.3)
    ax.axvline(K_max, color="gray", linestyle="--", alpha=0.6,
               label=f"d/2 = {K_max}")
    ax.legend(loc="lower right")

    ax2 = ax.twinx()
    ax2.plot(Ks, cum_var, color="steelblue", lw=1.5, linestyle=":",
             label="cum. eig$^2$ fraction")
    ax2.set_ylabel("cumulative $\\omega^2$ fraction", color="steelblue")
    ax2.set_ylim(0.0, 1.02)
    ax2.tick_params(axis="y", labelcolor="steelblue")

    ax.set_title("Truncation curve: how many disc games suffice?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    here = Path(__file__).resolve().parent
    ckpt_dir = here / "rps_behavioral_checkpoint"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint dir not found at {ckpt_dir}.\n"
            "Run examples/rps_behavioral_example.py first to produce it."
        )

    print("=" * 72)
    print("Behavioral FPTA — disc-game vs. empirical distance sanity check")
    print("=" * 72)
    print(f"\nLoading checkpoint: {ckpt_dir}/")
    ckpt = load_checkpoint(ckpt_dir)

    encoder = ckpt["encoder"]
    basis = ckpt["basis"]
    schur_vectors = ckpt["schur_vectors"]
    eigenvalues = np.asarray(ckpt["eigenvalues"])
    n_components = int(ckpt["n_components"])
    bot_names = ckpt["bot_names"]
    agent_data = ckpt["agent_data"]
    agent_mask = ckpt["agent_mask"]
    F_raw = np.asarray(ckpt["F_raw"], dtype=np.float64)
    N = F_raw.shape[0]
    d = schur_vectors.shape[0]

    print(f"  N agents       : {N}")
    print(f"  basis dim d    : {d}")
    print(f"  disc-game cmps : {n_components}")
    print(f"  trait_dim      : (re-encoded below)")

    # Symmetrise F the same way training does.
    F = 0.5 * (F_raw - F_raw.T)
    print(f"  ||F||_F (sym)  : {np.linalg.norm(F):.4f}")

    # Eigenvalue spectrum + cumulative omega^2 fraction
    print("\nDisc-game spectrum (omega_k, importance, cumulative):")
    om2 = eigenvalues ** 2
    om2_total = float(om2.sum())
    cum_frac = np.cumsum(om2) / max(om2_total, 1e-12)
    for k in range(n_components):
        print(f"  k={k+1:2d}   omega = {eigenvalues[k]:8.5f}   "
              f"omega^2 frac = {om2[k]/max(om2_total,1e-12):7.4f}   "
              f"cumulative   = {cum_frac[k]:7.4f}")

    # Closed-form embeddings (re-encode through the trained encoder + basis)
    print("\nRe-encoding all agents through encoder + basis ...")
    Y_full, b_mat = build_disc_game_embeddings(
        encoder, basis,
        agent_data=agent_data, agent_mask=agent_mask,
        schur_vectors=schur_vectors, eigenvalues=eigenvalues,
        n_components=n_components,
    )
    print(f"  Y_full shape   : {Y_full.shape}")
    print(f"  b matrix range : [{b_mat.min():+.3f}, {b_mat.max():+.3f}]")

    # Distance matrices
    rd_emp = rd_empirical(F)
    rd_close = rd_closed_from_embeddings(Y_full)
    print(f"\nRD_empirical : range [{rd_emp[rd_emp>0].min():.4f}, "
          f"{rd_emp.max():.4f}],  mean(off-diag) = {upper_offdiag(rd_emp).mean():.4f}")
    print(f"RD_closed    : range [{rd_close[rd_close>0].min():.4f}, "
          f"{rd_close.max():.4f}],  mean(off-diag) = {upper_offdiag(rd_close).mean():.4f}")

    # --- Sanity scatter ---
    sanity_path = here.parent / "rd_sanity_check.png"
    pear, spear, a, r2 = plot_scatter(rd_emp, rd_close, sanity_path)
    print(f"\nSaved {sanity_path}")
    print(f"  Pearson r       = {pear:.4f}")
    print(f"  Spearman rho    = {spear:.4f}")
    print(f"  y = a x fit     : a = {a:.4f},  R^2 (Pearson^2) = {r2:.4f}")
    if pear < 0.9:
        print(f"  NOTE: Pearson r below the 0.9 hope target.")

    # --- Truncation curve ---
    trunc_path = here.parent / "rd_truncation.png"
    rows = plot_truncation(rd_emp, Y_full, eigenvalues, trunc_path)
    print(f"\nSaved {trunc_path}")
    print(f"  K   Pearson r     Spearman rho")
    for K, p, s in rows:
        marker = "  <-- d/2" if K == max(1, n_components // 2) else ""
        print(f"  {K:2d}   {p:8.4f}     {s:8.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
