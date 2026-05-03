"""H5 / H6 — embedding orthogonality and equal magnitude.

For each dataset, fit FPTA at the richest basis, extract the disc-game
embeddings Y^(k)(x_i) ∈ ℝ², and check:

  H5: Cross-disc inner products  (1/N) Σ_i Y^(j)(x_i) ⋅ Y^(k)(x_i) ≈ 0  for j ≠ k
       Within-disc cross-coordinate (1/N) Σ_i Y_1^(k)(x_i) Y_2^(k)(x_i) ≈ 0
  H6: ‖Y_1^(k)‖²_π = ‖Y_2^(k)‖²_π = ω_k    (equal magnitude per disc)

We report the (top-k × top-k) Gram matrix as a heatmap to visualise H5,
and a bar chart of ‖Y_1‖² / ω vs ‖Y_2‖² / ω for H6 (target = 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.protocol import (  # noqa: F401
    MAX_M_FOR_FIT,
    fit_skew_C_train, orthonormalise, truncate_C,
)
from examples.classical_fpta_suite.data.blotto import build_blotto
from examples.classical_fpta_suite.data.kuhn import build_kuhn
from examples.classical_fpta_suite.data.rps import build_rps
from examples.classical_fpta_suite.data.tennis import build_tennis


_OUT = Path(__file__).resolve().parent.parent / "results"
_OUT.mkdir(exist_ok=True)


def disc_embeddings(B: np.ndarray, C: np.ndarray, k_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Y of shape (N, k_keep, 2) and the eigenvalues kept."""
    _, omegas, Q = truncate_C(C, k_keep)
    N = B.shape[0]
    Y = np.zeros((N, k_keep, 2))
    for j in range(k_keep):
        s = float(np.sqrt(max(omegas[j], 0.0)))
        Y[:, j, 0] = s * (B @ Q[:, 2 * j])
        Y[:, j, 1] = s * (B @ Q[:, 2 * j + 1])
    return Y, omegas[:k_keep]


def cross_gram(Y: np.ndarray) -> np.ndarray:
    """Return (k_keep, 2, k_keep, 2) inner products under empirical measure.

    Reshaped as a (2k, 2k) matrix where entries (2j+a, 2k+b) = ⟨Y_a^(j), Y_b^(k)⟩_π.
    """
    N, K, _ = Y.shape
    flat = Y.reshape(N, 2 * K)
    return (flat.T @ flat) / N


def run_dataset(ds, label: str, k_keep_max: int = 6) -> dict:
    bn = max(ds.available_bases, key=lambda n: ds.basis(n, MAX_M_FOR_FIT).shape[1])
    B_raw = ds.basis(bn, MAX_M_FOR_FIT)
    B = orthonormalise(B_raw)
    C = fit_skew_C_train(B, ds.F, ds.train_pairs, ridge=1e-4)

    from fptajax.decomposition import skew_symmetric_schur
    sch = skew_symmetric_schur(np.asarray(C, dtype=np.float64))
    nc_full = int(sch.n_components)
    k_keep = min(k_keep_max, nc_full)

    Y, omegas = disc_embeddings(B, C, k_keep)
    G = cross_gram(Y)                            # (2k, 2k)

    # H6: per-disc magnitudes
    mag1 = np.array([np.mean(Y[:, j, 0] ** 2) for j in range(k_keep)])
    mag2 = np.array([np.mean(Y[:, j, 1] ** 2) for j in range(k_keep)])

    # H5: off-diag mass / on-diag mass of cross-Gram
    diag_mass = float(np.sum(np.diag(G) ** 2))
    off_mass = float(np.sum(G ** 2) - diag_mass)

    return dict(
        label=label, basis=bn, k_keep=k_keep, omegas=omegas.tolist(),
        gram=G, mag1=mag1, mag2=mag2,
        h5_off_over_on=off_mass / max(diag_mass, 1e-12),
    )


def plot_dataset(res: dict, out_path: Path) -> None:
    G = res["gram"]; k = res["k_keep"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    # Cross-disc Gram heatmap
    vmax = max(abs(G).max(), 1e-12)
    im = axes[0].imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ticks = [f"Y$_{a}^{{({j+1})}}$" for j in range(k) for a in (1, 2)]
    axes[0].set_xticks(range(2 * k)); axes[0].set_xticklabels(ticks, fontsize=7, rotation=90)
    axes[0].set_yticks(range(2 * k)); axes[0].set_yticklabels(ticks, fontsize=7)
    axes[0].set_title(
        f"{res['label']} — H5 cross-disc Gram\n"
        f"off/on Frobenius² ratio = {res['h5_off_over_on']:.3e}"
    )
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Magnitude check: should match ω_k for both coordinates
    omegas = np.array(res["omegas"])
    xs = np.arange(k)
    axes[1].bar(xs - 0.2, res["mag1"], width=0.4, label=r"$\|Y_1^{(k)}\|^2$", color="C0")
    axes[1].bar(xs + 0.2, res["mag2"], width=0.4, label=r"$\|Y_2^{(k)}\|^2$", color="C1")
    axes[1].plot(xs, omegas, "kx-", label=r"$\omega_k$ (target)", markersize=8)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"k={i+1}" for i in xs])
    axes[1].set_title(f"{res['label']} — H6 per-disc magnitude vs ω_k")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    datasets = [
        ("blotto_small",      build_blotto("small")),
        ("blotto_medium",     build_blotto("medium")),
        ("kuhn_random",       build_kuhn("random",       N=150)),
        ("rps",               build_rps()),
        ("tennis",            build_tennis()),
    ]
    summary = []
    for label, ds in datasets:
        res = run_dataset(ds, label)
        # cast Gram to list for JSON
        rec = dict(res); rec["gram"] = res["gram"].tolist()
        rec["mag1"] = res["mag1"].tolist(); rec["mag2"] = res["mag2"].tolist()
        summary.append(rec)
        print(f"  {label:18s}  basis={res['basis']:18s}  k_keep={res['k_keep']}  "
              f"H5 off/on = {res['h5_off_over_on']:.3e}")
        plot_dataset(res, _OUT / f"h5_orthogonality_{label}.png")

    with open(_OUT / "h5_orthogonality.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
