"""H12 — FPTA ↔ PTA correspondence (Theorem 4.2).

Two claims tested per dataset:

  (a) For the empirical-measure FPTA solution  Y^FPTA  and the pointwise PTA
      solution  Y^PTA = Q Λ^{1/2}, the FPTA embeddings equal P_{B_X} · Y^PTA
      (the projection of PTA embeddings onto the column space of the basis
      matrix at the agents).

      Quantitatively:  ‖f − f̂‖²_{δ_X}  =  (1/N) ‖(I − P_{B_X}) Y^PTA‖²_F.

  (b) Corollary 4.3: when m = N and B_X is full-rank, FPTA recovers F
      exactly at sample pairs (P_{B_X} = I), so the residual is 0.

We use the engine's ``fpta_empirical`` (full-data FPTA) here — the train/test
split would obscure the projection identity, which is a *fitting* statement,
not a generalisation one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from examples.classical_fpta_suite.protocol import MAX_M_FOR_FIT

from fptajax.pta import pta

from examples.classical_fpta_suite.protocol import orthonormalise
from examples.classical_fpta_suite.data.blotto import build_blotto
from examples.classical_fpta_suite.data.kuhn import build_kuhn
from examples.classical_fpta_suite.data.rps import build_rps
from examples.classical_fpta_suite.data.tennis import build_tennis


_OUT = Path(__file__).resolve().parent.parent / "results"
_OUT.mkdir(exist_ok=True)


def projection_matrix(B: np.ndarray) -> np.ndarray:
    """Empirical-measure projector P_{B_X} = (1/N) B (Bᵀ B / N)^{-1} Bᵀ.

    With ``B`` already orthonormalised so that (1/N) Bᵀ B = I, this simplifies
    to ``(1/N) B Bᵀ``.
    """
    N = B.shape[0]
    return (1.0 / N) * (B @ B.T)


def projection_residual(P: np.ndarray, Y_pta: np.ndarray) -> float:
    """‖(I - P) Y^PTA‖²_Fro / N — paper's residual quantity."""
    N = Y_pta.shape[0]
    Y = Y_pta.reshape(N, -1)                      # (N, 2 k)
    R = Y - P @ Y
    return float(np.sum(R ** 2) / N)


def run_dataset(ds, label: str) -> dict:
    F = np.asarray(ds.F, dtype=np.float64)

    # Pointwise PTA on the full F
    pta_res = pta(F)
    Y_pta = np.asarray(pta_res.embeddings)         # (N, d, 2)

    rows = []
    # Sweep across all bases for this dataset
    from fptajax.decomposition import skew_symmetric_schur
    for bn in ds.available_bases:
        B_raw = np.asarray(ds.basis(bn, MAX_M_FOR_FIT), dtype=np.float64)
        B = orthonormalise(B_raw)                # rank-deficient-safe (eigendecomp+ridge)
        m = B.shape[1]
        N_ = B.shape[0]

        # FPTA on the full edge set via MC projection (paper eq. 4.2):
        #     C_ij = (1/N²) (Bᵀ F B)_ij,
        # using the orthonormalised basis. This avoids the engine's Cholesky
        # which fails on rank-deficient B.
        C = (B.T @ F @ B) / (N_ * N_)
        C = 0.5 * (C - C.T)
        sch = skew_symmetric_schur(np.asarray(C, dtype=np.float64))
        Q = np.asarray(sch.Q)
        omegas = np.asarray(sch.eigenvalues)
        nc = int(sch.n_components)
        Y = np.zeros((N_, nc, 2))
        for j in range(nc):
            s = float(np.sqrt(max(omegas[j], 0.0)))
            Y[:, j, 0] = s * (B @ Q[:, 2 * j])
            Y[:, j, 1] = s * (B @ Q[:, 2 * j + 1])
        F_hat = (np.einsum("ik,jk->ij", Y[:, :, 0], Y[:, :, 1])
               - np.einsum("ik,jk->ij", Y[:, :, 1], Y[:, :, 0]))
        lhs = float(np.mean((F - F_hat) ** 2))

        # Projection-identity prediction for the residual norm
        P = projection_matrix(B)
        rhs = projection_residual(P, Y_pta)

        rows.append(dict(
            dataset=label, basis=bn, m=int(m), N=int(N_),
            fit_error_lhs=lhs, projection_residual_rhs=rhs,
            ratio_lhs_rhs=lhs / max(rhs, 1e-15),
        ))
    return rows


def plot_lhs_vs_rhs(all_rows: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    cmap = plt.get_cmap("tab10")
    by_ds: dict[str, list] = {}
    for r in all_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for k, (label, rows) in enumerate(by_ds.items()):
        xs = [r["fit_error_lhs"] for r in rows]
        ys = [r["projection_residual_rhs"] for r in rows]
        ax.loglog(xs, ys, "o", color=cmap(k % 10), label=label, markersize=7,
                  alpha=0.85)
    lo = min(min(r["fit_error_lhs"] for r in all_rows),
             min(r["projection_residual_rhs"] for r in all_rows))
    hi = max(max(r["fit_error_lhs"] for r in all_rows),
             max(r["projection_residual_rhs"] for r in all_rows))
    if lo <= 0:
        lo = 1e-15
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, label="y = x (identity)")
    ax.set_xlabel(r"FPTA fit error  $\|F - \hat F\|^2 / N^2$")
    ax.set_ylabel(r"$\|(I-P_{B_X}) Y^{PTA}\|^2 / N$")
    ax.set_title("H12 — FPTA fit error vs PTA-projection residual\n"
                 "Theorem 4.2 says they are equal")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
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
    all_rows = []
    print(f"{'dataset':18s} {'basis':22s} {'m':>4s} {'N':>4s} "
          f"{'fit_lhs':>11s} {'rhs':>11s}  ratio")
    for label, ds in datasets:
        rows = run_dataset(ds, label)
        all_rows.extend(rows)
        for r in rows:
            print(f"  {label:18s} {r['basis']:22s} {r['m']:4d} {r['N']:4d} "
                  f"{r['fit_error_lhs']:11.5e} {r['projection_residual_rhs']:11.5e}  "
                  f"{r['ratio_lhs_rhs']:.4f}")

    with open(_OUT / "h12_fpta_vs_pta.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    plot_lhs_vs_rhs(all_rows, _OUT / "h12_fpta_vs_pta.png")
    print(f"\nSaved {_OUT/'h12_fpta_vs_pta.json'} and PNG.")


if __name__ == "__main__":
    main()
