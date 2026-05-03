"""H11 — composition invariance.

Theorem 3.15: If we reparameterise the trait space via x → M(x), the embeddings
of h(u, v) = f(M(u), M(v)) are exactly Y ∘ M (the original embeddings composed
with the new parameterisation).

Operationalisation: we apply an explicit invertible reparameterisation to the
trait space, refit FPTA, and compare embeddings at the *agent* locations
against the original embeddings.

Test on Blotto where the trait space is the simplex Δ^{K-1}: a natural
non-trivial M is the centred simplex coordinate transformation (Aitchison
log-ratio coordinates), which is invertible on the open simplex. We use a
simple linear M instead (centred & rescaled affine map) so it stays
well-defined at the simplex boundary too:

    M(x) = A x + c     where  A = I − (1/K) 1 1ᵀ + 0.3 R,
                       R is a fixed random orthogonal matrix.

Since this is a basis-invariant statement at the *embedding* level, the
embeddings under the same basis family computed in the new coordinates
should agree with the original embeddings up to a per-disc rotation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.protocol import (
    fit_skew_C_train, orthonormalise, truncate_C,
)
from examples.classical_fpta_suite.data.blotto import build_blotto, basis_monomials


_OUT = Path(__file__).resolve().parent.parent / "results"
_OUT.mkdir(exist_ok=True)


def _build_M(K: int, seed: int = 0) -> np.ndarray:
    """Invertible linear map on ℝ^K. Centring + small random rotation."""
    rng = np.random.RandomState(seed)
    A = np.eye(K) - (1.0 / K) * np.ones((K, K))
    R, _ = np.linalg.qr(rng.randn(K, K))
    return A + 0.3 * R


def run_blotto(setup: str = "small", k_keep: int = 4, basis_deg: int = 3) -> dict:
    ds = build_blotto(setup)
    K = ds.metadata["K"]

    # Build the original embeddings using monomial_d{deg} basis on traits.
    B_a_raw, _ = basis_monomials(ds.traits, basis_deg)
    B_a = orthonormalise(B_a_raw)
    C_a = fit_skew_C_train(B_a, ds.F, ds.train_pairs, ridge=1e-4)
    _, om_a, Q_a = truncate_C(C_a, k_keep)
    Y_a = np.zeros((ds.N, k_keep, 2))
    for j in range(k_keep):
        s = float(np.sqrt(max(om_a[j], 0.0)))
        Y_a[:, j, 0] = s * (B_a @ Q_a[:, 2 * j])
        Y_a[:, j, 1] = s * (B_a @ Q_a[:, 2 * j + 1])

    # Apply M to traits, refit with the same basis FAMILY in the new coordinates.
    M = _build_M(K, seed=0)
    new_traits = ds.traits @ M.T          # (N, K)

    B_b_raw, _ = basis_monomials(new_traits, basis_deg)
    B_b = orthonormalise(B_b_raw)
    C_b = fit_skew_C_train(B_b, ds.F, ds.train_pairs, ridge=1e-4)
    _, om_b, Q_b = truncate_C(C_b, k_keep)
    Y_b = np.zeros((ds.N, k_keep, 2))
    for j in range(k_keep):
        s = float(np.sqrt(max(om_b[j], 0.0)))
        Y_b[:, j, 0] = s * (B_b @ Q_b[:, 2 * j])
        Y_b[:, j, 1] = s * (B_b @ Q_b[:, 2 * j + 1])

    # Procrustes-align Y_b to Y_a per disc; report residuals.
    rows = []
    for j in range(k_keep):
        H = Y_a[:, j].T @ Y_b[:, j]
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = U @ Vt
        Y_b_rot = Y_b[:, j] @ R.T
        res = float(np.linalg.norm(Y_b_rot - Y_a[:, j]) / max(np.linalg.norm(Y_a[:, j]), 1e-12))
        rows.append(dict(
            k=j + 1, residual=res,
            omega_orig=float(om_a[j]), omega_new=float(om_b[j]),
        ))
    return dict(
        setup=setup, basis_deg=basis_deg, k_keep=k_keep,
        per_disc=rows,
    )


def plot_results(all_results: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    width = 0.25
    for i, res in enumerate(all_results):
        Ks = [d["k"] for d in res["per_disc"]]
        rs = [d["residual"] for d in res["per_disc"]]
        ax.bar([x + (i - 1) * width for x in Ks], rs, width=width,
               label=f"{res['setup']} (deg={res['basis_deg']})")
    ax.set_xlabel("disc game k"); ax.set_ylabel("Procrustes residual after rotation")
    ax.set_title("H11 — Composition invariance on Blotto\n"
                 "0 = perfect agreement under reparameterisation")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    out = []
    print(f"{'setup':10s} {'k':>2s} {'resid':>10s} {'ω_orig':>8s} {'ω_new':>8s}")
    for setup, deg in [("small", 3), ("medium", 3)]:
        res = run_blotto(setup, k_keep=4, basis_deg=deg)
        out.append(res)
        for d in res["per_disc"]:
            print(f"  {setup:10s} {d['k']:2d} {d['residual']:10.4e} "
                  f"{d['omega_orig']:8.4f} {d['omega_new']:8.4f}")
    plot_results(out, _OUT / "h11_composition_invariance.png")
    with open(_OUT / "h11_composition_invariance.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
