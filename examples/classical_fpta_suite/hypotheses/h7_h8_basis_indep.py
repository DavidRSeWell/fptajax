"""H7 / H8 — basis independence and uniqueness up to per-disc rotation.

Two same-span bases should produce the same embeddings up to a per-disc-plane
rotation. We test by fitting two bases that span the same polynomial space
on the dataset and Procrustes-aligning their disc-game embeddings.

Test pairs (per dataset):

  Blotto:  monomial_d3  vs  bernstein_d3      (both span deg-≤3 polys on simplex)
  Kuhn:    pairwise     vs  card_grouped       (both span the symmetric pair products
                                                — note ``card_grouped`` is a re-ordering
                                                of the same span)
  RPS:     deg2_self    vs  deg2_self          (with column shuffling)  — degenerate but
                                                still tests numerical robustness
  Tennis:  deg2         vs  pca10              (pca10 is a strict subspace, not equal
                                                span — we expect partial agreement and
                                                report it for completeness)

For each pair we report, per disc game k:

  - Procrustes residual after the best rotation R_k ∈ SO(2)
  - eigenvalue agreement |ω_k^A − ω_k^B|
  - the Procrustes rotation angle (for inspection)
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


def _embeddings(ds, basis_name: str, k_keep: int) -> tuple[np.ndarray, np.ndarray]:
    B_raw = ds.basis(basis_name, MAX_M_FOR_FIT)
    B = orthonormalise(B_raw)
    C = fit_skew_C_train(B, ds.F, ds.train_pairs, ridge=1e-4)
    _, omegas, Q = truncate_C(C, k_keep)
    N = B.shape[0]
    Y = np.zeros((N, k_keep, 2))
    for j in range(k_keep):
        s = float(np.sqrt(max(omegas[j], 0.0)))
        Y[:, j, 0] = s * (B @ Q[:, 2 * j])
        Y[:, j, 1] = s * (B @ Q[:, 2 * j + 1])
    return Y, omegas[:k_keep]


def _procrustes_2d(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Best 2D rotation that aligns A onto B in least-squares sense.

    Returns ``(R, residual, angle_radians)``.
    """
    H = B.T @ A                                # (2, 2)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt                                  # closest orthogonal
    if np.linalg.det(R) < 0:                   # ensure rotation, not reflection
        Vt[-1, :] *= -1
        R = U @ Vt
    A_rot = A @ R.T
    res = float(np.linalg.norm(A_rot - B) / max(np.linalg.norm(B), 1e-12))
    angle = float(np.arctan2(R[1, 0], R[0, 0]))
    return R, res, angle


# Pairs to compare per dataset
_PAIRS = {
    "blotto_small":   ("monomial_d3", "bernstein_d3"),
    "blotto_medium":  ("monomial_d3", "bernstein_d3"),
    "kuhn_random":    ("pairwise",    "card_grouped"),
    "rps":            ("deg2_self_and_opp", "deg2_self_and_opp"),  # noop control
    "tennis":         ("deg2",        "pca10"),
}


def run_dataset(label: str, ds, k_keep: int = 4) -> dict:
    a_name, b_name = _PAIRS[label]
    Y_a, om_a = _embeddings(ds, a_name, k_keep)
    Y_b, om_b = _embeddings(ds, b_name, k_keep)

    per_disc = []
    for j in range(k_keep):
        R, res, ang = _procrustes_2d(Y_a[:, j], Y_b[:, j])
        per_disc.append(dict(
            k=j + 1,
            residual=res,
            rotation_deg=float(np.degrees(ang)),
            omega_a=float(om_a[j]), omega_b=float(om_b[j]),
            omega_diff=float(om_a[j] - om_b[j]),
        ))
    return dict(label=label, basis_a=a_name, basis_b=b_name,
                k_keep=k_keep, per_disc=per_disc)


def plot_dataset(res: dict, out_path: Path):
    Ks = [d["k"] for d in res["per_disc"]]
    resids = [d["residual"] for d in res["per_disc"]]
    angs = [d["rotation_deg"] for d in res["per_disc"]]
    om_a = [d["omega_a"] for d in res["per_disc"]]
    om_b = [d["omega_b"] for d in res["per_disc"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    axes[0].bar(Ks, resids, color="crimson", alpha=0.8)
    axes[0].set_xlabel("disc game k"); axes[0].set_ylabel("Procrustes residual")
    axes[0].set_title(
        f"{res['label']} — H7 basis indep ({res['basis_a']} vs {res['basis_b']})"
    )
    axes[0].grid(True, alpha=0.3)
    for k, r, a in zip(Ks, resids, angs):
        axes[0].text(k, r, f"{a:.0f}°", ha="center", va="bottom", fontsize=7)

    xs = np.arange(len(Ks))
    axes[1].bar(xs - 0.2, om_a, width=0.4, color="C0", label=res["basis_a"])
    axes[1].bar(xs + 0.2, om_b, width=0.4, color="C1", label=res["basis_b"])
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"k={i+1}" for i in Ks])
    axes[1].set_title("ω_k agreement (H8: simple eigenvalues → unique)")
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
    print(f"{'dataset':18s} {'a':18s} {'b':18s} {'k':>2s} {'resid':>8s} {'rot°':>7s} "
          f"{'ω_a':>7s} {'ω_b':>7s}")
    for label, ds in datasets:
        res = run_dataset(label, ds)
        summary.append(res)
        for d in res["per_disc"]:
            print(f"  {label:18s} {res['basis_a']:18s} {res['basis_b']:18s} "
                  f"{d['k']:2d} {d['residual']:8.4f} {d['rotation_deg']:7.1f} "
                  f"{d['omega_a']:7.4f} {d['omega_b']:7.4f}")
        plot_dataset(res, _OUT / f"h7_basis_indep_{label}.png")

    with open(_OUT / "h7_basis_indep.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
