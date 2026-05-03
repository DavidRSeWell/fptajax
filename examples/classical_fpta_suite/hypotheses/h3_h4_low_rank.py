"""H3 / H4 — low-rank optimality + spectrum decay.

For each dataset, pick the *best* basis (largest available) and sweep the
truncation rank ``k`` from 1 to floor(m/2). Record train/test MSE at each
truncation level. Plot:

  - test MSE vs k (Eckart-Young: monotone non-increasing)
  - cumulative ω²ₖ / Σ ω² fraction (importance decay, H4)
  - random-rank-k baseline: pick a random skew-symmetric rank-2k coefficient
    matrix scaled to match Frobenius norm and compare its test MSE — the
    Schur truncation should beat it consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.protocol import (  # noqa: F401
    MAX_M_FOR_FIT,
    fit_eval, fit_skew_C_train, orthonormalise, predict, pair_mse, truncate_C,
)
from examples.classical_fpta_suite.data.blotto import build_blotto
from examples.classical_fpta_suite.data.kuhn import build_kuhn
from examples.classical_fpta_suite.data.rps import build_rps
from examples.classical_fpta_suite.data.tennis import build_tennis


_OUT = Path(__file__).resolve().parent.parent / "results"
_OUT.mkdir(exist_ok=True)


def _random_skew(m: int, target_fro: float, rng: np.random.RandomState) -> np.ndarray:
    A = rng.randn(m, m)
    S = 0.5 * (A - A.T)
    fro = np.linalg.norm(S)
    return S * (target_fro / max(fro, 1e-12))


def _random_rank_2k_skew(m: int, k: int, target_fro: float, rng) -> np.ndarray:
    """Random skew-symmetric matrix with rank exactly 2k."""
    if k <= 0:
        return np.zeros((m, m))
    Q, _ = np.linalg.qr(rng.randn(m, m))
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    U = np.zeros((m, m))
    eigs = np.abs(rng.randn(k))
    for j in range(k):
        U[2 * j:2 * j + 2, 2 * j:2 * j + 2] = eigs[j] * R
    S = Q @ U @ Q.T
    fro = np.linalg.norm(S)
    return S * (target_fro / max(fro, 1e-12))


def run_dataset(ds, label: str, n_random_baselines: int = 5) -> dict:
    # Pick richest basis available (highest m on hand)
    bn = max(ds.available_bases, key=lambda n: ds.basis(n, MAX_M_FOR_FIT).shape[1])
    B_raw = ds.basis(bn, MAX_M_FOR_FIT)
    B = orthonormalise(B_raw)
    m = B.shape[1]
    C_full = fit_skew_C_train(B, ds.F, ds.train_pairs, ridge=1e-4)
    F_hat_full = predict(B, C_full)

    from fptajax.decomposition import skew_symmetric_schur
    sch = skew_symmetric_schur(np.asarray(C_full, dtype=np.float64))
    omegas = np.asarray(sch.eigenvalues)
    nc_full = int(sch.n_components)
    fro_C = float(np.linalg.norm(C_full))

    # Sweep truncation rank
    rows = []
    rng = np.random.RandomState(0)
    for k in range(0, nc_full + 1):
        C_k, _, _ = truncate_C(C_full, k)
        F_hat_k = predict(B, C_k)
        train = pair_mse(ds.F, F_hat_k, ds.train_pairs)
        test  = pair_mse(ds.F, F_hat_k, ds.test_pairs)
        cum_omega2 = float(np.sum(omegas[:k] ** 2) / max(np.sum(omegas ** 2), 1e-12))

        # Random rank-2k baseline (mean test MSE over n_random_baselines)
        rand_mses = []
        for _ in range(n_random_baselines):
            C_rand = _random_rank_2k_skew(m, k, fro_C, rng)
            F_rand = predict(B, C_rand)
            rand_mses.append(pair_mse(ds.F, F_rand, ds.test_pairs))
        rand_test = float(np.mean(rand_mses)) if rand_mses else float("nan")

        rows.append({
            "dataset": label, "basis": bn, "m": m, "k": k,
            "train_mse": train, "test_mse": test,
            "norm_test": test / max(ds.f_norm_sq, 1e-12),
            "cum_omega2_frac": cum_omega2,
            "rand_test_mean": rand_test,
            "rand_norm_test_mean": rand_test / max(ds.f_norm_sq, 1e-12),
        })
    return {"label": label, "basis": bn, "m": m, "omegas": omegas.tolist(),
            "rows": rows}


def plot_dataset(res: dict, out_path: Path):
    rows = res["rows"]
    Ks = np.array([r["k"] for r in rows])
    test = np.array([r["norm_test"] for r in rows])
    train = np.array([r["train_mse"] / max(r.get("test_mse", 1e-12), 1e-12) * r["norm_test"]
                      if r["test_mse"] > 0 else 0.0
                      for r in rows])
    rand = np.array([r["rand_norm_test_mean"] for r in rows])
    cum = np.array([r["cum_omega2_frac"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
    axes[0].plot(Ks, test, "-o", color="crimson", label="Schur truncation (test)")
    axes[0].plot(Ks, rand, "--s", color="gray", label="random rank-2k (test, mean)")
    axes[0].axhline(1.0, color="gray", lw=0.6, ls=":")
    axes[0].set_xlabel("truncation rank k (disc games kept)")
    axes[0].set_ylabel("normalised test MSE")
    axes[0].set_title(f"{res['label']} — H3 truncation curve\nbasis={res['basis']}, m={res['m']}")
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(Ks, cum, "-o", color="navy")
    axes[1].axhline(0.9, color="gray", lw=0.5, ls="--", label="90% explained")
    axes[1].axhline(0.99, color="gray", lw=0.5, ls=":",  label="99% explained")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("cumulative ω² fraction")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"{res['label']} — H4 spectrum decay")
    axes[1].grid(True, alpha=0.3); axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    datasets = [
        ("blotto_small",      build_blotto("small")),
        ("blotto_medium",     build_blotto("medium")),
        ("blotto_large",      build_blotto("large")),
        ("kuhn_random",       build_kuhn("random",       N=150)),
        ("rps",               build_rps()),
        ("tennis",            build_tennis()),
    ]
    out_all = []
    print(f"{'dataset':18s} {'basis':24s} {'m':>4s} {'k':>3s} {'norm_test':>10s} "
          f"{'cum ω² frac':>12s}  random-rk")
    for label, ds in datasets:
        res = run_dataset(ds, label, n_random_baselines=5)
        out_all.append(res)
        for r in res["rows"]:
            print(f"  {label:18s} {res['basis']:24s} {res['m']:4d} {r['k']:3d} "
                  f"{r['norm_test']:10.5f} {r['cum_omega2_frac']:12.5f}  "
                  f"{r['rand_norm_test_mean']:8.5f}")
        plot_dataset(res, _OUT / f"h3_low_rank_{label}.png")

    with open(_OUT / "h3_low_rank.json", "w") as f:
        json.dump(out_all, f, indent=2)
    print(f"\nSaved {_OUT/'h3_low_rank.json'} and per-dataset PNGs.")


if __name__ == "__main__":
    main()
