"""H13 / H15 / H16 — eigengap stability + sample-rate scaling + obs-per-pair.

Three sub-experiments per dataset:

  H13 (eigengap stability) — Theorem 4.5: per-disc embedding perturbation
      ≤ ‖E‖₂ / Δ_k. We bootstrap the agent set, re-fit FPTA, and measure how
      much each disc-game embedding shifts. The shift should rank-correlate
      with 1/Δ_k (wider gap → smaller shift).

  H15 (sample-rate decay) — Var[E_X] = O(1/|ε| + |ε~ε|/|ε|²). For complete
      tournaments the rate is O(1/n). We sub-sample the agent set at sizes
      n ∈ {10, 20, 40, ...} and plot ‖C̃ − C_full‖ vs n on log-log axes;
      slope should be near −1 for complete tournaments.

  H16 (obs-per-pair, win-prob estimator) — only meaningful for datasets
      with stochastic outcomes. We synthesise i.i.d. observations of each
      pair's payoff (binarised to win/loss with prob (1+f)/2), re-estimate
      F at varying observations-per-pair K, and fit FPTA. Plot ‖C̃ − C‖ vs K;
      Hoeffding predicts exponential decay.

We run H13 + H15 on Blotto medium (clean, deterministic) and H16 on Blotto
medium (we synthesise observations from the deterministic f).
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fit_C(traits: np.ndarray, F: np.ndarray, max_deg: int = 3,
           ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Fit C on the FULL pair set (no train/test split — for stability we want
    the cleanest possible estimate)."""
    B_raw, _ = basis_monomials(traits, max_deg)
    B = orthonormalise(B_raw)
    N = traits.shape[0]
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i != j])
    C = fit_skew_C_train(B, F, pairs, ridge=ridge)
    return C, B


def _embeddings_from_C(B: np.ndarray, C: np.ndarray, k_keep: int):
    _, omegas, Q = truncate_C(C, k_keep)
    N = B.shape[0]
    Y = np.zeros((N, k_keep, 2))
    for j in range(k_keep):
        s = float(np.sqrt(max(omegas[j], 0.0)))
        Y[:, j, 0] = s * (B @ Q[:, 2 * j])
        Y[:, j, 1] = s * (B @ Q[:, 2 * j + 1])
    return Y, omegas[:k_keep]


def _procrustes_align(Y_a: np.ndarray, Y_b: np.ndarray) -> float:
    """2-D Procrustes per disc; return the per-disc residual vector."""
    K = Y_a.shape[1]
    out = np.zeros(K)
    for j in range(K):
        H = Y_b[:, j].T @ Y_a[:, j]
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = U @ Vt
        out[j] = float(np.linalg.norm(Y_a[:, j] - Y_b[:, j] @ R.T)
                       / max(np.linalg.norm(Y_a[:, j]), 1e-12))
    return out


# ---------------------------------------------------------------------------
# H13 — eigengap stability via bootstrap
# ---------------------------------------------------------------------------


def h13_bootstrap(traits: np.ndarray, F: np.ndarray,
                  n_boot: int = 60, frac: float = 0.8, k_keep: int = 4,
                  seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    N = traits.shape[0]
    C0, B0 = _fit_C(traits, F)
    Y0, om0 = _embeddings_from_C(B0, C0, k_keep)
    eigengaps = np.zeros(k_keep)
    om_pad = np.concatenate([[np.inf], om0, [0.0]])
    for j in range(k_keep):
        eigengaps[j] = min(om_pad[j] - om_pad[j + 1], om_pad[j + 1] - om_pad[j + 2])

    residuals = np.zeros((n_boot, k_keep))
    for b in range(n_boot):
        idx = rng.choice(N, size=int(frac * N), replace=False)
        sub_t = traits[idx]; sub_F = F[np.ix_(idx, idx)]
        try:
            C_b, B_b = _fit_C(sub_t, sub_F)
            Y_b, _ = _embeddings_from_C(B_b, C_b, k_keep)
            # Compare on the shared agent index set
            common = np.array([i for i in range(N) if i in idx])
            Y_a = Y0[common]
            # Y_b is computed on sub-N rows; align by mapping idx→sub-row position
            sub_pos = {a: p for p, a in enumerate(idx)}
            Y_b_aligned = Y_b[[sub_pos[a] for a in common]]
            residuals[b] = _procrustes_align(Y_a, Y_b_aligned)
        except Exception:
            residuals[b] = np.nan
    return dict(
        eigenvalues=om0.tolist(),
        eigengaps=eigengaps.tolist(),
        residuals_mean=np.nanmean(residuals, axis=0).tolist(),
        residuals_std=np.nanstd(residuals, axis=0).tolist(),
    )


def plot_h13(res: dict, out_path: Path):
    K = len(res["eigenvalues"])
    om = np.array(res["eigenvalues"])
    gaps = np.array(res["eigengaps"])
    res_mean = np.array(res["residuals_mean"])
    res_std  = np.array(res["residuals_std"])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.errorbar(np.arange(K), res_mean, yerr=res_std, fmt="o-",
                color="crimson", capsize=4, label="bootstrap residual ± std")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"k={k+1}\n(ω={om[k]:.3f})" for k in range(K)])
    ax2 = ax.twinx()
    ax2.plot(np.arange(K), 1.0 / np.maximum(gaps, 1e-12), "s--", color="navy",
             label="1 / eigengap Δ_k", alpha=0.7)
    ax2.set_ylabel(r"$1/\Delta_k$", color="navy"); ax2.tick_params(axis="y", labelcolor="navy")
    ax.set_ylabel("Procrustes residual (lower = more stable)")
    ax.set_title("H13 — embedding stability vs 1/eigengap\n(if H13 holds, both should track)")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# H15 — sample-rate scaling via sub-sampling agents
# ---------------------------------------------------------------------------


def h15_sample_rate(traits: np.ndarray, F: np.ndarray,
                    sample_sizes: list[int], n_reps: int = 8,
                    seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    N = traits.shape[0]
    C0, _ = _fit_C(traits, F)
    rows = []
    for n in sample_sizes:
        if n > N:
            continue
        diffs = []
        for r in range(n_reps):
            idx = rng.choice(N, size=n, replace=False)
            try:
                C_n, _ = _fit_C(traits[idx], F[np.ix_(idx, idx)])
                # Compare on a shared (Frobenius) basis: |‖C0‖ − ‖C_n‖|
                # We can't compare entrywise (different m), so we compare
                # via the Frobenius norm of the *embedding* of the shared agents.
                # Simpler: compare sorted spectra of the two coefficient matrices
                # — both invariant under rotations.
                from fptajax.decomposition import skew_symmetric_schur
                eig0 = np.asarray(skew_symmetric_schur(C0).eigenvalues)
                eig_n = np.asarray(skew_symmetric_schur(C_n).eigenvalues)
                k = min(len(eig0), len(eig_n))
                diffs.append(float(np.linalg.norm(eig0[:k] - eig_n[:k])))
            except Exception:
                continue
        rows.append(dict(n=n, n_reps=len(diffs),
                         mean_diff=float(np.mean(diffs)) if diffs else float("nan"),
                         std_diff=float(np.std(diffs)) if diffs else float("nan")))
    return dict(rows=rows)


def plot_h15(res: dict, out_path: Path):
    ns = np.array([r["n"] for r in res["rows"]])
    means = np.array([r["mean_diff"] for r in res["rows"]])
    stds = np.array([r["std_diff"] for r in res["rows"]])
    if len(ns) < 2:
        return
    log_n, log_m = np.log(ns), np.log(np.maximum(means, 1e-12))
    slope, intercept = np.polyfit(log_n, log_m, 1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.errorbar(ns, means, yerr=stds, fmt="o-", color="crimson", capsize=4)
    ax.plot(ns, np.exp(intercept) * ns ** slope, "k--", lw=0.7,
            label=f"slope = {slope:.2f}  (theory: −1 for complete tournaments)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# agents sampled (n)"); ax.set_ylabel("‖σ(C₀) − σ(C_n)‖")
    ax.set_title("H15 — sample-rate decay")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# H16 — observations-per-pair scaling (synthetic)
# ---------------------------------------------------------------------------


def h16_obs_per_pair(traits: np.ndarray, F_true: np.ndarray,
                     K_list: list[int], n_reps: int = 5, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    C_true, _ = _fit_C(traits, F_true)
    rows = []
    N = F_true.shape[0]
    for K in K_list:
        diffs = []
        for r in range(n_reps):
            # Synthesise K Bernoulli(0.5*(1+f)) win observations per pair, then
            # reconstruct F̂ as 2*win_freq − 1.
            p = np.clip(0.5 * (1.0 + F_true), 0.0, 1.0)
            wins = rng.binomial(K, p) / K
            F_hat = 2.0 * wins - 1.0
            np.fill_diagonal(F_hat, 0.0)
            F_hat = 0.5 * (F_hat - F_hat.T)
            try:
                C_n, _ = _fit_C(traits, F_hat)
                from fptajax.decomposition import skew_symmetric_schur
                eig0 = np.asarray(skew_symmetric_schur(C_true).eigenvalues)
                eig_n = np.asarray(skew_symmetric_schur(C_n).eigenvalues)
                k = min(len(eig0), len(eig_n))
                diffs.append(float(np.linalg.norm(eig0[:k] - eig_n[:k])))
            except Exception:
                continue
        rows.append(dict(K=K, mean_diff=float(np.mean(diffs)),
                         std_diff=float(np.std(diffs))))
    return dict(rows=rows)


def plot_h16(res: dict, out_path: Path):
    Ks = np.array([r["K"] for r in res["rows"]])
    means = np.array([r["mean_diff"] for r in res["rows"]])
    stds = np.array([r["std_diff"] for r in res["rows"]])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.errorbar(Ks, means, yerr=stds, fmt="o-", color="crimson", capsize=4,
                label="observed")
    # Hoeffding shape ~ exp(-c K)/√K → on a log-y plot vs K it should be roughly linear-ish
    if len(Ks) >= 2 and means.min() > 0:
        slope, intercept = np.polyfit(Ks, np.log(np.maximum(means, 1e-12)), 1)
        ax.plot(Ks, np.exp(intercept + slope * Ks), "k--", lw=0.7,
                label=f"exp fit slope = {slope:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("observations per pair (K)")
    ax.set_ylabel("‖σ(C_true) − σ(C_K)‖ (log scale)")
    ax.set_title("H16 — Hoeffding-style decay vs observations/pair")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ds = build_blotto("medium")     # deterministic, K=5, B=20, N=300
    traits = ds.traits; F = ds.F

    print("=" * 60)
    print("H13 bootstrap stability ...")
    h13 = h13_bootstrap(traits, F, n_boot=40, frac=0.8, k_keep=4)
    plot_h13(h13, _OUT / "h13_eigengap_stability.png")
    for k, (om, gap, rm, rs) in enumerate(zip(
        h13["eigenvalues"], h13["eigengaps"], h13["residuals_mean"], h13["residuals_std"],
    )):
        print(f"  k={k+1}  ω={om:.4f}  Δ={gap:.4f}  resid={rm:.3e} ± {rs:.3e}")

    print("\nH15 sample-rate scaling ...")
    sample_sizes = [20, 40, 80, 160, 240]
    h15 = h15_sample_rate(traits, F, sample_sizes, n_reps=6)
    plot_h15(h15, _OUT / "h15_sample_rate.png")
    for r in h15["rows"]:
        print(f"  n={r['n']:3d}  ‖Δσ‖={r['mean_diff']:.3e} ± {r['std_diff']:.3e}")

    print("\nH16 observations per pair ...")
    K_list = [5, 10, 25, 50, 100, 250]
    h16 = h16_obs_per_pair(traits, F, K_list, n_reps=4)
    plot_h16(h16, _OUT / "h16_obs_per_pair.png")
    for r in h16["rows"]:
        print(f"  K={r['K']:4d}  ‖Δσ‖={r['mean_diff']:.3e} ± {r['std_diff']:.3e}")

    out = dict(h13=h13, h15=h15, h16=h16)
    with open(_OUT / "h13_h15_h16_sampling_stability.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
