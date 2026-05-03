"""H1 / H9 / H10 — empirical-risk minimisation, basis sweep, convergence rate.

For every dataset and every registered basis, sweep basis dimension ``m``
through its full range and record train/test MSE. Plot:

  - normalised test MSE vs m, one curve per basis family per dataset
  - same on log-log axes — slope is the empirical convergence rate (H9)

Confirms H1 (ERM works) and lets us read off whether bases matched to the
domain (e.g. Bernstein on Blotto's simplex) outperform mismatched ones (H10).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.protocol import fit_eval, normalised_test_mse, MAX_M_FOR_FIT
from examples.classical_fpta_suite.data.blotto import build_blotto
from examples.classical_fpta_suite.data.kuhn import build_kuhn
from examples.classical_fpta_suite.data.rps import build_rps
from examples.classical_fpta_suite.data.tennis import build_tennis


_OUT = Path(__file__).resolve().parent.parent / "results"
_OUT.mkdir(exist_ok=True)


def m_grid_for(N: int, m_max: int) -> list[int]:
    """Geometric grid 2, 3, 5, 8, 12, ... up to min(m_max, N-1, MAX_M_FOR_FIT)."""
    cap = min(m_max, max(2, N - 2), MAX_M_FOR_FIT)
    grid = sorted({int(round(x)) for x in np.geomspace(2, cap, 8)})
    return [g for g in grid if g <= cap]


def run_dataset(ds, label: str) -> dict:
    rows = []
    for bn in ds.available_bases:
        m_full = ds.basis(bn, MAX_M_FOR_FIT).shape[1]
        for m in m_grid_for(ds.N, m_full):
            try:
                r = fit_eval(ds, bn, m, k_trunc=None, ridge=1e-4)
            except Exception as exc:
                continue
            rows.append({
                "dataset": label, "basis": bn, "m": int(m),
                "train_mse": r["train_mse"], "test_mse": r["test_mse"],
                "norm_train": r["train_mse"] / max(ds.f_norm_sq, 1e-12),
                "norm_test":  r["test_mse"]  / max(ds.f_norm_sq, 1e-12),
                "f_norm_sq":  ds.f_norm_sq,
            })
    return rows


def plot_dataset(rows: list[dict], label: str, out_path: Path) -> None:
    bases = sorted({r["basis"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
    cmap = plt.get_cmap("tab10")
    for k, bn in enumerate(bases):
        sel = [r for r in rows if r["basis"] == bn]
        sel.sort(key=lambda r: r["m"])
        ms = [r["m"] for r in sel]
        nt = [r["norm_test"] for r in sel]
        ntr = [r["norm_train"] for r in sel]
        c = cmap(k % 10)
        axes[0].plot(ms, nt, "-o", color=c, label=f"{bn} (test)")
        axes[0].plot(ms, ntr, "--", color=c, alpha=0.5)
        axes[1].loglog(ms, np.maximum(nt, 1e-8), "-o", color=c, label=bn)
    axes[0].axhline(1.0, color="gray", lw=0.7, ls=":")
    axes[0].set_title(f"{label} — normalised MSE vs basis dim m")
    axes[0].set_xlabel("m"); axes[0].set_ylabel("MSE / ‖f‖²")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(f"{label} — log-log (slope = convergence rate)")
    axes[1].set_xlabel("m"); axes[1].set_ylabel("normalised test MSE")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    datasets = [
        ("blotto_small",      build_blotto("small")),
        ("blotto_medium",     build_blotto("medium")),
        ("blotto_large",      build_blotto("large")),
        ("kuhn_random",       build_kuhn("random",       N=150)),
        ("kuhn_nash_family",  build_kuhn("nash_family",  N=150)),
        ("rps",               build_rps()),
        ("tennis",            build_tennis()),
    ]
    all_rows = []
    print(f"{'dataset':22s} {'basis':24s} {'m':>4s} {'norm_train':>11s} {'norm_test':>11s}")
    for label, ds in datasets:
        rows = run_dataset(ds, label)
        all_rows.extend(rows)
        for r in rows:
            print(f"  {label:22s} {r['basis']:24s} {r['m']:4d} "
                  f"{r['norm_train']:11.5f} {r['norm_test']:11.5f}")
        plot_dataset(rows, label, _OUT / f"h1_basis_rate_{label}.png")

    with open(_OUT / "h1_basis_rate.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {_OUT / 'h1_basis_rate.json'} and 7 PNGs in {_OUT}/")


if __name__ == "__main__":
    main()
