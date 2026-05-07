"""Aggregate multi-seed benchmark + trait-recovery JSONs.

Reads all per-seed JSON outputs from a sweep, prints mean ± std tables for:

  1. Predictive performance (norm_test): classical FPTA at each (degree, k)
     setting, plus behavioural FPTA (best-step, single number per seed).
  2. Trait-recovery joint R^2: per-trait OLS + LASSO recovery for both BFPTA
     and PTA disc games.

Saves a one-page summary PNG plus an aggregated JSON.

Usage:
    python -m examples.iblotto.aggregate_seed_sweep \
        --benchmark_glob "results/benchmark_main_v1_seed*.json" \
        --bfpta_recovery_glob "results/trait_recovery_main_v1_seed*/trait_recovery.json" \
        --pta_recovery_json results/pta_trait_recovery_main_v1_seed0/pta_trait_recovery.json \
        --out_dir results/seed_sweep_summary --tag main_v1
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_TRAIT_NAMES = ("learning_rate", "win_reinvestment", "loss_disinvestment",
                "opponent_allocation", "innovation_noise")


def _stat(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std())


def aggregate_benchmark(json_paths: list[Path]) -> dict:
    """Mean ± std across seeds for each (method, basis, m, k) row."""
    by_key = defaultdict(list)
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        for r in d.get("classical_rows", []):
            key = ("classical", r["basis"], r["m"], str(r["k_trunc"]))
            by_key[key].append(r["norm_test"])
        b = d.get("behavioural")
        if b is not None:
            if "norm_test" in b:
                v = b["norm_test"]
            else:
                v = b.get("norm_test_best", b.get("norm_test_final"))
            by_key[("behavioural", "hier skill+disc",
                    b["m"], str(b["k_trunc"]))].append(v)
    rows = []
    for (method, basis, m, k), vals in by_key.items():
        mean, std = _stat(vals)
        rows.append(dict(
            method=method, basis=basis, m=int(m), k_trunc=k,
            n_seeds=len(vals),
            norm_test_mean=mean, norm_test_std=std,
        ))
    rows.sort(key=lambda r: r["norm_test_mean"])
    return rows


def aggregate_recovery(bfpta_paths: list[Path]) -> dict:
    """Mean ± std across seeds for per-trait OLS+LASSO joint R²."""
    ols_by_trait = defaultdict(list)
    lasso_by_trait = defaultdict(list)
    for p in bfpta_paths:
        with open(p) as f:
            d = json.load(f)
        for j, name in enumerate(d.get("trait_names", _TRAIT_NAMES)):
            ols_by_trait[name].append(d["R2_joint_ols"][j])
            lasso_by_trait[name].append(d["R2_joint_lasso"][j])
    out = {}
    for name in _TRAIT_NAMES:
        m_o, s_o = _stat(ols_by_trait[name])
        m_l, s_l = _stat(lasso_by_trait[name])
        out[name] = dict(
            ols_mean=m_o, ols_std=s_o,
            lasso_mean=m_l, lasso_std=s_l,
            n_seeds=len(ols_by_trait[name]),
        )
    return out


def load_pta_recovery(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for j, name in enumerate(d.get("trait_names", _TRAIT_NAMES)):
        out[name] = dict(
            ols=d["R2_joint_ols"][j],
            lasso=d["R2_joint_lasso"][j],
        )
    return out


def main(benchmark_glob: str, bfpta_recovery_glob: str,
         pta_recovery_json: Path, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_paths = [Path(p) for p in sorted(glob.glob(benchmark_glob))]
    bfpta_paths     = [Path(p) for p in sorted(glob.glob(bfpta_recovery_glob))]
    if not benchmark_paths:
        raise SystemExit(f"No benchmark JSONs match {benchmark_glob}")
    if not bfpta_paths:
        raise SystemExit(f"No trait-recovery JSONs match {bfpta_recovery_glob}")

    print(f"Aggregating {len(benchmark_paths)} benchmark seeds and "
          f"{len(bfpta_paths)} trait-recovery seeds")

    bench_rows  = aggregate_benchmark(benchmark_paths)
    bfpta_recov = aggregate_recovery(bfpta_paths)
    pta_recov   = load_pta_recovery(pta_recovery_json)

    # ----- Print headline performance table -----
    print(f"\n=== Predictive performance (norm_test, mean ± std over "
          f"{len(benchmark_paths)} seeds) ===\n")
    print(f"  {'method':>13s}  {'basis':>22s}  {'m':>4s}  {'k':>5s}  "
          f"{'norm_test':>16s}  {'n_seeds':>8s}")
    print("  " + "-" * 80)
    for r in bench_rows:
        nt = (f"{r['norm_test_mean']:.4f} ± {r['norm_test_std']:.4f}"
              if r['norm_test_std'] > 0 else f"{r['norm_test_mean']:.4f}")
        print(f"  {r['method']:>13s}  {r['basis']:>22s}  {r['m']:>4d}  "
              f"{r['k_trunc']:>5s}  {nt:>16s}  {r['n_seeds']:>8d}")

    # ----- Print trait recovery table -----
    print(f"\n=== Trait recovery joint R² (mean ± std over "
          f"{len(bfpta_paths)} seeds) ===\n")
    print(f"  {'trait':>20s}  {'BFPTA OLS':>18s}  {'PTA OLS':>10s}  "
          f"{'BFPTA LASSO':>18s}  {'PTA LASSO':>10s}")
    print("  " + "-" * 88)
    for name in _TRAIT_NAMES:
        b = bfpta_recov[name]
        p = pta_recov.get(name, {})
        bfpta_ols = (f"{b['ols_mean']:.3f} ± {b['ols_std']:.3f}"
                     if b['ols_std'] > 0 else f"{b['ols_mean']:.3f}")
        bfpta_las = (f"{b['lasso_mean']:.3f} ± {b['lasso_std']:.3f}"
                     if b['lasso_std'] > 0 else f"{b['lasso_mean']:.3f}")
        pta_ols = f"{p.get('ols', float('nan')):.3f}" if p else "—"
        pta_las = f"{p.get('lasso', float('nan')):.3f}" if p else "—"
        print(f"  {name:>20s}  {bfpta_ols:>18s}  {pta_ols:>10s}  "
              f"{bfpta_las:>18s}  {pta_las:>10s}")

    # ----- Plot 1: predictive performance -----
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(bench_rows))), dpi=120)
    means = np.array([r["norm_test_mean"] for r in bench_rows])
    stds  = np.array([r["norm_test_std"]  for r in bench_rows])
    labels = [f"{r['basis']} k={r['k_trunc']}"
              if r['method'] == 'classical' else "behavioural (best step)"
              for r in bench_rows]
    colours = ["#3a7ca5" if r["method"] == "classical" else "#c0504d"
               for r in bench_rows]
    y = np.arange(len(bench_rows))
    ax.barh(y, means, xerr=stds, color=colours, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color="gray", lw=0.7, ls=":", label="null")
    ax.set_xlim(0, max(1.05, means.max() + stds.max() + 0.05))
    ax.set_xlabel("normalised test MSE  (lower = better)")
    ax.set_title(f"iblotto FPTA — n_seeds={len(benchmark_paths)} ({tag})")
    from matplotlib.patches import Patch
    legend = [Patch(color="#3a7ca5", label="classical"),
              Patch(color="#c0504d", label="behavioural")]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", bbox_inches="tight")
    plt.close(fig)

    # ----- Plot 2: trait recovery (BFPTA vs PTA) -----
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    xs = np.arange(len(_TRAIT_NAMES))
    bfpta_means = np.array([bfpta_recov[n]["ols_mean"] for n in _TRAIT_NAMES])
    bfpta_stds  = np.array([bfpta_recov[n]["ols_std"]  for n in _TRAIT_NAMES])
    pta_vals    = np.array([pta_recov.get(n, {}).get("ols", np.nan)
                            for n in _TRAIT_NAMES])
    width = 0.38
    ax.bar(xs - width / 2, bfpta_means, width=width, yerr=bfpta_stds,
           label=f"BFPTA OLS (n={len(bfpta_paths)} seeds)",
           color="firebrick", alpha=0.85)
    ax.bar(xs + width / 2, pta_vals, width=width,
           label="PTA OLS (single seed, dense F)",
           color="navy", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(_TRAIT_NAMES, rotation=20, fontsize=8)
    ax.set_ylabel("joint $R^2$")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Trait-recovery joint R² — BFPTA across seeds vs PTA on dense F")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trait_recovery.png", bbox_inches="tight")
    plt.close(fig)

    # ----- Save aggregated JSON -----
    out_json = out_dir / "aggregate.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            tag=tag,
            n_seeds_benchmark=len(benchmark_paths),
            n_seeds_recovery=len(bfpta_paths),
            performance=bench_rows,
            bfpta_recovery=bfpta_recov,
            pta_recovery=pta_recov,
        ), f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"Saved {out_dir / 'performance.png'}")
    print(f"Saved {out_dir / 'trait_recovery.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_glob", required=True,
                   help="glob pattern for per-seed benchmark JSONs")
    p.add_argument("--bfpta_recovery_glob", required=True,
                   help="glob pattern for per-seed trait_recovery.json files")
    p.add_argument("--pta_recovery_json", type=Path, required=True,
                   help="single PTA trait-recovery JSON (same across seeds)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--tag", default="seed_sweep")
    args = p.parse_args()
    main(args.benchmark_glob, args.bfpta_recovery_glob,
         args.pta_recovery_json, args.out_dir, args.tag)
