"""Pretty-print a benchmark JSON file and render a comparison PNG.

Takes one or more ``benchmark_*.json`` files (the output of
``benchmark.py --output_json``) and produces:

  - stdout: a sorted text table of all (method, basis, m, k) rows by
    norm_test (lower = better). If multiple JSON files are passed (e.g.
    multi-seed), aggregates mean ± std across seeds.
  - PNG: ``<json_stem>_summary.png`` (or ``<tag>_seeds.png`` for multi).

Usage:
    # Single seed:
    python -m examples.iblotto.summarize_benchmark \
        --json results/benchmark_main_v1.json

    # Multi-seed aggregate:
    python -m examples.iblotto.summarize_benchmark \
        --json results/benchmark_seed*.json --tag main_v1
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _row_label(r: dict) -> str:
    if r["method"] == "classical":
        return f"{r['basis']} k={r['k_trunc']}"
    return f"{r['method']} ({'final' if r.get('best_step') is None else 'best'})"


def load_runs(json_paths: list[Path]) -> list[dict]:
    """Each JSON is one full run (one seed). Returns a list of dicts."""
    runs = []
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        runs.append(d)
    return runs


def aggregate_rows(runs: list[dict]) -> list[dict]:
    """Combine classical_rows + behavioural across seeds into one list of
    aggregated rows with ``norm_test_mean`` / ``norm_test_std`` per method."""
    by_key: dict[str, list[dict]] = defaultdict(list)

    for run in runs:
        # classical sweep
        for r in run.get("classical_rows", []):
            key = f"classical|{r['basis']}|m{r['m']}|k{r['k_trunc']}"
            by_key[key].append({
                "label": f"{r['basis']} k={r['k_trunc']}",
                "method": "classical",
                "m": r["m"], "k": str(r["k_trunc"]),
                "train": r["train_mse"], "test": r["test_mse"],
                "norm_test": r["norm_test"],
            })

        # behavioural — record both 'final' and 'best' as separate rows
        b = run.get("behavioural")
        if b is not None:
            for kind, key_suffix in (("final", "final"), ("best", "best")):
                key = f"behavioural|{kind}"
                by_key[key].append({
                    "label": f"behavioural ({kind}, step={b.get('best_step', '?') if kind == 'best' else 'last'})",
                    "method": f"behavioural-{kind}",
                    "m": b["m"], "k": str(b["k_trunc"]),
                    "train": b[f"train_mse_{kind}"], "test": b[f"test_mse_{kind}"],
                    "norm_test": b[f"norm_test_{kind}"],
                })

    out = []
    for key, items in by_key.items():
        nt = np.array([it["norm_test"] for it in items])
        tm = np.array([it["test"] for it in items])
        out.append(dict(
            label=items[0]["label"], method=items[0]["method"],
            m=items[0]["m"], k=items[0]["k"],
            n_seeds=len(items),
            test_mean=float(tm.mean()), test_std=float(tm.std()),
            norm_test_mean=float(nt.mean()), norm_test_std=float(nt.std()),
        ))
    out.sort(key=lambda r: r["norm_test_mean"])
    return out


def print_table(rows: list[dict], n_seeds: int):
    print(f"\n{'rank':>4s}  {'method':>20s}  {'m':>4s}  {'k':>5s}  "
          f"{'test MSE':>12s}  {'norm_test':>16s}")
    seed_str = f"(±std over {n_seeds} seeds)" if n_seeds > 1 else ""
    print("-" * 78)
    for i, r in enumerate(rows):
        if r["norm_test_std"] > 0:
            nt = f"{r['norm_test_mean']:.4f} ± {r['norm_test_std']:.4f}"
            tm = f"{r['test_mean']:.1f} ± {r['test_std']:.1f}"
        else:
            nt = f"{r['norm_test_mean']:.4f}"
            tm = f"{r['test_mean']:.1f}"
        print(f"  {i+1:2d}.  {r['method']:>20s}  {r['m']:>4d}  {r['k']:>5s}  "
              f"{tm:>12s}  {nt:>16s}")
    print(f"\n{seed_str}")


def plot_summary(rows: list[dict], out_path: Path, title: str = ""):
    """Bar chart: norm_test for each method, classical-vs-behavioural shaded."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(rows))), dpi=120)
    labels = [r["label"] for r in rows]
    means  = np.array([r["norm_test_mean"] for r in rows])
    stds   = np.array([r["norm_test_std"]  for r in rows])
    colours = ["#3a7ca5" if r["method"] == "classical" else "#c0504d"
               for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=stds, color=colours, alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normalised test MSE  (lower = better)")
    ax.axvline(1.0, color="gray", lw=0.7, ls=":", label="null (predict 0)")
    ax.set_xlim(0, max(1.05, means.max() + (stds.max() if len(stds) else 0) + 0.05))
    ax.set_title(title or "Classical vs behavioural FPTA on iblotto")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#3a7ca5", label="classical"),
        Patch(color="#c0504d", label="behavioural"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", required=True,
                   help="one or more benchmark JSON files (globs OK)")
    p.add_argument("--tag", default=None,
                   help="output PNG tag (default = first JSON's stem)")
    args = p.parse_args()

    paths = []
    for pattern in args.json:
        paths.extend(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No JSON files matched {args.json}")
    paths = sorted(paths)
    print(f"Loaded {len(paths)} run(s):")
    for p in paths:
        print(f"  {p}")

    runs = load_runs(paths)
    rows = aggregate_rows(runs)

    print(f"\n=== Comparison ranked by norm_test_mean ===")
    n_seeds = max((r["n_seeds"] for r in rows), default=1)
    print_table(rows, n_seeds=n_seeds)

    tag = args.tag or paths[0].stem.replace("benchmark_", "")
    out_path = paths[0].parent / f"{tag}_summary.png"
    plot_summary(rows, out_path,
                 title=f"iblotto FPTA benchmark — {tag} "
                       f"(n_seeds={n_seeds})")
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
