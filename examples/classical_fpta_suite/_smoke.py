"""Smoke test: fit/eval one basis on every dataset to verify protocol works."""

from __future__ import annotations

import numpy as np

from examples.classical_fpta_suite.protocol import fit_eval, normalised_test_mse
from examples.classical_fpta_suite.data.blotto import build_blotto
from examples.classical_fpta_suite.data.kuhn import build_kuhn
from examples.classical_fpta_suite.data.rps import build_rps
from examples.classical_fpta_suite.data.tennis import build_tennis


def main():
    datasets = [
        build_blotto("small"),
        build_blotto("medium"),
        build_kuhn("random", N=120),
        build_kuhn("nash_family", N=120),
        build_rps(),
        build_tennis(),
    ]
    print(f"{'dataset':22s} {'basis':24s} {'m':>4s} {'k':>3s}  "
          f"{'train':>9s} {'test':>9s} {'norm_test':>11s}  spectrum (top-4 ω)")
    print("-" * 110)
    for ds in datasets:
        for bn in ds.available_bases[:2]:        # quickly try first two bases
            try:
                m = ds.basis(bn, 1000).shape[1]
                res = fit_eval(ds, bn, m, k_trunc=None, ridge=1e-4)
            except Exception as exc:
                print(f"  {ds.name:22s} {bn:24s} FAILED: {exc!r}")
                continue
            top = np.asarray(res["spectrum"][:4])
            top_str = " ".join(f"{v:7.4f}" for v in top)
            print(
                f"  {ds.name:22s} {bn:24s} {res['m']:4d} {res['k_trunc']:3d}  "
                f"{res['train_mse']:.5f} {res['test_mse']:.5f}  "
                f"{normalised_test_mse(res):.5f}    [{top_str}]"
            )


if __name__ == "__main__":
    main()
