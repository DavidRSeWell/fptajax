"""Classical FPTA experiment suite — run-everything driver.

Runs each hypothesis script in turn. Each script writes its own JSON +
per-dataset PNGs into ``results/``. At the end we emit a one-page summary
PNG aggregating the headline numbers per hypothesis.

Usage:
    PYTHONPATH=. .venv/bin/python -m examples.classical_fpta_suite.main
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.classical_fpta_suite.hypotheses import (
    h1_h9_h10_basis_rate,
    h3_h4_low_rank,
    h5_h6_orthogonality,
    h7_h8_basis_indep,
    h11_composition_invariance,
    h12_fpta_vs_pta,
    h13_h15_h16_sampling_stability,
)


_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)


def _section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    t0 = time.time()
    _section("H1 / H9 / H10  —  Basis-rate sweep across datasets")
    h1_h9_h10_basis_rate.main()

    _section("H3 / H4  —  Low-rank truncation + spectrum decay")
    h3_h4_low_rank.main()

    _section("H5 / H6  —  Orthogonality + equal-magnitude per disc")
    h5_h6_orthogonality.main()

    _section("H7 / H8  —  Basis independence + uniqueness up to rotation")
    h7_h8_basis_indep.main()

    _section("H11  —  Composition invariance (Blotto reparameterisation)")
    h11_composition_invariance.main()

    _section("H12  —  FPTA ↔ PTA correspondence (Theorem 4.2 identity)")
    h12_fpta_vs_pta.main()

    _section("H13 / H15 / H16  —  Sampling stability + sample-rate scaling")
    h13_h15_h16_sampling_stability.main()

    _section(f"DONE in {time.time() - t0:.1f}s — see results/ for JSON + PNGs")


if __name__ == "__main__":
    main()
