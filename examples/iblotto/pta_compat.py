"""Wrapper around ``fptajax.pta`` adding the stability bound from
``Perform_Principal_Tradeoff_Analysis.m``.

Returns the PTA result plus per-disc-game embedding-error bounds derived
from the per-entry uncertainties in ``F`` via Bandeira–van Handel's
operator-norm expectation bound (2016).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fptajax.pta import pta as _pta_engine


@dataclass
class PTAReport:
    """Result of running PTA on an iterated-Blotto F-matrix."""
    Y: np.ndarray                   # (N, 2K_eff) embedding coordinates
    omegas: np.ndarray              # (N,) raw eigenvalues (paired ±ω)
    importances: np.ndarray         # (K,) ω²-fractions of variance
    cumulative_importances: np.ndarray
    effective_rank: int             # smallest K covering 95% variance
    band_gaps: np.ndarray           # eigengap per disc game
    error_norm_bound: float         # operator-norm bound on E[‖E‖]
    embedding_error_bounds: np.ndarray  # per-disc embedding error bounds


def perform_pta(F: np.ndarray, F_std: np.ndarray, variance_target: float = 0.95) -> PTAReport:
    """Run PTA on ``F`` with per-entry standard errors ``F_std``.

    Args:
        F:        skew-symmetric (N, N) payoff matrix.
        F_std:    per-entry standard errors of ``F``.
        variance_target: fraction of cumulative ω² to capture for the
            effective-rank statistic.
    """
    F = np.asarray(F, dtype=np.float64)
    F_std = np.asarray(F_std, dtype=np.float64)
    N = F.shape[0]

    # PTA via the engine (real-skew Schur on F directly)
    res = _pta_engine(F)
    omegas_paired = np.asarray(res.eigenvalues)        # already non-negative |ω| per disc
    nc = int(res.n_components)

    # Build the same paired-eigenvalue vector the MATLAB returns (length N,
    # values ±ω alternating), purely for reporting compatibility.
    omegas_signed = np.zeros(N)
    for k in range(nc):
        omegas_signed[2 * k]     =  omegas_paired[k]
        omegas_signed[2 * k + 1] = -omegas_paired[k]

    # Importances: ω_k² / Σ ω²
    om2 = omegas_paired ** 2
    importances = om2 / max(np.sum(om2), 1e-12)
    cum = np.cumsum(importances)
    eff_rank = int(np.searchsorted(cum, variance_target) + 1)
    eff_rank = min(eff_rank, nc)

    # Embedding coords: rotate each disc plane so its centroid lies on +x
    Y = np.zeros((N, 2 * eff_rank))
    embeddings = np.asarray(res.embeddings)            # (N, nc, 2)
    for k in range(eff_rank):
        coords = embeddings[:, k, :].copy()
        centroid = coords.mean(axis=0)
        angle = np.arctan2(centroid[1], centroid[0])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, s], [-s, c]])
        Y[:, 2 * k:2 * k + 2] = coords @ rot.T

    # Band-gap calculation (Theorem 4.5 of the FPTA paper, in MATLAB form)
    band_gaps = np.zeros(nc)
    if nc >= 2:
        band_gaps[0] = abs(omegas_paired[0] - omegas_paired[1])
        band_gaps[-1] = (omegas_paired[-2] - omegas_paired[-1]) if nc >= 2 else 0.0
        for k in range(1, nc - 1):
            up = abs(omegas_paired[k - 1] - omegas_paired[k])
            dn = abs(omegas_paired[k] - omegas_paired[k + 1])
            band_gaps[k] = min(up, dn)

    # Bandeira–van Handel operator-norm bound on E[‖E‖]
    failure_prob = 0.1
    sigma_row  = float(np.max(np.sqrt(np.sum(F_std ** 2, axis=0))))
    sigma_col  = float(np.max(np.sqrt(np.sum(F_std ** 2, axis=1))))
    sigma_star = float(np.max(F_std))

    epsilons = np.linspace(0.01, 0.5, 100)
    bounds = (
        (1.0 + epsilons) * (sigma_row + sigma_col)
        + (6.0 / np.sqrt(np.log1p(epsilons))) * sigma_star * np.sqrt(np.log(N))
        + sigma_star * np.sqrt(2.0 * abs(np.log(failure_prob)))
    )
    error_norm_bound = float(np.min(bounds))

    # Per-disc stability ratio + embedding error bound
    safe_gap = np.where(band_gaps > 1e-12, band_gaps, np.inf)
    stability_ratio = np.minimum(2.0 * error_norm_bound / safe_gap, 1.0 - 1e-12)
    embedding_err = np.abs(np.log(1.0 - stability_ratio)) / 2.0

    return PTAReport(
        Y=Y,
        omegas=omegas_signed,
        importances=importances,
        cumulative_importances=cum,
        effective_rank=eff_rank,
        band_gaps=band_gaps,
        error_norm_bound=error_norm_bound,
        embedding_error_bounds=embedding_err,
    )
