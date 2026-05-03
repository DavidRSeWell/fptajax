"""Tennis dataset for the classical FPTA suite (hand-crafted features).

Reuses the MCP CSV files at ``/tmp/mcp/``. Per-player aggregate features
(no shot-level parsing — this is the *classical*, not behavioral, pipeline):

  - serve_win_pct      : own service points won / own service points
  - return_win_pct     : non-service points won / non-service points
  - hold_pct           : service-game-equivalents won (rough proxy)
  - log_n_points       : log10(total points played) — exposure normaliser
  - hard_win_pct       : pts won on hard surface / pts on hard
  - clay_win_pct       : pts won on clay / pts on clay
  - grass_win_pct      : pts won on grass / pts on grass

Feature vector is 7-dim (after we drop NaNs / surface gaps). Bases:

  - "raw"            : 7-d  (the features themselves)
  - "raw_const"      : 8-d  (raw + constant)
  - "deg2"           : 36-d (1 ∪ raw ∪ pairwise products)
  - "pca5", "pca10"  : low-rank PCA projections of "deg2" (for H7 same-span check)

F is built from pair-level point-share differential, exactly like
``examples/tennis/loader.py`` does — we reuse that pipeline.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.classical_fpta_suite.protocol import FPTADataset, random_pair_split


_MCP_DIR = Path("/tmp/mcp")
_MATCHES_CSV = _MCP_DIR / "charting-m-matches.csv"
_POINTS_CSVS = [
    _MCP_DIR / "charting-m-points-2020s.csv",
    _MCP_DIR / "charting-m-points-2010s.csv",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_match_meta() -> dict:
    out = {}
    with open(_MATCHES_CSV) as f:
        for row in csv.DictReader(f):
            mid = row.get("match_id", "").strip()
            if not mid:
                continue
            out[mid] = dict(
                player1=row.get("Player 1", "").strip(),
                player2=row.get("Player 2", "").strip(),
                surface=row.get("Surface", "").strip(),
            )
    return out


def _load_points_grouped() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in _POINTS_CSVS:
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                mid = row.get("match_id", "").strip()
                if mid:
                    groups[mid].append(row)
    return groups


# ---------------------------------------------------------------------------
# Per-player aggregation
# ---------------------------------------------------------------------------


def _accumulate_stats(min_matches: int = 30, max_players: int | None = 40):
    """Aggregate per-player stats across all usable matches.

    Returns:
        ``(player_names, feats, F, pair_counts)`` where ``feats`` is the (N, 7)
        feature matrix, ``F`` the (N, N) skew-symmetric pair-level point-share
        differential, ``pair_counts`` counts of charted matchups.
    """
    meta = _load_match_meta()
    points = _load_points_grouped()
    usable = sorted(set(meta) & set(points))

    # Count matches per player to filter
    match_count: dict[str, int] = defaultdict(int)
    for mid in usable:
        m = meta[mid]
        if m["player1"]:
            match_count[m["player1"]] += 1
        if m["player2"]:
            match_count[m["player2"]] += 1
    eligible = sorted(
        [p for p, c in match_count.items() if c >= min_matches],
        key=lambda p: (-match_count[p], p),
    )
    if max_players is not None:
        eligible = eligible[:max_players]
    name_to_idx = {n: i for i, n in enumerate(eligible)}
    eligible_set = set(eligible)
    N = len(eligible)

    # Aggregators per player
    n_pts = np.zeros(N)
    n_won = np.zeros(N)
    n_serve_pts = np.zeros(N)
    n_serve_won = np.zeros(N)
    n_return_pts = np.zeros(N)
    n_return_won = np.zeros(N)
    by_surface_pts = {s: np.zeros(N) for s in ("Hard", "Clay", "Grass")}
    by_surface_won = {s: np.zeros(N) for s in ("Hard", "Clay", "Grass")}

    # F building (same convention as examples/tennis/loader.py)
    pair_pts_total = np.zeros((N, N))
    pair_pts_scored = np.zeros((N, N))     # [i, j] = pts i won vs j
    pair_counts = np.zeros((N, N), dtype=np.int64)

    for mid in usable:
        m = meta[mid]
        p1, p2, surf = m["player1"], m["player2"], m["surface"]
        if p1 not in eligible_set or p2 not in eligible_set:
            continue
        i, j = name_to_idx[p1], name_to_idx[p2]
        rows = points[mid]
        n_pts_total = 0
        n_p1 = 0
        for r in rows:
            try:
                svr = int(r.get("Svr", 0))
                w = int(r.get("PtWinner", 0))
            except ValueError:
                continue
            if w not in (1, 2):
                continue
            n_pts_total += 1
            won_by_p1 = (w == 1)
            served_by_p1 = (svr == 1)
            if won_by_p1:
                n_p1 += 1

            for who, idx, is_them in [(1, i, won_by_p1), (2, j, not won_by_p1)]:
                n_pts[idx] += 1
                if is_them:
                    n_won[idx] += 1
                if (who == 1) == served_by_p1:
                    # this player served on this point
                    n_serve_pts[idx] += 1
                    if is_them:
                        n_serve_won[idx] += 1
                else:
                    n_return_pts[idx] += 1
                    if is_them:
                        n_return_won[idx] += 1
                if surf in by_surface_pts:
                    by_surface_pts[surf][idx] += 1
                    if is_them:
                        by_surface_won[surf][idx] += 1

        if n_pts_total > 0:
            pair_pts_total[i, j] += n_pts_total
            pair_pts_total[j, i] += n_pts_total
            pair_pts_scored[i, j] += n_p1
            pair_pts_scored[j, i] += (n_pts_total - n_p1)
            pair_counts[i, j] += 1
            pair_counts[j, i] += 1

    # Build feature matrix
    def _safe_div(a, b):
        out = np.where(b > 0, a / np.maximum(b, 1), 0.0)
        return out

    serve_win = _safe_div(n_serve_won, n_serve_pts)
    return_win = _safe_div(n_return_won, n_return_pts)
    hold_pct  = serve_win                 # rough proxy under iid points
    log_n_pts = np.log10(np.maximum(n_pts, 1.0))
    hard_win  = _safe_div(by_surface_won["Hard"],  by_surface_pts["Hard"])
    clay_win  = _safe_div(by_surface_won["Clay"],  by_surface_pts["Clay"])
    grass_win = _safe_div(by_surface_won["Grass"], by_surface_pts["Grass"])

    feats = np.stack([
        serve_win, return_win, hold_pct,
        log_n_pts / log_n_pts.max() if log_n_pts.max() > 0 else log_n_pts,
        hard_win, clay_win, grass_win,
    ], axis=1).astype(np.float64)

    # F[i, j] = (pts_i - pts_j) / total
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.where(pair_pts_total > 0, pair_pts_total, 1.0)
        F_full = ((2.0 * pair_pts_scored - pair_pts_total) / denom).astype(np.float64)
        F = np.where(pair_pts_total > 0, F_full, 0.0)
    F = 0.5 * (F - F.T)

    return eligible, feats, F, pair_counts


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


def _basis_raw(feats: np.ndarray) -> np.ndarray:
    return feats.copy()


def _basis_raw_const(feats: np.ndarray) -> np.ndarray:
    N = feats.shape[0]
    return np.concatenate([np.ones((N, 1)), feats], axis=1)


def _basis_deg2(feats: np.ndarray) -> np.ndarray:
    N, T = feats.shape
    cols = [np.ones(N)]
    cols.extend(feats[:, k] for k in range(T))
    for i in range(T):
        for j in range(i, T):
            cols.append(feats[:, i] * feats[:, j])
    return np.stack(cols, axis=1)


def _basis_pca(feats: np.ndarray, k: int) -> np.ndarray:
    """PCA projection of the deg-2 basis to k dims (centered, no whitening).

    Same span structure as ``deg2`` if ``k == ndim``; smaller k is a strict
    subspace which is useful for H7 / H1 truncation studies.
    """
    deg2 = _basis_deg2(feats)
    centered = deg2 - deg2.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:k].T


# ---------------------------------------------------------------------------
# Public dataset constructor
# ---------------------------------------------------------------------------


def build_tennis(
    min_matches: int = 30, max_players: int = 40, frac_train: float = 0.8, seed: int = 0,
) -> FPTADataset:
    names, feats, F, pair_counts = _accumulate_stats(min_matches, max_players)
    N = F.shape[0]
    train_pairs, test_pairs = random_pair_split(N, frac_train=frac_train, seed=seed)

    available = ("raw", "raw_const", "deg2", "pca5", "pca10")

    def _basis(name: str, m_target: int) -> np.ndarray:
        if name == "raw":
            B_ = _basis_raw(feats)
        elif name == "raw_const":
            B_ = _basis_raw_const(feats)
        elif name == "deg2":
            B_ = _basis_deg2(feats)
        elif name == "pca5":
            B_ = _basis_pca(feats, k=5)
        elif name == "pca10":
            B_ = _basis_pca(feats, k=10)
        else:
            raise KeyError(f"Unknown basis: {name}")
        if 0 < m_target < B_.shape[1]:
            B_ = B_[:, :m_target]
        return B_

    return FPTADataset(
        name="tennis",
        traits=feats,
        F=F,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        available_bases=available,
        basis_fn=_basis,
        eval_f=None,
        metadata=dict(player_names=names, pair_counts=pair_counts,
                      feature_names=("serve_win", "return_win", "hold_pct",
                                     "log_n_pts_norm", "hard_win", "clay_win",
                                     "grass_win")),
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ds = build_tennis()
    print(
        f"  {ds.name:8s}  N={ds.N}  ||F||²/N²={ds.f_norm_sq:.4f}  "
        f"skew err={np.max(np.abs(ds.F + ds.F.T)):.2e}  "
        f"train/test={len(ds.train_pairs)}/{len(ds.test_pairs)}"
    )
    print("  feature names:", ds.metadata["feature_names"])
    print("  feats range:", ds.traits.min(), ds.traits.max())
    for bn in ds.available_bases:
        B_ = ds.basis(bn, 1000)
        print(f"      basis {bn:14s}: shape={B_.shape}")
