"""Random-Forest baseline for the tennis Behavioral FPTA experiment.

Computes per-player hand-crafted shot statistics from the already-parsed
token tensors, builds pair features, and fits a RandomForestRegressor to
predict F[i, j]. Uses the same train/test pair split as the FPTA model
so the two numbers are directly comparable.

Feature layout reminder (57-dim token vector, from parser.py):
    0..18   shot_type one-hot  (0=serve, then f/b/r/s/v/z/o/p/u/y/l/m/h/i/j/k/t/q)
    19..25  direction one-hot  (0=unknown, 1-3 rally, 4-6 serve)
    26..29  depth one-hot      (0=none, then 7/8/9)
    30..38  error_type one-hot (0=none, ...)
    39..48  ending one-hot     (0=none, 1=ace, 2=unreturnable, 3=winner,
                                4=unforced, 5=forced, 6=fault, 7=double_fault,
                                8=challenge, 9=time_violation)
    49..56  binary: is_serve, is_approach, is_net_pos, is_baseline_pos,
                    is_net_cord, is_drop_volley, is_serve_and_volley,
                    is_self_action

Per-player stats: mean of self-action tokens ++ mean of opp-action tokens,
dropping the is_self_action channel (degenerate by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Channel indices (see parser.shot_to_vector). Kept hard-coded here so this
# module does not need to reparse anything; if parser.py changes the layout,
# update these offsets.
_IS_SELF_CH = 56  # last of the 8 binary flags

# Named channel offsets for interpretable feature naming.
_SHOT_TYPE_NAMES = [
    "serve", "fh", "bh", "fh_slice", "bh_slice",
    "fh_volley", "bh_volley", "fh_overhead", "bh_overhead",
    "fh_drop", "bh_drop", "fh_lob", "bh_lob",
    "fh_half_volley", "bh_half_volley",
    "fh_swing_volley", "bh_swing_volley", "trick", "unknown_shot",
]
_DIRECTION_NAMES = [
    "dir_unknown", "rally_dir_1", "rally_dir_2", "rally_dir_3",
    "serve_wide", "serve_body", "serve_t",
]
_DEPTH_NAMES = ["depth_none", "depth_7", "depth_8", "depth_9"]
_ERROR_NAMES = [
    "err_none", "err_net", "err_wide", "err_deep", "err_both",
    "err_shank", "err_foot", "err_unknown", "err_time_violation",
]
_ENDING_NAMES = [
    "end_none", "ace", "unreturnable", "winner", "unforced",
    "forced", "fault", "double_fault", "challenge", "time_violation",
]
_BINARY_NAMES = [
    "is_serve", "is_approach", "is_net_pos", "is_baseline_pos",
    "is_net_cord", "is_drop_volley", "is_serve_and_volley", "is_self_action",
]


def _channel_names() -> list[str]:
    """Return the 57 per-channel names in order."""
    return (
        _SHOT_TYPE_NAMES
        + _DIRECTION_NAMES
        + _DEPTH_NAMES
        + _ERROR_NAMES
        + _ENDING_NAMES
        + _BINARY_NAMES
    )


@dataclass
class RFBaselineResult:
    train_mse: float
    test_mse: float
    null_train_mse: float
    null_test_mse: float
    feature_names: list[str]
    feature_importance: np.ndarray  # (n_features,)
    model: object  # the fitted sklearn regressor
    # Predictions + labels so downstream plots / ablations don't have to refit.
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    pred_train: np.ndarray = None
    pred_test: np.ndarray = None
    train_pair_tuples: np.ndarray = None
    test_pair_tuples: np.ndarray = None


def compute_player_stats(ds) -> tuple[np.ndarray, list[str]]:
    """Compute per-player hand-crafted stat vectors.

    For each player, averages the self-action token vectors over all their
    shots (across all games), and separately averages the opp-action token
    vectors — this captures what the player does AND how opponents play
    against them. Drops the is_self_action channel (always 1 in self, 0 in
    opp — degenerate).

    Args:
        ds: TennisDataset from loader.build_tennis_dataset.

    Returns:
        stats: (N, 2*56) float array. First 56 cols are self-token means,
            next 56 are opp-token means.
        names: list of 112 feature names.
    """
    games = ds.agent_games           # (N, G, L, 57)
    tmask = ds.agent_token_mask      # (N, G, L)
    is_self = games[..., _IS_SELF_CH] > 0.5  # (N, G, L) bool

    self_mask = tmask & is_self
    opp_mask = tmask & (~is_self)

    # Aggregate over (G, L) for each player
    self_sums = (games * self_mask[..., None]).sum(axis=(1, 2))  # (N, 57)
    opp_sums = (games * opp_mask[..., None]).sum(axis=(1, 2))    # (N, 57)

    self_counts = self_mask.sum(axis=(1, 2))  # (N,)
    opp_counts = opp_mask.sum(axis=(1, 2))    # (N,)

    # Mean (avoid div-by-zero)
    self_means = self_sums / np.maximum(self_counts, 1)[:, None]
    opp_means = opp_sums / np.maximum(opp_counts, 1)[:, None]

    # Drop is_self_action channel (trivially 1 in self, 0 in opp)
    ch_names = _channel_names()
    keep = [i for i in range(57) if i != _IS_SELF_CH]
    self_means = self_means[:, keep]
    opp_means = opp_means[:, keep]
    kept_names = [ch_names[i] for i in keep]

    stats = np.concatenate([self_means, opp_means], axis=1).astype(np.float32)
    names = (
        [f"self_{n}" for n in kept_names]
        + [f"opp_{n}" for n in kept_names]
    )
    assert stats.shape[1] == len(names) == 2 * 56
    return stats, names


def build_pair_features(
    stats: np.ndarray,
    pair_tuples: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for each observed pair (i, j).

    Features are [stats_i, stats_j, stats_i - stats_j]. The difference block
    gives the RF access to the natural skew-symmetric signal (since
    F[i,j] = -F[j,i] ↔ stats_i - stats_j flips sign).

    Args:
        stats: (N, D) per-player stats.
        pair_tuples: (P, 2) int array of (i, j) pairs.
        feature_names: D names for the columns of stats.

    Returns:
        X: (P, 3*D) float32 feature matrix.
        names: list of 3*D feature names.
    """
    i_idx = pair_tuples[:, 0]
    j_idx = pair_tuples[:, 1]
    xi = stats[i_idx]
    xj = stats[j_idx]
    X = np.concatenate([xi, xj, xi - xj], axis=1).astype(np.float32)
    names = (
        [f"i_{n}" for n in feature_names]
        + [f"j_{n}" for n in feature_names]
        + [f"diff_{n}" for n in feature_names]
    )
    return X, names


def train_rf_baseline(
    ds,
    train_pair_tuples: np.ndarray,
    test_pair_tuples: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> RFBaselineResult:
    """Train a Random Forest regressor on F[i,j] using hand-crafted features.

    Args:
        ds: TennisDataset.
        train_pair_tuples: (P_train, 2) int array of (i, j) pairs for training.
        test_pair_tuples: (P_test, 2) int array of (i, j) pairs for evaluation.
        n_estimators, max_depth, min_samples_leaf: RF hyperparameters.
        random_state: RF seed.
        verbose: print diagnostics.

    Returns:
        RFBaselineResult with MSEs and the fitted model.
    """
    from sklearn.ensemble import RandomForestRegressor

    # Per-player stats
    stats, stat_names = compute_player_stats(ds)
    if verbose:
        print(f"  Per-player stats: {stats.shape}")

    # Pair features for train/test
    X_train, feat_names = build_pair_features(stats, train_pair_tuples, stat_names)
    X_test, _ = build_pair_features(stats, test_pair_tuples, stat_names)
    y_train = np.array([ds.F[i, j] for i, j in train_pair_tuples],
                       dtype=np.float32)
    y_test = np.array([ds.F[i, j] for i, j in test_pair_tuples],
                      dtype=np.float32)

    if verbose:
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Fit
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Predict + MSE
    pred_train = rf.predict(X_train)
    pred_test = rf.predict(X_test)
    train_mse = float(np.mean((y_train - pred_train) ** 2))
    test_mse = float(np.mean((y_test - pred_test) ** 2))

    # Null baseline: predict the training-set mean for all pairs
    y_mean = float(np.mean(y_train))
    null_train_mse = float(np.mean((y_train - y_mean) ** 2))
    null_test_mse = float(np.mean((y_test - y_mean) ** 2))

    if verbose:
        print(f"  Null (predict train-mean={y_mean:+.4f}): "
              f"train={null_train_mse:.6f}, test={null_test_mse:.6f}")
        print(f"  Random Forest: train={train_mse:.6f}, test={test_mse:.6f}")

    return RFBaselineResult(
        train_mse=train_mse,
        test_mse=test_mse,
        null_train_mse=null_train_mse,
        null_test_mse=null_test_mse,
        feature_names=feat_names,
        feature_importance=rf.feature_importances_,
        model=rf,
        y_train=y_train,
        y_test=y_test,
        pred_train=pred_train,
        pred_test=pred_test,
        train_pair_tuples=np.asarray(train_pair_tuples),
        test_pair_tuples=np.asarray(test_pair_tuples),
    )


def top_feature_importance(result: RFBaselineResult, k: int = 15) -> list[tuple[str, float]]:
    """Return the top-k most important features by RF importance."""
    imp = result.feature_importance
    order = np.argsort(imp)[::-1][:k]
    return [(result.feature_names[i], float(imp[i])) for i in order]


# ---------------------------------------------------------------------------
# CLI: run the baseline stand-alone (rebuilds dataset)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _HERE = _Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    from loader import build_tennis_dataset  # type: ignore

    MCP_DIR = _Path("/tmp/mcp")
    print("Loading tennis dataset...")
    ds = build_tennis_dataset(
        matches_csv=MCP_DIR / "charting-m-matches.csv",
        points_csvs=[
            MCP_DIR / "charting-m-points-2020s.csv",
            MCP_DIR / "charting-m-points-2010s.csv",
        ],
        min_matches_per_player=30,
        max_players=40,
        max_games_per_player=50,
        max_shots_per_game=200,
        verbose=True,
    )

    # Same split as train.py
    observed = np.array(ds.observed_pairs)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pairs = observed[perm[:split]]
    test_pairs = observed[perm[split:]]
    print(f"\nTrain pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    print("\nTraining Random Forest baseline...")
    result = train_rf_baseline(ds, train_pairs, test_pairs)

    print("\nTop 15 features by importance:")
    for name, imp in top_feature_importance(result, k=15):
        print(f"  {imp:.4f}  {name}")
