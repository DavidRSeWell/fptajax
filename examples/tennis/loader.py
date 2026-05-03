"""Convert MCP CSV files into hierarchical-FPTA tensors.

Outputs match the shapes expected by ``hierarchical_behavioral_fpta``:

  agent_games      : (N, G_max, L_max, token_dim)  float32
  agent_token_mask : (N, G_max, L_max)              bool
  agent_game_mask  : (N, G_max)                     bool
  F                : (N, N)                         float32  (skew-symmetric)
  observed_pairs   : list[(i, j)] of pairs with >=1 match

For each player X:
  - We collect every match they played.
  - For each match we produce one "game" tensor of up to L_max shots, where
    each shot is a fixed-size feature vector. The per-shot features include
    an ``is_self_action`` bit that is True for shots X hit and False for
    shots their opponent hit.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from parser import (  # type: ignore
    ParsedPoint,
    Shot,
    parse_point,
    shot_feature_dim,
    shot_to_vector,
)


# ---------------------------------------------------------------------------
# Match loading
# ---------------------------------------------------------------------------


@dataclass
class MatchMeta:
    match_id: str
    player1: str
    player2: str
    date: str
    surface: str
    tournament: str


def load_matches_csv(path: str | Path) -> dict[str, MatchMeta]:
    """Load the charting-?-matches.csv file and return a match_id -> meta map."""
    out: dict[str, MatchMeta] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("match_id", "").strip()
            if not mid:
                continue
            out[mid] = MatchMeta(
                match_id=mid,
                player1=row.get("Player 1", "").strip(),
                player2=row.get("Player 2", "").strip(),
                date=row.get("Date", "").strip(),
                surface=row.get("Surface", "").strip(),
                tournament=row.get("Tournament", "").strip(),
            )
    return out


def load_points_csvs(paths: list[str | Path]) -> dict[str, list[dict]]:
    """Load one or more points CSVs and group rows by match_id.

    Returns a dict mapping match_id -> list of point-row dicts, in point order.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("match_id", "").strip()
                if mid:
                    groups[mid].append(row)
    # Keep match points in order of the Pt column (integer)
    for mid, rows in groups.items():
        def _pt(r):
            try:
                return int(r.get("Pt", 0))
            except ValueError:
                return 0
        rows.sort(key=_pt)
    return groups


# ---------------------------------------------------------------------------
# Per-match shot sequence (from a specified player's perspective)
# ---------------------------------------------------------------------------


def build_match_shot_sequence(
    match_meta: MatchMeta,
    match_points: list[dict],
    target_player: str,
    max_shots: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build the (L_max, token_dim) shot tensor for a single match from the
    target_player's perspective.

    Returns:
        tokens: (L_max, token_dim) float32.
        mask: (L_max,) bool.
        pts_target: total points target_player won in the match.
        pts_total: total points played.
    """
    token_dim = shot_feature_dim()
    tokens = np.zeros((max_shots, token_dim), dtype=np.float32)
    mask = np.zeros(max_shots, dtype=bool)

    if target_player == match_meta.player1:
        target_num = 1
    elif target_player == match_meta.player2:
        target_num = 2
    else:
        return tokens, mask, 0, 0

    n_pts_total = 0
    n_pts_target = 0
    t = 0  # token index

    for row in match_points:
        first_cell = row.get("1st", "") or ""
        second_cell = row.get("2nd", "") or ""
        try:
            server = int(row.get("Svr", 0))
        except ValueError:
            server = 0
        try:
            pt_winner = int(row.get("PtWinner", 0))
        except ValueError:
            pt_winner = 0

        n_pts_total += 1
        if pt_winner == target_num:
            n_pts_target += 1

        pp = parse_point(first_cell, second_cell)
        if not pp.shots:
            continue

        # Determine who played each shot.
        # With a 1st-serve fault, shots[0] is the (fault) 1st serve, shots[1]
        # is the 2nd serve, both played by the server. Shots[2] is the return.
        # Otherwise shots[0] is the (in-play) serve and shots[1] is the return.
        # In both cases, shots alternate between server and returner starting
        # from the serve, except that all consecutive leading "serve" shots are
        # played by the server.
        for k, shot in enumerate(pp.shots):
            if t >= max_shots:
                break
            # server_plays_this_shot: True if shot is played by server
            if shot.is_serve:
                server_plays = True
            else:
                # Count how many of the preceding shots were serves
                n_prior_serves = sum(1 for s in pp.shots[:k] if s.is_serve)
                # The first non-serve shot (return) is played by the returner.
                # Then alternates.
                idx_after_serves = (k - n_prior_serves)
                server_plays = (idx_after_serves % 2 == 1)
            player_num_of_shot = server if server_plays else (3 - server)
            is_self = (player_num_of_shot == target_num)

            vec = shot_to_vector(shot, is_self_action=is_self)
            tokens[t] = vec
            mask[t] = True
            t += 1

        if t >= max_shots:
            break

    return tokens, mask, n_pts_target, n_pts_total


# ---------------------------------------------------------------------------
# Full dataset build
# ---------------------------------------------------------------------------


@dataclass
class TennisDataset:
    """Bundled dataset for hierarchical FPTA."""
    player_names: list[str]
    agent_games: np.ndarray       # (N, G_max, L_max, token_dim)
    agent_token_mask: np.ndarray  # (N, G_max, L_max)
    agent_game_mask: np.ndarray   # (N, G_max)
    F: np.ndarray                 # (N, N) float32, skew-symmetric
    pair_counts: np.ndarray       # (N, N) int — number of matches observed
    pair_points_scored: np.ndarray  # (N, N) float — [i,j] = total points i won vs j
    pair_points_total: np.ndarray   # (N, N) float — [i,j] = total points played in i-vs-j matches
    observed_pairs: list[tuple[int, int]]
    token_dim: int
    L_max: int
    match_counts: dict[str, int]


def build_tennis_dataset(
    matches_csv: str | Path,
    points_csvs: list[str | Path],
    min_matches_per_player: int = 30,
    max_players: int | None = None,
    max_games_per_player: int = 60,
    max_shots_per_game: int = 400,
    surface_filter: str | None = None,
    verbose: bool = True,
) -> TennisDataset:
    """Build a complete TennisDataset ready for hierarchical_behavioral_fpta.

    Args:
        matches_csv: path to charting-?-matches.csv.
        points_csvs: paths to charting-?-points-*.csv files.
        min_matches_per_player: drop players with fewer charted matches.
        max_players: if set, keep only the top-M players by match count.
        max_games_per_player: G_max — cap number of games stored per player.
        max_shots_per_game: L_max — cap shots per game tensor.
        surface_filter: if set, keep only matches of this surface
            (e.g., "Hard", "Clay", "Grass").
        verbose: print diagnostics.

    Returns:
        TennisDataset.
    """
    if verbose:
        print(f"Loading matches from {matches_csv} ...")
    meta_map = load_matches_csv(matches_csv)
    if surface_filter:
        meta_map = {mid: m for mid, m in meta_map.items()
                    if m.surface == surface_filter}
    if verbose:
        print(f"  matches in metadata: {len(meta_map)}")

    if verbose:
        print(f"Loading points from {len(points_csvs)} csvs ...")
    points_by_match = load_points_csvs(points_csvs)
    if verbose:
        print(f"  matches with points data: {len(points_by_match)}")

    # Intersect: only keep matches that have both metadata and points
    usable_mids = set(meta_map.keys()) & set(points_by_match.keys())
    if verbose:
        print(f"  usable matches (meta + points): {len(usable_mids)}")

    # Count matches per player
    match_counts: Counter = Counter()
    for mid in usable_mids:
        m = meta_map[mid]
        if m.player1:
            match_counts[m.player1] += 1
        if m.player2:
            match_counts[m.player2] += 1

    eligible = [p for p, c in match_counts.items() if c >= min_matches_per_player]
    eligible.sort(key=lambda p: (-match_counts[p], p))
    if max_players is not None:
        eligible = eligible[:max_players]
    player_names = eligible
    N = len(player_names)
    if verbose:
        print(f"  eligible players (>= {min_matches_per_player} matches): {N}")
        for name in player_names[:10]:
            print(f"    {name}: {match_counts[name]}")

    name_to_idx = {n: i for i, n in enumerate(player_names)}
    eligible_set = set(player_names)

    # Collect matches each player has (paired with eligible opponents)
    player_matches: dict[str, list[str]] = {n: [] for n in player_names}
    pair_points_target = np.zeros((N, N), dtype=np.float64)
    pair_points_total = np.zeros((N, N), dtype=np.float64)
    pair_counts = np.zeros((N, N), dtype=np.int64)

    token_dim = shot_feature_dim()
    L_max = max_shots_per_game
    G_max = max_games_per_player

    agent_games = np.zeros((N, G_max, L_max, token_dim), dtype=np.float32)
    agent_token_mask = np.zeros((N, G_max, L_max), dtype=bool)
    agent_game_mask = np.zeros((N, G_max), dtype=bool)
    g_counters = np.zeros(N, dtype=np.int32)

    # Shuffle-friendly iteration order (deterministic by match_id sort)
    mids_sorted = sorted(usable_mids)

    for mid in mids_sorted:
        m = meta_map[mid]
        if m.player1 not in eligible_set or m.player2 not in eligible_set:
            continue

        i = name_to_idx[m.player1]
        j = name_to_idx[m.player2]
        rows = points_by_match[mid]

        # Add game tensor from each player's perspective (if we still have slots)
        for player_name, idx, other_idx in [
            (m.player1, i, j), (m.player2, j, i),
        ]:
            gi = int(g_counters[idx])
            if gi >= G_max:
                continue
            tokens, mask, pts_target, pts_total = build_match_shot_sequence(
                m, rows, target_player=player_name, max_shots=L_max,
            )
            if pts_total == 0:
                continue
            agent_games[idx, gi] = tokens
            agent_token_mask[idx, gi] = mask
            agent_game_mask[idx, gi] = True
            g_counters[idx] = gi + 1
            # Only count pair stats once per match (when processing player1)
            if player_name == m.player1:
                pair_points_target[i, j] += pts_target
                pair_points_total[i, j] += pts_total
                pair_points_target[j, i] += (pts_total - pts_target)
                pair_points_total[j, i] += pts_total
                pair_counts[i, j] += 1
                pair_counts[j, i] += 1
            player_matches[player_name].append(mid)

    # Build payoff matrix F[i, j] = (pts_i - pts_j) / total_pts
    F = np.zeros((N, N), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        # F[i,j] = (pts_i_vs_j - pts_j_vs_i) / total
        total = pair_points_total  # already doubled: both [i,j] and [j,i] store totals
        pts_i = pair_points_target
        pts_j = total - pts_i
        denom = np.where(total > 0, total, 1)
        F_full = (pts_i - pts_j) / denom
        F = np.where(total > 0, F_full, 0.0).astype(np.float32)
    # Enforce exact skew-symmetry to eliminate rounding
    F = 0.5 * (F - F.T)

    observed = np.argwhere(pair_counts > 0)
    observed_pairs = [(int(i), int(j)) for i, j in observed if i != j]

    if verbose:
        games_per_player = agent_game_mask.sum(axis=1)
        print(f"\nDataset built: N={N} players, "
              f"G_max={G_max}, L_max={L_max}, token_dim={token_dim}")
        print(f"  Games/player: min={games_per_player.min()}, "
              f"max={games_per_player.max()}, mean={games_per_player.mean():.1f}")
        print(f"  Observed ordered pairs (i,j), i!=j: {len(observed_pairs)}")
        print(f"  F range: [{F.min():.3f}, {F.max():.3f}]")

    return TennisDataset(
        player_names=player_names,
        agent_games=agent_games,
        agent_token_mask=agent_token_mask,
        agent_game_mask=agent_game_mask,
        F=F,
        pair_counts=pair_counts,
        pair_points_scored=pair_points_target.astype(np.float32),
        pair_points_total=pair_points_total.astype(np.float32),
        observed_pairs=observed_pairs,
        token_dim=token_dim,
        L_max=L_max,
        match_counts=dict(match_counts),
    )


# ---------------------------------------------------------------------------
# CLI diagnostic
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    MCP_DIR = Path("/tmp/mcp")
    ds = build_tennis_dataset(
        matches_csv=MCP_DIR / "charting-m-matches.csv",
        points_csvs=[MCP_DIR / "charting-m-points-2020s.csv"],
        min_matches_per_player=20,
        max_players=40,
        max_games_per_player=40,
        max_shots_per_game=300,
        verbose=True,
    )
    print(f"\nTop 10 by match count: {ds.player_names[:10]}")
    print(f"Tokens shape: {ds.agent_games.shape}")
    print(f"F shape: {ds.F.shape}, nonzero fraction: "
          f"{(ds.pair_counts > 0).mean():.3f}")
