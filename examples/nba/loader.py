"""Build hierarchical-FPTA tensors from NBA PBP V3.

Agent = team-season (30 agents for one regular season).
Game (FPTA sense) = one NBA game viewed from one team's perspective —
a chronological sequence of the game's possessions where an ``is_self_offense``
bit distinguishes own offence from own defence.

Output matches the shape contract consumed by ``hierarchical_skill_fpta``:

    agent_games      : (N, G_max, L_max, token_dim)  float32
    agent_token_mask : (N, G_max, L_max)              bool
    agent_game_mask  : (N, G_max)                     bool
    F                : (N, N)                         float32  (skew-symmetric)

F[i,j] is the net points-per-possession differential for team i vs team j
averaged across direct regular-season matchups (3–4 games per pair).

    F[i,j] = mean over (i_vs_j games) of [ PPP(i offense) - PPP(j offense) ]

Missing-pair F values are zero (with pair_counts=0 flag), but for a full
regular season every (i,j) pair has ≥1 direct matchup.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fetch import fetch_pbp, fetch_season_games   # type: ignore
from possessions import Possession, segment_game  # type: ignore


# ---------------------------------------------------------------------------
# Token encoding
# ---------------------------------------------------------------------------

START_CTX_VOCAB = [
    "period_start", "after_made_fg", "after_def_reb",
    "after_turnover", "after_fta", "other",
]
END_TYPE_VOCAB = [
    "made_2", "made_3", "turnover", "missed_def_reb",
    "fta_made", "period_end", "other",
]
_START_IDX = {s: i for i, s in enumerate(START_CTX_VOCAB)}
_END_IDX = {s: i for i, s in enumerate(END_TYPE_VOCAB)}

N_PERIOD_ONEHOT = 5   # Q1..Q4 + OT (anything >=5 maps to index 4)
PERIOD_LEN_SEC = 720.0
SCORE_DIFF_CLIP = 20.0
POSS_DUR_CLIP = 24.0


def token_dim() -> int:
    return (
        1                      # is_self_offense
        + N_PERIOD_ONEHOT
        + 1                    # time_norm
        + 1                    # score_diff_norm (self perspective)
        + len(START_CTX_VOCAB)
        + len(END_TYPE_VOCAB)
        + 1                    # points (offense) / 3
        + 1                    # duration / 24
    )


def possession_to_token(p: Possession, self_team: int) -> np.ndarray:
    """Encode one possession as a token from ``self_team``'s perspective."""
    tok = np.zeros(token_dim(), dtype=np.float32)
    pos = 0

    is_self_off = float(p.offensive_team == self_team)
    tok[pos] = is_self_off; pos += 1

    period_idx = min(max(p.period - 1, 0), N_PERIOD_ONEHOT - 1)
    tok[pos + period_idx] = 1.0; pos += N_PERIOD_ONEHOT

    tok[pos] = np.clip(p.start_clock_sec / PERIOD_LEN_SEC, 0.0, 1.0); pos += 1

    score_diff_self = p.score_diff_start if p.offensive_team == self_team \
        else -p.score_diff_start
    tok[pos] = float(np.clip(score_diff_self, -SCORE_DIFF_CLIP, SCORE_DIFF_CLIP)
                     / SCORE_DIFF_CLIP); pos += 1

    si = _START_IDX.get(p.start_context, _START_IDX["other"])
    tok[pos + si] = 1.0; pos += len(START_CTX_VOCAB)

    ei = _END_IDX.get(p.end_type, _END_IDX["other"])
    tok[pos + ei] = 1.0; pos += len(END_TYPE_VOCAB)

    tok[pos] = float(np.clip(p.points, 0.0, 4.0) / 3.0); pos += 1
    tok[pos] = float(np.clip(p.duration_sec, 0.0, POSS_DUR_CLIP)
                     / POSS_DUR_CLIP); pos += 1

    assert pos == token_dim()
    return tok


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class NBADataset:
    team_ids: list[int]
    team_names: list[str]     # tricodes aligned with team_ids
    agent_games: np.ndarray       # (N, G_max, L_max, token_dim)
    agent_token_mask: np.ndarray  # (N, G_max, L_max)
    agent_game_mask: np.ndarray   # (N, G_max)
    F: np.ndarray                 # (N, N) float32, skew-symmetric
    pair_counts: np.ndarray       # (N, N) int — direct matchups observed
    pair_off_poss: np.ndarray     # (N, N) int   — [i,j] = poss i on off vs j
    pair_off_pts:  np.ndarray     # (N, N) float — [i,j] = pts i on off vs j
    observed_pairs: list[tuple[int, int]]
    token_dim: int
    L_max: int
    season: str
    game_ids: list[str]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_nba_dataset(
    season: str = "2023-24",
    season_type: str = "Regular Season",
    max_games: int | None = None,
    L_max: int = 260,
    verbose: bool = True,
) -> NBADataset:
    """Build a full NBADataset for the requested season.

    Args:
        season: e.g. "2023-24".
        season_type: "Regular Season" | "Playoffs".
        max_games: if set, only process this many unique GAME_IDs (smoke tests).
        L_max: cap on possessions per game tensor.
    """
    if verbose:
        print(f"Loading game list for {season} {season_type} ...")
    games_df = fetch_season_games(season=season, season_type=season_type)
    games_df = games_df.drop_duplicates("GAME_ID").sort_values("GAME_DATE")
    game_ids = list(games_df["GAME_ID"].astype(str))

    # Team roster = every distinct team_id that appears in the season.
    team_rows = fetch_season_games(season=season, season_type=season_type)
    team_rows = team_rows.drop_duplicates("TEAM_ID")[["TEAM_ID", "TEAM_ABBREVIATION"]]
    team_ids = sorted(int(t) for t in team_rows["TEAM_ID"].tolist())
    tid_to_abbrev = {int(r["TEAM_ID"]): r["TEAM_ABBREVIATION"]
                     for _, r in team_rows.iterrows()}
    team_names = [tid_to_abbrev[t] for t in team_ids]
    N = len(team_ids)
    tid_to_idx = {t: i for i, t in enumerate(team_ids)}
    if verbose:
        print(f"  {N} teams, {len(game_ids)} games")

    if max_games is not None:
        game_ids = game_ids[:max_games]
        if verbose:
            print(f"  (limited to {len(game_ids)} games for smoke test)")

    G_max = 82 if max_games is None else max_games
    td = token_dim()

    agent_games = np.zeros((N, G_max, L_max, td), dtype=np.float32)
    agent_token_mask = np.zeros((N, G_max, L_max), dtype=bool)
    agent_game_mask = np.zeros((N, G_max), dtype=bool)
    g_counters = np.zeros(N, dtype=np.int32)

    pair_counts = np.zeros((N, N), dtype=np.int64)
    pair_off_poss = np.zeros((N, N), dtype=np.int64)
    pair_off_pts = np.zeros((N, N), dtype=np.float64)

    used_game_ids: list[str] = []

    for gk, gid in enumerate(game_ids):
        pbp = fetch_pbp(gid)
        if pbp is None:
            if verbose:
                print(f"  skipped {gid} (PBP fetch failed)")
            continue
        try:
            poss = segment_game(pbp)
        except Exception as exc:
            if verbose:
                print(f"  skipped {gid} (segment failure: {exc!r})")
            continue
        if not poss:
            continue

        game_team_ids = sorted({p.offensive_team for p in poss})
        if len(game_team_ids) != 2 or any(t not in tid_to_idx for t in game_team_ids):
            continue
        t1, t2 = game_team_ids
        i1, i2 = tid_to_idx[t1], tid_to_idx[t2]

        # Per-team tokens for this game (whole possession stream from each
        # team's perspective; is_self_offense bit flips accordingly).
        for self_team, self_idx in [(t1, i1), (t2, i2)]:
            gi = int(g_counters[self_idx])
            if gi >= G_max:
                continue
            for k, p in enumerate(poss[:L_max]):
                agent_games[self_idx, gi, k] = possession_to_token(p, self_team)
                agent_token_mask[self_idx, gi, k] = True
            agent_game_mask[self_idx, gi] = True
            g_counters[self_idx] = gi + 1

        # Pair accounting (add to both orderings: i-vs-j and j-vs-i).
        for p in poss:
            off_t = p.offensive_team
            def_t = p.defensive_team
            if off_t not in tid_to_idx or def_t not in tid_to_idx:
                continue
            i = tid_to_idx[off_t]; j = tid_to_idx[def_t]
            pair_off_poss[i, j] += 1
            pair_off_pts[i, j] += p.points
        pair_counts[i1, i2] += 1
        pair_counts[i2, i1] += 1
        used_game_ids.append(gid)

        if verbose and (gk + 1) % 50 == 0:
            print(f"  processed {gk + 1}/{len(game_ids)}")

    # F[i,j] = PPP(i on offense vs j) - PPP(j on offense vs i)
    with np.errstate(invalid="ignore", divide="ignore"):
        ppp_ij = np.where(pair_off_poss > 0,
                          pair_off_pts / np.maximum(pair_off_poss, 1),
                          0.0)
    F = (ppp_ij - ppp_ij.T).astype(np.float32)
    # Enforce exact skew-symmetry (it already is by construction; belt & braces).
    F = 0.5 * (F - F.T)

    observed = np.argwhere(pair_counts > 0)
    observed_pairs = [(int(i), int(j)) for i, j in observed if i != j]

    if verbose:
        games_per_team = agent_game_mask.sum(axis=1)
        print(f"\nNBADataset built: N={N}, G_max={G_max}, L_max={L_max}, "
              f"token_dim={td}")
        print(f"  Games/team: min={games_per_team.min()}, "
              f"max={games_per_team.max()}, mean={games_per_team.mean():.1f}")
        print(f"  Observed ordered pairs: {len(observed_pairs)} / {N*(N-1)}")
        nz = (pair_off_poss > 0).sum()
        print(f"  Pair density (has direct matchup): {nz / (N*N):.3f}")
        print(f"  F range: [{F.min():.4f}, {F.max():.4f}], "
              f"|F|_mean={np.abs(F).mean():.4f}")

    return NBADataset(
        team_ids=team_ids,
        team_names=team_names,
        agent_games=agent_games,
        agent_token_mask=agent_token_mask,
        agent_game_mask=agent_game_mask,
        F=F,
        pair_counts=pair_counts,
        pair_off_poss=pair_off_poss,
        pair_off_pts=pair_off_pts.astype(np.float32),
        observed_pairs=observed_pairs,
        token_dim=td,
        L_max=L_max,
        season=season,
        game_ids=used_game_ids,
    )


if __name__ == "__main__":
    ds = build_nba_dataset(season="2023-24", max_games=20, L_max=260, verbose=True)
    print(f"\nTeams: {ds.team_names}")
    print(f"agent_games shape: {ds.agent_games.shape}")
    print(f"F nonzero entries: {(ds.F != 0).sum()}")
    # Compare team PPP diffs against a quick sanity check: highest-scoring
    # team in our sample should have a positive row-mean in F.
    row_mean = ds.F.mean(axis=1)
    order = np.argsort(-row_mean)
    print("\nTop by F row-mean (proxy for quality in this sample):")
    for idx in order[:8]:
        print(f"  {ds.team_names[idx]:>4s}  row_mean={row_mean[idx]:+.4f}")
