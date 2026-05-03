"""Fetch NBA regular-season game list + per-game play-by-play via nba_api.

Caches everything to disk so repeated runs don't re-hit stats.nba.com.
Rate-limited and retried because the endpoint is flaky under load.

Usage:
    from examples.nba.fetch import fetch_season_games, fetch_pbp

    games = fetch_season_games("2023-24")          # DataFrame, 1230 rows
    pbp   = fetch_pbp(games.iloc[0]["GAME_ID"])    # DataFrame, ~450 events

Cache layout:
    {CACHE_DIR}/games_{season}.parquet
    {CACHE_DIR}/pbp/{game_id}.parquet
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(os.environ.get("NBA_CACHE_DIR", "/tmp/nba_pbp"))
REQUEST_SLEEP_SEC = 0.6
MAX_RETRIES = 5


def _ensure_dirs() -> None:
    (CACHE_DIR / "pbp").mkdir(parents=True, exist_ok=True)


def _sleep_with_jitter(base: float) -> None:
    time.sleep(base + random.uniform(0.0, 0.3))


def fetch_season_games(
    season: str = "2023-24",
    season_type: str = "Regular Season",
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with one row per team-game (so 2 rows per game).

    Columns include GAME_ID, TEAM_ID, TEAM_ABBREVIATION, MATCHUP, WL, PTS, ...
    """
    _ensure_dirs()
    cache_path = CACHE_DIR / f"games_{season.replace('-', '_')}_{season_type.replace(' ', '_')}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    from nba_api.stats.endpoints import leaguegamefinder

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    df.to_parquet(cache_path, index=False)
    return df


def fetch_pbp(
    game_id: str,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """Return PBP events for one game, or None on persistent failure.

    Pads short GAME_IDs to the 10-digit form nba_api expects.
    """
    _ensure_dirs()
    gid = str(game_id).zfill(10)
    cache_path = CACHE_DIR / "pbp" / f"{gid}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    from nba_api.stats.endpoints import playbyplayv3

    for attempt in range(MAX_RETRIES):
        try:
            resp = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=30)
            df = resp.get_data_frames()[0]
            df.to_parquet(cache_path, index=False)
            _sleep_with_jitter(REQUEST_SLEEP_SEC)
            return df
        except Exception as exc:
            wait = 2.0 ** attempt + random.uniform(0, 1.0)
            print(f"  [fetch_pbp] {gid} attempt {attempt + 1} failed ({exc!r}); sleeping {wait:.1f}s")
            time.sleep(wait)
    print(f"  [fetch_pbp] {gid} giving up after {MAX_RETRIES} retries")
    return None


def fetch_all_pbp(
    game_ids: list[str],
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch PBP for every game_id, returning a dict keyed by padded id."""
    out: dict[str, pd.DataFrame] = {}
    for k, gid in enumerate(game_ids):
        pbp = fetch_pbp(gid)
        if pbp is not None:
            out[str(gid).zfill(10)] = pbp
        if verbose and (k + 1) % 50 == 0:
            print(f"  fetched {k + 1}/{len(game_ids)} games")
    return out


if __name__ == "__main__":
    games = fetch_season_games("2023-24")
    unique_games = games.drop_duplicates("GAME_ID")
    print(f"Season 2023-24 RS: {len(unique_games)} unique games, {len(games)} team-game rows.")
    print(unique_games[["GAME_ID", "GAME_DATE", "MATCHUP", "PTS"]].head(5))

    first_gid = unique_games.iloc[0]["GAME_ID"]
    pbp = fetch_pbp(first_gid)
    print(f"\nPBP for {first_gid}: {len(pbp)} events, columns={list(pbp.columns)[:10]}...")
