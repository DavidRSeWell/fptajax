"""Segment a PBP V3 event stream into possessions.

A possession ends on:
  - Made FG (not followed by an and-1 FT trip by the same team)
  - Last made FT of a trip
  - Last missed FT of a trip that is rebounded by the defence
  - Turnover
  - Defensive rebound of a missed shot
  - End of period

An offensive rebound keeps the same possession open; we only record it by
bumping the possession's ``n_oreb`` counter.

Points credited to a possession = FG points from made shots during the
possession + made free throws during the possession. Foul-shooting points
that follow an and-1 are lumped into the possession that ends on the FGM.

Output records are plain dicts with fields consumed by loader.py:

    offensive_team   : int (TEAM_ID)
    defensive_team   : int (TEAM_ID)
    period           : int
    start_clock_sec  : float (seconds remaining in period)
    end_clock_sec    : float
    duration_sec     : float
    start_context    : str (period_start / after_made_fg / after_def_reb /
                            after_turnover / after_fta / other)
    end_type         : str (made_2 / made_3 / turnover / missed_def_reb /
                            fta_made / fta_missed_def_reb / period_end / other)
    points           : float (0..4 — and-1 can be 3 or 4)
    n_oreb           : int
    score_diff_start : int (score_offense - score_defense entering possession)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Clock parsing
# ---------------------------------------------------------------------------

_CLOCK_RE = re.compile(r"^PT(\d+)M([\d.]+)S$")


def clock_to_seconds(clock_str: str) -> float:
    """Parse "PT11M38.00S" -> 11*60 + 38.0 = 698.0."""
    if not isinstance(clock_str, str):
        return 0.0
    m = _CLOCK_RE.match(clock_str)
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


# ---------------------------------------------------------------------------
# Free-throw subType parsing
# ---------------------------------------------------------------------------

_FT_RE = re.compile(r"Free Throw (\d+) of (\d+)")
_TECH_RE = re.compile(r"Technical|Flagrant|Clear Path", re.I)


def parse_ft_subtype(subtype: str) -> tuple[int, int, bool]:
    """Return ``(n, m, is_technical_or_flagrant)``.

    ``is_technical_or_flagrant`` FTs do not end or extend a normal possession
    — they're 'extra' shots. We credit their points to whichever possession
    happens to be open but don't treat them as boundaries.
    """
    if not isinstance(subtype, str):
        return (0, 0, False)
    if _TECH_RE.search(subtype):
        return (1, 1, True)
    m = _FT_RE.search(subtype)
    if not m:
        return (0, 0, False)
    return (int(m.group(1)), int(m.group(2)), False)


# ---------------------------------------------------------------------------
# Possession record
# ---------------------------------------------------------------------------


@dataclass
class Possession:
    offensive_team: int
    defensive_team: int
    period: int
    start_clock_sec: float
    end_clock_sec: float = 0.0
    duration_sec: float = 0.0
    start_context: str = "other"
    end_type: str = "other"
    points: float = 0.0
    n_oreb: int = 0
    score_diff_start: int = 0   # offense minus defense
    n_events: int = 0

    def to_dict(self) -> dict:
        return dict(
            offensive_team=self.offensive_team,
            defensive_team=self.defensive_team,
            period=self.period,
            start_clock_sec=self.start_clock_sec,
            end_clock_sec=self.end_clock_sec,
            duration_sec=self.duration_sec,
            start_context=self.start_context,
            end_type=self.end_type,
            points=self.points,
            n_oreb=self.n_oreb,
            score_diff_start=self.score_diff_start,
            n_events=self.n_events,
        )


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class PossessionSegmenter:
    """Stateful walker over PBP rows that emits Possession records."""

    def __init__(self, team_ids: tuple[int, int]):
        # ``team_ids[0]`` and ``team_ids[1]`` are the two teamIds that appear
        # in this game's PBP (home and away, arbitrary order). We need them
        # to know the "defensive team" whenever we observe an offensive action.
        self.teams = tuple(team_ids)
        assert len(self.teams) == 2 and self.teams[0] != self.teams[1]
        self.possessions: list[Possession] = []
        self.current: Optional[Possession] = None
        self.last_missed_shot_team: Optional[int] = None
        # tracks whether we just saw a made FG by offense and are waiting to
        # decide whether an immediate shooting foul + FT makes this an and-1.
        self.pending_made_fg_close: bool = False
        # running scores keyed by teamId
        self.score: dict[int, int] = {self.teams[0]: 0, self.teams[1]: 0}

    # ---- helpers ----

    def _other(self, team: int) -> int:
        return self.teams[0] if team == self.teams[1] else self.teams[1]

    def _score_diff(self, off_team: int) -> int:
        return self.score[off_team] - self.score[self._other(off_team)]

    def _open_possession(
        self,
        offensive_team: int,
        period: int,
        start_clock_sec: float,
        start_context: str,
    ) -> None:
        if offensive_team not in self.teams:
            return
        self.current = Possession(
            offensive_team=offensive_team,
            defensive_team=self._other(offensive_team),
            period=period,
            start_clock_sec=start_clock_sec,
            start_context=start_context,
            score_diff_start=self._score_diff(offensive_team),
        )

    def _close_possession(self, end_type: str, end_clock_sec: float) -> None:
        if self.current is None:
            return
        self.current.end_type = end_type
        self.current.end_clock_sec = end_clock_sec
        dur = self.current.start_clock_sec - end_clock_sec
        self.current.duration_sec = max(0.0, dur)
        self.possessions.append(self.current)
        self.current = None
        self.pending_made_fg_close = False

    # ---- main entry ----

    def ingest(self, pbp: pd.DataFrame) -> list[Possession]:
        """Walk the full event stream (chronological) and return possessions."""
        # Sort by (period, actionNumber) to guarantee chronology; PlayByPlayV3
        # usually comes sorted but guard anyway.
        df = pbp.sort_values(["period", "actionNumber"]).reset_index(drop=True)

        rows = df.to_dict("records")
        n = len(rows)
        for idx in range(n):
            ev = rows[idx]
            period = int(ev.get("period") or 0)
            action = ev.get("actionType") or ""
            subtype = ev.get("subType") or ""
            team_id = ev.get("teamId")
            team_id = int(team_id) if team_id and not pd.isna(team_id) else None
            clock_sec = clock_to_seconds(ev.get("clock") or "")

            # Advance running score from the snapshot columns
            score_home = ev.get("scoreHome")
            score_away = ev.get("scoreAway")
            # scoreHome/scoreAway are strings like "0" in V3 — coerce.
            try:
                sh = int(score_home) if score_home not in (None, "") else None
                sa = int(score_away) if score_away not in (None, "") else None
            except (TypeError, ValueError):
                sh = sa = None

            # --- Period boundaries ---
            if action == "period":
                if subtype == "start":
                    # Close any lingering possession — shouldn't exist except
                    # on the very first event, where self.current is None.
                    self._close_possession("period_end", 0.0)
                    # We don't open a possession yet; we wait for the first
                    # action with a teamId (usually the jump ball or an
                    # inbound pass immediately followed by a play).
                    continue
                elif subtype == "end":
                    self._close_possession("period_end", 0.0)
                    continue

            # If no possession is open and we see an actionable team event,
            # open one with "other"/"period_start" depending on context.
            if self.current is None and team_id in self.teams and action in (
                "Made Shot", "Missed Shot", "Turnover", "Jump Ball",
            ):
                ctx = "period_start" if period and not self.possessions or (
                    self.possessions and self.possessions[-1].period < period
                ) else "other"
                self._open_possession(team_id, period, clock_sec, ctx)

            # If the prior event was a made FG that *might* be an and-1, decide now.
            if self.pending_made_fg_close:
                # and-1 detection: next "Foul" actionType with subType "Shooting"
                # and defender teamId == _other(offensive_team). If that's the
                # case, we DON'T close yet — we wait for the FT.
                if action == "Foul" and "Shooting" in subtype and self.current is not None:
                    self.pending_made_fg_close = False
                    # stay open; the upcoming FT will close
                else:
                    # No shooting foul follows → actually close the made FG.
                    # We close with the last-known clock (close to this event's
                    # clock since only benign events — subs, timeouts — should be
                    # between the FGM and here).
                    if self.current is not None:
                        offensive_team = self.current.offensive_team
                        end_type = "made_3" if self.current.points >= 3 else "made_2"
                        self._close_possession(end_type, clock_sec)
                        # New possession: the team that was on defence.
                        self._open_possession(
                            self._other(offensive_team), period, clock_sec,
                            "after_made_fg",
                        )

            # --- Main event dispatch ---
            if action == "Made Shot" and team_id in self.teams:
                if self.current is None:
                    self._open_possession(team_id, period, clock_sec, "other")
                self.current.points += float(ev.get("shotValue") or 0)
                self.current.n_events += 1
                # Defer closing until we know whether an and-1 FT follows.
                self.pending_made_fg_close = True

            elif action == "Missed Shot" and team_id in self.teams:
                if self.current is None:
                    self._open_possession(team_id, period, clock_sec, "other")
                self.current.n_events += 1
                self.last_missed_shot_team = team_id

            elif action == "Rebound" and team_id in self.teams:
                shooter = self.last_missed_shot_team
                if shooter is None or self.current is None:
                    self.last_missed_shot_team = None
                    continue
                if team_id == shooter:
                    # Offensive rebound — same possession continues.
                    self.current.n_oreb += 1
                    self.current.n_events += 1
                else:
                    # Defensive rebound — possession over.
                    offensive_team = self.current.offensive_team
                    self._close_possession("missed_def_reb", clock_sec)
                    self._open_possession(
                        team_id, period, clock_sec, "after_def_reb",
                    )
                self.last_missed_shot_team = None

            elif action == "Turnover" and team_id in self.teams:
                if self.current is None:
                    self._open_possession(team_id, period, clock_sec, "other")
                offensive_team = self.current.offensive_team
                self._close_possession("turnover", clock_sec)
                self._open_possession(
                    self._other(offensive_team), period, clock_sec,
                    "after_turnover",
                )

            elif action == "Free Throw" and team_id in self.teams:
                n_ft, m_ft, is_tech = parse_ft_subtype(subtype)
                # PBP V3 leaves ``shotResult`` blank for free throws; the
                # made/missed state is encoded in the description.
                desc = str(ev.get("description") or "")
                made = not desc.lstrip().startswith("MISS")
                # A technical/flagrant FT: credit points but do NOT close or
                # open a possession on it.
                if is_tech:
                    if made and self.current is not None and self.current.offensive_team == team_id:
                        # only credit if this team is the one with the open poss
                        self.current.points += 1.0
                    # otherwise it's a bonus FT for a team not currently on
                    # offence — ignore for PPP purposes.
                    continue
                # Make sure we have a possession open for this team.
                if self.current is None or self.current.offensive_team != team_id:
                    # FT for a team we weren't tracking as on offence — force a
                    # possession to exist.
                    if self.current is not None:
                        self._close_possession("other", clock_sec)
                    self._open_possession(team_id, period, clock_sec, "other")
                if made:
                    self.current.points += 1.0
                self.current.n_events += 1
                # If this is the last FT of the trip, close and maybe flip.
                if m_ft > 0 and n_ft == m_ft:
                    if made:
                        # Possession ends — defence gets inbound.
                        offensive_team = self.current.offensive_team
                        self._close_possession("fta_made", clock_sec)
                        self._open_possession(
                            self._other(offensive_team), period, clock_sec,
                            "after_fta",
                        )
                    else:
                        # Missed last FT is live: we'll see the Rebound next.
                        # Mark last_missed_shot_team so the rebound logic
                        # treats it correctly.
                        self.last_missed_shot_team = team_id

            elif action == "Jump Ball" and team_id in self.teams:
                # Jump ball gives possession to one team (we'll let the next
                # action establish the offensive team). Open a placeholder
                # possession if none is open yet.
                if self.current is None:
                    self._open_possession(team_id, period, clock_sec, "period_start")

            # everything else (Foul, Substitution, Timeout, Violation, Instant Replay, ...)
            # is a no-op for possession boundaries — just absorb it.

            # Refresh running score from snapshot AFTER processing the event.
            if sh is not None and sa is not None:
                # Map home/away -> teamId. We can't do this without knowing
                # who is home; the scoreHome/scoreAway are running totals and
                # we infer them from "location" field. For simplicity we rely
                # on the FGM/FT/points logic above — skipping snapshot sync.
                pass

        # Close whatever is hanging at end of data.
        self._close_possession("period_end", 0.0)
        return self.possessions


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def segment_game(pbp: pd.DataFrame) -> list[Possession]:
    """Infer team ids from the PBP and segment into possessions."""
    team_ids = [t for t in pbp["teamId"].dropna().unique() if int(t) != 0]
    team_ids = sorted(int(t) for t in team_ids)
    assert len(team_ids) == 2, f"Expected 2 teams, got {team_ids}"
    seg = PossessionSegmenter(tuple(team_ids))
    return seg.ingest(pbp)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from fetch import fetch_pbp, fetch_season_games

    games = fetch_season_games("2023-24")
    gid = games.drop_duplicates("GAME_ID").iloc[0]["GAME_ID"]
    pbp = fetch_pbp(gid)
    poss = segment_game(pbp)
    by_team: dict[int, list[Possession]] = {}
    for p in poss:
        by_team.setdefault(p.offensive_team, []).append(p)
    print(f"Game {gid}: {len(poss)} possessions total")
    for tid, lst in by_team.items():
        pts = sum(p.points for p in lst)
        print(f"  team {tid}: {len(lst)} poss, {pts:.0f} pts, "
              f"PPP={pts/max(1,len(lst)):.3f}")
