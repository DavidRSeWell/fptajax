"""Parser for the Jeff Sackmann Match Charting Project shot-code format.

The MCP encodes each point as a compact string. Example:

    ``4sj=+1b2z2b2z3r2t2v1*``  means:
    serve wide (4), backhand slice (s), forehand swinging volley (j)
    at baseline (=), serve-and-volley (+), direction 1, backhand (b)
    direction 2, backhand volley (z), ... winner (*).

Grammar (summarized from the MatchChart 0.3.2.xlsm "Instructions" tab):

SERVE (always the first shot)
  direction    : 4=wide, 5=body, 6=T, 0=unknown
  serve-volley : '+' after the digit
  fault types  : n=net, w=wide, d=deep, x=both, g=foot, e=unknown, !=shank, V=time violation
  outcomes     : *=ace, #=unreturnable, @=unforced-err on return

RALLY SHOT (shots 1..L-1)
  shot letter:
    f=forehand, b=backhand groundstroke
    r=fh slice, s=bh slice
    v=fh volley, z=bh volley
    o=overhead, p=bh overhead
    u=fh drop, y=bh drop
    l=fh lob, m=bh lob
    h=fh half-volley, i=bh half-volley
    j=fh swinging-volley, k=bh swinging-volley
    t=trick shot, q=unknown
  modifiers  (any subset, before direction): '+' approach, '-' at net, '=' at baseline, ';' net cord, '^' drop-volley
  direction  : 0-3 (0=unknown, 1=to RH-FH, 2=middle, 3=to RH-BH)
  depth      : 7/8/9 after direction on RETURNS only (shot 1)
  error type : n/w/d/x/!/e (for error shots)
  ending     : *=winner, @=unforced, #=forced, C=incorrect challenge

SPECIAL
  leading 'c' characters (any number) indicate let serves
  'S' / 'R' / 'P' / 'Q' in the cell replace the whole point (server-won skip,
    returner-won skip, point penalty against server, point penalty against returner)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Shot-type vocabulary
# ---------------------------------------------------------------------------

SHOT_TYPE_LETTERS = "fbrsvzopuylmhijktq"
# index 0 is reserved for 'serve'; indices 1+ correspond to SHOT_TYPE_LETTERS.
SHOT_TYPE_VOCAB = ["serve"] + list(SHOT_TYPE_LETTERS)
SHOT_TYPE_TO_IDX = {s: i for i, s in enumerate(SHOT_TYPE_VOCAB)}
N_SHOT_TYPES = len(SHOT_TYPE_VOCAB)  # 19

# Error type vocabulary (0 = none)
ERROR_TYPE_VOCAB = ["none", "net", "wide", "deep", "both", "shank", "foot", "unknown", "time_violation"]
ERROR_TYPE_TO_IDX = {s: i for i, s in enumerate(ERROR_TYPE_VOCAB)}
_ERROR_CHAR_MAP = {
    "n": "net", "w": "wide", "d": "deep", "x": "both",
    "g": "foot", "e": "unknown", "!": "shank",
}

# Ending vocabulary
ENDING_VOCAB = ["none", "ace", "unreturnable", "winner", "unforced", "forced",
                "fault", "double_fault", "challenge", "time_violation"]
ENDING_TO_IDX = {s: i for i, s in enumerate(ENDING_VOCAB)}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    """One shot in a point."""
    shot_type: str = "serve"           # from SHOT_TYPE_VOCAB
    direction: int = 0                  # 0=unknown; 1-3 rally, 4-6 serve
    depth: int = 0                      # 0=unknown; 7/8/9 on return depth
    is_serve: bool = False
    is_approach: bool = False           # '+' after shot letter
    is_net_position: bool = False       # '-' explicit net position
    is_baseline_position: bool = False  # '=' explicit baseline position
    is_net_cord: bool = False           # ';' net cord
    is_drop_volley: bool = False        # '^' stop/drop volley
    is_serve_and_volley: bool = False   # '+' after serve digit
    error_type: str = "none"            # from ERROR_TYPE_VOCAB
    ending: str = "none"                # from ENDING_VOCAB

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ParsedPoint:
    """A parsed point as a sequence of shots."""
    shots: list[Shot] = field(default_factory=list)
    # True if the SERVER (player who served this point) won the point.
    server_won: Optional[bool] = None
    # True if 1st serve was a fault requiring a second serve.
    first_serve_fault: bool = False
    # True if it's a double fault.
    double_fault: bool = False
    # True if point could not be parsed.
    unparsed: bool = False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_SHOT_TYPE_SET = set(SHOT_TYPE_LETTERS)
_MODIFIER_SET = set("+-=;^")
_ERROR_CHAR_SET = set("nwdxge!")


def _parse_serve_prefix(text: str, i: int) -> tuple[Optional[Shot], int]:
    """Parse a serve at position i. Returns (shot, new_i) or (None, i) if absent."""
    n = len(text)
    # Skip lets
    while i < n and text[i] == "c":
        i += 1
    if i >= n:
        return None, i

    # Time violation
    if text[i] == "V":
        s = Shot(shot_type="serve", is_serve=True,
                 error_type="time_violation", ending="time_violation")
        return s, i + 1

    # Shank on serve
    if text[i] == "!":
        s = Shot(shot_type="serve", is_serve=True,
                 error_type="shank", ending="fault")
        return s, i + 1

    # Serve direction digit
    if text[i] not in "0123456":
        return None, i

    direction = int(text[i])
    # Map 0 -> 0 (unknown), 4-6 to their own values (wide/body/T); note that
    # 1-3 as "direction" values for a serve are unusual, but we keep as-is.
    i += 1

    shot = Shot(
        shot_type="serve", direction=direction, is_serve=True,
        depth=0, error_type="none", ending="none",
    )

    # Optional serve-and-volley marker
    if i < n and text[i] == "+":
        shot.is_serve_and_volley = True
        i += 1

    # Optional fault letter / outcome char
    if i < n and text[i] in _ERROR_CHAR_SET:
        shot.error_type = _ERROR_CHAR_MAP[text[i]]
        shot.ending = "fault"
        return shot, i + 1
    if i < n and text[i] == "*":
        shot.ending = "ace"
        return shot, i + 1
    if i < n and text[i] == "#":
        shot.ending = "unreturnable"
        return shot, i + 1
    if i < n and text[i] == "@":
        # Unforced error directly off the serve (unusual)
        shot.ending = "unforced"
        return shot, i + 1

    return shot, i


def _parse_rally_shot(text: str, i: int, shot_idx: int) -> tuple[Optional[Shot], int]:
    """Parse one rally shot starting at position i.

    shot_idx is 1 for the service return, 2+ for subsequent rally shots.
    Returns (shot, new_i) or (None, i) if nothing parsable.
    """
    n = len(text)
    if i >= n or text[i] not in _SHOT_TYPE_SET:
        return None, i

    shot = Shot(
        shot_type=text[i],
        direction=0, depth=0, is_serve=False,
        error_type="none", ending="none",
    )
    i += 1

    # Modifiers immediately after shot letter
    while i < n and text[i] in _MODIFIER_SET:
        c = text[i]
        if c == "+":
            shot.is_approach = True
        elif c == "-":
            shot.is_net_position = True
        elif c == "=":
            shot.is_baseline_position = True
        elif c == ";":
            shot.is_net_cord = True
        elif c == "^":
            shot.is_drop_volley = True
        i += 1

    # Direction digit (1-3 or 0). NB: returns may also have depth 7/8/9.
    if i < n and text[i] in "0123":
        shot.direction = int(text[i])
        i += 1

    # Return depth (only on the service return, shot_idx == 1)
    if shot_idx == 1 and i < n and text[i] in "0789":
        shot.depth = int(text[i])
        i += 1

    # Error type before ending
    if i < n and text[i] in _ERROR_CHAR_SET:
        shot.error_type = _ERROR_CHAR_MAP[text[i]]
        i += 1

    # Ending marker
    if i < n:
        c = text[i]
        if c == "*":
            shot.ending = "winner"; i += 1
        elif c == "@":
            shot.ending = "unforced"; i += 1
        elif c == "#":
            shot.ending = "forced"; i += 1
        elif c == "C":
            shot.ending = "challenge"; i += 1

    return shot, i


def _ending_means_server_won(
    shot: Shot, server_shot_parity: int, shot_idx_in_sequence: int,
) -> Optional[bool]:
    """Given a terminating shot, return True if server won, False if lost.

    server_shot_parity tells us which shot indices belong to the server:
      - server plays shots with idx % 2 == 0 (i.e. 0, 2, 4, ...)
      - returner plays shots with idx % 2 == 1
    So the player who hit the terminal shot is:
      - if shot_idx_in_sequence % 2 == 0 -> server
      - else -> returner
    """
    played_by_server = (shot_idx_in_sequence % 2 == 0)

    if shot.ending == "ace":
        return True
    if shot.ending == "unreturnable":
        return True
    if shot.ending == "winner":
        return played_by_server
    if shot.ending in ("unforced", "forced"):
        # Error by the player who hit the terminal shot
        return not played_by_server
    if shot.ending == "fault":
        # Won't happen here (handled earlier as fault path), but be safe
        return None
    return None


def parse_point(first_cell: str, second_cell: str = "") -> ParsedPoint:
    """Parse the ``1st`` / ``2nd`` cells of a point into an ordered shot list.

    Args:
        first_cell: contents of the '1st' column for this point. May be a full
            point (1st serve in + rally), a fault-only code (e.g. ``5d``), or a
            special single-letter code ('S', 'R', 'P', 'Q').
        second_cell: contents of the '2nd' column. Non-empty when the 1st was a
            fault; contains the 2nd serve and the rally.

    Returns:
        ParsedPoint with the full sequence of shots. An empty list of shots
        means the point was a special/skipped point.
    """
    pp = ParsedPoint()

    first_cell = (first_cell or "").strip()
    second_cell = (second_cell or "").strip()

    # Special point codes
    if first_cell in ("S", "R"):
        pp.server_won = (first_cell == "S")
        return pp
    if first_cell in ("P", "Q"):
        pp.server_won = (first_cell == "Q")  # P = penalty vs server
        return pp

    # Determine which cell has the rally
    serve1, after1 = _parse_serve_prefix(first_cell, 0) if first_cell else (None, 0)
    first_was_fault = serve1 is not None and serve1.ending == "fault"
    pp.first_serve_fault = first_was_fault

    if first_was_fault and not second_cell:
        # 1st was a fault, nothing in 2nd: treat as double fault (or incomplete)
        pp.double_fault = True
        pp.shots.append(serve1)
        pp.server_won = False
        return pp

    if first_was_fault:
        # Use the 2nd serve + rally from second_cell
        serve2, j = _parse_serve_prefix(second_cell, 0)
        if serve2 is None:
            pp.unparsed = True
            return pp
        # Record 1st serve as a fault (marked is_serve), then 2nd serve onward
        pp.shots.append(serve1)
        current_serve = serve2
        current_cell = second_cell
        start = j
    else:
        if serve1 is None:
            pp.unparsed = True
            return pp
        current_serve = serve1
        current_cell = first_cell
        start = after1

    # Now current_serve is the serve-in-play (or the sole serve info)
    pp.shots.append(current_serve)

    # If the serve already terminated the point (ace / unreturnable / unforced)
    if current_serve.ending in ("ace", "unreturnable", "unforced"):
        if current_serve.ending == "ace":
            pp.server_won = True
        elif current_serve.ending == "unreturnable":
            pp.server_won = True
        else:  # unforced error on the return
            pp.server_won = True
        return pp

    if current_serve.ending == "fault":
        # Double fault
        pp.double_fault = True
        pp.server_won = False
        return pp

    # Parse rally shots
    i = start
    shot_idx = 1  # this is the service return
    while i < len(current_cell):
        shot, new_i = _parse_rally_shot(current_cell, i, shot_idx)
        if shot is None:
            # Couldn't parse further; stop
            break
        pp.shots.append(shot)
        i = new_i
        if shot.ending not in ("none", "challenge"):
            # rally terminated
            server_won = _ending_means_server_won(
                shot, server_shot_parity=0, shot_idx_in_sequence=shot_idx,
            )
            pp.server_won = server_won
            break
        shot_idx += 1

    # If we reached end of string without a terminating marker, leave server_won
    # as None — downstream code can fall back to the PtWinner column if needed.
    return pp


# ---------------------------------------------------------------------------
# Tokenization: Shot -> feature vector
# ---------------------------------------------------------------------------


def shot_feature_dim() -> int:
    """Return the per-shot token dimension."""
    # One-hots:
    #   shot_type      (19)
    #   direction      (7)  -> [unknown, 1, 2, 3, 4, 5, 6]
    #   depth          (4)  -> [none, 7, 8, 9]
    #   error_type     (9)
    #   ending         (10)
    # Binary features:
    #   is_serve, is_approach, is_net_position, is_baseline_position,
    #   is_net_cord, is_drop_volley, is_serve_and_volley,
    #   is_self_action (the target player hit this shot)
    # Total: 19 + 7 + 4 + 9 + 10 + 8 = 57
    return N_SHOT_TYPES + 7 + 4 + len(ERROR_TYPE_VOCAB) + len(ENDING_VOCAB) + 8


def shot_to_vector(shot: Shot, is_self_action: bool) -> list[float]:
    """Encode one shot as a fixed-length feature vector.

    Args:
        shot: parsed shot.
        is_self_action: True if the target player played this shot.
    """
    vec = [0.0] * shot_feature_dim()
    off = 0

    # Shot type (19)
    vec[off + SHOT_TYPE_TO_IDX[shot.shot_type]] = 1.0
    off += N_SHOT_TYPES

    # Direction (7: 0=unknown, 1-3 rally, 4-6 serve)
    dir_idx = shot.direction if 0 <= shot.direction <= 6 else 0
    vec[off + dir_idx] = 1.0
    off += 7

    # Depth (4: 0=none, 7, 8, 9 -> indices 0, 1, 2, 3)
    depth_idx = {0: 0, 7: 1, 8: 2, 9: 3}.get(shot.depth, 0)
    vec[off + depth_idx] = 1.0
    off += 4

    # Error type
    vec[off + ERROR_TYPE_TO_IDX[shot.error_type]] = 1.0
    off += len(ERROR_TYPE_VOCAB)

    # Ending
    vec[off + ENDING_TO_IDX[shot.ending]] = 1.0
    off += len(ENDING_VOCAB)

    # Binary features
    vec[off + 0] = 1.0 if shot.is_serve else 0.0
    vec[off + 1] = 1.0 if shot.is_approach else 0.0
    vec[off + 2] = 1.0 if shot.is_net_position else 0.0
    vec[off + 3] = 1.0 if shot.is_baseline_position else 0.0
    vec[off + 4] = 1.0 if shot.is_net_cord else 0.0
    vec[off + 5] = 1.0 if shot.is_drop_volley else 0.0
    vec[off + 6] = 1.0 if shot.is_serve_and_volley else 0.0
    vec[off + 7] = 1.0 if is_self_action else 0.0

    return vec


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    examples = [
        "6*",                   # ace down the T
        "4f1n#",                # wide serve, fh return dir-1 net, forced
        "5b39w@",               # body, bh dir-3 depth-9 wide unforced
        "6f28f3n@",
        "4b37f+1f1n#",          # serve-in, return bh, fh approach, fh net forced
        "6s37f+1l2o1*",         # overhead winner
        "4sj=+1b2z2b2z3r2t2v1*",
    ]
    for e in examples:
        pp = parse_point(e)
        print(f"\n{e!r} -> ({len(pp.shots)} shots, server_won={pp.server_won})")
        for k, s in enumerate(pp.shots):
            print(f"  [{k}] {s.shot_type:>7} dir={s.direction} depth={s.depth}"
                  f" err={s.error_type:>8} end={s.ending:>10}"
                  f"{' SV' if s.is_serve_and_volley else ''}"
                  f"{' approach' if s.is_approach else ''}")
    print(f"\nshot_feature_dim = {shot_feature_dim()}")
