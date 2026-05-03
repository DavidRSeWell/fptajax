"""Lure v0 — 7x7 gridworld with a central pressure cell.

Focal agent (F) and opponent (O) move simultaneously on a 7x7 grid.
Walls at rows 2 and 4 (except column 3) force any agent crossing
between the north half and south half of the grid to pass through
column 3, whose middle cell is the pressure cell P = (3, 3).

Layout::

         0  1  2  3  4  5  6
       +---------------------+
     0 | F  .  .  R1 .  .  . |     F: focal start
     1 | .  .  .  .  .  .  . |     O: opponent start
     2 | #  #  #  .  #  #  # |     R1: resource (north)
     3 | .  .  .  P  .  .  . |     R2: resource (south)
     4 | #  #  #  .  #  #  # |     P:  pressure cell (focal compel target)
     5 | .  .  .  .  .  .  . |     #:  wall
     6 | .  .  .  R2 .  .  O |
       +---------------------+

Focal reward:
    +1 per resource collected, -1 per resource the opponent collects,
    +1 per step the opponent occupies P (the compel bonus).

The setup is asymmetric: the opponent has no incentive to be on P; the
opponent's reward structure depends only on its scripted heuristic. So
f(focal, opp) != -f(opp, focal) — this is why we use the unconstrained-C
g-FPTA variant downstream rather than the skew-symmetric form.

Actions per agent::

    0: N  (-1, 0)
    1: S  (+1, 0)
    2: E  ( 0,+1)
    3: W  ( 0,-1)
    4: stay

Simultaneous-move collision rule: both agents submit actions; tentative
new positions are computed independently; if the tentatives collide
(same cell), both agents stay. Swapping cells is allowed (not a
collision in this formulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 7
EPISODE_LENGTH = 25

ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),   # N
    1: (1, 0),    # S
    2: (0, 1),    # E
    3: (0, -1),   # W
    4: (0, 0),    # stay
}
ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "stay"}
N_ACTIONS = 5

# Walls block rows 2 and 4 everywhere except column 3 — creating a
# single vertical corridor through P = (3, 3).
WALLS: frozenset[tuple[int, int]] = frozenset(
    [(2, c) for c in range(GRID_SIZE) if c != 3]
    + [(4, c) for c in range(GRID_SIZE) if c != 3]
)

FOCAL_START: tuple[int, int] = (0, 0)
OPP_START: tuple[int, int] = (6, 6)
RESOURCE_CELLS: tuple[tuple[int, int], ...] = ((0, 3), (6, 3))  # R1 (north), R2 (south)
PRESSURE_CELL: tuple[int, int] = (3, 3)
OPP_HOME: frozenset[tuple[int, int]] = frozenset({(5, 5), (5, 6), (6, 5), (6, 6)})

RESOURCE_RESPAWN_DELAY = 5     # steps between collection and respawn
RESOURCE_REWARD = 1.0          # per resource collected by focal
RESOURCE_PENALTY = 1.0         # per resource collected by opp (from focal's view)
COMPEL_BONUS_PER_STEP = 1.0    # focal gains this per step opp occupies P


# ---------------------------------------------------------------------------
# State and transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LureState:
    focal_pos: tuple[int, int]
    opp_pos: tuple[int, int]
    # One entry per cell in RESOURCE_CELLS: (available, respawn_countdown).
    # respawn_countdown is meaningful only when available=False.
    resource_state: tuple[tuple[bool, int], ...]
    step: int


def initial_state() -> LureState:
    return LureState(
        focal_pos=FOCAL_START,
        opp_pos=OPP_START,
        resource_state=tuple((True, 0) for _ in RESOURCE_CELLS),
        step=0,
    )


def _apply_action(pos: tuple[int, int], action: int) -> tuple[int, int]:
    """Return tentative new position after taking ``action``.

    Out-of-bounds or into-wall moves are silently blocked (agent stays).
    """
    dr, dc = ACTIONS[action]
    new_pos = (pos[0] + dr, pos[1] + dc)
    if not (0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE):
        return pos
    if new_pos in WALLS:
        return pos
    return new_pos


def step(
    state: LureState, focal_action: int, opp_action: int,
) -> tuple[LureState, float, float, bool, dict]:
    """Advance the environment one step.

    Returns ``(next_state, focal_reward, opp_reward, done, info)``.
    ``focal_reward`` includes the compel bonus; ``opp_reward`` only
    counts resources from the opponent's perspective.
    """
    f_new = _apply_action(state.focal_pos, focal_action)
    o_new = _apply_action(state.opp_pos, opp_action)

    # Collision: both tried to land on the same cell -> both stay.
    # (Swaps — where f_new = opp_current and o_new = focal_current —
    # are *not* collisions and both agents move.)
    if f_new == o_new:
        f_new = state.focal_pos
        o_new = state.opp_pos

    # Resource collection and respawn.
    focal_reward = 0.0
    opp_reward = 0.0
    new_resource_state: list[tuple[bool, int]] = []
    for cell, (avail, respawn) in zip(RESOURCE_CELLS, state.resource_state):
        if avail:
            on_f = (f_new == cell)
            on_o = (o_new == cell)
            if on_f and not on_o:
                focal_reward += RESOURCE_REWARD
                new_resource_state.append((False, RESOURCE_RESPAWN_DELAY))
            elif on_o and not on_f:
                opp_reward += RESOURCE_REWARD
                focal_reward -= RESOURCE_PENALTY
                new_resource_state.append((False, RESOURCE_RESPAWN_DELAY))
            else:
                # Either nobody is there, or both (ruled out by the collision
                # rule above — both on the same cell means both tried to move
                # to it and were blocked; but defensively leave resource
                # available).
                new_resource_state.append((True, 0))
        else:
            if respawn > 0:
                new_resource_state.append((False, respawn - 1))
            else:
                new_resource_state.append((True, 0))

    # Compel bonus.
    if o_new == PRESSURE_CELL:
        focal_reward += COMPEL_BONUS_PER_STEP

    new_state = LureState(
        focal_pos=f_new,
        opp_pos=o_new,
        resource_state=tuple(new_resource_state),
        step=state.step + 1,
    )
    done = new_state.step >= EPISODE_LENGTH

    info = {
        "opp_on_P": o_new == PRESSURE_CELL,
        "focal_action": focal_action,
        "opp_action": opp_action,
    }
    return new_state, focal_reward, opp_reward, done, info


# ---------------------------------------------------------------------------
# Tokenisation (for the FPTA encoder pipeline)
# ---------------------------------------------------------------------------

# state: focal_pos (2) + opp_pos (2) + resource_avail (len(RESOURCE_CELLS)) + step_progress (1)
STATE_FEATURE_DIM = 2 + 2 + len(RESOURCE_CELLS) + 1   # = 7
TOKEN_DIM = STATE_FEATURE_DIM + 2 * N_ACTIONS         # = 17


def state_features(state: LureState) -> np.ndarray:
    """Pack a LureState into a fixed-length feature vector (normalised)."""
    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    features[0] = state.focal_pos[0] / (GRID_SIZE - 1)
    features[1] = state.focal_pos[1] / (GRID_SIZE - 1)
    features[2] = state.opp_pos[0] / (GRID_SIZE - 1)
    features[3] = state.opp_pos[1] / (GRID_SIZE - 1)
    for i, (avail, _) in enumerate(state.resource_state):
        features[4 + i] = 1.0 if avail else 0.0
    features[-1] = state.step / EPISODE_LENGTH
    return features


def make_token(
    state: LureState, self_action: int, opp_action: int,
) -> np.ndarray:
    """Produce a ``(TOKEN_DIM,)`` token for the FPTA encoder.

    Concatenates normalised state features with two action one-hots.
    For focal-perspective tokens, ``self_action`` is the focal action
    and ``opp_action`` is the opponent action; for opponent-perspective
    tokens, swap.
    """
    s = state_features(state)
    self_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    self_oh[self_action] = 1.0
    opp_oh = np.zeros(N_ACTIONS, dtype=np.float32)
    opp_oh[opp_action] = 1.0
    return np.concatenate([s, self_oh, opp_oh])


# ---------------------------------------------------------------------------
# Convenience: run a single episode given two bots
# ---------------------------------------------------------------------------


def run_episode(focal_bot, opp_bot, record: bool = True) -> dict:
    """Run a single episode. Both bots must expose ``.act(state) -> int``.

    ``focal_bot`` receives the state as-is (focal is the distinguished
    agent). ``opp_bot`` may treat ``state.opp_pos`` as its position.

    Returns a dict with keys ``focal_reward, opp_reward, focal_actions,
    opp_actions, focal_positions, opp_positions, states, steps_on_P``.
    """
    state = initial_state()
    focal_actions: list[int] = []
    opp_actions: list[int] = []
    focal_positions: list[tuple[int, int]] = [state.focal_pos]
    opp_positions: list[tuple[int, int]] = [state.opp_pos]
    states: list[LureState] = [state]
    focal_total = 0.0
    opp_total = 0.0
    steps_on_P = 0

    while True:
        f_act = focal_bot.act(state)
        o_act = opp_bot.act(state)
        next_state, f_r, o_r, done, info = step(state, f_act, o_act)
        focal_actions.append(f_act)
        opp_actions.append(o_act)
        focal_total += f_r
        opp_total += o_r
        if info["opp_on_P"]:
            steps_on_P += 1
        state = next_state
        focal_positions.append(state.focal_pos)
        opp_positions.append(state.opp_pos)
        states.append(state)
        if done:
            break

    return {
        "focal_reward": focal_total,
        "opp_reward": opp_total,
        "focal_actions": focal_actions,
        "opp_actions": opp_actions,
        "focal_positions": focal_positions,
        "opp_positions": opp_positions,
        "states": states if record else None,
        "steps_on_P": steps_on_P,
    }
