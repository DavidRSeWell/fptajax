"""Iterated Blotto game mechanics — JAX port of Single_Blotto_Round.m.

Pure-functional, jit-compatible. All randomness threaded through ``key``.

Game options are bundled in :class:`GameOptions`. The single-round function
:func:`single_round` is the workhorse; per round it returns the payouts to
each player, the resources returned (per side, per zone), and the
information observed (per-zone win record + opponent allocations seen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Game options (static struct; passed as static_argnames in jit)
# ---------------------------------------------------------------------------


# Encode mode strings as small integer codes so the option struct can be a
# jax-pytree-friendly NamedTuple. Comments map back to MATLAB strings.

CSF_AUCTION              = 0   # 'auction'
CSF_THRESHOLD            = 1   # 'win by threshold auction'
CSF_LOTTERY              = 2   # 'lottery'

RR_NONE                  = 0   # 'none'
RR_KEEP_INVESTMENT       = 1   # 'keep investment'
RR_KEEP_INV_DIFFERENCE   = 2   # 'keep investment difference'
RR_WIN_OPPONENT_BID      = 3   # 'win opponent bid'
RR_KEEP_LOSING_BIDS      = 4   # 'keep losing bids'

REALLOC_STAY_IN_ZONE     = 0   # 'stay in zone'
REALLOC_REALLOCATE       = 1   # 'reallocate'

INFO_ALL_INVESTMENTS     = 0   # 'all investments'
INFO_TOTAL_INVESTED      = 1   # 'total invested'  (currently not used by policy)


@dataclass(frozen=True)
class GameOptions:
    n_zones: int
    zone_values: Array                 # (n_zones,)
    csf_mode: int = CSF_AUCTION
    csf_threshold: float = 0.0
    csf_temperature: float = 1.0
    resource_return_mode: int = RR_NONE
    reallocation_mode: int = REALLOC_STAY_IN_ZONE
    depreciation: float = 0.5
    info_mode: int = INFO_ALL_INVESTMENTS
    info_noise: float = 0.0
    n_rounds: int = 100                # 'fixed end' stopping


# ---------------------------------------------------------------------------
# Single-round outputs as a NamedTuple
# ---------------------------------------------------------------------------


class RoundOutcome(NamedTuple):
    """Per-round outputs.

    Attributes:
        payouts:           (2,)               player payouts this round
        resources_returned: (2, n_zones)      resources returned to each player by zone
        zone_outcomes:     (n_zones,)         per-zone outcome from P1's perspective:
                                              +1 = P1 won, -1 = P2 won, 0 = tied
        opp_alloc_seen:    (2, n_zones)       P1's view of P2's alloc, P2's view of P1's
    """
    payouts: Array
    resources_returned: Array
    zone_outcomes: Array
    opp_alloc_seen: Array


# ---------------------------------------------------------------------------
# CSF: who wins each zone
# ---------------------------------------------------------------------------


def _csf_decide(diff: Array, opts: GameOptions, key: Array) -> tuple[Array, Array, Array]:
    """Return (won_by_p1, lost_by_p1, tied) boolean masks of shape (n_zones,)."""
    if opts.csf_mode == CSF_AUCTION:
        won  = diff > 0
        lost = diff < 0
        tied = diff == 0
    elif opts.csf_mode == CSF_THRESHOLD:
        thr = opts.csf_threshold
        won  = diff > thr
        lost = -diff > thr
        tied = jnp.zeros_like(diff, dtype=bool)
    elif opts.csf_mode == CSF_LOTTERY:
        win_prob = jax.nn.sigmoid(diff / opts.csf_temperature)
        zeta = jax.random.uniform(key, shape=diff.shape)
        won  = zeta < win_prob
        lost = ~won
        tied = jnp.zeros_like(diff, dtype=bool)
    else:
        raise ValueError(f"Unknown CSF mode: {opts.csf_mode}")
    return won, lost, tied


# ---------------------------------------------------------------------------
# Resource return: who keeps what after a round
# ---------------------------------------------------------------------------


def _compute_resources_returned(
    allocations: Array, won: Array, lost: Array, tied: Array, opts: GameOptions,
) -> Array:
    """Return (2, n_zones) resources flowing back to each player."""
    a1 = allocations[0]
    a2 = allocations[1]
    diff = a1 - a2
    n = a1.shape[0]
    out = jnp.zeros((2, n))

    mode = opts.resource_return_mode
    if mode == RR_NONE:
        pass
    elif mode == RR_KEEP_INVESTMENT:
        out = out.at[0].set(jnp.where(won, a1, jnp.where(tied, 0.5 * a1, 0.0)))
        out = out.at[1].set(jnp.where(lost, a2, jnp.where(tied, 0.5 * a1, 0.0)))   # MATLAB: 0.5*a1 (sic)
    elif mode == RR_KEEP_INV_DIFFERENCE:
        out = out.at[0].set(jnp.where(won, diff, 0.0))
        out = out.at[1].set(jnp.where(lost, -diff, 0.0))
    elif mode == RR_WIN_OPPONENT_BID:
        out = out.at[0].set(jnp.where(won, a2, 0.0))
        out = out.at[1].set(jnp.where(lost, a1, 0.0))
    elif mode == RR_KEEP_LOSING_BIDS:
        out = out.at[0].set(jnp.where(lost, a1, jnp.where(tied, 0.5 * a1, 0.0)))
        out = out.at[1].set(jnp.where(won, a2, jnp.where(tied, 0.5 * a2, 0.0)))
    else:
        raise ValueError(f"Unknown resource_return_mode: {mode}")

    return opts.depreciation * out


# ---------------------------------------------------------------------------
# Single round
# ---------------------------------------------------------------------------


def single_round(allocations: Array, opts: GameOptions, key: Array) -> RoundOutcome:
    """Simulate one Blotto round.

    Args:
        allocations: (2, n_zones) — non-negative bids per player per zone.
        opts:        :class:`GameOptions`.
        key:         PRNG key (consumed only by lottery CSF / info noise).

    Returns:
        :class:`RoundOutcome`.
    """
    a1 = allocations[0]
    a2 = allocations[1]
    diff = a1 - a2
    zone_values = opts.zone_values

    k_csf, k_info = jax.random.split(key)
    won, lost, tied = _csf_decide(diff, opts, k_csf)

    payouts = jnp.stack([
        jnp.sum(jnp.where(won,  zone_values, 0.0)) + 0.5 * jnp.sum(jnp.where(tied, zone_values, 0.0)),
        jnp.sum(jnp.where(lost, zone_values, 0.0)) + 0.5 * jnp.sum(jnp.where(tied, zone_values, 0.0)),
    ])

    resources_returned = _compute_resources_returned(allocations, won, lost, tied, opts)

    zone_outcomes = jnp.where(won, 1.0, jnp.where(lost, -1.0, 0.0))

    # Information observation: each player sees the OTHER's bids (with optional noise).
    if opts.info_mode == INFO_ALL_INVESTMENTS:
        n = a1.shape[0]
        noise = opts.info_noise * jax.random.normal(k_info, shape=(2, n))
        opp_alloc_seen = jnp.stack([a2, a1]) + noise
    else:
        # Total-invested mode: caller can sum opp_alloc_seen and ignore the
        # per-zone resolution; kept here for API parity.
        n = a1.shape[0]
        noise = opts.info_noise * jax.random.normal(k_info, shape=(2, n))
        opp_alloc_seen = jnp.stack([a2, a1]) + noise

    return RoundOutcome(
        payouts=payouts,
        resources_returned=resources_returned,
        zone_outcomes=zone_outcomes,
        opp_alloc_seen=opp_alloc_seen,
    )
