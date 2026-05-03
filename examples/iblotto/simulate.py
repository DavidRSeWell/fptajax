"""Iterated Blotto game simulator — JAX port of Iterated_Blotto_Autoregressive.m.

Memory-=1 only (matching the driver scripts). The full game is a
``lax.scan`` over rounds so the whole trajectory can be jit + vmap.

The carry state per round is:

    budgets          (2,)
    policies         (2, n+1)        autoregressive policy parameters
    preexisting      (2, n)          carry-over allocations (locked-in zones)
    last_outcomes    (2, n)          per-zone outcomes from each player's view
    last_opp_alloc   (2, n)          last-round opponent allocations seen
    cum_payouts      (2,)            running total payouts

Per-round outputs (collected via scan into a trajectory):

    allocations      (2, n)
    payouts_round    (2,)
    budgets          (2,)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from examples.iblotto.game import (
    GameOptions, RoundOutcome, single_round,
    REALLOC_REALLOCATE, REALLOC_STAY_IN_ZONE,
)
from examples.iblotto.policy import (
    autoregressive_update, deterministic_allocation, initial_policy,
)


class GameState(NamedTuple):
    budgets: Array            # (2,)
    policies: Array           # (2, n + 1)
    preexisting: Array        # (2, n)
    last_outcomes: Array      # (2, n)
    last_opp_alloc: Array     # (2, n)
    cum_payouts: Array        # (2,)


class StepOutput(NamedTuple):
    allocations: Array        # (2, n)
    payouts_round: Array      # (2,)
    budgets: Array            # (2,)


def initial_state(
    p1_params: Array, p2_params: Array, opts: GameOptions,
    p1_budget: float, p2_budget: float,
    p1_inv_frac: float, p2_inv_frac: float,
    key: Array,
) -> tuple[GameState, Array]:
    """Sample initial policies + allocations.

    Returns ``(state, first_allocations)`` where ``first_allocations`` is
    used as the round-0 bid (and not produced by the autoregressive update).
    """
    k1, k2 = jax.random.split(key)
    n = opts.n_zones
    pol1, alloc1 = initial_policy(n, p1_params, p1_inv_frac, p1_budget, k1)
    pol2, alloc2 = initial_policy(n, p2_params, p2_inv_frac, p2_budget, k2)

    state = GameState(
        budgets       = jnp.array([p1_budget, p2_budget], dtype=jnp.float64),
        policies      = jnp.stack([pol1, pol2]),
        preexisting   = jnp.zeros((2, n)),
        last_outcomes = jnp.zeros((2, n)),
        last_opp_alloc = jnp.zeros((2, n)),
        cum_payouts   = jnp.zeros((2,)),
    )
    first_allocations = jnp.stack([alloc1, alloc2])
    return state, first_allocations


def _step(
    state: GameState, first_allocations: Array | None,
    p1_params: Array, p2_params: Array, opts: GameOptions, key: Array,
) -> tuple[GameState, StepOutput]:
    """One round of play.  ``first_allocations`` shortcircuits the policy update on round 0."""
    k_pol1, k_pol2, k_round = jax.random.split(key, 3)

    if first_allocations is None:
        new_pol1 = autoregressive_update(
            state.policies[0], state.last_outcomes[0],
            state.last_opp_alloc[0], p1_params, k_pol1,
        )
        new_pol2 = autoregressive_update(
            state.policies[1], state.last_outcomes[1],
            state.last_opp_alloc[1], p2_params, k_pol2,
        )
        alloc1 = deterministic_allocation(new_pol1, state.budgets[0], p1_params)
        alloc2 = deterministic_allocation(new_pol2, state.budgets[1], p2_params)
        new_policies = jnp.stack([new_pol1, new_pol2])
    else:
        alloc1, alloc2 = first_allocations[0], first_allocations[1]
        new_policies = state.policies

    allocations = jnp.stack([alloc1, alloc2]) + state.preexisting

    outcome: RoundOutcome = single_round(allocations, opts, k_round)

    # Budget update: subtract own raw allocation (NOT including preexisting)
    new_budgets = jnp.maximum(state.budgets - jnp.stack([
        jnp.sum(alloc1), jnp.sum(alloc2),
    ]), 0.0)

    # Resource return + reallocation handling
    if opts.reallocation_mode == REALLOC_REALLOCATE:
        new_budgets = new_budgets + jnp.sum(outcome.resources_returned, axis=1)
        new_preexisting = jnp.zeros_like(state.preexisting)
    else:   # REALLOC_STAY_IN_ZONE
        new_preexisting = outcome.resources_returned

    # Information update: P1 sees zone_outcomes as-is; P2 sees them flipped.
    new_last_outcomes = jnp.stack([outcome.zone_outcomes, -outcome.zone_outcomes])
    new_last_opp_alloc = outcome.opp_alloc_seen

    new_state = GameState(
        budgets        = new_budgets,
        policies       = new_policies,
        preexisting    = new_preexisting,
        last_outcomes  = new_last_outcomes,
        last_opp_alloc = new_last_opp_alloc,
        cum_payouts    = state.cum_payouts + outcome.payouts,
    )
    output = StepOutput(
        allocations   = allocations,
        payouts_round = outcome.payouts,
        budgets       = new_budgets,
    )
    return new_state, output


def simulate_iblotto(
    p1_params: Array, p2_params: Array,
    opts: GameOptions,
    key: Array,
    p1_budget: float = 1000.0,
    p2_budget: float = 1000.0,
    p1_inv_frac: float = 0.1,
    p2_inv_frac: float = 0.1,
    return_history: bool = False,
):
    """Run a full ``opts.n_rounds``-round game and return the cumulative payouts.

    Args:
        p1_params, p2_params: shape (6,) policy parameter vectors.
        opts:                 :class:`GameOptions`.
        key:                  PRNG key.
        p1_budget, p2_budget: per-player initial budget.
        p1_inv_frac, p2_inv_frac: initial investment fraction (used by the
            uniform initial-policy sampler).
        return_history: if True, also return per-round trajectories.

    Returns:
        If ``return_history=False``: ``cum_payouts`` shape ``(2,)``.
        Else: ``(cum_payouts, history)`` where history has fields with leading
        dimension ``opts.n_rounds``.
    """
    k_init, k_round0, k_scan = jax.random.split(key, 3)
    state, first_allocations = initial_state(
        p1_params, p2_params, opts,
        p1_budget, p2_budget, p1_inv_frac, p2_inv_frac, k_init,
    )

    # Round 0 uses the Dirichlet-sampled initial allocations rather than
    # the autoregressive update.
    state, out0 = _step(state, first_allocations, p1_params, p2_params, opts, k_round0)

    if opts.n_rounds <= 1:
        if return_history:
            history = StepOutput(
                allocations   = out0.allocations[None],
                payouts_round = out0.payouts_round[None],
                budgets       = out0.budgets[None],
            )
            return state.cum_payouts, history
        return state.cum_payouts

    # Remaining rounds use the autoregressive update.
    keys_rest = jax.random.split(k_scan, opts.n_rounds - 1)

    def scan_step(carry, k):
        new_carry, out = _step(carry, None, p1_params, p2_params, opts, k)
        return new_carry, out

    state, history = jax.lax.scan(scan_step, state, keys_rest)

    if return_history:
        # Prepend round-0 output so history has length n_rounds
        full_history = StepOutput(
            allocations   = jnp.concatenate([out0.allocations[None], history.allocations]),
            payouts_round = jnp.concatenate([out0.payouts_round[None], history.payouts_round]),
            budgets       = jnp.concatenate([out0.budgets[None], history.budgets]),
        )
        return state.cum_payouts, full_history
    return state.cum_payouts
