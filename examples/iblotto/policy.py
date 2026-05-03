"""Autoregressive policies — JAX port of Autoregressive_Policy.m.

The agent's policy is parametrised as a non-negative vector
``policy ∈ ℝ^{n+1}_{≥0}``: the first ``n`` entries are zone weights, the
last entry is a ``no-invest`` weight which controls the per-round
investment fraction. Sum is held at ``concentration`` throughout.

Each round the policy is updated as a convex combination of the previous
policy and an ``innovation`` driven by:

    1. wins on the previous round → reinvest into the same zones;
    2. losses on the previous round → disinvest from those zones;
    3. opponent's last-round allocation → pursue (or avoid) where they put
       resources;
    4. Gaussian innovation noise.

After the update, zone weights are clipped at zero, renormalised so that
``policy[:n].sum() = concentration − policy[n]`` (i.e., the no-invest
weight is fixed across rounds), and the whole vector is rescaled to sum
to ``concentration``.

The deterministic allocation rule is ``budget · policy[:n] / concentration``
(the MATLAB had a Dirichlet-sample option commented out; we mirror the
selected, deterministic-by-default branch).

Six parameters per agent, packed in a flat array:

    params[0]  learning_rate     — convex-combination weight on innovation
    params[1]  win_reinvestment  — multiplier on past_policy[zone] for won zones
    params[2]  loss_disinvestment — same, for lost zones (note: subtracted)
    params[3]  opponent_allocation — pursue (+) or avoid (−) opponent's bids
    params[4]  innovation_noise  — std-dev of zero-mean Gaussian noise
    params[5]  concentration     — target sum of policy vector
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


_LR, _WIN, _LOSS, _OPP, _NOISE, _CONC = range(6)


# ---------------------------------------------------------------------------
# Initial policy + initial allocation (Dirichlet-sampled like the MATLAB)
# ---------------------------------------------------------------------------


def initial_policy(
    n_zones: int, params: Array, investment_fraction: float, budget: float,
    key: Array,
) -> tuple[Array, Array]:
    """Sample an agent's first allocation.

    Mirrors the ``policy.initial.strategy = 'uniform'`` branch in
    ``Iterated_Blotto_Autoregressive.m``: build a uniform policy with the
    no-invest weight set so that the expected fraction of budget allocated
    is ``investment_fraction``, normalise to ``concentration``, then sample
    a Dirichlet of shape ``policy + 1``.

    Returns:
        ``(policy_params, allocation)`` where ``policy_params`` has shape
        ``(n_zones + 1,)`` and ``allocation`` has shape ``(n_zones,)``.
    """
    n = n_zones
    conc = params[_CONC]
    # Uniform zone weights, no-invest weight chosen for target investment fraction
    no_invest = n * (1.0 - investment_fraction) / investment_fraction
    raw = jnp.concatenate([jnp.ones((n,)), jnp.array([no_invest])])
    policy = conc * raw / jnp.sum(raw)              # policy.sum() == conc

    # Dirichlet sample: gamrnd(policy + 1, 1) / sum
    sample = jax.random.dirichlet(key, alpha=policy + 1.0)
    allocation = budget * sample[:n]
    return policy, allocation


# ---------------------------------------------------------------------------
# Autoregressive update (one round)
# ---------------------------------------------------------------------------


def autoregressive_update(
    past_policy: Array,
    zone_outcomes: Array,           # shape (n,), values in {-1, 0, +1}
    opp_alloc: Array,               # shape (n,)
    params: Array,
    key: Array,
) -> Array:
    """Apply the autoregressive update rule to ``past_policy``.

    All inputs/outputs are jit-friendly; randomness threaded through ``key``.
    """
    n = past_policy.shape[0] - 1
    lr      = params[_LR]
    win_re  = params[_WIN]
    loss_di = params[_LOSS]
    opp_p   = params[_OPP]
    innov_s = params[_NOISE]
    conc    = params[_CONC]

    past_zone     = past_policy[:n]
    past_no_inv   = past_policy[n]
    past_zone_sum = jnp.sum(past_zone)

    # Decay
    decayed_zone = (1.0 - lr) * past_zone

    # Win/loss innovation
    innov_w = jnp.where(zone_outcomes > 0,  win_re  * past_zone,  0.0)
    innov_l = jnp.where(zone_outcomes < 0, -loss_di * past_zone,  0.0)
    innovation = innov_w + innov_l

    # Opponent-allocation innovation (centred).  In normal operation
    # ``opp_alloc`` comes from the previous round's actual bids and has
    # positive sum; if a previously-collapsed agent's NaN allocation has
    # propagated into ``opp_alloc``, NaN flows through here and downstream.
    opp_sum  = jnp.sum(opp_alloc)
    opp_mean = jnp.mean(opp_alloc)
    innovation = innovation + (
        opp_p * past_zone_sum / opp_sum * (opp_alloc - opp_mean)
    )

    # Gaussian innovation noise
    noise = jax.random.normal(key, shape=(n,))
    innovation = innovation + past_zone_sum * innov_s * noise

    # Move
    new_zone = decayed_zone + lr * innovation
    new_zone = jnp.maximum(new_zone, 0.0)

    # Renormalise zone weights to (conc - past_no_inv).  When the zones all
    # clip to zero the MATLAB produces 0/0 = NaN here; we mirror that exactly
    # because the downstream behaviour (NaN allocations → 0 payouts to both
    # players) is what determines the tournament outcome for unstable agents.
    z_sum     = jnp.sum(new_zone)
    new_zone  = (conc - past_no_inv) * new_zone / z_sum

    new_policy = jnp.concatenate([new_zone, past_no_inv[None]])
    # Final renormalisation; in normal operation this is a no-op (sum already
    # equals ``conc``). When ``new_zone`` contains NaN the whole vector goes
    # NaN, which propagates as expected.
    new_policy = conc * new_policy / jnp.sum(new_policy)
    return new_policy


def deterministic_allocation(policy: Array, budget: Array, params: Array) -> Array:
    """The MATLAB's deterministic allocation rule.

    ``allocation = budget · policy[:n] / concentration``.
    """
    n = policy.shape[0] - 1
    conc = params[_CONC]
    return budget * policy[:n] / conc
