"""Tests for behavioral FPTA (DeepSets encoder + learned traits)."""

import jax
import jax.numpy as jnp
import pytest

from fptajax.behavioral import (
    SetEncoder,
    _BehavioralModel,
    BehavioralFPTAResult,
    behavioral_fpta,
    _behavioral_loss,
)
from fptajax.neural import NeuralBasis, SkewParam, TrainConfig


# ---------------------------------------------------------------------------
# SetEncoder tests
# ---------------------------------------------------------------------------


def test_set_encoder_shape():
    """SetEncoder should produce correct output shapes."""
    key = jax.random.PRNGKey(0)
    enc = SetEncoder(sa_dim=4, trait_dim=3, phi_hidden=(16,), rho_hidden=(8,), key=key)

    # Single agent with 5 state-action pairs
    sa = jax.random.normal(key, (5, 4))
    trait = enc(sa)
    assert trait.shape == (3,), f"Expected (3,), got {trait.shape}"


def test_set_encoder_batch():
    """SetEncoder batch encoding should work."""
    key = jax.random.PRNGKey(0)
    enc = SetEncoder(sa_dim=4, trait_dim=3, phi_hidden=(16,), rho_hidden=(8,), key=key)

    # 10 agents, each with 5 (s,a) pairs
    sa_batch = jax.random.normal(key, (10, 5, 4))
    traits = enc.encode_batch(sa_batch)
    assert traits.shape == (10, 3)


def test_set_encoder_masked():
    """SetEncoder should handle masked inputs."""
    key = jax.random.PRNGKey(0)
    enc = SetEncoder(sa_dim=2, trait_dim=4, phi_hidden=(16,), rho_hidden=(8,), key=key)

    sa = jax.random.normal(key, (5, 2))
    mask = jnp.array([True, True, True, False, False])

    trait_masked = enc(sa, mask)
    assert trait_masked.shape == (4,)

    # Masked result should differ from unmasked
    trait_full = enc(sa)
    assert not jnp.allclose(trait_masked, trait_full, atol=1e-3)


def test_set_encoder_permutation_invariant():
    """DeepSets should be approximately permutation invariant."""
    key = jax.random.PRNGKey(42)
    enc = SetEncoder(sa_dim=3, trait_dim=2, phi_hidden=(16,), rho_hidden=(8,), key=key)

    sa = jax.random.normal(key, (6, 3))

    # Original order
    t1 = enc(sa)

    # Permuted order
    perm = jnp.array([3, 1, 5, 0, 4, 2])
    t2 = enc(sa[perm])

    assert jnp.allclose(t1, t2, atol=1e-5), \
        f"Not permutation invariant: diff = {jnp.max(jnp.abs(t1 - t2))}"


def test_set_encoder_batch_masked():
    """Batch encoding with masks."""
    key = jax.random.PRNGKey(0)
    enc = SetEncoder(sa_dim=2, trait_dim=3, phi_hidden=(16,), rho_hidden=(8,), key=key)

    sa_batch = jax.random.normal(key, (4, 5, 2))
    mask = jnp.array([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, False, False, False],
        [True, False, False, False, False],
    ])
    traits = enc.encode_batch(sa_batch, mask)
    assert traits.shape == (4, 3)


# ---------------------------------------------------------------------------
# Behavioral loss test
# ---------------------------------------------------------------------------


def test_behavioral_loss_computes():
    """Behavioral loss should compute without errors."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    model = _BehavioralModel(
        encoder=SetEncoder(sa_dim=3, trait_dim=4, phi_hidden=(8,), rho_hidden=(8,), key=k1),
        basis=NeuralBasis(trait_dim=4, d=6, hidden_dims=(8,), key=k2),
        skew=SkewParam(d=6, key=k3),
    )

    sa_i = jax.random.normal(key, (8, 5, 3))
    sa_j = jax.random.normal(key, (8, 5, 3))
    f_ij = jax.random.normal(key, (8,))

    loss, metrics = _behavioral_loss(model, sa_i, sa_j, None, None, f_ij, 0.1)
    assert jnp.isfinite(loss)
    assert "mse" in metrics
    assert "ortho" in metrics


# ---------------------------------------------------------------------------
# End-to-end behavioral FPTA test
# ---------------------------------------------------------------------------


def _make_synthetic_rps_data(key, n_agents=12, k_per_agent=20):
    """Create synthetic RPS-like behavioral data.

    Agents have a hidden 'type' (rock-biased, paper-biased, scissors-biased).
    Their behavior data reflects which action they tend to play.
    The payoff is determined by RPS rules applied to their type distributions.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # 3 types of agents: rock, paper, scissors biased
    n_per_type = n_agents // 3
    types = []  # action distribution per agent
    for t in range(3):
        # Each type has a dominant action
        probs = jnp.ones(3) * 0.1
        probs = probs.at[t].set(0.8)
        types.extend([probs] * n_per_type)

    # Fill remaining agents
    while len(types) < n_agents:
        types.append(jnp.ones(3) / 3.0)
    types = jnp.stack(types)  # (N, 3)

    # Generate behavior data: for each agent, sample state-action pairs
    # State = random context (irrelevant), Action = one-hot from their distribution
    sa_dim = 3 + 3  # 3 state dims + 3 action dims (one-hot)
    agent_data = jnp.zeros((n_agents, k_per_agent, sa_dim))

    for i in range(n_agents):
        k_i = jax.random.fold_in(k2, i)
        states = jax.random.normal(k_i, (k_per_agent, 3)) * 0.1
        actions_idx = jax.random.choice(
            jax.random.fold_in(k3, i), 3, shape=(k_per_agent,), p=types[i]
        )
        actions = jax.nn.one_hot(actions_idx, 3)
        agent_data = agent_data.at[i].set(jnp.concatenate([states, actions], axis=-1))

    # Build payoff matrix from expected matchups
    # E[payoff(i,j)] based on type distributions — like weighted RPS
    rps = jnp.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=jnp.float32)
    F = types @ rps @ types.T  # (N, N) expected payoff
    F = 0.5 * (F - F.T)

    return agent_data, None, F


def test_behavioral_fpta_synthetic():
    """Behavioral FPTA on synthetic RPS data should produce valid output."""
    key = jax.random.PRNGKey(123)
    agent_data, agent_mask, F = _make_synthetic_rps_data(key, n_agents=9, k_per_agent=15)

    config = TrainConfig(
        lr=1e-3,
        n_steps=300,
        batch_size=32,
        ortho_weight=0.05,
        c_correction_every=100,
        log_every=300,
    )

    result = behavioral_fpta(
        agent_data, agent_mask, F,
        sa_dim=6,
        trait_dim=4,
        d=6,
        phi_hidden=(16,),
        rho_hidden=(16,),
        basis_hidden=(16,),
        config=config,
        key=jax.random.PRNGKey(0),
        verbose=False,
    )

    # Should produce at least 1 component
    assert result.n_components >= 1

    # Embeddings should have correct shape
    Y = result.embed(agent_data, agent_mask)
    assert Y.shape == (9, result.n_components, 2)

    # Traits should have correct shape
    traits = result.encode(agent_data, agent_mask)
    assert traits.shape == (9, 4)

    # Importance should be non-negative
    imp = result.get_importance()
    assert jnp.all(imp >= -1e-6)


def test_behavioral_fpta_result_methods():
    """All BehavioralFPTAResult methods should work."""
    key = jax.random.PRNGKey(0)
    agent_data, agent_mask, F = _make_synthetic_rps_data(key, n_agents=6, k_per_agent=10)

    config = TrainConfig(n_steps=100, batch_size=16, log_every=200)
    result = behavioral_fpta(
        agent_data, agent_mask, F,
        sa_dim=6, trait_dim=3, d=4,
        phi_hidden=(8,), rho_hidden=(8,), basis_hidden=(8,),
        config=config, key=jax.random.PRNGKey(0), verbose=False,
    )

    # predict should return (N, M) payoff
    F_pred = result.predict(agent_data, agent_data, agent_mask, agent_mask)
    assert F_pred.shape == (6, 6)

    # Prediction should be approximately skew-symmetric
    assert jnp.allclose(F_pred, -F_pred.T, atol=1e-4)

    # embed_from_traits should work
    traits = result.encode(agent_data, agent_mask)
    Y = result.embed_from_traits(traits)
    assert Y.shape[1] == result.n_components
    assert Y.shape[2] == 2
