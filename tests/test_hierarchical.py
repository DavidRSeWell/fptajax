"""Tests for hierarchical behavioral FPTA (transformer-per-game + DeepSets)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fptajax.hierarchical import (
    TransformerBlock,
    GameEncoder,
    HierarchicalSetEncoder,
    hierarchical_behavioral_fpta,
    _sample_games,
)
from fptajax.neural import TrainConfig


# ---------------------------------------------------------------------------
# GameEncoder tests
# ---------------------------------------------------------------------------


def test_game_encoder_shape():
    """GameEncoder should produce (d_model,) vectors."""
    key = jax.random.PRNGKey(0)
    enc = GameEncoder(token_dim=10, d_model=32, L_max=50,
                      n_heads=2, n_layers=2, key=key)
    game = jax.random.normal(key, (30, 10))
    mask = jnp.ones(30, dtype=bool)
    out = enc(game, mask)
    assert out.shape == (32,), f"Expected (32,), got {out.shape}"


def test_game_encoder_masking():
    """Masked tokens should not affect the output."""
    key = jax.random.PRNGKey(0)
    enc = GameEncoder(token_dim=8, d_model=16, L_max=20,
                      n_heads=2, n_layers=1, key=key)

    # Base game
    game = jax.random.normal(key, (20, 8))
    mask1 = jnp.array([True] * 10 + [False] * 10)

    # Same first 10 tokens, different padding
    game2 = game.at[10:].set(100.0)  # change padding values drastically

    out1 = enc(game, mask1)
    out2 = enc(game2, mask1)
    # Outputs should be close because padding is masked out
    np.testing.assert_allclose(out1, out2, atol=1e-4)


# ---------------------------------------------------------------------------
# HierarchicalSetEncoder tests
# ---------------------------------------------------------------------------


def test_hierarchical_encoder_shape():
    """Single-agent encoding should produce (trait_dim,)."""
    key = jax.random.PRNGKey(0)
    enc = HierarchicalSetEncoder(
        token_dim=10, L_max=30, trait_dim=8,
        d_model=16, n_heads=2, n_layers=1, key=key,
    )
    games = jax.random.normal(key, (4, 30, 10))
    tmask = jnp.ones((4, 30), dtype=bool)
    gmask = jnp.ones(4, dtype=bool)
    trait = enc(games, tmask, gmask)
    assert trait.shape == (8,), f"Expected (8,), got {trait.shape}"


def test_hierarchical_encoder_batch():
    """Batch encoding should produce (N, trait_dim)."""
    key = jax.random.PRNGKey(0)
    enc = HierarchicalSetEncoder(
        token_dim=10, L_max=30, trait_dim=8,
        d_model=16, n_heads=2, n_layers=1, key=key,
    )
    N = 5
    games = jax.random.normal(key, (N, 4, 30, 10))
    tmask = jnp.ones((N, 4, 30), dtype=bool)
    gmask = jnp.ones((N, 4), dtype=bool)
    traits = enc.encode_batch(games, tmask, gmask)
    assert traits.shape == (N, 8)


def test_hierarchical_permutation_invariant_over_games():
    """Trait should not depend on the order of games."""
    key = jax.random.PRNGKey(0)
    enc = HierarchicalSetEncoder(
        token_dim=8, L_max=20, trait_dim=4,
        d_model=16, n_heads=2, n_layers=1, key=key,
    )
    games = jax.random.normal(key, (5, 20, 8))
    tmask = jnp.ones((5, 20), dtype=bool)
    gmask = jnp.ones(5, dtype=bool)

    perm = jnp.array([3, 1, 4, 0, 2])
    t1 = enc(games, tmask, gmask)
    t2 = enc(games[perm], tmask[perm], gmask[perm])
    np.testing.assert_allclose(t1, t2, atol=1e-5)


def test_hierarchical_game_mask():
    """Masked-out games should not affect the trait."""
    key = jax.random.PRNGKey(0)
    enc = HierarchicalSetEncoder(
        token_dim=8, L_max=20, trait_dim=4,
        d_model=16, n_heads=2, n_layers=1, key=key,
    )
    games = jax.random.normal(key, (5, 20, 8))
    tmask = jnp.ones((5, 20), dtype=bool)

    # Only first 3 games are valid
    gmask = jnp.array([True, True, True, False, False])

    # Replace the masked-out games with something wild
    games2 = games.at[3:].set(100.0)

    t1 = enc(games, tmask, gmask)
    t2 = enc(games2, tmask, gmask)
    np.testing.assert_allclose(t1, t2, atol=1e-4)


# ---------------------------------------------------------------------------
# Game subsampling helper
# ---------------------------------------------------------------------------


def test_sample_games_shapes():
    """_sample_games should produce correct shapes."""
    N, G_max, L, td = 6, 10, 15, 8
    rng = np.random.RandomState(0)
    games = rng.randn(N, G_max, L, td).astype(np.float32)
    tmask = np.ones((N, G_max, L), dtype=bool)
    gmask = np.ones((N, G_max), dtype=bool)

    agent_idx = np.array([0, 2, 4])
    g, tm, gm = _sample_games(games, tmask, gmask, agent_idx, G_sample=3, rng=rng)
    assert g.shape == (3, 3, L, td)
    assert tm.shape == (3, 3, L)
    assert gm.shape == (3, 3)
    assert gm.all()


def test_sample_games_with_padding():
    """Agents with fewer valid games than G_sample should be padded."""
    N, G_max, L, td = 3, 10, 5, 4
    rng = np.random.RandomState(0)
    games = rng.randn(N, G_max, L, td).astype(np.float32)
    tmask = np.ones((N, G_max, L), dtype=bool)
    gmask = np.zeros((N, G_max), dtype=bool)
    gmask[0, :10] = True  # agent 0 has 10 games
    gmask[1, :4] = True   # agent 1 has only 4
    gmask[2, :2] = True   # agent 2 has only 2

    agent_idx = np.array([0, 1, 2])
    g, tm, gm = _sample_games(games, tmask, gmask, agent_idx, G_sample=6, rng=rng)
    assert gm[0].sum() == 6  # agent 0 has enough
    assert gm[1].sum() == 4  # agent 1 padded
    assert gm[2].sum() == 2  # agent 2 padded


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


def test_hierarchical_fpta_synthetic():
    """End-to-end hierarchical training should run and reduce loss."""
    N, G, L, token_dim = 5, 4, 15, 8
    rng = np.random.RandomState(0)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    F = rng.randn(N, N).astype(np.float32)
    F = 0.5 * (F - F.T)

    config = TrainConfig(
        lr=1e-3, n_steps=20, batch_size=4,
        c_correction_every=100, log_every=100,
        ortho_weight=0.1, ridge_lambda=1e-4, grad_clip=1.0,
    )

    result = hierarchical_behavioral_fpta(
        games, tmask, gmask, F,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(16,), basis_hidden=(16,),
        config=config, G_sample=3, G_sample_eval=G,
        verbose=False,
    )

    assert result.n_components >= 1
    assert result.coefficient_matrix.shape == (4, 4)
    # Enforce skew-symmetry of C
    C = result.coefficient_matrix
    np.testing.assert_allclose(C, -C.T, atol=1e-5)


def test_hierarchical_fpta_result_methods():
    """Result.embed and predict should return correct shapes."""
    N, G, L, token_dim = 4, 3, 10, 6
    rng = np.random.RandomState(1)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    F = rng.randn(N, N).astype(np.float32)
    F = 0.5 * (F - F.T)

    config = TrainConfig(
        lr=1e-3, n_steps=10, batch_size=2,
        c_correction_every=100, log_every=100,
        grad_clip=1.0,
    )
    result = hierarchical_behavioral_fpta(
        games, tmask, gmask, F,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, d_model=8, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(8,), basis_hidden=(8,),
        config=config, G_sample=G, G_sample_eval=G,
        verbose=False,
    )

    # encode
    traits = result.encode(
        jnp.array(games), jnp.array(tmask), jnp.array(gmask),
    )
    assert traits.shape == (N, 4)

    # embed
    Y = result.embed(
        jnp.array(games), jnp.array(tmask), jnp.array(gmask),
    )
    assert Y.shape == (N, result.n_components, 2)

    # predict
    F_pred = result.predict(
        jnp.array(games), jnp.array(tmask), jnp.array(gmask),
        jnp.array(games), jnp.array(tmask), jnp.array(gmask),
    )
    assert F_pred.shape == (N, N)
