"""Tests for hierarchical skill + disc-game FPTA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fptajax.hierarchical_skill import (
    SkillHead,
    hierarchical_skill_fpta,
)
from fptajax.neural import TrainConfig


# ---------------------------------------------------------------------------
# SkillHead
# ---------------------------------------------------------------------------


def test_skill_head_shape():
    """SkillHead should return a scalar per trait."""
    key = jax.random.PRNGKey(0)
    head = SkillHead(trait_dim=8, hidden=(16,), key=key)
    trait = jax.random.normal(key, (8,))
    out = head(trait)
    assert out.shape == (), f"Expected scalar (shape ()), got {out.shape}"


def test_skill_head_batch():
    """vmapped over a batch should give (B,)."""
    key = jax.random.PRNGKey(0)
    head = SkillHead(trait_dim=6, hidden=(12,), key=key)
    traits = jax.random.normal(key, (5, 6))
    out = jax.vmap(head)(traits)
    assert out.shape == (5,)


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


def test_hierarchical_skill_fpta_synthetic():
    """Smoke test: end-to-end training runs, returns expected shapes."""
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
        skill_centering_weight=1.0,
    )

    result = hierarchical_skill_fpta(
        games, tmask, gmask, F,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(16,), basis_hidden=(16,), skill_hidden=(16,),
        config=config, G_sample=3, G_sample_eval=G,
        verbose=False,
    )

    assert result.n_components >= 1
    assert result.coefficient_matrix.shape == (4, 4)
    # Enforce skew-symmetry of C
    C = np.asarray(result.coefficient_matrix)
    np.testing.assert_allclose(C, -C.T, atol=1e-5)


def test_hierarchical_skill_fpta_result_methods():
    """Result.encode/skills/embed/predict/decompose_F should return correct shapes."""
    N, G, L, token_dim = 4, 3, 10, 6
    rng = np.random.RandomState(1)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    F = rng.randn(N, N).astype(np.float32)
    F = 0.5 * (F - F.T)

    config = TrainConfig(
        lr=1e-3, n_steps=10, batch_size=2,
        c_correction_every=100, log_every=100, grad_clip=1.0,
        skill_centering_weight=1.0,
    )
    result = hierarchical_skill_fpta(
        games, tmask, gmask, F,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, d_model=8, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(8,), basis_hidden=(8,), skill_hidden=(8,),
        config=config, G_sample=G, G_sample_eval=G,
        verbose=False,
    )

    games_j = jnp.array(games); tmask_j = jnp.array(tmask); gmask_j = jnp.array(gmask)

    traits = result.encode(games_j, tmask_j, gmask_j)
    assert traits.shape == (N, 4)

    skills = result.skills(games_j, tmask_j, gmask_j)
    assert skills.shape == (N,)

    Y = result.embed(games_j, tmask_j, gmask_j)
    assert Y.shape == (N, result.n_components, 2)

    F_pred = result.predict(
        games_j, tmask_j, gmask_j,
        games_j, tmask_j, gmask_j,
    )
    assert F_pred.shape == (N, N)

    F_skill, F_disc, F_total = result.decompose_F(games_j, tmask_j, gmask_j)
    assert F_skill.shape == F_disc.shape == F_total.shape == (N, N)
    np.testing.assert_allclose(
        np.asarray(F_total),
        np.asarray(F_skill) + np.asarray(F_disc),
        atol=1e-5,
    )
    # F_skill should be skew-symmetric: s_i - s_j = -(s_j - s_i)
    np.testing.assert_allclose(
        np.asarray(F_skill),
        -np.asarray(F_skill).T,
        atol=1e-5,
    )

    # Variance decomposition
    stats = result.variance_decomposition(games_j, tmask_j, gmask_j)
    assert "skill_frac" in stats and "disc_frac" in stats
    assert 0.0 <= stats["skill_frac"] <= 1.0 + 1e-6
    assert 0.0 <= stats["disc_frac"] <= 1.0 + 1e-6


def test_skill_head_absorbs_transitive_synthetic():
    """With a purely transitive F, the skill head should explain almost all of
    the predicted variance and the disc-game component should be small.

    We build a synthetic F where F[i, j] = r_i - r_j for a random rank vector
    r, so there is *no* genuine cyclic structure. After training, the
    variance carried by the skill part should dwarf the disc-game part.
    """
    N, G, L, token_dim = 8, 6, 12, 8
    rng = np.random.RandomState(7)
    # Per-agent "ranks" embedded in the games: each game for agent i has token
    # values concentrated at r_i, so the encoder can read r_i off easily.
    ranks = rng.randn(N).astype(np.float32)
    games = rng.randn(N, G, L, token_dim).astype(np.float32) * 0.1
    games[:, :, :, 0] += ranks[:, None, None]            # plant the rank signal
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    F = (ranks[:, None] - ranks[None, :]).astype(np.float32)
    F = 0.5 * (F - F.T)

    config = TrainConfig(
        lr=3e-3, n_steps=400, batch_size=8,
        c_correction_every=100, log_every=100, grad_clip=1.0,
        ortho_weight=0.1, ridge_lambda=1e-4,
        skill_centering_weight=1.0,
    )
    result = hierarchical_skill_fpta(
        games, tmask, gmask, F,
        token_dim=token_dim, L_max=L,
        trait_dim=8, d=4, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(16,), basis_hidden=(32,), skill_hidden=(16,),
        config=config, G_sample=3, G_sample_eval=G,
        verbose=False,
    )
    games_j = jnp.array(games); tmask_j = jnp.array(tmask); gmask_j = jnp.array(gmask)
    stats = result.variance_decomposition(games_j, tmask_j, gmask_j)
    # Skill should carry the bulk of predicted variance on a transitive target.
    # Allow some slop (training is short), but skill should be comfortably
    # larger than disc-game variance.
    assert stats["skill_frac"] > stats["disc_frac"], (
        f"Expected skill_frac > disc_frac on a transitive target, got "
        f"skill={stats['skill_frac']:.3f}, disc={stats['disc_frac']:.3f}"
    )
