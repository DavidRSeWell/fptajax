"""Tests for direct-g-prediction hierarchical FPTA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fptajax.hierarchical_g import (
    VectorSkillHead,
    hierarchical_g_fpta,
)
from fptajax.neural import TrainConfig


# ---------------------------------------------------------------------------
# VectorSkillHead
# ---------------------------------------------------------------------------


def test_vector_skill_head_shape():
    key = jax.random.PRNGKey(0)
    head = VectorSkillHead(trait_dim=8, skill_dim=4, hidden=(16,), key=key)
    trait = jax.random.normal(key, (8,))
    out = head(trait)
    assert out.shape == (4,)


def test_vector_skill_head_batch():
    key = jax.random.PRNGKey(0)
    head = VectorSkillHead(trait_dim=6, skill_dim=3, hidden=(12,), key=key)
    traits = jax.random.normal(key, (5, 6))
    out = jax.vmap(head)(traits)
    assert out.shape == (5, 3)


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


def test_g_fpta_synthetic_shape():
    N, G, L, token_dim = 5, 4, 12, 8
    rng = np.random.RandomState(0)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    # Arbitrary non-skew-symmetric g labels: 10 + rank_diff + rank_sum
    ranks = rng.randn(N).astype(np.float32)
    G_lab = (10 + (ranks[:, None] - ranks[None, :])
             + 0.5 * (ranks[:, None] + ranks[None, :])).astype(np.float32)

    config = TrainConfig(
        lr=1e-3, n_steps=20, batch_size=4,
        c_correction_every=100, log_every=100, grad_clip=1.0,
        ortho_weight=0.1, ridge_lambda=1e-4,
    )
    result = hierarchical_g_fpta(
        games, tmask, gmask, G_lab,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, skill_dim=3,
        d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(16,), basis_hidden=(16,), skill_hidden=(16,),
        config=config, G_sample=3, G_sample_eval=G,
        verbose=False,
    )
    assert result.coupling_matrix.shape == (4, 4)
    # coefficient_matrix is the skew-symmetric projection
    C_skew = np.asarray(result.coefficient_matrix)
    np.testing.assert_allclose(C_skew, -C_skew.T, atol=1e-5)


def test_g_fpta_f_antisymmetric_by_construction():
    """f_hat predicted via two passes through g must be exactly skew-sym."""
    N, G, L, token_dim = 6, 3, 10, 6
    rng = np.random.RandomState(1)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    G_lab = rng.randn(N, N).astype(np.float32) + 5.0

    config = TrainConfig(
        lr=1e-3, n_steps=10, batch_size=2,
        c_correction_every=100, log_every=100, grad_clip=1.0,
    )
    result = hierarchical_g_fpta(
        games, tmask, gmask, G_lab,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, skill_dim=2,
        d_model=8, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(8,), basis_hidden=(8,), skill_hidden=(8,),
        config=config, G_sample=G, G_sample_eval=G, verbose=False,
    )
    games_j = jnp.array(games); tmask_j = jnp.array(tmask); gmask_j = jnp.array(gmask)
    F_pred = result.predict_f(
        games_j, tmask_j, gmask_j,
        games_j, tmask_j, gmask_j,
    )
    F_pred = np.asarray(F_pred)
    np.testing.assert_allclose(F_pred, -F_pred.T, atol=1e-5)
    # Diagonal should be exactly zero
    np.testing.assert_allclose(np.diag(F_pred), 0.0, atol=1e-5)


def test_g_fpta_result_methods():
    N, G, L, token_dim = 4, 3, 10, 6
    rng = np.random.RandomState(1)
    games = rng.randn(N, G, L, token_dim).astype(np.float32)
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    G_lab = rng.randn(N, N).astype(np.float32)

    config = TrainConfig(lr=1e-3, n_steps=8, batch_size=2,
                         c_correction_every=100, log_every=100, grad_clip=1.0)
    result = hierarchical_g_fpta(
        games, tmask, gmask, G_lab,
        token_dim=token_dim, L_max=L,
        trait_dim=4, d=4, skill_dim=3,
        d_model=8, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(8,), basis_hidden=(8,), skill_hidden=(8,),
        config=config, G_sample=G, G_sample_eval=G, verbose=False,
    )
    games_j = jnp.array(games); tmask_j = jnp.array(tmask); gmask_j = jnp.array(gmask)

    traits = result.encode(games_j, tmask_j, gmask_j)
    assert traits.shape == (N, 4)

    s = result.skills(games_j, tmask_j, gmask_j)
    assert s.shape == (N, 3)

    # effective scalar skill has the right shape
    eff = result.effective_scalar_skill(games_j, tmask_j, gmask_j)
    assert eff.shape == (N,)

    g_pred = result.predict_g(games_j, tmask_j, gmask_j, games_j, tmask_j, gmask_j)
    assert g_pred.shape == (N, N)

    f_pred = result.predict_f(games_j, tmask_j, gmask_j, games_j, tmask_j, gmask_j)
    assert f_pred.shape == (N, N)

    Y = result.embed(games_j, tmask_j, gmask_j)
    assert Y.shape == (N, result.n_components, 2)

    g_sx, g_sy, g_bilin, g_b, g_tot = result.decompose_g(games_j, tmask_j, gmask_j)
    assert g_tot.shape == (N, N)
    np.testing.assert_allclose(
        np.asarray(g_tot),
        np.asarray(g_sx) + np.asarray(g_sy) + np.asarray(g_bilin) + np.asarray(g_b),
        atol=1e-5,
    )

    f_sk, f_di, f_tot = result.decompose_f(games_j, tmask_j, gmask_j)
    np.testing.assert_allclose(
        np.asarray(f_tot),
        np.asarray(f_sk) + np.asarray(f_di),
        atol=1e-5,
    )
    # f_total should be equal to predict_f output
    np.testing.assert_allclose(np.asarray(f_tot), np.asarray(f_pred), atol=1e-5)

    vd_g = result.variance_decomposition_g(games_j, tmask_j, gmask_j)
    vd_f = result.variance_decomposition_f(games_j, tmask_j, gmask_j)
    for vd in (vd_g, vd_f):
        assert vd["total_var"] >= 0.0


def test_g_fpta_learns_symmetric_component():
    """Build g = c_0 * s(x) + c_1 * s(y) + small_bilinear with c_0 != -c_1,
    so the label has a nonzero symmetric component. Train briefly, and
    check that the model has learned c_0 + c_1 != 0 (i.e. the symmetric
    component is captured), distinguishing this from a purely
    antisymmetric F-only target.
    """
    N, G, L, token_dim = 10, 5, 15, 8
    rng = np.random.RandomState(42)
    ranks = rng.randn(N).astype(np.float32)
    # Plant rank signal in channel 0 so the encoder can read it off
    games = rng.randn(N, G, L, token_dim).astype(np.float32) * 0.1
    games[:, :, :, 0] += ranks[:, None, None]
    tmask = np.ones((N, G, L), dtype=bool)
    gmask = np.ones((N, G), dtype=bool)
    # g(x, y) = 0.7 * rank_x + 0.3 * rank_y + 10  (c_0=0.7, c_1=0.3 ⇒ symmetric-heavy)
    G_lab = 0.7 * ranks[:, None] + 0.3 * ranks[None, :] + 10.0

    config = TrainConfig(
        lr=3e-3, n_steps=400, batch_size=10,
        c_correction_every=100, log_every=100, grad_clip=1.0,
        ortho_weight=0.1, ridge_lambda=1e-4,
    )
    result = hierarchical_g_fpta(
        games, tmask, gmask, G_lab,
        token_dim=token_dim, L_max=L,
        trait_dim=8, d=4, skill_dim=2,
        d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
        rho_hidden=(16,), basis_hidden=(16,), skill_hidden=(16,),
        config=config, G_sample=3, G_sample_eval=G, verbose=False,
    )
    c0 = np.asarray(result.c_0)
    c1 = np.asarray(result.c_1)
    # The true c_0 + c_1 = 1.0 (symmetric component), c_0 - c_1 = 0.4.
    # After training, the model should have c_0 and c_1 same-signed (not
    # exactly antisymmetric). This checks that the symmetric component was
    # actually fit.
    # We don't check exact recovery since s is multi-dim and there is gauge
    # freedom — we just check that (c_0 + c_1) has non-trivial magnitude.
    sum_norm = float(np.linalg.norm(c0 + c1))
    diff_norm = float(np.linalg.norm(c0 - c1))
    # The symmetric component (sum) should be non-trivial relative to the
    # antisymmetric component (diff).
    assert sum_norm > 0.2 * diff_norm, (
        f"Expected symmetric c0+c1 to be recoverable: "
        f"||c0+c1||={sum_norm:.3f}, ||c0-c1||={diff_norm:.3f}"
    )
