"""Tests for neural basis FPTA."""

import jax
import jax.numpy as jnp
import pytest

from fptajax.neural import (
    NeuralBasis,
    SkewParam,
    TrainConfig,
    _Model,
    neural_fpta,
    neural_fpta_from_matrix,
    _compute_loss,
    _closed_form_c,
)


# ---------------------------------------------------------------------------
# NeuralBasis tests
# ---------------------------------------------------------------------------


def test_neural_basis_shape():
    """NeuralBasis should produce correct output shapes."""
    key = jax.random.PRNGKey(0)
    basis = NeuralBasis(trait_dim=1, d=8, hidden_dims=(32, 32), key=key)

    # Single point
    x = jnp.array([0.5])
    out = basis(x)
    assert out.shape == (8,)

    # Batch
    x_batch = jnp.linspace(0, 1, 20)
    out_batch = basis.evaluate_batch(x_batch)
    assert out_batch.shape == (20, 8)


def test_neural_basis_multidim():
    """NeuralBasis should work with multi-dimensional traits."""
    key = jax.random.PRNGKey(0)
    basis = NeuralBasis(trait_dim=3, d=8, hidden_dims=(32,), key=key)

    x = jnp.ones((10, 3))
    out = basis.evaluate_batch(x)
    assert out.shape == (10, 8)


# ---------------------------------------------------------------------------
# SkewParam tests
# ---------------------------------------------------------------------------


def test_skew_param_antisymmetric():
    """SkewParam.C should always be skew-symmetric."""
    key = jax.random.PRNGKey(0)
    skew = SkewParam(d=6, key=key)
    C = skew.C
    assert jnp.allclose(C, -C.T, atol=1e-7)


def test_skew_param_zero_diagonal():
    """Skew-symmetric matrix should have zero diagonal."""
    skew = SkewParam(d=4, key=jax.random.PRNGKey(0))
    assert jnp.allclose(jnp.diag(skew.C), 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


def test_loss_decreases():
    """Loss should decrease after a gradient step."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    import optax
    import equinox as eqx

    basis = NeuralBasis(trait_dim=1, d=4, hidden_dims=(16,), key=k1)
    skew = SkewParam(d=4, key=k2)
    model = _Model(basis=basis, skew=skew)

    # Synthetic data: f(x,y) = sin(x - y)
    x = jax.random.uniform(k3, (64, 1)) * 2 * jnp.pi
    k3, k4 = jax.random.split(k3)
    y = jax.random.uniform(k4, (64, 1)) * 2 * jnp.pi
    f_xy = jnp.sin(x[:, 0] - y[:, 0])

    loss_before, _ = _compute_loss(model, x, y, f_xy, 0.1)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    (loss, _), grads = eqx.filter_value_and_grad(
        lambda m: _compute_loss(m, x, y, f_xy, 0.1),
        has_aux=True,
    )(model)

    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array),
    )
    model2 = eqx.apply_updates(model, updates)

    loss_after, _ = _compute_loss(model2, x, y, f_xy, 0.1)

    assert float(loss_after) < float(loss_before), \
        f"Loss did not decrease: {loss_before} -> {loss_after}"


# ---------------------------------------------------------------------------
# Closed-form C correction tests
# ---------------------------------------------------------------------------


def test_closed_form_c_skew():
    """Closed-form C should be skew-symmetric."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    basis = NeuralBasis(trait_dim=1, d=4, hidden_dims=(16,), key=k1)
    x = jax.random.uniform(k2, (50, 1))
    y = jax.random.uniform(k3, (50, 1))
    f_xy = jnp.sin(x[:, 0] - y[:, 0])

    C = _closed_form_c(basis, x, y, f_xy, ridge_lambda=1e-3)
    assert jnp.allclose(C, -C.T, atol=1e-5), f"C not skew-symmetric: max diff={jnp.max(jnp.abs(C + C.T))}"


# ---------------------------------------------------------------------------
# End-to-end training tests
# ---------------------------------------------------------------------------


def test_neural_fpta_sine():
    """Neural FPTA should learn f(x,y) = sin(x - y) with low error."""
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)

    N = 500
    x = jax.random.uniform(k1, (N,)) * 2 * jnp.pi
    y = jax.random.uniform(k2, (N,)) * 2 * jnp.pi
    f_xy = jnp.sin(x - y)

    config = TrainConfig(
        lr=1e-3,
        n_steps=500,
        batch_size=128,
        ortho_weight=0.05,
        c_correction_every=100,
        log_every=500,  # quiet
    )

    result = neural_fpta(
        x, y, f_xy,
        d=8, trait_dim=1, hidden_dims=(32, 32),
        config=config, key=jax.random.PRNGKey(0),
        verbose=False,
    )

    # Should have at least 1 disc game component
    assert result.n_components >= 1

    # Reconstruction should be reasonable
    x_test = jnp.linspace(0, 2 * jnp.pi, 20)
    y_test = jnp.linspace(0, 2 * jnp.pi, 20)
    F_hat = result.predict(x_test, y_test)
    xx, yy = jnp.meshgrid(x_test, y_test, indexing='ij')
    F_true = jnp.sin(xx - yy)

    rmse = float(jnp.sqrt(jnp.mean((F_true - F_hat) ** 2)))
    assert rmse < 0.5, f"RMSE too high: {rmse}"


def test_neural_fpta_from_matrix_rps():
    """Neural FPTA from RPS payoff matrix should produce valid embeddings."""
    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])

    config = TrainConfig(
        lr=1e-3,
        n_steps=300,
        batch_size=9,  # all pairs
        ortho_weight=0.05,
        c_correction_every=100,
        log_every=300,
    )

    result = neural_fpta_from_matrix(
        F, d=4, hidden_dims=(16, 16),
        config=config, key=jax.random.PRNGKey(0),
        verbose=False,
    )

    assert result.n_components >= 1
    assert result.coefficient_matrix.shape == (4, 4)

    # Embeddings should have correct shape
    traits = jnp.arange(3, dtype=jnp.float32)
    Y = result.embed(traits)
    assert Y.shape[1] == result.n_components
    assert Y.shape[2] == 2


def test_neural_fpta_result_methods():
    """NeuralFPTAResult methods should not error."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    x = jax.random.uniform(k1, (100,)) * 2 * jnp.pi
    y = jax.random.uniform(k2, (100,)) * 2 * jnp.pi
    f_xy = jnp.sin(x - y)

    config = TrainConfig(n_steps=100, log_every=200)
    result = neural_fpta(
        x, y, f_xy, d=4, config=config,
        key=jax.random.PRNGKey(0), verbose=False,
    )

    # All methods should run without error
    imp = result.get_importance()
    assert imp.shape[0] == result.n_components

    cum = result.get_cumulative_importance()
    assert cum.shape[0] == result.n_components

    x_new = jnp.linspace(0, 2 * jnp.pi, 10)
    F_hat = result.reconstruct(x_new, x_new)
    assert F_hat.shape == (10, 10)

    F_pred = result.predict(x_new, x_new)
    assert F_pred.shape == (10, 10)

    # Reconstruction should be approximately skew-symmetric
    assert jnp.allclose(F_hat, -F_hat.T, atol=1e-4)
