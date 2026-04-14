"""Neural basis learning for FPTA.

Learns basis functions b_theta: X -> R^d as a neural network, jointly with
a skew-symmetric coefficient matrix C, such that:

    f(x, y) ≈ b(x)^T C b(y)

After training, C is decomposed via Schur decomposition to extract disc game
embeddings, seamlessly integrating with the rest of fptajax.

Training approach (hybrid end-to-end + periodic C correction):
    1. Parameterize C = A - A^T for unconstrained A (automatic skew-symmetry)
    2. Train (theta, A) jointly via Adam with backprop through the full pipeline
    3. Every K steps, recompute the optimal C in closed form given fixed b_theta
    4. Regularize with orthogonality penalty on the Gram matrix of b(X)

Requires: pip install fptajax[neural]  (equinox, optax)

Inspired by:
    Ingebrand et al. "Function Encoders: A Principled Approach to Transfer
    Learning in Hilbert Spaces." (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

try:
    import equinox as eqx
    import optax
except ImportError:
    raise ImportError(
        "Neural FPTA requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.decomposition import skew_symmetric_schur
from fptajax.utils import importance, cumulative_importance


# ---------------------------------------------------------------------------
# Neural basis model
# ---------------------------------------------------------------------------


class NeuralBasis(eqx.Module):
    """Neural network basis: b_theta: R^T -> R^d.

    An MLP that maps trait vectors to d-dimensional basis representations.
    Designed to be used with NeuralFPTA for learning disc game embeddings.

    Attributes:
        mlp: the underlying Equinox MLP.
        d: output dimensionality (number of basis functions).
    """

    mlp: eqx.nn.MLP
    d: int = eqx.field(static=True)

    def __init__(
        self,
        trait_dim: int,
        d: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        key: Array | None = None,
        activation: Callable = jax.nn.gelu,
    ):
        """Initialize neural basis.

        Args:
            trait_dim: dimensionality of input traits (1 for scalar traits).
            d: number of basis functions (output dimension).
            hidden_dims: hidden layer sizes.
            key: PRNG key. If None, uses jax.random.PRNGKey(0).
            activation: activation function for hidden layers.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        self.d = d

        # Build MLP: trait_dim -> hidden -> ... -> hidden -> d
        depth = len(hidden_dims)
        self.mlp = eqx.nn.MLP(
            in_size=trait_dim,
            out_size=d,
            width_size=hidden_dims[0] if hidden_dims else 64,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        """Evaluate basis at a single trait point.

        Args:
            x: trait vector, shape (T,) or scalar reshaped to (1,).

        Returns:
            Basis values, shape (d,).
        """
        x = jnp.atleast_1d(x)
        return self.mlp(x)

    def evaluate_batch(self, x: Array) -> Array:
        """Evaluate basis at multiple trait points.

        Args:
            x: trait array, shape (N,) or (N, T).

        Returns:
            Basis matrix, shape (N, d). Row i = b(x_i).
        """
        x = jnp.atleast_1d(x)
        if x.ndim == 1:
            x = x[:, None]  # (N,) -> (N, 1) for MLP
        return jax.vmap(self)(x)


# ---------------------------------------------------------------------------
# Skew-symmetric coefficient parameterization
# ---------------------------------------------------------------------------


class SkewParam(eqx.Module):
    """Parameterize a skew-symmetric matrix C = A - A^T.

    Stores the free parameter A and computes C on the fly.
    """

    A: Array  # (d, d) unconstrained

    def __init__(self, d: int, key: Array | None = None, scale: float = 0.01):
        if key is None:
            key = jax.random.PRNGKey(1)
        self.A = scale * jax.random.normal(key, (d, d))

    @property
    def C(self) -> Array:
        """The skew-symmetric matrix C = A - A^T."""
        return self.A - self.A.T


# ---------------------------------------------------------------------------
# Combined trainable model (basis + skew in one Equinox module)
# ---------------------------------------------------------------------------


class _Model(eqx.Module):
    """Internal wrapper combining NeuralBasis + SkewParam for clean optax usage."""
    basis: NeuralBasis
    skew: SkewParam


# ---------------------------------------------------------------------------
# Training config and loss
# ---------------------------------------------------------------------------


class TrainConfig(NamedTuple):
    """Hyperparameters for neural FPTA training."""
    lr: float = 1e-3
    n_steps: int = 2000
    batch_size: int = 256
    ortho_weight: float = 0.1
    ridge_lambda: float = 1e-4
    c_correction_every: int = 200
    grad_clip: float = 1.0
    log_every: int = 100


def _compute_loss(
    model: _Model,
    x: Array,
    y: Array,
    f_xy: Array,
    ortho_weight: float,
) -> tuple[Array, dict]:
    """Compute training loss.

    Loss = MSE(f, b(x)^T C b(y)) + ortho_weight * ||B^T B / N - I||^2

    Args:
        model: combined basis + skew model.
        x: first agent traits, shape (B, T) or (B, 1).
        y: second agent traits, shape (B, T) or (B, 1).
        f_xy: performance values f(x, y), shape (B,).
        ortho_weight: weight for orthogonality regularizer.

    Returns:
        (total_loss, metrics_dict).
    """
    C = model.skew.C  # (d, d)

    # Evaluate basis
    bx = model.basis.evaluate_batch(x)  # (B, d)
    by = model.basis.evaluate_batch(y)  # (B, d)

    # Prediction: f_hat = b(x)^T C b(y) = sum_ij C_ij b_i(x) b_j(y)
    # Efficient: f_hat = (bx @ C * by).sum(-1)
    f_hat = jnp.sum(bx @ C * by, axis=-1)  # (B,)

    # MSE loss
    mse = jnp.mean((f_xy - f_hat) ** 2)

    # Orthogonality regularizer: ||B^TB/N - I||_F^2
    # Use combined batch of x and y for better Gram estimate
    B_all = jnp.concatenate([bx, by], axis=0)  # (2B, d)
    N = B_all.shape[0]
    gram = (B_all.T @ B_all) / N  # (d, d)
    ortho_loss = jnp.mean((gram - jnp.eye(model.basis.d)) ** 2)

    total = mse + ortho_weight * ortho_loss

    metrics = {
        "loss": total,
        "mse": mse,
        "ortho": ortho_loss,
        "C_norm": jnp.linalg.norm(C),
    }
    return total, metrics


def _closed_form_c(
    basis: NeuralBasis,
    x_all: Array,
    y_all: Array,
    f_all: Array,
    ridge_lambda: float = 1e-4,
) -> Array:
    """Solve for optimal skew-symmetric C given fixed basis.

    Given b fixed, solve:  min_C sum (f(x,y) - b(x)^T C b(y))^2  s.t. C = -C^T

    This is a linear least-squares problem. We solve the unconstrained version
    and project to skew-symmetric.

    For pairwise data, we vectorize and solve via normal equations.
    """
    bx = jax.lax.stop_gradient(basis.evaluate_batch(x_all))  # (N, d)
    by = jax.lax.stop_gradient(basis.evaluate_batch(y_all))  # (N, d)
    d = bx.shape[1]

    # Outer products: for each sample, K_n = bx_n ⊗ by_n, shape (d, d)
    # Vectorized: K_n = vec(bx_n by_n^T), and f_hat = vec(C)^T vec(bx_n by_n^T)
    # So the problem is: min ||f - M vec(C)||^2 where M[n, :] = vec(bx_n by_n^T)
    # M has shape (N, d^2)
    M = (bx[:, :, None] * by[:, None, :]).reshape(-1, d * d)  # (N, d^2)

    # Ridge-regularized normal equations
    MtM = M.T @ M + ridge_lambda * jnp.eye(d * d)
    Mtf = M.T @ f_all
    c_vec = jnp.linalg.solve(MtM, Mtf)  # (d^2,)

    C_raw = c_vec.reshape(d, d)
    # Project to skew-symmetric
    C = 0.5 * (C_raw - C_raw.T)
    return C


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


@dataclass
class NeuralFPTAResult:
    """Result of neural FPTA.

    After training the neural basis and coefficient matrix, the Schur
    decomposition of C yields disc game embeddings. The neural basis
    can evaluate embeddings at any trait value.

    Attributes:
        basis: trained NeuralBasis module.
        coefficient_matrix: learned skew-symmetric C, shape (d, d).
        eigenvalues: omega_k from Schur decomposition, shape (n_components,).
        schur_vectors: Q from C = QUQ^T, shape (d, d).
        n_components: number of disc game components.
        f_norm_sq: estimated ||f||^2 from training data.
        train_history: list of metric dicts from training.
    """
    basis: NeuralBasis
    coefficient_matrix: Array
    eigenvalues: Array
    schur_vectors: Array
    n_components: int
    f_norm_sq: float | None = None
    train_history: list | None = None

    def embed(self, x: Array) -> Array:
        """Evaluate disc game embeddings at arbitrary trait values.

        Y^(k)(x) = sqrt(omega_k) * b(x)^T [q_{2k-1}, q_{2k}]

        Args:
            x: trait values, shape (N,) or (N, T).

        Returns:
            Embeddings, shape (N, n_components, 2).
        """
        B = self.basis.evaluate_batch(x)  # (N, d)
        Q = self.schur_vectors
        omegas = self.eigenvalues
        d = self.n_components

        embeddings = []
        for k in range(d):
            q1 = Q[:, 2 * k]
            q2 = Q[:, 2 * k + 1]
            y1 = jnp.sqrt(omegas[k]) * (B @ q1)
            y2 = jnp.sqrt(omegas[k]) * (B @ q2)
            embeddings.append(jnp.stack([y1, y2], axis=-1))

        return jnp.stack(embeddings, axis=1)

    def reconstruct(
        self,
        x: Array,
        y: Array,
        n_components: int | None = None,
    ) -> Array:
        """Reconstruct f_hat(x, y) from disc game embeddings.

        Args:
            x: first agent traits, shape (N,).
            y: second agent traits, shape (M,).
            n_components: number of disc games. If None, uses all.

        Returns:
            Reconstructed payoff, shape (N, M).
        """
        Y_x = self.embed(x)  # (N, d, 2)
        Y_y = self.embed(y)  # (M, d, 2)

        nc = n_components or self.n_components
        Y_x = Y_x[:, :nc, :]
        Y_y = Y_y[:, :nc, :]

        F_hat = jnp.einsum('ik,jk->ij', Y_x[:, :, 0], Y_y[:, :, 1]) \
              - jnp.einsum('ik,jk->ij', Y_x[:, :, 1], Y_y[:, :, 0])
        return F_hat

    def predict(self, x: Array, y: Array) -> Array:
        """Predict f(x, y) directly via b(x)^T C b(y).

        Uses the full learned C (not truncated Schur). Useful for
        evaluating the raw model before decomposition.

        Args:
            x: first agent traits, shape (N,) or (N, T).
            y: second agent traits, shape (M,) or (M, T).

        Returns:
            Predicted payoff, shape (N, M).
        """
        bx = self.basis.evaluate_batch(x)  # (N, d)
        by = self.basis.evaluate_batch(y)  # (M, d)
        return bx @ self.coefficient_matrix @ by.T

    def get_importance(self) -> Array:
        """Relative importance of each disc game."""
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        """Cumulative explained variance."""
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


def neural_fpta(
    x_data: Array,
    y_data: Array,
    f_data: Array,
    d: int = 16,
    trait_dim: int = 1,
    hidden_dims: tuple[int, ...] = (64, 64),
    config: TrainConfig | None = None,
    key: Array | None = None,
    n_components: int | None = None,
    verbose: bool = True,
) -> NeuralFPTAResult:
    """Train a neural basis for FPTA via hybrid end-to-end optimization.

    Learns b_theta: R^T -> R^d and skew-symmetric C such that
    f(x, y) ≈ b(x)^T C b(y), then decomposes C into disc games.

    Training procedure:
        1. Joint Adam optimization of (theta, A) where C = A - A^T
        2. Every c_correction_every steps, recompute optimal C in closed form
        3. Orthogonality regularization: ||B^TB/N - I||^2

    Args:
        x_data: first agent traits, shape (N,) or (N, T).
        y_data: second agent traits, shape (N,) or (N, T).
        f_data: performance values f(x, y), shape (N,).
        d: basis dimension (number of basis functions).
        trait_dim: dimensionality of trait space.
        hidden_dims: MLP hidden layer sizes.
        config: training hyperparameters. If None, uses defaults.
        key: PRNG key. If None, uses PRNGKey(42).
        n_components: max disc game components to extract. If None, keeps all.
        verbose: if True, print training progress.

    Returns:
        NeuralFPTAResult with trained basis, C, and disc game decomposition.
    """
    if config is None:
        config = TrainConfig()
    if key is None:
        key = jax.random.PRNGKey(42)

    # Reshape inputs
    x_data = jnp.atleast_1d(jnp.array(x_data, dtype=jnp.float32))
    y_data = jnp.atleast_1d(jnp.array(y_data, dtype=jnp.float32))
    f_data = jnp.array(f_data, dtype=jnp.float32)

    if x_data.ndim == 1:
        x_data = x_data[:, None]
    if y_data.ndim == 1:
        y_data = y_data[:, None]

    N = x_data.shape[0]

    # Initialize model
    key, k1, k2 = jax.random.split(key, 3)
    basis = NeuralBasis(trait_dim, d, hidden_dims, key=k1)
    skew = SkewParam(d, key=k2)
    model = _Model(basis=basis, skew=skew)

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT-compiled training step
    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _compute_loss(
                m, x_batch, y_batch, f_batch, config.ortho_weight,
            ),
            has_aux=True,
        )(model)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

    # Training loop
    history = []
    for step in range(config.n_steps):
        # Sample batch
        key, batch_key = jax.random.split(key)
        idx = jax.random.choice(batch_key, N, shape=(config.batch_size,))
        x_batch = x_data[idx]
        y_batch = y_data[idx]
        f_batch = f_data[idx]

        # Gradient step
        model, opt_state, metrics = train_step(
            model, opt_state, x_batch, y_batch, f_batch,
        )

        # Periodic closed-form C correction
        if (step + 1) % config.c_correction_every == 0 and step > 0:
            C_optimal = _closed_form_c(
                model.basis, x_data, y_data, f_data, config.ridge_lambda,
            )
            # Update skew.A such that A - A^T = C_optimal
            # A = C_optimal / 2 works since (C/2) - (C/2)^T = C/2 + C/2 = C
            new_A = C_optimal / 2.0
            model = eqx.tree_at(lambda m: m.skew.A, model, new_A)
            # Re-init optimizer state to avoid stale momentum
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

            if verbose:
                print(f"  [step {step+1}] C correction applied, "
                      f"||C||={float(jnp.linalg.norm(C_optimal)):.4f}")

        # Logging
        if verbose and (step % config.log_every == 0 or step == config.n_steps - 1):
            m = {k: float(v) for k, v in metrics.items()}
            history.append({"step": step, **m})
            print(f"  step {step:5d} | loss={m['loss']:.6f} | "
                  f"mse={m['mse']:.6f} | ortho={m['ortho']:.6f} | "
                  f"||C||={m['C_norm']:.4f}")

    # Final C correction
    basis = model.basis
    C_final = _closed_form_c(basis, x_data, y_data, f_data, config.ridge_lambda)
    C_final = 0.5 * (C_final - C_final.T)  # enforce exact skew-symmetry

    # Schur decomposition of learned C
    schur = skew_symmetric_schur(C_final)

    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    # Estimate ||f||^2 from data
    f_norm_sq = float(jnp.mean(f_data ** 2))

    if verbose:
        print(f"\nNeural FPTA complete:")
        print(f"  Disc game components: {nc}")
        print(f"  Eigenvalues: {schur.eigenvalues[:nc]}")
        imp = importance(schur.eigenvalues[:nc], f_norm_sq)
        print(f"  Importance: {imp}")

    return NeuralFPTAResult(
        basis=basis,
        coefficient_matrix=C_final,
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        n_components=nc,
        f_norm_sq=f_norm_sq,
        train_history=history,
    )


def neural_fpta_from_matrix(
    F: Array,
    traits: Array | None = None,
    d: int = 16,
    hidden_dims: tuple[int, ...] = (64, 64),
    config: TrainConfig | None = None,
    key: Array | None = None,
    n_components: int | None = None,
    verbose: bool = True,
) -> NeuralFPTAResult:
    """Convenience: train neural FPTA from a payoff matrix.

    Converts an N x N payoff matrix into pairwise training data and
    runs neural_fpta. Useful when you have discrete agents with known
    traits and want to learn a continuous embedding.

    Args:
        F: payoff matrix, shape (N, N). Should be approximately skew-symmetric.
        traits: agent trait vectors, shape (N,) or (N, T).
            If None, uses indices [0, 1, ..., N-1] as traits.
        d: basis dimension.
        hidden_dims: MLP hidden layer sizes.
        config: training hyperparameters.
        key: PRNG key.
        n_components: max disc game components.
        verbose: print training progress.

    Returns:
        NeuralFPTAResult.
    """
    N = F.shape[0]

    # Enforce skew-symmetry
    F = 0.5 * (F - F.T)

    if traits is None:
        traits = jnp.arange(N, dtype=jnp.float32)

    traits = jnp.atleast_1d(jnp.array(traits, dtype=jnp.float32))
    trait_dim = 1 if traits.ndim == 1 else traits.shape[1]

    # Expand to all pairs
    idx_x, idx_y = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    idx_x = idx_x.ravel()
    idx_y = idx_y.ravel()

    if traits.ndim == 1:
        x_data = traits[idx_x][:, None]
        y_data = traits[idx_y][:, None]
    else:
        x_data = traits[idx_x]
        y_data = traits[idx_y]

    f_data = F[idx_x, idx_y]

    return neural_fpta(
        x_data, y_data, f_data,
        d=d, trait_dim=trait_dim, hidden_dims=hidden_dims,
        config=config, key=key, n_components=n_components,
        verbose=verbose,
    )
