"""Behavioral FPTA: learn agent traits from behavior data.

When agent traits are not directly observed, we learn them from behavioral
data D_i = {(s, a)_1, ..., (s, a)_K} — sets of state-action pairs showing
how each agent plays. The full pipeline is:

    D_i → φ (set encoder) → x_i (traits) → b (basis NN) → b(x_i) ∈ R^d
                                                               ↓
                                  f̂(i,j) = b(x_i)^T C b(x_j)

All three components — φ, b, C — are trained jointly end-to-end.

The set encoder uses DeepSets (Zaheer et al. 2017):
    φ_elem: R^{S+A} → R^H          (per-element MLP)
    φ_agg:  R^H → R^T              (post-aggregation MLP)
    x_i = φ_agg( mean_k φ_elem(s_k, a_k) )

Requires: pip install fptajax[neural]  (equinox, optax)
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
        "Behavioral FPTA requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.decomposition import skew_symmetric_schur
from fptajax.neural import NeuralBasis, SkewParam, TrainConfig
from fptajax.utils import importance, cumulative_importance


# ---------------------------------------------------------------------------
# DeepSets encoder: {(s, a)} → trait vector
# ---------------------------------------------------------------------------


class SetEncoder(eqx.Module):
    """DeepSets encoder: maps a set of (state, action) pairs to a trait vector.

    Architecture (Zaheer et al. 2017):
        x = rho( mean_k phi(sa_k) )

    where phi is a per-element MLP and rho is a post-aggregation MLP.

    Attributes:
        phi: per-element MLP, R^{sa_dim} → R^{hidden}.
        rho: aggregation MLP, R^{hidden} → R^{trait_dim}.
        trait_dim: output dimensionality of trait vectors.
    """

    phi: eqx.nn.MLP
    rho: eqx.nn.MLP
    trait_dim: int = eqx.field(static=True)

    def __init__(
        self,
        sa_dim: int,
        trait_dim: int,
        phi_hidden: tuple[int, ...] = (64, 64),
        rho_hidden: tuple[int, ...] = (64,),
        key: Array | None = None,
        activation: Callable = jax.nn.gelu,
    ):
        """Initialize DeepSets encoder.

        Args:
            sa_dim: dimensionality of each (state, action) vector.
            trait_dim: dimensionality of output trait vectors.
            phi_hidden: hidden layer sizes for the per-element MLP.
            rho_hidden: hidden layer sizes for the aggregation MLP.
            key: PRNG key.
            activation: activation function.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        self.trait_dim = trait_dim

        # Per-element encoder: (s, a) → hidden representation
        phi_out = phi_hidden[-1] if phi_hidden else 64
        self.phi = eqx.nn.MLP(
            in_size=sa_dim,
            out_size=phi_out,
            width_size=phi_hidden[0] if phi_hidden else 64,
            depth=len(phi_hidden),
            activation=activation,
            key=k1,
        )

        # Aggregation network: pooled representation → traits
        self.rho = eqx.nn.MLP(
            in_size=phi_out,
            out_size=trait_dim,
            width_size=rho_hidden[0] if rho_hidden else 64,
            depth=len(rho_hidden),
            activation=activation,
            key=k2,
        )

    def __call__(self, sa_set: Array, mask: Array | None = None) -> Array:
        """Encode a single agent's behavior data into a trait vector.

        Args:
            sa_set: state-action pairs, shape (K, sa_dim).
            mask: boolean mask, shape (K,). True = valid entry.
                If None, all entries are treated as valid.

        Returns:
            Trait vector, shape (trait_dim,).
        """
        # Apply phi to each element
        h = jax.vmap(self.phi)(sa_set)  # (K, hidden)

        # Masked mean pooling
        if mask is not None:
            mask_f = mask.astype(h.dtype)[:, None]  # (K, 1)
            h = h * mask_f
            n_valid = jnp.maximum(mask_f.sum(), 1.0)
            pooled = h.sum(axis=0) / n_valid  # (hidden,)
        else:
            pooled = h.mean(axis=0)  # (hidden,)

        # Aggregation
        return self.rho(pooled)  # (trait_dim,)

    def encode_batch(self, sa_batch: Array, mask: Array | None = None) -> Array:
        """Encode a batch of agents' behavior data.

        Args:
            sa_batch: padded behavior data, shape (N, K_max, sa_dim).
            mask: boolean mask, shape (N, K_max). True = valid.
                If None, all entries treated as valid.

        Returns:
            Trait vectors, shape (N, trait_dim).
        """
        if mask is not None:
            return jax.vmap(self.__call__)(sa_batch, mask)
        else:
            return jax.vmap(self.__call__)(sa_batch)


# ---------------------------------------------------------------------------
# Combined behavioral model
# ---------------------------------------------------------------------------


class _BehavioralModel(eqx.Module):
    """Full pipeline: SetEncoder → NeuralBasis → SkewParam."""
    encoder: SetEncoder
    basis: NeuralBasis
    skew: SkewParam


# ---------------------------------------------------------------------------
# Loss and closed-form C
# ---------------------------------------------------------------------------


def _behavioral_loss(
    model: _BehavioralModel,
    sa_i: Array,
    sa_j: Array,
    mask_i: Array | None,
    mask_j: Array | None,
    f_ij: Array,
    ortho_weight: float,
) -> tuple[Array, dict]:
    """Compute loss for behavioral FPTA.

    Pipeline: D_i → encoder → x_i → basis → b(x_i), then f̂ = b^T C b.

    Args:
        model: full behavioral model.
        sa_i: behavior data for agent i, shape (B, K_max, sa_dim).
        sa_j: behavior data for agent j, shape (B, K_max, sa_dim).
        mask_i: mask for agent i, shape (B, K_max) or None.
        mask_j: mask for agent j, shape (B, K_max) or None.
        f_ij: true payoff values, shape (B,).
        ortho_weight: weight for orthogonality regularization.

    Returns:
        (total_loss, metrics_dict).
    """
    C = model.skew.C  # (d, d)

    # Encode behavior → traits
    traits_i = model.encoder.encode_batch(sa_i, mask_i)  # (B, trait_dim)
    traits_j = model.encoder.encode_batch(sa_j, mask_j)  # (B, trait_dim)

    # Traits → basis values
    bi = model.basis.evaluate_batch(traits_i)  # (B, d)
    bj = model.basis.evaluate_batch(traits_j)  # (B, d)

    # Prediction
    f_hat = jnp.sum(bi @ C * bj, axis=-1)  # (B,)

    # MSE
    mse = jnp.mean((f_ij - f_hat) ** 2)

    # Orthogonality regularizer on basis outputs
    B_all = jnp.concatenate([bi, bj], axis=0)  # (2B, d)
    N = B_all.shape[0]
    gram = (B_all.T @ B_all) / N
    ortho_loss = jnp.mean((gram - jnp.eye(model.basis.d)) ** 2)

    total = mse + ortho_weight * ortho_loss

    metrics = {
        "loss": total,
        "mse": mse,
        "ortho": ortho_loss,
        "C_norm": jnp.linalg.norm(C),
    }
    return total, metrics


def _eval_mse(
    model: _BehavioralModel,
    agent_data: Array,
    agent_mask: Array | None,
    idx_i: Array,
    idx_j: Array,
    f_pairs: Array,
) -> float:
    """Compute MSE over a set of pairs (no grad, no regularization)."""
    sa_i = agent_data[idx_i]
    sa_j = agent_data[idx_j]
    mask_i = agent_mask[idx_i] if agent_mask is not None else None
    mask_j = agent_mask[idx_j] if agent_mask is not None else None

    traits_i = model.encoder.encode_batch(sa_i, mask_i)
    traits_j = model.encoder.encode_batch(sa_j, mask_j)
    bi = model.basis.evaluate_batch(traits_i)
    bj = model.basis.evaluate_batch(traits_j)
    C = model.skew.C
    f_hat = jnp.sum(bi @ C * bj, axis=-1)
    return float(jnp.mean((f_pairs - f_hat) ** 2))


def _behavioral_closed_form_c(
    model: _BehavioralModel,
    agent_data: Array,
    agent_mask: Array | None,
    F: Array,
    ridge_lambda: float = 1e-4,
) -> Array:
    """Closed-form optimal C given fixed encoder + basis.

    Computes basis values for all agents, then solves the linear system.
    """
    # Encode all agents
    traits = jax.lax.stop_gradient(
        model.encoder.encode_batch(agent_data, agent_mask)
    )  # (N, trait_dim)

    # Basis values for all agents
    B = jax.lax.stop_gradient(
        model.basis.evaluate_batch(traits)
    )  # (N, d)

    N_agents = B.shape[0]
    d = B.shape[1]

    # Flatten the payoff matrix to pairwise data
    idx_i, idx_j = jnp.meshgrid(
        jnp.arange(N_agents), jnp.arange(N_agents), indexing='ij'
    )
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    f_flat = F[idx_i, idx_j]

    bi = B[idx_i]  # (N^2, d)
    bj = B[idx_j]  # (N^2, d)

    # M[n, :] = vec(bi_n bj_n^T)
    M = (bi[:, :, None] * bj[:, None, :]).reshape(-1, d * d)

    # Ridge-regularized solve
    MtM = M.T @ M + ridge_lambda * jnp.eye(d * d)
    Mtf = M.T @ f_flat
    c_vec = jnp.linalg.solve(MtM, Mtf)

    C_raw = c_vec.reshape(d, d)
    return 0.5 * (C_raw - C_raw.T)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BehavioralFPTAResult:
    """Result of behavioral FPTA.

    Contains the trained encoder, basis, and coefficient matrix, plus the
    Schur decomposition into disc game components.

    Attributes:
        encoder: trained SetEncoder (behavior → traits).
        basis: trained NeuralBasis (traits → basis values).
        coefficient_matrix: learned skew-symmetric C, shape (d, d).
        eigenvalues: omega_k from Schur decomposition.
        schur_vectors: Q from C = QUQ^T, shape (d, d).
        n_components: number of disc game components.
        f_norm_sq: estimated ||f||^2 from training data.
        train_history: training metrics log.
    """
    encoder: SetEncoder
    basis: NeuralBasis
    coefficient_matrix: Array
    eigenvalues: Array
    schur_vectors: Array
    n_components: int
    f_norm_sq: float | None = None
    train_history: list | None = None

    def encode(self, sa_data: Array, mask: Array | None = None) -> Array:
        """Encode behavior data into trait vectors.

        Args:
            sa_data: shape (N, K, sa_dim) — padded behavior data.
            mask: shape (N, K) — boolean mask, or None.

        Returns:
            Trait vectors, shape (N, trait_dim).
        """
        return self.encoder.encode_batch(sa_data, mask)

    def embed(self, sa_data: Array, mask: Array | None = None) -> Array:
        """Full pipeline: behavior data → disc game embeddings.

        Args:
            sa_data: shape (N, K, sa_dim).
            mask: shape (N, K) or None.

        Returns:
            Embeddings, shape (N, n_components, 2).
        """
        traits = self.encode(sa_data, mask)  # (N, trait_dim)
        B = self.basis.evaluate_batch(traits)  # (N, d)

        Q = self.schur_vectors
        omegas = self.eigenvalues

        embeddings = []
        for k in range(self.n_components):
            q1 = Q[:, 2 * k]
            q2 = Q[:, 2 * k + 1]
            y1 = jnp.sqrt(omegas[k]) * (B @ q1)
            y2 = jnp.sqrt(omegas[k]) * (B @ q2)
            embeddings.append(jnp.stack([y1, y2], axis=-1))

        return jnp.stack(embeddings, axis=1)

    def embed_from_traits(self, traits: Array) -> Array:
        """Evaluate embeddings from pre-computed trait vectors.

        Useful when you've already called encode() and want to reuse traits.

        Args:
            traits: shape (N, trait_dim).

        Returns:
            Embeddings, shape (N, n_components, 2).
        """
        B = self.basis.evaluate_batch(traits)
        Q = self.schur_vectors
        omegas = self.eigenvalues

        embeddings = []
        for k in range(self.n_components):
            q1 = Q[:, 2 * k]
            q2 = Q[:, 2 * k + 1]
            y1 = jnp.sqrt(omegas[k]) * (B @ q1)
            y2 = jnp.sqrt(omegas[k]) * (B @ q2)
            embeddings.append(jnp.stack([y1, y2], axis=-1))

        return jnp.stack(embeddings, axis=1)

    def predict(
        self,
        sa_i: Array,
        sa_j: Array,
        mask_i: Array | None = None,
        mask_j: Array | None = None,
    ) -> Array:
        """Predict f(i, j) from behavior data.

        Args:
            sa_i: agent i behavior, shape (N, K, sa_dim).
            sa_j: agent j behavior, shape (M, K, sa_dim).
            mask_i, mask_j: boolean masks or None.

        Returns:
            Predicted payoff, shape (N, M).
        """
        ti = self.encode(sa_i, mask_i)
        tj = self.encode(sa_j, mask_j)
        bi = self.basis.evaluate_batch(ti)
        bj = self.basis.evaluate_batch(tj)
        return bi @ self.coefficient_matrix @ bj.T

    def get_importance(self) -> Array:
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def behavioral_fpta(
    agent_data: Array,
    agent_mask: Array | None,
    F: Array,
    sa_dim: int,
    trait_dim: int = 8,
    d: int = 16,
    phi_hidden: tuple[int, ...] = (64, 64),
    rho_hidden: tuple[int, ...] = (64,),
    basis_hidden: tuple[int, ...] = (64, 64),
    config: TrainConfig | None = None,
    key: Array | None = None,
    n_components: int | None = None,
    train_pairs: Array | None = None,
    test_pairs: Array | None = None,
    eval_every: int = 100,
    verbose: bool = True,
) -> BehavioralFPTAResult:
    """Train behavioral FPTA: learn traits from behavior data + decompose game.

    Full pipeline:
        D_i → SetEncoder(φ, ρ) → x_i → NeuralBasis(b) → b(x_i)
        f̂(i,j) = b(x_i)^T C b(x_j)

    All parameters (encoder φ/ρ, basis b, coefficient A where C=A-A^T)
    are trained jointly via Adam with periodic closed-form C correction.

    Args:
        agent_data: padded behavior data, shape (N, K_max, sa_dim).
            Agent i's data is agent_data[i, :, :] = [(s,a)_1, ..., (s,a)_K].
        agent_mask: boolean mask, shape (N, K_max). True = valid entry.
            If None, all entries treated as valid.
        F: payoff matrix, shape (N, N). Should be approximately skew-symmetric.
        sa_dim: dimensionality of each (state, action) vector.
        trait_dim: dimensionality of inferred trait vectors.
        d: basis dimension (number of basis functions).
        phi_hidden: hidden layers for DeepSets per-element MLP.
        rho_hidden: hidden layers for DeepSets aggregation MLP.
        basis_hidden: hidden layers for the basis network.
        config: training hyperparameters.
        key: PRNG key.
        n_components: max disc game components.
        train_pairs: flat indices into the N×N pair grid to train on, shape (M,).
            If None, uses all N² pairs.
        test_pairs: flat indices for held-out evaluation, shape (M_test,).
            If None, no test evaluation is performed.
        eval_every: evaluate train/test MSE every this many steps.
        verbose: print training progress.

    Returns:
        BehavioralFPTAResult with trained encoder, basis, C, and embeddings.
    """
    if config is None:
        config = TrainConfig()
    if key is None:
        key = jax.random.PRNGKey(42)

    # Validate and prepare data
    agent_data = jnp.array(agent_data, dtype=jnp.float32)
    F = jnp.array(F, dtype=jnp.float32)
    F = 0.5 * (F - F.T)  # enforce skew-symmetry

    if agent_mask is not None:
        agent_mask = jnp.array(agent_mask, dtype=jnp.bool_)

    N = F.shape[0]
    assert agent_data.shape[0] == N, \
        f"agent_data has {agent_data.shape[0]} agents but F is {N}x{N}"

    # Build all pairwise indices
    idx_i, idx_j = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    f_pairs = F[idx_i, idx_j]  # (N^2,)

    # Train/test split over pair indices
    if train_pairs is not None:
        train_pairs = jnp.array(train_pairs)
        train_idx_i = idx_i[train_pairs]
        train_idx_j = idx_j[train_pairs]
        train_f = f_pairs[train_pairs]
    else:
        train_pairs = jnp.arange(N * N)
        train_idx_i = idx_i
        train_idx_j = idx_j
        train_f = f_pairs
    N_train = train_pairs.shape[0]

    if test_pairs is not None:
        test_pairs = jnp.array(test_pairs)
        test_idx_i = idx_i[test_pairs]
        test_idx_j = idx_j[test_pairs]
        test_f = f_pairs[test_pairs]

    # Initialize model
    key, k1, k2, k3 = jax.random.split(key, 4)
    encoder = SetEncoder(sa_dim, trait_dim, phi_hidden, rho_hidden, key=k1)
    basis = NeuralBasis(trait_dim, d, basis_hidden, key=k2)
    skew = SkewParam(d, key=k3)
    model = _BehavioralModel(encoder=encoder, basis=basis, skew=skew)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT-compiled training step
    @eqx.filter_jit
    def train_step(model, opt_state, sa_i, sa_j, mask_i, mask_j, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _behavioral_loss(
                m, sa_i, sa_j, mask_i, mask_j, f_batch, config.ortho_weight,
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
        # Sample a batch of agent pairs from train set
        key, batch_key = jax.random.split(key)
        batch_sel = jax.random.choice(batch_key, N_train, shape=(config.batch_size,))
        i_idx = train_idx_i[batch_sel]
        j_idx = train_idx_j[batch_sel]

        sa_i_batch = agent_data[i_idx]  # (B, K_max, sa_dim)
        sa_j_batch = agent_data[j_idx]  # (B, K_max, sa_dim)
        mask_i_batch = agent_mask[i_idx] if agent_mask is not None else None
        mask_j_batch = agent_mask[j_idx] if agent_mask is not None else None
        f_batch = train_f[batch_sel]

        # Gradient step
        model, opt_state, metrics = train_step(
            model, opt_state,
            sa_i_batch, sa_j_batch, mask_i_batch, mask_j_batch,
            f_batch,
        )

        # Periodic closed-form C correction
        if (step + 1) % config.c_correction_every == 0 and step > 0:
            C_optimal = _behavioral_closed_form_c(
                model, agent_data, agent_mask, F, config.ridge_lambda,
            )
            new_A = C_optimal / 2.0
            model = eqx.tree_at(lambda m: m.skew.A, model, new_A)
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

            if verbose:
                print(f"  [step {step+1}] C correction applied, "
                      f"||C||={float(jnp.linalg.norm(C_optimal)):.4f}")

        # Logging
        if step % config.log_every == 0 or step == config.n_steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            record = {"step": step, **m}

            # Periodic full eval on train/test sets
            if step % eval_every == 0 or step == config.n_steps - 1:
                train_mse = _eval_mse(
                    model, agent_data, agent_mask,
                    train_idx_i, train_idx_j, train_f,
                )
                record["train_mse"] = train_mse
                if test_pairs is not None:
                    test_mse = _eval_mse(
                        model, agent_data, agent_mask,
                        test_idx_i, test_idx_j, test_f,
                    )
                    record["test_mse"] = test_mse

            history.append(record)

            if verbose:
                line = (f"  step {step:5d} | loss={m['loss']:.6f} | "
                        f"mse={m['mse']:.6f} | ortho={m['ortho']:.6f} | "
                        f"||C||={m['C_norm']:.4f}")
                if "train_mse" in record:
                    line += f" | train_mse={record['train_mse']:.6f}"
                if "test_mse" in record:
                    line += f" | test_mse={record['test_mse']:.6f}"
                print(line)

    # Final C correction
    C_final = _behavioral_closed_form_c(
        model, agent_data, agent_mask, F, config.ridge_lambda,
    )
    C_final = 0.5 * (C_final - C_final.T)

    # Schur decomposition
    schur = skew_symmetric_schur(C_final)

    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    f_norm_sq = float(jnp.mean(F ** 2))

    if verbose:
        # Print learned traits for inspection
        traits = model.encoder.encode_batch(agent_data, agent_mask)
        print(f"\nBehavioral FPTA complete:")
        print(f"  Disc game components: {nc}")
        print(f"  Eigenvalues: {schur.eigenvalues[:nc]}")
        print(f"  Importance: {importance(schur.eigenvalues[:nc], f_norm_sq)}")
        print(f"  Learned trait range: [{float(traits.min()):.3f}, "
              f"{float(traits.max()):.3f}]")

    return BehavioralFPTAResult(
        encoder=model.encoder,
        basis=model.basis,
        coefficient_matrix=C_final,
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        n_components=nc,
        f_norm_sq=f_norm_sq,
        train_history=history,
    )
