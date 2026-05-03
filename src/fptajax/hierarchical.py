"""Hierarchical Behavioral FPTA: transformer-per-game + DeepSets-over-games.

An alternative to the flat DeepSets encoder in behavioral.py. Rather than
treating each (state, action) pair independently, this encoder:

  1. Encodes each full game trajectory (state_t, self_action_t, opp_action_t)
     via a transformer, producing a single game-level vector.
  2. Mean-pools game vectors across all games an agent played, then applies
     an aggregation MLP to produce the agent's trait.

The target player's actions occupy the "self" slot and the opponent's actions
the "opp" slot. The same match data is used twice — once with each player as
"self" — to produce per-agent behavioral data.

Requires: pip install fptajax[neural]  (equinox, optax)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

try:
    import equinox as eqx
    import optax
except ImportError:
    raise ImportError(
        "Hierarchical behavioral FPTA requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.neural import NeuralBasis, SkewParam, TrainConfig
from fptajax.decomposition import skew_symmetric_schur
from fptajax.utils import importance, cumulative_importance


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------


class TransformerBlock(eqx.Module):
    """Standard pre-norm transformer block: LayerNorm → MHA → residual →
    LayerNorm → MLP → residual.
    """

    attn: eqx.nn.MultiheadAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        key: Array = None,
    ):
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key=k1,
        )
        self.ln1 = eqx.nn.LayerNorm(d_model)
        self.ln2 = eqx.nn.LayerNorm(d_model)
        self.mlp = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=d_model * mlp_ratio,
            depth=1,
            activation=jax.nn.gelu,
            key=k2,
        )

    def __call__(self, x: Array, attn_mask: Array) -> Array:
        # x: (L, d_model), attn_mask: (L, L) bool
        h = jax.vmap(self.ln1)(x)
        h = self.attn(h, h, h, mask=attn_mask)
        x = x + h
        h = jax.vmap(self.ln2)(x)
        h = jax.vmap(self.mlp)(h)
        return x + h


class GameEncoder(eqx.Module):
    """Encode a single game trajectory into a vector.

    Pipeline: project tokens → add learned positional embeddings →
    transformer blocks → final layernorm → masked mean pool.

    Attributes:
        proj: linear projection from token_dim → d_model.
        pos_emb: learned positional embeddings, shape (L_max, d_model).
        blocks: tuple of TransformerBlocks.
        ln_final: final layernorm.
        d_model: model dimension (static).
        L_max: maximum supported sequence length (static).
    """

    proj: eqx.nn.Linear
    pos_emb: Array
    blocks: tuple
    ln_final: eqx.nn.LayerNorm
    d_model: int = eqx.field(static=True)
    L_max: int = eqx.field(static=True)

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        L_max: int,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 4,
        key: Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, n_layers + 2)
        self.d_model = d_model
        self.L_max = L_max

        self.proj = eqx.nn.Linear(token_dim, d_model, key=keys[0])
        self.pos_emb = 0.02 * jax.random.normal(keys[1], (L_max, d_model))
        self.blocks = tuple(
            TransformerBlock(d_model, n_heads, mlp_ratio, key=keys[2 + i])
            for i in range(n_layers)
        )
        self.ln_final = eqx.nn.LayerNorm(d_model)

    def __call__(self, game: Array, token_mask: Array) -> Array:
        """Encode a single game.

        Args:
            game: token sequence, shape (L, token_dim).
            token_mask: boolean mask, shape (L,). True = valid token.

        Returns:
            Game vector, shape (d_model,).
        """
        L = game.shape[0]
        x = jax.vmap(self.proj)(game)  # (L, d_model)
        x = x + self.pos_emb[:L]

        # Attention mask: queries attend to valid keys only.
        # Shape (L, L): mask[i, j] = token_mask[j].
        attn_mask = jnp.broadcast_to(token_mask[None, :], (L, L))

        for block in self.blocks:
            x = block(x, attn_mask)

        x = jax.vmap(self.ln_final)(x)

        # Masked mean pool over valid tokens
        mf = token_mask[:, None].astype(x.dtype)
        pooled = (x * mf).sum(axis=0) / jnp.maximum(mf.sum(), 1.0)
        return pooled  # (d_model,)


class HierarchicalSetEncoder(eqx.Module):
    """Hierarchical encoder: transformer-per-game + mean-pool-over-games + rho.

    Maps a set of games (variable count) to a trait vector:
        games → {game_encoder(game_m)} → mean pool → rho → trait

    Attributes:
        game_encoder: GameEncoder for per-game encoding.
        rho: aggregation MLP, d_model → trait_dim.
        trait_dim: output dimension (static).
    """

    game_encoder: GameEncoder
    rho: eqx.nn.MLP
    trait_dim: int = eqx.field(static=True)

    def __init__(
        self,
        token_dim: int,
        L_max: int,
        trait_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 4,
        rho_hidden: tuple[int, ...] = (64,),
        key: Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.trait_dim = trait_dim

        self.game_encoder = GameEncoder(
            token_dim=token_dim,
            d_model=d_model,
            L_max=L_max,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            key=k1,
        )
        self.rho = eqx.nn.MLP(
            in_size=d_model,
            out_size=trait_dim,
            width_size=rho_hidden[0] if rho_hidden else d_model,
            depth=len(rho_hidden) if rho_hidden else 1,
            activation=jax.nn.gelu,
            key=k2,
        )

    def __call__(
        self,
        games: Array,
        token_mask: Array,
        game_mask: Array,
    ) -> Array:
        """Encode a single agent's set of games into a trait vector.

        Args:
            games: shape (G, L, token_dim).
            token_mask: shape (G, L), bool.
            game_mask: shape (G,), bool (True = valid game).

        Returns:
            Trait vector, shape (trait_dim,).
        """
        # Per-game encoding
        game_vecs = jax.vmap(self.game_encoder)(games, token_mask)  # (G, d_model)

        # Masked mean pool over games
        mf = game_mask[:, None].astype(game_vecs.dtype)
        pooled = (game_vecs * mf).sum(axis=0) / jnp.maximum(mf.sum(), 1.0)

        return self.rho(pooled)  # (trait_dim,)

    def encode_batch(
        self,
        games: Array,
        token_mask: Array,
        game_mask: Array,
    ) -> Array:
        """Encode a batch of agents.

        Args:
            games: shape (N, G, L, token_dim).
            token_mask: shape (N, G, L).
            game_mask: shape (N, G).

        Returns:
            Traits, shape (N, trait_dim).
        """
        return jax.vmap(self.__call__)(games, token_mask, game_mask)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


class _HierarchicalBehavioralModel(eqx.Module):
    """Full pipeline: HierarchicalSetEncoder → NeuralBasis → SkewParam."""
    encoder: HierarchicalSetEncoder
    basis: NeuralBasis
    skew: SkewParam


# ---------------------------------------------------------------------------
# Loss and closed-form C
# ---------------------------------------------------------------------------


def _hierarchical_loss(
    model: _HierarchicalBehavioralModel,
    games_i: Array,
    token_mask_i: Array,
    game_mask_i: Array,
    games_j: Array,
    token_mask_j: Array,
    game_mask_j: Array,
    f_ij: Array,
    ortho_weight: float,
    spread_weight: float = 0.0,
    spread_target: float = 1.0,
) -> tuple[Array, dict]:
    """Compute loss for hierarchical behavioral FPTA.

    Loss = MSE + ortho_weight * ||B^T B / N - I||^2 + spread_weight * spread_penalty

    Spread penalty is VICReg-style: for each trait dimension, hinge-penalise
    std below ``spread_target``. Prevents the encoder from collapsing all
    agents to a near-constant trait vector.
    """
    C = model.skew.C  # (d, d)

    traits_i = model.encoder.encode_batch(games_i, token_mask_i, game_mask_i)
    traits_j = model.encoder.encode_batch(games_j, token_mask_j, game_mask_j)

    bi = model.basis.evaluate_batch(traits_i)  # (B, d)
    bj = model.basis.evaluate_batch(traits_j)  # (B, d)

    f_hat = jnp.sum(bi @ C * bj, axis=-1)  # (B,)
    mse = jnp.mean((f_ij - f_hat) ** 2)

    # Orthogonality regularizer
    B_all = jnp.concatenate([bi, bj], axis=0)
    N = B_all.shape[0]
    gram = (B_all.T @ B_all) / N
    ortho_loss = jnp.mean((gram - jnp.eye(model.basis.d)) ** 2)

    # VICReg-style per-dim std hinge on TRAITS (not basis outputs).
    traits_all = jnp.concatenate([traits_i, traits_j], axis=0)  # (2B, trait_dim)
    std_per_dim = jnp.sqrt(jnp.var(traits_all, axis=0) + 1e-6)
    spread_loss = jnp.mean(jax.nn.relu(spread_target - std_per_dim))

    total = mse + ortho_weight * ortho_loss + spread_weight * spread_loss
    metrics = {
        "loss": total,
        "mse": mse,
        "ortho": ortho_loss,
        "spread": spread_loss,
        "trait_std_mean": jnp.mean(std_per_dim),
        "C_norm": jnp.linalg.norm(C),
    }
    return total, metrics


def _hierarchical_closed_form_c(
    model: _HierarchicalBehavioralModel,
    all_games: Array,
    all_token_mask: Array,
    all_game_mask: Array,
    F: Array,
    ridge_lambda: float = 1e-4,
) -> Array:
    """Closed-form optimal C given fixed encoder + basis."""
    traits = jax.lax.stop_gradient(
        model.encoder.encode_batch(all_games, all_token_mask, all_game_mask)
    )  # (N, trait_dim)

    B = jax.lax.stop_gradient(model.basis.evaluate_batch(traits))  # (N, d)

    N_agents, d = B.shape
    idx_i, idx_j = jnp.meshgrid(
        jnp.arange(N_agents), jnp.arange(N_agents), indexing='ij'
    )
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    f_flat = F[idx_i, idx_j]

    bi = B[idx_i]
    bj = B[idx_j]
    M = (bi[:, :, None] * bj[:, None, :]).reshape(-1, d * d)

    MtM = M.T @ M + ridge_lambda * jnp.eye(d * d)
    Mtf = M.T @ f_flat
    c_vec = jnp.linalg.solve(MtM, Mtf)
    C_raw = c_vec.reshape(d, d)
    return 0.5 * (C_raw - C_raw.T)


def _hierarchical_eval_mse(
    model: _HierarchicalBehavioralModel,
    all_games: Array,
    all_token_mask: Array,
    all_game_mask: Array,
    idx_i: Array,
    idx_j: Array,
    f_pairs: Array,
) -> float:
    """Compute MSE over a set of pairs using all available games per agent."""
    # Encode all agents once
    traits = model.encoder.encode_batch(all_games, all_token_mask, all_game_mask)
    B = model.basis.evaluate_batch(traits)
    C = model.skew.C
    bi = B[idx_i]
    bj = B[idx_j]
    f_hat = jnp.sum(bi @ C * bj, axis=-1)
    return float(jnp.mean((f_pairs - f_hat) ** 2))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalBehavioralFPTAResult:
    """Result of hierarchical behavioral FPTA.

    Attributes:
        encoder: trained HierarchicalSetEncoder.
        basis: trained NeuralBasis.
        coefficient_matrix: learned skew-symmetric C.
        eigenvalues: omega_k from Schur decomposition.
        schur_vectors: Q from C = QUQ^T.
        n_components: number of disc game components.
        f_norm_sq: estimated ||f||^2 from training data.
        train_history: training metrics log.
    """
    encoder: HierarchicalSetEncoder
    basis: NeuralBasis
    coefficient_matrix: Array
    eigenvalues: Array
    schur_vectors: Array
    n_components: int
    f_norm_sq: float | None = None
    train_history: list | None = None

    def encode(self, games: Array, token_mask: Array, game_mask: Array) -> Array:
        return self.encoder.encode_batch(games, token_mask, game_mask)

    def embed(self, games: Array, token_mask: Array, game_mask: Array) -> Array:
        traits = self.encode(games, token_mask, game_mask)
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
        games_i: Array, token_mask_i: Array, game_mask_i: Array,
        games_j: Array, token_mask_j: Array, game_mask_j: Array,
    ) -> Array:
        ti = self.encode(games_i, token_mask_i, game_mask_i)
        tj = self.encode(games_j, token_mask_j, game_mask_j)
        bi = self.basis.evaluate_batch(ti)
        bj = self.basis.evaluate_batch(tj)
        return bi @ self.coefficient_matrix @ bj.T

    def get_importance(self) -> Array:
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


# ---------------------------------------------------------------------------
# Game subsampling helper
# ---------------------------------------------------------------------------


def _sample_games(
    all_games: np.ndarray,
    all_token_mask: np.ndarray,
    all_game_mask: np.ndarray,
    agent_idx: np.ndarray,
    G_sample: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each agent in agent_idx, randomly subsample G_sample games.

    Args:
        all_games: (N, G_max, L, token_dim).
        all_token_mask: (N, G_max, L) bool.
        all_game_mask: (N, G_max) bool — True = valid game slot.
        agent_idx: (B,) — batch of agent indices.
        G_sample: number of games to sample per agent.
        rng: numpy RandomState.

    Returns:
        sampled_games: (B, G_sample, L, token_dim).
        sampled_token_mask: (B, G_sample, L).
        sampled_game_mask: (B, G_sample).
    """
    B = len(agent_idx)
    L = all_games.shape[2]
    td = all_games.shape[3]
    sampled_games = np.zeros((B, G_sample, L, td), dtype=all_games.dtype)
    sampled_tmask = np.zeros((B, G_sample, L), dtype=bool)
    sampled_gmask = np.zeros((B, G_sample), dtype=bool)

    for b in range(B):
        a = int(agent_idx[b])
        valid_idx = np.where(all_game_mask[a])[0]
        n_valid = len(valid_idx)
        if n_valid >= G_sample:
            sel = rng.choice(valid_idx, size=G_sample, replace=False)
            sampled_games[b] = all_games[a, sel]
            sampled_tmask[b] = all_token_mask[a, sel]
            sampled_gmask[b, :] = True
        else:
            # Take all valid games, pad the rest (masked out)
            sampled_games[b, :n_valid] = all_games[a, valid_idx]
            sampled_tmask[b, :n_valid] = all_token_mask[a, valid_idx]
            sampled_gmask[b, :n_valid] = True

    return sampled_games, sampled_tmask, sampled_gmask


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def hierarchical_behavioral_fpta(
    agent_games: np.ndarray,
    agent_token_mask: np.ndarray,
    agent_game_mask: np.ndarray,
    F: Array,
    token_dim: int,
    L_max: int,
    trait_dim: int = 32,
    d: int = 12,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    mlp_ratio: int = 4,
    rho_hidden: tuple[int, ...] = (64,),
    basis_hidden: tuple[int, ...] = (128, 128),
    config: TrainConfig | None = None,
    key: Array | None = None,
    n_components: int | None = None,
    train_pairs: Array | None = None,
    test_pairs: Array | None = None,
    eval_every: int = 200,
    G_sample: int = 16,
    G_sample_eval: int = 32,
    numpy_seed: int = 0,
    pretrained_encoder: "HierarchicalSetEncoder | None" = None,
    verbose: bool = True,
) -> HierarchicalBehavioralFPTAResult:
    """Train hierarchical behavioral FPTA.

    Pipeline: games → Transformer-per-game → mean-pool → rho → trait → basis → C.

    Args:
        agent_games: padded game tensors, shape (N, G_max, L_max, token_dim).
            Each token is typically concat(state, self_action_oh, opp_action_oh).
        agent_token_mask: (N, G_max, L_max) bool, True = valid token.
        agent_game_mask: (N, G_max) bool, True = valid game slot.
        F: payoff matrix, shape (N, N). Will be symmetrized.
        token_dim: per-token feature dimension.
        L_max: max game length (for positional embeddings).
        trait_dim: trait vector dimension.
        d: basis dimension.
        d_model: transformer hidden dim.
        n_heads: number of attention heads.
        n_layers: number of transformer layers.
        mlp_ratio: MLP width ratio in transformer blocks.
        rho_hidden: hidden layers for aggregation MLP.
        basis_hidden: hidden layers for neural basis.
        config: training hyperparameters.
        key: PRNG key.
        n_components: max disc game components.
        train_pairs: flat pair indices for training.
        test_pairs: flat pair indices for held-out eval.
        eval_every: step interval for train/test MSE eval.
        G_sample: number of games to subsample per agent per training step.
        G_sample_eval: number of games for closed-form C correction and MSE eval
            (uses more games for stability; set >= max games per agent for full).
        numpy_seed: seed for numpy-based game subsampling.
        verbose: print progress.

    Returns:
        HierarchicalBehavioralFPTAResult.
    """
    if config is None:
        config = TrainConfig()
    if key is None:
        key = jax.random.PRNGKey(42)

    # Prepare data
    agent_games = np.asarray(agent_games, dtype=np.float32)
    agent_token_mask = np.asarray(agent_token_mask, dtype=bool)
    agent_game_mask = np.asarray(agent_game_mask, dtype=bool)
    F = jnp.array(F, dtype=jnp.float32)
    F = 0.5 * (F - F.T)

    N = F.shape[0]
    assert agent_games.shape[0] == N, \
        f"agent_games has {agent_games.shape[0]} agents but F is {N}x{N}"

    # Pair indices
    idx_i_all, idx_j_all = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    idx_i_all = idx_i_all.ravel()
    idx_j_all = idx_j_all.ravel()
    f_pairs = F[idx_i_all, idx_j_all]

    if train_pairs is not None:
        train_pairs = np.asarray(train_pairs)
        train_idx_i = np.asarray(idx_i_all[train_pairs])
        train_idx_j = np.asarray(idx_j_all[train_pairs])
        train_f = jnp.array(f_pairs[train_pairs])
    else:
        train_idx_i = np.asarray(idx_i_all)
        train_idx_j = np.asarray(idx_j_all)
        train_f = f_pairs
    N_train = len(train_idx_i)

    if test_pairs is not None:
        test_pairs = jnp.array(test_pairs)
        test_idx_i = idx_i_all[test_pairs]
        test_idx_j = idx_j_all[test_pairs]
        test_f = f_pairs[test_pairs]
    train_idx_i_j = jnp.array(train_idx_i)
    train_idx_j_j = jnp.array(train_idx_j)

    # For eval: prepare a padded tensor with G_sample_eval games per agent
    rng = np.random.RandomState(numpy_seed)
    G_sample_eval = min(G_sample_eval, agent_games.shape[1])
    eval_games, eval_tmask, eval_gmask = _sample_games(
        agent_games, agent_token_mask, agent_game_mask,
        np.arange(N), G_sample_eval, rng,
    )
    eval_games_j = jnp.array(eval_games)
    eval_tmask_j = jnp.array(eval_tmask)
    eval_gmask_j = jnp.array(eval_gmask)

    # Build model. If a pretrained encoder is supplied, reuse its parameters
    # (useful for warm-starting from contrastive pretraining).
    key, k1, k2, k3 = jax.random.split(key, 4)
    if pretrained_encoder is not None:
        encoder = pretrained_encoder
        if verbose:
            print("  Using pretrained encoder (warm start).")
    else:
        encoder = HierarchicalSetEncoder(
            token_dim=token_dim,
            L_max=L_max,
            trait_dim=trait_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            rho_hidden=rho_hidden,
            key=k1,
        )
    basis = NeuralBasis(trait_dim, d, basis_hidden, key=k2)
    skew = SkewParam(d, key=k3)
    model = _HierarchicalBehavioralModel(encoder=encoder, basis=basis, skew=skew)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # JIT'd training step
    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _hierarchical_loss(
                m, gi, tmi, gmi, gj, tmj, gmj, f_batch,
                config.ortho_weight,
                config.spread_weight,
                config.spread_target,
            ),
            has_aux=True,
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

    # Training loop
    history: list[dict] = []
    for step in range(config.n_steps):
        # Sample pairs from train set
        pair_sel = rng.randint(0, N_train, size=config.batch_size)
        i_agent_idx = train_idx_i[pair_sel]
        j_agent_idx = train_idx_j[pair_sel]
        f_batch_np = np.asarray(train_f)[pair_sel]

        # Subsample games per agent
        gi, tmi, gmi = _sample_games(
            agent_games, agent_token_mask, agent_game_mask,
            i_agent_idx, G_sample, rng,
        )
        gj, tmj, gmj = _sample_games(
            agent_games, agent_token_mask, agent_game_mask,
            j_agent_idx, G_sample, rng,
        )

        # Gradient step
        model, opt_state, metrics = train_step(
            model, opt_state,
            jnp.array(gi), jnp.array(tmi), jnp.array(gmi),
            jnp.array(gj), jnp.array(tmj), jnp.array(gmj),
            jnp.array(f_batch_np),
        )

        # Periodic closed-form C correction (uses G_sample_eval games per agent)
        if (step + 1) % config.c_correction_every == 0 and step > 0:
            C_optimal = _hierarchical_closed_form_c(
                model, eval_games_j, eval_tmask_j, eval_gmask_j,
                F, config.ridge_lambda,
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

            if step % eval_every == 0 or step == config.n_steps - 1:
                train_mse = _hierarchical_eval_mse(
                    model, eval_games_j, eval_tmask_j, eval_gmask_j,
                    train_idx_i_j, train_idx_j_j, train_f,
                )
                record["train_mse"] = train_mse
                if test_pairs is not None:
                    test_mse = _hierarchical_eval_mse(
                        model, eval_games_j, eval_tmask_j, eval_gmask_j,
                        test_idx_i, test_idx_j, test_f,
                    )
                    record["test_mse"] = test_mse

            history.append(record)

            if verbose:
                line = (f"  step {step:5d} | loss={m['loss']:.6f} | "
                        f"mse={m['mse']:.6f} | ortho={m['ortho']:.6f} | "
                        f"||C||={m['C_norm']:.4f} | "
                        f"trait_std={m.get('trait_std_mean', 0.0):.4f}")
                if config.spread_weight > 0:
                    line += f" | spread_loss={m['spread']:.4f}"
                if "train_mse" in record:
                    line += f" | train_mse={record['train_mse']:.6f}"
                if "test_mse" in record:
                    line += f" | test_mse={record['test_mse']:.6f}"
                print(line)

    # Final C correction
    C_final = _hierarchical_closed_form_c(
        model, eval_games_j, eval_tmask_j, eval_gmask_j,
        F, config.ridge_lambda,
    )
    C_final = 0.5 * (C_final - C_final.T)

    # Schur decomposition
    schur = skew_symmetric_schur(C_final)
    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    # f_norm_sq = total explained variance in the neural basis = 2 * sum(omega_k^2)
    # This normalises importance_k = 2*omega_k^2 / f_norm_sq so they sum to 1.
    f_norm_sq = float(2.0 * jnp.sum(schur.eigenvalues ** 2))

    if verbose:
        print(f"\nHierarchical Behavioral FPTA complete:")
        print(f"  Disc game components: {nc}")
        print(f"  Eigenvalues: {schur.eigenvalues[:nc]}")
        print(f"  Importance: {importance(schur.eigenvalues[:nc], f_norm_sq)}")

    return HierarchicalBehavioralFPTAResult(
        encoder=model.encoder,
        basis=model.basis,
        coefficient_matrix=C_final,
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        n_components=nc,
        f_norm_sq=f_norm_sq,
        train_history=history,
    )
