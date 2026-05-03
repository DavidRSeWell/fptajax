"""MLP-head baseline for the hierarchical encoder.

Ablation study: does the skew-symmetric bilinear form (NeuralBasis + C)
constrain the model, or is the encoder itself the bottleneck?

Architecture:
    games  ->  HierarchicalSetEncoder  ->  trait phi
    (phi_i, phi_j)  ->  AntisymMLPHead  ->  f_hat[i, j]

where AntisymMLPHead wraps an unconstrained MLP g(.) to be antisymmetric:

    f_hat(x, y) = g(phi(x) || phi(y)) - g(phi(y) || phi(x))

This matches F's skew-symmetry by construction (and also enforces
f(x, x) = 0), so the only difference from FPTA is that the predictor is
not forced into a rank-d bilinear form.

Training is plain MSE + gradient descent end-to-end; no orthogonality
regulariser, no closed-form C correction.

Requires: pip install fptajax[neural]  (equinox, optax)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

try:
    import equinox as eqx
    import optax
except ImportError:
    raise ImportError(
        "MLP baseline requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games
from fptajax.neural import TrainConfig


# ---------------------------------------------------------------------------
# Antisymmetric MLP head
# ---------------------------------------------------------------------------


class AntisymMLPHead(eqx.Module):
    """Unconstrained MLP wrapped to be antisymmetric in its two arguments.

        f_hat(x, y) = g(x || y) - g(y || x)

    Where g is a plain feed-forward MLP mapping R^{2*trait_dim} -> R.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        trait_dim: int,
        hidden: tuple[int, ...] = (128, 128),
        key: Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.mlp = eqx.nn.MLP(
            in_size=2 * trait_dim,
            out_size=1,
            width_size=hidden[0] if hidden else 2 * trait_dim,
            depth=max(len(hidden), 1),
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, ti: Array, tj: Array) -> Array:
        """Predict scalar f_hat(i, j) from two trait vectors."""
        fwd = self.mlp(jnp.concatenate([ti, tj]))[0]
        rev = self.mlp(jnp.concatenate([tj, ti]))[0]
        return fwd - rev


class _HierarchicalMLPModel(eqx.Module):
    encoder: HierarchicalSetEncoder
    head: AntisymMLPHead


# ---------------------------------------------------------------------------
# Loss + eval
# ---------------------------------------------------------------------------


def _mlp_loss(
    model: _HierarchicalMLPModel,
    gi: Array, tmi: Array, gmi: Array,
    gj: Array, tmj: Array, gmj: Array,
    f_ij: Array,
) -> tuple[Array, dict]:
    ti = model.encoder.encode_batch(gi, tmi, gmi)  # (B, trait_dim)
    tj = model.encoder.encode_batch(gj, tmj, gmj)
    f_hat = jax.vmap(model.head)(ti, tj)  # (B,)
    mse = jnp.mean((f_ij - f_hat) ** 2)
    return mse, {"loss": mse, "mse": mse, "pred_std": jnp.std(f_hat)}


def _mlp_eval_mse(
    model: _HierarchicalMLPModel,
    all_games: Array,
    all_tmask: Array,
    all_gmask: Array,
    idx_i: Array,
    idx_j: Array,
    f_pairs: Array,
) -> float:
    traits = model.encoder.encode_batch(all_games, all_tmask, all_gmask)
    ti = traits[idx_i]
    tj = traits[idx_j]
    f_hat = jax.vmap(model.head)(ti, tj)
    return float(jnp.mean((f_pairs - f_hat) ** 2))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalMLPResult:
    """Result of hierarchical-encoder + antisymmetric MLP head training."""
    encoder: HierarchicalSetEncoder
    head: AntisymMLPHead
    train_history: list | None = None

    def encode(self, games: Array, token_mask: Array, game_mask: Array) -> Array:
        return self.encoder.encode_batch(games, token_mask, game_mask)

    def predict(
        self,
        games_i: Array, token_mask_i: Array, game_mask_i: Array,
        games_j: Array, token_mask_j: Array, game_mask_j: Array,
    ) -> Array:
        """Return N_i x N_j prediction matrix."""
        ti = self.encode(games_i, token_mask_i, game_mask_i)
        tj = self.encode(games_j, token_mask_j, game_mask_j)
        # Full outer product via double vmap
        return jax.vmap(lambda a: jax.vmap(lambda b: self.head(a, b))(tj))(ti)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def hierarchical_mlp_baseline(
    agent_games: np.ndarray,
    agent_token_mask: np.ndarray,
    agent_game_mask: np.ndarray,
    F: Array,
    token_dim: int,
    L_max: int,
    trait_dim: int = 24,
    d_model: int = 32,
    n_heads: int = 2,
    n_layers: int = 1,
    mlp_ratio: int = 2,
    rho_hidden: tuple[int, ...] = (32,),
    head_hidden: tuple[int, ...] = (128, 128),
    config: TrainConfig | None = None,
    key: Array | None = None,
    train_pairs: Array | None = None,
    test_pairs: Array | None = None,
    eval_every: int = 200,
    G_sample: int = 4,
    G_sample_eval: int = 20,
    numpy_seed: int = 0,
    verbose: bool = True,
) -> HierarchicalMLPResult:
    """Train HierarchicalSetEncoder + AntisymMLPHead end-to-end on MSE(F).

    Args mirror ``hierarchical_behavioral_fpta`` for apples-to-apples
    comparison. Any FPTA-specific knobs (``d``, ``basis_hidden``,
    ``ortho_weight``, ``c_correction_every``) have no analog here.

    Returns:
        HierarchicalMLPResult with the trained encoder + head and a training
        history identical in shape to FPTA's (so the same plotting code works).
    """
    if config is None:
        config = TrainConfig()
    if key is None:
        key = jax.random.PRNGKey(42)

    agent_games = np.asarray(agent_games, dtype=np.float32)
    agent_token_mask = np.asarray(agent_token_mask, dtype=bool)
    agent_game_mask = np.asarray(agent_game_mask, dtype=bool)
    F = jnp.array(F, dtype=jnp.float32)
    F = 0.5 * (F - F.T)

    N = F.shape[0]
    assert agent_games.shape[0] == N

    # Flat pair indices
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

    # Eval tensor: a padded slice with G_sample_eval games per agent
    rng = np.random.RandomState(numpy_seed)
    G_sample_eval = min(G_sample_eval, agent_games.shape[1])
    eval_games, eval_tmask, eval_gmask = _sample_games(
        agent_games, agent_token_mask, agent_game_mask,
        np.arange(N), G_sample_eval, rng,
    )
    eval_games_j = jnp.array(eval_games)
    eval_tmask_j = jnp.array(eval_tmask)
    eval_gmask_j = jnp.array(eval_gmask)

    # Build model
    key, k1, k2 = jax.random.split(key, 3)
    encoder = HierarchicalSetEncoder(
        token_dim=token_dim, L_max=L_max, trait_dim=trait_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        mlp_ratio=mlp_ratio, rho_hidden=rho_hidden, key=k1,
    )
    head = AntisymMLPHead(trait_dim, head_hidden, key=k2)
    model = _HierarchicalMLPModel(encoder=encoder, head=head)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            _mlp_loss, has_aux=True,
        )(model, gi, tmi, gmi, gj, tmj, gmj, f_batch)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), new_opt_state, metrics

    history: list[dict] = []
    for step in range(config.n_steps):
        pair_sel = rng.randint(0, N_train, size=config.batch_size)
        i_agent_idx = train_idx_i[pair_sel]
        j_agent_idx = train_idx_j[pair_sel]
        f_batch_np = np.asarray(train_f)[pair_sel]

        gi, tmi, gmi = _sample_games(
            agent_games, agent_token_mask, agent_game_mask,
            i_agent_idx, G_sample, rng,
        )
        gj, tmj, gmj = _sample_games(
            agent_games, agent_token_mask, agent_game_mask,
            j_agent_idx, G_sample, rng,
        )

        model, opt_state, metrics = train_step(
            model, opt_state,
            jnp.array(gi), jnp.array(tmi), jnp.array(gmi),
            jnp.array(gj), jnp.array(tmj), jnp.array(gmj),
            jnp.array(f_batch_np),
        )

        if step % config.log_every == 0 or step == config.n_steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            record = {"step": step, **m}

            if step % eval_every == 0 or step == config.n_steps - 1:
                train_mse = _mlp_eval_mse(
                    model, eval_games_j, eval_tmask_j, eval_gmask_j,
                    train_idx_i_j, train_idx_j_j, train_f,
                )
                record["train_mse"] = train_mse
                if test_pairs is not None:
                    test_mse = _mlp_eval_mse(
                        model, eval_games_j, eval_tmask_j, eval_gmask_j,
                        test_idx_i, test_idx_j, test_f,
                    )
                    record["test_mse"] = test_mse

            history.append(record)

            if verbose:
                line = (f"  step {step:5d} | loss={m['loss']:.6f} | "
                        f"pred_std={m['pred_std']:.4f}")
                if "train_mse" in record:
                    line += f" | train_mse={record['train_mse']:.6f}"
                if "test_mse" in record:
                    line += f" | test_mse={record['test_mse']:.6f}"
                print(line)

    if verbose:
        print(f"\nHierarchical MLP baseline complete.")

    return HierarchicalMLPResult(
        encoder=model.encoder,
        head=model.head,
        train_history=history,
    )
