"""Hierarchical skill + disc-game FPTA.

Extends :func:`hierarchical_behavioral_fpta` with an explicit transitive
(Elo-style) skill head:

    f_hat(x, y) = (s(x) - s(y)) + B(phi(x))^T C B(phi(y))

  - ``phi``: HierarchicalSetEncoder (same as before)                 --> trait
  - ``s``:   small scalar-valued MLP on top of the trait             --> skill
  - ``B``:   NeuralBasis on top of the trait                         --> basis
  - ``C``:   SkewParam                                                --> d x d skew-symmetric

A mean-zero centering regulariser on ``B`` keeps the skill and disc-game
subspaces orthogonal in the L^2 skew-symmetric function space. Without it,
the bilinear term trivially absorbs transitive structure ("disc game with
the constant direction"), making the skill head redundant.

Result object exposes ``skills()``, ``embed()``, ``predict()`` and a
``decompose_F()`` helper that splits any prediction grid into its skill and
disc-game contributions.

Requires: ``pip install fptajax[neural]`` (equinox, optax).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

try:
    import equinox as eqx
    import optax
except ImportError:
    raise ImportError(
        "Hierarchical skill FPTA requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.neural import NeuralBasis, SkewParam, TrainConfig
from fptajax.decomposition import skew_symmetric_schur
from fptajax.utils import importance, cumulative_importance
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games


# ---------------------------------------------------------------------------
# Skill head
# ---------------------------------------------------------------------------


class SkillHead(eqx.Module):
    """Scalar-valued MLP on top of a trait vector.

    Maps ``trait ∈ R^{trait_dim}`` to a single scalar skill. Inside the
    training loop this is always called per-agent via ``jax.vmap``.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        trait_dim: int,
        hidden: tuple[int, ...] = (64,),
        key: Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.mlp = eqx.nn.MLP(
            in_size=trait_dim,
            out_size=1,
            width_size=hidden[0] if hidden else trait_dim,
            depth=max(len(hidden), 1),
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, trait: Array) -> Array:
        """``trait`` shape ``(trait_dim,)`` -> scalar (shape ``()``)."""
        return self.mlp(trait)[0]


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


class _HierarchicalSkillModel(eqx.Module):
    """Encoder + skill head + basis + skew-sym coupling."""
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    basis: NeuralBasis
    skew: SkewParam


# ---------------------------------------------------------------------------
# Loss + closed-form C
# ---------------------------------------------------------------------------


def _skill_loss(
    model: _HierarchicalSkillModel,
    games_i: Array,
    token_mask_i: Array,
    game_mask_i: Array,
    games_j: Array,
    token_mask_j: Array,
    game_mask_j: Array,
    f_ij: Array,
    ortho_weight: float,
    centering_weight: float,
) -> tuple[Array, dict]:
    """Loss = MSE + ortho_weight · basis-ortho + centering_weight · mean-zero-B."""
    C = model.skew.C  # (d, d)

    traits_i = model.encoder.encode_batch(games_i, token_mask_i, game_mask_i)
    traits_j = model.encoder.encode_batch(games_j, token_mask_j, game_mask_j)

    s_i = jax.vmap(model.skill_head)(traits_i)       # (B,)
    s_j = jax.vmap(model.skill_head)(traits_j)       # (B,)

    bi = model.basis.evaluate_batch(traits_i)        # (B, d)
    bj = model.basis.evaluate_batch(traits_j)        # (B, d)

    disc = jnp.sum(bi @ C * bj, axis=-1)             # (B,)
    f_hat = (s_i - s_j) + disc
    mse = jnp.mean((f_ij - f_hat) ** 2)

    # Basis orthonormality — same as hierarchical_behavioral_fpta
    B_all = jnp.concatenate([bi, bj], axis=0)
    N = B_all.shape[0]
    gram = (B_all.T @ B_all) / N
    ortho_loss = jnp.mean((gram - jnp.eye(model.basis.d)) ** 2)

    # Mean-zero centering on B: keeps skill & disc-game subspaces
    # orthogonal in L^2. Without this, a "disc game with the constant
    # direction" would absorb transitive structure and the skill head
    # would be redundant.
    mean_b = jnp.mean(B_all, axis=0)                 # (d,)
    centering_loss = jnp.sum(mean_b ** 2)

    total = (
        mse
        + ortho_weight * ortho_loss
        + centering_weight * centering_loss
    )

    skill_all = jnp.concatenate([s_i, s_j], axis=0)
    metrics = {
        "loss": total,
        "mse": mse,
        "ortho": ortho_loss,
        "centering": centering_loss,
        "C_norm": jnp.linalg.norm(C),
        "skill_std": jnp.std(skill_all),
        "disc_std": jnp.std(disc),
    }
    return total, metrics


def _skill_closed_form_c(
    model: _HierarchicalSkillModel,
    all_games: Array,
    all_token_mask: Array,
    all_game_mask: Array,
    F: Array,
    ridge_lambda: float = 1e-4,
) -> Array:
    """Solve for optimal skew-symmetric C given fixed (encoder, skill_head, basis).

    Minimises ``sum_{i,j} (F[i,j] - (s_i - s_j) - bi^T C bj)^2`` over
    skew-symmetric ``C``, with ridge ``lambda``. Reuses the encoder's
    traits and skill values as fixed; only ``C`` is updated.
    """
    traits = jax.lax.stop_gradient(
        model.encoder.encode_batch(all_games, all_token_mask, all_game_mask)
    )
    skills = jax.lax.stop_gradient(jax.vmap(model.skill_head)(traits))   # (N,)
    B = jax.lax.stop_gradient(model.basis.evaluate_batch(traits))        # (N, d)

    N_agents, d = B.shape
    idx_i, idx_j = jnp.meshgrid(
        jnp.arange(N_agents), jnp.arange(N_agents), indexing="ij",
    )
    idx_i = idx_i.ravel(); idx_j = idx_j.ravel()
    # Residual after subtracting the skill part: we want to fit it with the
    # bilinear term.
    f_residual = F[idx_i, idx_j] - (skills[idx_i] - skills[idx_j])

    bi = B[idx_i]; bj = B[idx_j]
    M = (bi[:, :, None] * bj[:, None, :]).reshape(-1, d * d)

    MtM = M.T @ M + ridge_lambda * jnp.eye(d * d)
    Mtf = M.T @ f_residual
    c_vec = jnp.linalg.solve(MtM, Mtf)
    C_raw = c_vec.reshape(d, d)
    return 0.5 * (C_raw - C_raw.T)


def _skill_eval_mse(
    model: _HierarchicalSkillModel,
    all_games: Array,
    all_tmask: Array,
    all_gmask: Array,
    idx_i: Array,
    idx_j: Array,
    f_pairs: Array,
) -> float:
    traits = model.encoder.encode_batch(all_games, all_tmask, all_gmask)
    skills = jax.vmap(model.skill_head)(traits)
    B = model.basis.evaluate_batch(traits)
    bi = B[idx_i]; bj = B[idx_j]
    disc = jnp.sum(bi @ model.skew.C * bj, axis=-1)
    f_hat = (skills[idx_i] - skills[idx_j]) + disc
    return float(jnp.mean((f_pairs - f_hat) ** 2))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalSkillFPTAResult:
    """Result of hierarchical skill + disc-game FPTA training."""
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    basis: NeuralBasis
    coefficient_matrix: Array           # skew-sym C
    eigenvalues: Array                  # Schur eigenvalues of C (|ω_k|)
    schur_vectors: Array                # Q in C = Q U Q^T
    n_components: int
    f_norm_sq: float | None = None
    train_history: list | None = None

    # ---- core ----

    def encode(self, games: Array, token_mask: Array, game_mask: Array) -> Array:
        return self.encoder.encode_batch(games, token_mask, game_mask)

    def skills(self, games: Array, token_mask: Array, game_mask: Array) -> Array:
        """Per-agent scalar skill values — the Elo axis. Shape (N,)."""
        traits = self.encode(games, token_mask, game_mask)
        return jax.vmap(self.skill_head)(traits)

    def embed(
        self, games: Array, token_mask: Array, game_mask: Array,
    ) -> Array:
        """Disc-game embeddings, shape (N, n_components, 2)."""
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
        """Full predicted F matrix: shape (N_i, N_j)."""
        traits_i = self.encode(games_i, token_mask_i, game_mask_i)
        traits_j = self.encode(games_j, token_mask_j, game_mask_j)
        s_i = jax.vmap(self.skill_head)(traits_i)
        s_j = jax.vmap(self.skill_head)(traits_j)
        bi = self.basis.evaluate_batch(traits_i)
        bj = self.basis.evaluate_batch(traits_j)
        skill = s_i[:, None] - s_j[None, :]
        disc = bi @ self.coefficient_matrix @ bj.T
        return skill + disc

    # ---- decomposition ----

    def decompose_F(
        self,
        games_i: Array, token_mask_i: Array, game_mask_i: Array,
        games_j: Array | None = None,
        token_mask_j: Array | None = None,
        game_mask_j: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Return ``(F_skill, F_disc, F_total)`` on the N_i × N_j grid.

        ``F_total = F_skill + F_disc`` pointwise. If the j-side args are
        omitted, the decomposition is evaluated on the same agents on
        both sides (self-play grid).
        """
        if games_j is None:
            games_j, token_mask_j, game_mask_j = games_i, token_mask_i, game_mask_i
        traits_i = self.encode(games_i, token_mask_i, game_mask_i)
        traits_j = self.encode(games_j, token_mask_j, game_mask_j)
        s_i = jax.vmap(self.skill_head)(traits_i)
        s_j = jax.vmap(self.skill_head)(traits_j)
        bi = self.basis.evaluate_batch(traits_i)
        bj = self.basis.evaluate_batch(traits_j)
        F_skill = s_i[:, None] - s_j[None, :]
        F_disc = bi @ self.coefficient_matrix @ bj.T
        return F_skill, F_disc, F_skill + F_disc

    def variance_decomposition(
        self,
        games: Array, token_mask: Array, game_mask: Array,
        pair_tuples: np.ndarray | None = None,
    ) -> dict:
        """Fraction of predicted F variance carried by skill vs disc games.

        If ``pair_tuples`` is provided (shape ``(P, 2)`` of agent indices),
        restricts the variance computation to those pairs. Otherwise uses
        the full off-diagonal N×N grid.
        """
        F_skill, F_disc, F_total = self.decompose_F(games, token_mask, game_mask)
        F_skill = np.asarray(F_skill)
        F_disc = np.asarray(F_disc)
        F_total = np.asarray(F_total)
        if pair_tuples is not None:
            pair_tuples = np.asarray(pair_tuples)
            i_idx = pair_tuples[:, 0]; j_idx = pair_tuples[:, 1]
            s_flat = F_skill[i_idx, j_idx]
            d_flat = F_disc[i_idx, j_idx]
            t_flat = F_total[i_idx, j_idx]
        else:
            N = F_total.shape[0]
            off = ~np.eye(N, dtype=bool)
            s_flat = F_skill[off]; d_flat = F_disc[off]; t_flat = F_total[off]
        total_var = float(np.var(t_flat))
        skill_var = float(np.var(s_flat))
        disc_var = float(np.var(d_flat))
        return {
            "skill_var": skill_var,
            "disc_var": disc_var,
            "total_var": total_var,
            "skill_frac": skill_var / max(total_var, 1e-12),
            "disc_frac": disc_var / max(total_var, 1e-12),
        }

    # ---- disc-game importance ----

    def get_importance(self) -> Array:
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self) -> Array:
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def hierarchical_skill_fpta(
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
    skill_hidden: tuple[int, ...] = (64,),
    config: TrainConfig | None = None,
    key: Array | None = None,
    n_components: int | None = None,
    train_pairs: Array | None = None,
    test_pairs: Array | None = None,
    eval_every: int = 200,
    G_sample: int = 16,
    G_sample_eval: int = 32,
    numpy_seed: int = 0,
    pretrained_encoder: Optional["HierarchicalSetEncoder"] = None,
    verbose: bool = True,
) -> HierarchicalSkillFPTAResult:
    """Train hierarchical skill + disc-game FPTA end-to-end.

    Same arguments as :func:`hierarchical_behavioral_fpta` plus:

    Args:
        skill_hidden: hidden-layer widths for the scalar skill MLP.
        config.skill_centering_weight: weight on the mean-zero-basis
            regulariser. Default 1.0 is usually fine.

    Returns:
        HierarchicalSkillFPTAResult.
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

    # Pair indices
    idx_i_all, idx_j_all = jnp.meshgrid(
        jnp.arange(N), jnp.arange(N), indexing="ij",
    )
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

    # Eval tensor
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
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    if pretrained_encoder is not None:
        encoder = pretrained_encoder
        if verbose:
            print("  Using pretrained encoder (warm start).")
    else:
        encoder = HierarchicalSetEncoder(
            token_dim=token_dim, L_max=L_max, trait_dim=trait_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            mlp_ratio=mlp_ratio, rho_hidden=rho_hidden, key=k1,
        )
    skill_head = SkillHead(trait_dim, hidden=skill_hidden, key=k2)
    basis = NeuralBasis(trait_dim, d, basis_hidden, key=k3)
    skew = SkewParam(d, key=k4)
    model = _HierarchicalSkillModel(
        encoder=encoder, skill_head=skill_head, basis=basis, skew=skew,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _skill_loss(
                m, gi, tmi, gmi, gj, tmj, gmj, f_batch,
                config.ortho_weight,
                config.skill_centering_weight,
            ),
            has_aux=True,
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

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

        # Periodic closed-form C correction (same cadence as hierarchical_fpta)
        if (
            config.c_correction_every > 0
            and (step + 1) % config.c_correction_every == 0
            and step > 0
        ):
            C_optimal = _skill_closed_form_c(
                model, eval_games_j, eval_tmask_j, eval_gmask_j,
                F, config.ridge_lambda,
            )
            new_A = C_optimal / 2.0
            model = eqx.tree_at(lambda m: m.skew.A, model, new_A)
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            if verbose:
                print(f"  [step {step+1}] C correction applied, "
                      f"||C||={float(jnp.linalg.norm(C_optimal)):.4f}")

        if step % config.log_every == 0 or step == config.n_steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            record = {"step": step, **m}

            if step % eval_every == 0 or step == config.n_steps - 1:
                train_mse = _skill_eval_mse(
                    model, eval_games_j, eval_tmask_j, eval_gmask_j,
                    train_idx_i_j, train_idx_j_j, train_f,
                )
                record["train_mse"] = train_mse
                if test_pairs is not None:
                    test_mse = _skill_eval_mse(
                        model, eval_games_j, eval_tmask_j, eval_gmask_j,
                        test_idx_i, test_idx_j, test_f,
                    )
                    record["test_mse"] = test_mse
            history.append(record)

            if verbose:
                line = (
                    f"  step {step:5d} | loss={m['loss']:.6f} | "
                    f"mse={m['mse']:.6f} | ortho={m['ortho']:.5f} | "
                    f"cent={m['centering']:.5f} | "
                    f"||C||={m['C_norm']:.3f} | "
                    f"skill_std={m['skill_std']:.4f} | "
                    f"disc_std={m['disc_std']:.4f}"
                )
                if "train_mse" in record:
                    line += f" | train_mse={record['train_mse']:.6f}"
                if "test_mse" in record:
                    line += f" | test_mse={record['test_mse']:.6f}"
                print(line)

    # Final C correction (even if c_correction_every is 0, give C its best
    # closed-form value given the final encoder/skill state).
    C_final = _skill_closed_form_c(
        model, eval_games_j, eval_tmask_j, eval_gmask_j,
        F, config.ridge_lambda,
    )
    C_final = 0.5 * (C_final - C_final.T)

    schur = skew_symmetric_schur(C_final)
    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)

    # f_norm_sq — same convention as hierarchical_behavioral_fpta
    f_norm_sq = float(2.0 * jnp.sum(schur.eigenvalues ** 2))

    if verbose:
        print(f"\nHierarchical skill FPTA complete:")
        print(f"  Disc game components: {nc}")
        print(f"  Eigenvalues: {np.asarray(schur.eigenvalues[:nc])}")
        print(f"  Importance:  {np.asarray(importance(schur.eigenvalues[:nc], f_norm_sq))}")

    return HierarchicalSkillFPTAResult(
        encoder=model.encoder,
        skill_head=model.skill_head,
        basis=model.basis,
        coefficient_matrix=C_final,
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        n_components=nc,
        f_norm_sq=f_norm_sq,
        train_history=history,
    )
