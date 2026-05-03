"""Direct-g-prediction hierarchical FPTA.

Supervises the model on a per-pair scalar ``g(x, y)`` (e.g. raw points
player x scored against y, or total wins agent x got against y) rather
than the skew-symmetric differential ``f(x, y)``. The parametrisation is

    g(x, y) = c_0^T s(x) + c_1^T s(y) + B(phi(x))^T C B(phi(y))

with:
  - ``s``:   a *vector-valued* skill head        -> R^{skill_dim}
  - ``c_0``, ``c_1``: learned coefficient vectors -> R^{skill_dim} each
  - ``B``:   NeuralBasis                          -> R^d
  - ``C``:   *unconstrained* d x d coupling (not skew-symmetric)

After training, the corresponding skew-symmetric ``f`` is recovered by
antisymmetrisation:

    f(x, y) = g(x, y) - g(y, x)
            = (c_0 - c_1)^T [s(x) - s(y)] + B(x)^T (C - C^T) B(y).

Disc games of f are extracted via Schur of ``C - C^T`` (post-hoc).

Key differences from :func:`hierarchical_skill_fpta`:
  - Labels are NOT required to be skew-symmetric.
  - The *symmetric* part of g is now observable and supervised, so there
    is no gauge ambiguity to fix with a mean-zero-basis regulariser.
  - Skill is vector-valued, giving multiple "ratings" per agent that can
    be weighted differently on each side of the matchup.

Training signal density per pair increases because each pair contributes
g(i, j) and g(j, i) as two independent scalars (vs a single F[i,j]) —
unless the label happens to be exactly antisymmetric, in which case it
collapses back to the single-label case.

Requires: ``pip install fptajax[neural]``.
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
        "Hierarchical g-FPTA requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.neural import NeuralBasis, TrainConfig
from fptajax.decomposition import skew_symmetric_schur
from fptajax.utils import importance, cumulative_importance
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games


# ---------------------------------------------------------------------------
# Vector skill head
# ---------------------------------------------------------------------------


class VectorSkillHead(eqx.Module):
    """MLP that produces a vector-valued skill s(x) in R^{skill_dim}."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        trait_dim: int,
        skill_dim: int = 4,
        hidden: tuple[int, ...] = (64,),
        key: Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.mlp = eqx.nn.MLP(
            in_size=trait_dim,
            out_size=skill_dim,
            width_size=hidden[0] if hidden else trait_dim,
            depth=max(len(hidden), 1),
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, trait: Array) -> Array:
        """trait: (trait_dim,) -> (skill_dim,)."""
        return self.mlp(trait)


# ---------------------------------------------------------------------------
# Unconstrained coupling parameter
# ---------------------------------------------------------------------------


class _CouplingParam(eqx.Module):
    """Unconstrained d x d coupling matrix, initialised with small values."""
    M: Array
    d: int = eqx.field(static=True)

    def __init__(self, d: int, key: Array):
        self.d = d
        self.M = 0.01 * jax.random.normal(key, (d, d))


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


class _HierarchicalGModel(eqx.Module):
    encoder: HierarchicalSetEncoder
    skill_head: VectorSkillHead
    basis: NeuralBasis
    coupling: _CouplingParam
    c_0: Array          # (skill_dim,)
    c_1: Array          # (skill_dim,)
    g_bias: Array       # () — scalar bias absorbing overall mean of g


# ---------------------------------------------------------------------------
# Loss + eval
# ---------------------------------------------------------------------------


def _g_predict_batch(
    model: _HierarchicalGModel,
    traits_i: Array,         # (B, trait_dim)
    traits_j: Array,         # (B, trait_dim)
) -> Array:
    s_i = jax.vmap(model.skill_head)(traits_i)        # (B, skill_dim)
    s_j = jax.vmap(model.skill_head)(traits_j)
    bi = model.basis.evaluate_batch(traits_i)         # (B, d)
    bj = model.basis.evaluate_batch(traits_j)
    skill_term = s_i @ model.c_0 + s_j @ model.c_1    # (B,)
    bilinear = jnp.sum(bi @ model.coupling.M * bj, axis=-1)
    return skill_term + bilinear + model.g_bias


def _g_loss(
    model: _HierarchicalGModel,
    games_i: Array, token_mask_i: Array, game_mask_i: Array,
    games_j: Array, token_mask_j: Array, game_mask_j: Array,
    g_ij: Array,
    ortho_weight: float,
) -> tuple[Array, dict]:
    """MSE on direct g labels + basis-orthonormality regulariser."""
    traits_i = model.encoder.encode_batch(games_i, token_mask_i, game_mask_i)
    traits_j = model.encoder.encode_batch(games_j, token_mask_j, game_mask_j)

    g_hat = _g_predict_batch(model, traits_i, traits_j)
    mse = jnp.mean((g_ij - g_hat) ** 2)

    # Basis orthonormality (same role as in other hierarchical variants)
    bi = model.basis.evaluate_batch(traits_i)
    bj = model.basis.evaluate_batch(traits_j)
    B_all = jnp.concatenate([bi, bj], axis=0)
    N = B_all.shape[0]
    gram = (B_all.T @ B_all) / N
    ortho_loss = jnp.mean((gram - jnp.eye(model.basis.d)) ** 2)

    total = mse + ortho_weight * ortho_loss

    s_all = jnp.concatenate([
        jax.vmap(model.skill_head)(traits_i),
        jax.vmap(model.skill_head)(traits_j),
    ], axis=0)
    metrics = {
        "loss": total,
        "mse": mse,
        "ortho": ortho_loss,
        "coupling_norm": jnp.linalg.norm(model.coupling.M),
        "skill_std": jnp.std(s_all),
        "c0_norm": jnp.linalg.norm(model.c_0),
        "c1_norm": jnp.linalg.norm(model.c_1),
    }
    return total, metrics


def _g_closed_form_c_and_bias(
    model: _HierarchicalGModel,
    all_games: Array,
    all_tmask: Array,
    all_gmask: Array,
    G: Array,
    train_idx_i: Array,
    train_idx_j: Array,
    ridge: float = 1e-4,
) -> tuple[Array, Array]:
    """Ridge LS for (vec(C), bias) given fixed (encoder, skill_head, basis,
    c_0, c_1).

    Solves
        G[i, j] - c_0^T s(x_i) - c_1^T s(x_j)  ≈  B(x_i)^T C B(x_j) + bias

    i.e. fits the bilinear coupling plus offset on the residual after
    subtracting the current skill-head contribution. ``c_0`` and ``c_1``
    are NOT re-fit — gradient descent handles them on the next step. This
    avoids the ill-conditioned joint LS (columns for c_0 / c_1 become
    near-collinear when s(x) has small magnitude, causing blow-up in the
    unconstrained-solve case).
    """
    traits = jax.lax.stop_gradient(
        model.encoder.encode_batch(all_games, all_tmask, all_gmask)
    )
    S = jax.lax.stop_gradient(jax.vmap(model.skill_head)(traits))   # (N, k)
    B = jax.lax.stop_gradient(model.basis.evaluate_batch(traits))   # (N, d)

    d = B.shape[1]
    # Current skill contributions per agent (scalar).
    c0_contrib = S @ model.c_0              # (N,) = c_0^T s(x)
    c1_contrib = S @ model.c_1              # (N,) = c_1^T s(x)

    # Per-pair residual that (C, bias) must explain.
    residual = (
        G[train_idx_i, train_idx_j]
        - c0_contrib[train_idx_i]
        - c1_contrib[train_idx_j]
    )

    bi = B[train_idx_i]
    bj = B[train_idx_j]
    outer = (bi[:, :, None] * bj[:, None, :]).reshape(-1, d * d)
    ones = jnp.ones((len(train_idx_i), 1))

    M = jnp.concatenate([outer, ones], axis=1)   # (P, d^2 + 1)
    MtM = M.T @ M + ridge * jnp.eye(M.shape[1])
    Mty = M.T @ residual
    theta = jnp.linalg.solve(MtM, Mty)

    C = theta[:d * d].reshape(d, d)
    bias = theta[-1]
    return C, bias


def _g_eval_mse(
    model: _HierarchicalGModel,
    all_games: Array, all_tmask: Array, all_gmask: Array,
    idx_i: Array, idx_j: Array,
    g_pairs: Array,
) -> float:
    traits = model.encoder.encode_batch(all_games, all_tmask, all_gmask)
    g_hat = _g_predict_batch(model, traits[idx_i], traits[idx_j])
    return float(jnp.mean((g_pairs - g_hat) ** 2))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalGFPTAResult:
    """Result of g-supervised hierarchical FPTA."""
    encoder: HierarchicalSetEncoder
    skill_head: VectorSkillHead
    basis: NeuralBasis
    c_0: Array                      # (skill_dim,)
    c_1: Array                      # (skill_dim,)
    coupling_matrix: Array          # (d, d), unconstrained — in original g-scale
    g_bias: Array                   # scalar — overall g offset, in original units
    # Skew-symmetric projection of the coupling (for disc-game extraction)
    coefficient_matrix: Array       # (d, d) = C - C^T, skew-symmetric
    eigenvalues: Array              # Schur eigenvalues of coefficient_matrix
    schur_vectors: Array
    n_components: int
    f_norm_sq: float | None = None
    train_history: list | None = None
    # Training-time label normalisation. Stored for reference; all prediction
    # methods return values in the original (un-normalised) g-scale already.
    # train_history entries record MSE in *normalised* units — multiply by
    # g_scale**2 to convert back to original g² units.
    g_mean: float = 0.0
    g_scale: float = 1.0

    # ---- basic state ----

    def encode(self, games, tmask, gmask):
        return self.encoder.encode_batch(games, tmask, gmask)

    def skills(self, games, tmask, gmask):
        """Vector-valued per-agent skill, shape (N, skill_dim)."""
        traits = self.encode(games, tmask, gmask)
        return jax.vmap(self.skill_head)(traits)

    def effective_scalar_skill(self, games, tmask, gmask):
        """(c_0 - c_1)^T s(x) — the scalar combination that actually enters f."""
        s = self.skills(games, tmask, gmask)
        return s @ (self.c_0 - self.c_1)

    # ---- g predictions (direct) ----

    def predict_g(
        self,
        games_i, tmask_i, gmask_i,
        games_j, tmask_j, gmask_j,
    ):
        """g_hat(x, y) on (N_i, N_j) grid — the direct training target."""
        ti = self.encode(games_i, tmask_i, gmask_i)
        tj = self.encode(games_j, tmask_j, gmask_j)
        s_i = jax.vmap(self.skill_head)(ti)
        s_j = jax.vmap(self.skill_head)(tj)
        bi = self.basis.evaluate_batch(ti)
        bj = self.basis.evaluate_batch(tj)
        # Shape out (N_i, N_j): skill_i part broadcasts over j, and vice versa.
        skill_part = (s_i @ self.c_0)[:, None] + (s_j @ self.c_1)[None, :]
        bilinear = bi @ self.coupling_matrix @ bj.T
        return skill_part + bilinear + self.g_bias

    def decompose_g(self, games, tmask, gmask):
        """Split g_hat on the self-play grid into (g_skill_x, g_skill_y,
        g_bilinear, g_bias, g_total)."""
        t = self.encode(games, tmask, gmask)
        s = jax.vmap(self.skill_head)(t)
        B = self.basis.evaluate_batch(t)
        N = s.shape[0]
        g_skill_x = jnp.broadcast_to((s @ self.c_0)[:, None], (N, N))
        g_skill_y = jnp.broadcast_to((s @ self.c_1)[None, :], (N, N))
        g_bilinear = B @ self.coupling_matrix @ B.T
        g_bias = jnp.broadcast_to(self.g_bias, (N, N))
        g_total = g_skill_x + g_skill_y + g_bilinear + g_bias
        return g_skill_x, g_skill_y, g_bilinear, g_bias, g_total

    # ---- f (antisymmetric) predictions ----

    def predict_f(
        self,
        games_i, tmask_i, gmask_i,
        games_j, tmask_j, gmask_j,
    ):
        """f_hat(x, y) = g_hat(x, y) - g_hat(y, x)."""
        g_xy = self.predict_g(games_i, tmask_i, gmask_i,
                              games_j, tmask_j, gmask_j)
        g_yx = self.predict_g(games_j, tmask_j, gmask_j,
                              games_i, tmask_i, gmask_i)
        return g_xy - g_yx.T   # both are (N_i, N_j) after .T

    def decompose_f(self, games, tmask, gmask):
        """Split f_hat on the self-play grid into (f_skill, f_disc, f_total)."""
        t = self.encode(games, tmask, gmask)
        s = jax.vmap(self.skill_head)(t)
        B = self.basis.evaluate_batch(t)
        # f = (c_0 - c_1)^T (s(x) - s(y)) + B^T (C - C.T) B
        delta_c = self.c_0 - self.c_1
        eff_skill = s @ delta_c
        f_skill = eff_skill[:, None] - eff_skill[None, :]
        C_skew = self.coefficient_matrix
        f_disc = B @ C_skew @ B.T
        return f_skill, f_disc, f_skill + f_disc

    def embed(self, games, tmask, gmask):
        """Disc-game embeddings via Schur of (C - C^T), shape (N, n_components, 2)."""
        t = self.encode(games, tmask, gmask)
        B = self.basis.evaluate_batch(t)
        Q = self.schur_vectors
        omegas = self.eigenvalues
        embs = []
        for k in range(self.n_components):
            q1 = Q[:, 2 * k]; q2 = Q[:, 2 * k + 1]
            y1 = jnp.sqrt(omegas[k]) * (B @ q1)
            y2 = jnp.sqrt(omegas[k]) * (B @ q2)
            embs.append(jnp.stack([y1, y2], axis=-1))
        return jnp.stack(embs, axis=1)

    # ---- variance decomposition ----

    def variance_decomposition_g(
        self, games, tmask, gmask,
        pair_tuples: np.ndarray | None = None,
    ) -> dict:
        """Variance fractions of g_hat carried by each component."""
        g_skill_x, g_skill_y, g_bilinear, g_bias, g_total = self.decompose_g(
            games, tmask, gmask,
        )
        g_skill = np.asarray(g_skill_x) + np.asarray(g_skill_y)
        g_bilinear = np.asarray(g_bilinear)
        g_total = np.asarray(g_total)
        N = g_total.shape[0]
        if pair_tuples is not None:
            pair_tuples = np.asarray(pair_tuples)
            i = pair_tuples[:, 0]; j = pair_tuples[:, 1]
            gt = g_total[i, j]; gs = g_skill[i, j]; gb = g_bilinear[i, j]
        else:
            off = ~np.eye(N, dtype=bool)
            gt = g_total[off]; gs = g_skill[off]; gb = g_bilinear[off]
        total_var = float(np.var(gt))
        return {
            "skill_var": float(np.var(gs)),
            "bilinear_var": float(np.var(gb)),
            "total_var": total_var,
            "skill_frac": float(np.var(gs)) / max(total_var, 1e-12),
            "bilinear_frac": float(np.var(gb)) / max(total_var, 1e-12),
        }

    def variance_decomposition_f(
        self, games, tmask, gmask,
        pair_tuples: np.ndarray | None = None,
    ) -> dict:
        """Variance fractions of the antisymmetric f_hat."""
        f_skill, f_disc, f_total = self.decompose_f(games, tmask, gmask)
        f_skill = np.asarray(f_skill)
        f_disc = np.asarray(f_disc)
        f_total = np.asarray(f_total)
        N = f_total.shape[0]
        if pair_tuples is not None:
            pair_tuples = np.asarray(pair_tuples)
            i = pair_tuples[:, 0]; j = pair_tuples[:, 1]
            ft = f_total[i, j]; fs = f_skill[i, j]; fd = f_disc[i, j]
        else:
            off = ~np.eye(N, dtype=bool)
            ft = f_total[off]; fs = f_skill[off]; fd = f_disc[off]
        total_var = float(np.var(ft))
        return {
            "skill_var": float(np.var(fs)),
            "disc_var": float(np.var(fd)),
            "total_var": total_var,
            "skill_frac": float(np.var(fs)) / max(total_var, 1e-12),
            "disc_frac": float(np.var(fd)) / max(total_var, 1e-12),
        }

    def get_importance(self):
        return importance(self.eigenvalues, self.f_norm_sq)

    def get_cumulative_importance(self):
        return cumulative_importance(self.eigenvalues, self.f_norm_sq)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def hierarchical_g_fpta(
    agent_games: np.ndarray,
    agent_token_mask: np.ndarray,
    agent_game_mask: np.ndarray,
    G: Array,
    token_dim: int,
    L_max: int,
    trait_dim: int = 32,
    d: int = 12,
    skill_dim: int = 4,
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
) -> HierarchicalGFPTAResult:
    """Train direct-g-supervised hierarchical FPTA.

    Labels ``G[i, j]`` are per-pair scalars and are NOT required to be
    skew-symmetric. A natural choice for tennis is the total points that
    player i scored against j, summed across all their matches. For RPS
    tournament data it is the total rounds i won against j.

    The closed-form correction at ``c_correction_every`` jointly re-fits
    ``(c_0, c_1, C, g_bias)`` via ridge LS.
    """
    if config is None:
        config = TrainConfig()
    if key is None:
        key = jax.random.PRNGKey(42)

    agent_games = np.asarray(agent_games, dtype=np.float32)
    agent_token_mask = np.asarray(agent_token_mask, dtype=bool)
    agent_game_mask = np.asarray(agent_game_mask, dtype=bool)
    G_raw = jnp.array(G, dtype=jnp.float32)

    N = G_raw.shape[0]
    assert agent_games.shape[0] == N
    assert G_raw.shape == (N, N)

    # Internal label normalisation. Raw g-labels (e.g. total points scored,
    # total wins) can be in the hundreds or thousands, which destabilises the
    # closed-form joint LS on (c_0, c_1, C). We z-score against the TRAIN
    # subset, train in normalised units, and un-normalise predictions only
    # when they leave the module (via the stored ``g_mean`` / ``g_scale``
    # inside the returned ``HierarchicalGFPTAResult``).
    g_train_values = (
        G_raw.ravel()[np.asarray(train_pairs)]
        if train_pairs is not None else G_raw.ravel()
    )
    g_mean = float(jnp.mean(g_train_values))
    g_scale = float(jnp.std(g_train_values))
    g_scale = max(g_scale, 1e-8)  # guard against degenerate constant labels
    if verbose:
        print(f"  G label stats (train): mean={g_mean:.3f}, std={g_scale:.3f}. "
              f"Training in normalised units (g' = (g - mean) / std).")
    G = (G_raw - g_mean) / g_scale

    # Flat pair indices
    idx_i_all, idx_j_all = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing="ij")
    idx_i_all = idx_i_all.ravel()
    idx_j_all = idx_j_all.ravel()
    g_pairs_flat = G[idx_i_all, idx_j_all]

    if train_pairs is not None:
        train_pairs = np.asarray(train_pairs)
        train_idx_i = np.asarray(idx_i_all[train_pairs])
        train_idx_j = np.asarray(idx_j_all[train_pairs])
        train_g = jnp.array(g_pairs_flat[train_pairs])
    else:
        train_idx_i = np.asarray(idx_i_all)
        train_idx_j = np.asarray(idx_j_all)
        train_g = g_pairs_flat
    N_train = len(train_idx_i)

    if test_pairs is not None:
        test_pairs = jnp.array(test_pairs)
        test_idx_i = idx_i_all[test_pairs]
        test_idx_j = idx_j_all[test_pairs]
        test_g = g_pairs_flat[test_pairs]
    train_idx_i_j = jnp.array(train_idx_i)
    train_idx_j_j = jnp.array(train_idx_j)

    # Eval tensor: padded with G_sample_eval games per agent
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
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
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
    skill_head = VectorSkillHead(trait_dim, skill_dim=skill_dim,
                                  hidden=skill_hidden, key=k2)
    basis = NeuralBasis(trait_dim, d, basis_hidden, key=k3)
    coupling = _CouplingParam(d=d, key=k4)
    # Initialise c_0, c_1 with small values. Start with c_0 ≈ -c_1 so that at
    # init we are near the antisymmetric regime, but optimiser is free to
    # diverge.
    c_0_init = 0.1 * jax.random.normal(k5, (skill_dim,))
    c_1_init = -c_0_init
    # Initialise bias to the mean of (normalised) training labels. Since we
    # z-scored above, this is essentially 0 — kept for symmetry with the
    # un-normalised path.
    g_bias_init = jnp.array(float(jnp.mean(train_g)))
    model = _HierarchicalGModel(
        encoder=encoder, skill_head=skill_head, basis=basis, coupling=coupling,
        c_0=c_0_init, c_1=c_1_init, g_bias=g_bias_init,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, g_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _g_loss(
                m, gi, tmi, gmi, gj, tmj, gmj, g_batch, config.ortho_weight,
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
        g_batch_np = np.asarray(train_g)[pair_sel]

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
            jnp.array(g_batch_np),
        )

        # Periodic closed-form re-fit of (C, bias). c_0 and c_1 stay on
        # gradient-descent updates — refitting them here as part of a joint
        # LS becomes ill-conditioned under trait collapse and blows up.
        if (
            config.c_correction_every > 0
            and (step + 1) % config.c_correction_every == 0
            and step > 0
        ):
            C_new, bias_new = _g_closed_form_c_and_bias(
                model, eval_games_j, eval_tmask_j, eval_gmask_j,
                G, train_idx_i_j, train_idx_j_j, config.ridge_lambda,
            )
            model = eqx.tree_at(lambda m: m.coupling.M, model, C_new)
            model = eqx.tree_at(lambda m: m.g_bias, model, bias_new)
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            if verbose:
                print(f"  [step {step+1}] (C, bias) correction: "
                      f"||C||={float(jnp.linalg.norm(C_new)):.3f} "
                      f"bias={float(bias_new):.3f} "
                      f"||c_0||={float(jnp.linalg.norm(model.c_0)):.3f} "
                      f"||c_1||={float(jnp.linalg.norm(model.c_1)):.3f}")

        if step % config.log_every == 0 or step == config.n_steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            record = {"step": step, **m}

            if step % eval_every == 0 or step == config.n_steps - 1:
                train_mse = _g_eval_mse(
                    model, eval_games_j, eval_tmask_j, eval_gmask_j,
                    train_idx_i_j, train_idx_j_j, train_g,
                )
                record["train_mse"] = train_mse
                if test_pairs is not None:
                    test_mse = _g_eval_mse(
                        model, eval_games_j, eval_tmask_j, eval_gmask_j,
                        test_idx_i, test_idx_j, test_g,
                    )
                    record["test_mse"] = test_mse
            history.append(record)

            if verbose:
                line = (
                    f"  step {step:5d} | loss={m['loss']:.4f} | "
                    f"mse={m['mse']:.4f} | ortho={m['ortho']:.5f} | "
                    f"||C||={m['coupling_norm']:.3f} | "
                    f"skill_std={m['skill_std']:.4f} | "
                    f"||c0||={m['c0_norm']:.3f} ||c1||={m['c1_norm']:.3f}"
                )
                if "train_mse" in record:
                    line += f" | train_mse={record['train_mse']:.4f}"
                if "test_mse" in record:
                    line += f" | test_mse={record['test_mse']:.4f}"
                print(line)

    # Final closed-form correction for (C, bias).
    C_f, bias_f = _g_closed_form_c_and_bias(
        model, eval_games_j, eval_tmask_j, eval_gmask_j,
        G, train_idx_i_j, train_idx_j_j, config.ridge_lambda,
    )
    model = eqx.tree_at(lambda m: m.coupling.M, model, C_f)
    model = eqx.tree_at(lambda m: m.g_bias, model, bias_f)

    # Un-normalise MSE entries in history so consumers see values in the
    # original g-scale (multiplying MSE by g_scale**2 reverses the standardise).
    _g_scale2 = g_scale ** 2
    for record in history:
        if "mse" in record:
            record["mse"] *= _g_scale2
        if "loss" in record:
            # "loss" is MSE + orthoweight * ortho_loss + basis-space stuff;
            # scale only the MSE component approximately by assuming ortho is
            # roughly unchanged. We rescale both for consistency, acknowledging
            # it's approximate for the ortho term.
            record["loss"] *= _g_scale2
        if "train_mse" in record:
            record["train_mse"] *= _g_scale2
        if "test_mse" in record:
            record["test_mse"] *= _g_scale2

    # Before returning, un-normalise the linear heads back into the original
    # g-scale so downstream predictions are in the user's units. Because the
    # training-time prediction is
    #     g'_hat = c_0'^T s + c_1'^T s' + B^T C' B + bias',
    # and g' = (g - g_mean) / g_scale, the user-facing ("un-normalised")
    # parameters satisfy:
    #     c_0   = g_scale * c_0'
    #     c_1   = g_scale * c_1'
    #     C     = g_scale * C'
    #     bias  = g_mean + g_scale * bias'
    model = eqx.tree_at(lambda m: m.c_0,        model, g_scale * model.c_0)
    model = eqx.tree_at(lambda m: m.c_1,        model, g_scale * model.c_1)
    model = eqx.tree_at(lambda m: m.coupling.M, model, g_scale * model.coupling.M)
    model = eqx.tree_at(
        lambda m: m.g_bias, model,
        jnp.array(g_mean + g_scale * float(model.g_bias), dtype=model.g_bias.dtype),
    )

    # Skew-symmetric FPTA coefficient matrix of f = g(x,y) - g(y,x).
    # Working through:
    #   f(x,y) = (c_0 - c_1)^T (s(x) - s(y)) + B(x)^T (C - C^T) B(y),
    # so the FPTA coefficient matrix of f is exactly (C - C^T), not halved.
    C = model.coupling.M
    C_skew = C - C.T
    schur = skew_symmetric_schur(C_skew)
    nc = schur.n_components
    if n_components is not None:
        nc = min(n_components, nc)
    f_norm_sq = float(2.0 * jnp.sum(schur.eigenvalues ** 2))

    if verbose:
        print(f"\nHierarchical g-FPTA complete:")
        print(f"  Coupling ||C||: {float(jnp.linalg.norm(C)):.3f}, "
              f"||C_skew||: {float(jnp.linalg.norm(C_skew)):.3f}, "
              f"||C_sym||: {float(jnp.linalg.norm(0.5 * (C + C.T))):.3f}")
        print(f"  c_0: {np.asarray(model.c_0)}")
        print(f"  c_1: {np.asarray(model.c_1)}")
        print(f"  c_0 - c_1: {np.asarray(model.c_0 - model.c_1)}")
        print(f"  Disc-game components: {nc}")
        print(f"  Eigenvalues: {np.asarray(schur.eigenvalues[:nc])}")

    return HierarchicalGFPTAResult(
        encoder=model.encoder,
        skill_head=model.skill_head,
        basis=model.basis,
        c_0=model.c_0,
        c_1=model.c_1,
        coupling_matrix=model.coupling.M,
        g_bias=model.g_bias,
        coefficient_matrix=C_skew,
        eigenvalues=schur.eigenvalues[:nc],
        schur_vectors=schur.Q,
        n_components=nc,
        f_norm_sq=f_norm_sq,
        train_history=history,
        g_mean=g_mean,
        g_scale=g_scale,
    )
