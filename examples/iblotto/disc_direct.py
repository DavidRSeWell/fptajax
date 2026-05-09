"""Direct-disc FPTA: encoder maps straight into disc-game coordinates.

No basis, no SkewParam, no closed-form C correction. The encoder produces
a trait vector; a small disc-head MLP maps the trait to ``2K`` numbers
interpreted as ``K`` planar disc embeddings ``(u_k, v_k)``. Pairwise F
is predicted as

    F_hat(i, j) = (s(i) - s(j))                                 [skill term]
                + sum_k [ u_k(i) v_k(j) - v_k(i) u_k(j) ]       [K disc games]

Skew-symmetry is automatic by construction. Training is end-to-end MSE.

Two variants are supported:

  --use_skill 1   :  skill term + disc games   (default; classical FPTA form)
  --use_skill 0   :  pure-disc model (no skill head); tests whether skill is
                     pulling weight on iblotto

Output checkpoint format mirrors ``ablate_basis_rbf.py`` so the existing
trait-recovery pipeline can be extended to read it via
``meta.pkl["basis_kind"] = "disc_direct"`` (the disc head replaces the
basis network).

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -m examples.iblotto.disc_direct \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --out_dir disc_direct_checkpoints/main_v1_with_skill_seed0 \
        --K 10 --n_steps 20000 --use_skill 1 --seed 0 \
        --output_json results/disc_direct_with_skill_seed0.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from examples.iblotto.behavioral import drop_dead_agents
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games
from fptajax.hierarchical_skill import SkillHead


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DiscFPTAModel(eqx.Module):
    """Encoder + skill head + disc head.

    ``skill_head`` is always present (Equinox doesn't love Optional leaves);
    its contribution to the prediction is gated by a static ``use_skill``
    flag passed at the prediction call site. When ``use_skill=False`` the
    skill_head still exists but never receives gradient (its output is
    multiplied by 0 in the loss); we just leave it in the saved checkpoint.
    """
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    disc_head: eqx.nn.MLP
    K: int = eqx.field(static=True)


def _build_model(sa_dim, L_max, trait_dim, d_model, n_layers, n_heads,
                 disc_hidden, K, key):
    k1, k2, k3 = jax.random.split(key, 3)
    encoder = HierarchicalSetEncoder(
        token_dim=sa_dim, L_max=L_max, trait_dim=trait_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        mlp_ratio=4, rho_hidden=(64,), key=k1,
    )
    skill_head = SkillHead(trait_dim, hidden=(64,), key=k2)
    disc_head = eqx.nn.MLP(
        in_size=trait_dim, out_size=2 * K,
        width_size=disc_hidden, depth=2,
        activation=jax.nn.gelu, key=k3,
    )
    return DiscFPTAModel(encoder=encoder, skill_head=skill_head,
                         disc_head=disc_head, K=K)


# ---------------------------------------------------------------------------
# Prediction + loss
# ---------------------------------------------------------------------------


def _disc_predict(z_i: Array, z_j: Array) -> Array:
    """Sum of K planar disc games. ``z`` shape ``(B, 2K)`` reshaped to
    ``(B, K, 2)`` so axis -1 indexes ``(u, v)``."""
    B = z_i.shape[0]
    K = z_i.shape[-1] // 2
    z_i = z_i.reshape(B, K, 2)
    z_j = z_j.reshape(B, K, 2)
    # per-disc bracket [u_i v_j - v_i u_j], then sum over K
    return jnp.sum(z_i[..., 0] * z_j[..., 1] - z_i[..., 1] * z_j[..., 0],
                   axis=-1)


def _disc_subspace_ortho_penalty(Z: Array, K: int) -> Array:
    """Sum of squared off-block-diagonal entries of ``Z^T Z``.

    ``Z`` has shape ``(M, 2K)`` with columns arranged
    ``(u_1, v_1, ..., u_K, v_K)``. Reshapes ``G = Z^T Z`` to
    ``(K, 2, K, 2)`` so the 2x2 block at ``(k, l)`` is the cross-disc
    Gram between disc k and disc l. Diagonal blocks (k == l) encode
    within-disc scale and shape — left free. Off-diagonal blocks measure
    subspace overlap in R^M; this penalty drives them to zero, which is
    the Grassmannian generalisation of PCA's axis orthogonality.

    Rotation-invariant under per-disc SO(2): rotating ``(u_k, v_k)`` by
    ``R_k`` left-multiplies block ``G_kl`` by ``R_k`` (and right-multiplies
    by ``R_l^T``), preserving its Frobenius norm.
    """
    G = Z.T @ Z                                          # (2K, 2K)
    G = G.reshape(K, 2, K, 2)
    block_diags = jnp.diagonal(G, axis1=0, axis2=2)      # (2, 2, K)
    return jnp.sum(G ** 2) - jnp.sum(block_diags ** 2)


def _disc_loss(
    model: DiscFPTAModel,
    games_i: Array, token_mask_i: Array, game_mask_i: Array,
    games_j: Array, token_mask_j: Array, game_mask_j: Array,
    f_batch: Array, use_skill: bool, ortho_weight: float,
) -> tuple[Array, dict]:
    traits_i = model.encoder.encode_batch(games_i, token_mask_i, game_mask_i)
    traits_j = model.encoder.encode_batch(games_j, token_mask_j, game_mask_j)
    z_i = jax.vmap(model.disc_head)(traits_i)
    z_j = jax.vmap(model.disc_head)(traits_j)
    disc = _disc_predict(z_i, z_j)
    if use_skill:
        s_i = jax.vmap(model.skill_head)(traits_i)
        s_j = jax.vmap(model.skill_head)(traits_j)
        f_hat = (s_i - s_j) + disc
        skill_term_std = jnp.std(s_i - s_j)
    else:
        f_hat = disc
        skill_term_std = jnp.array(0.0)
    mse = jnp.mean((f_batch - f_hat) ** 2)
    # Subspace-orthogonality penalty over the batch (i and j combined).
    Z_batch = jnp.concatenate([z_i, z_j], axis=0)        # (2B, 2K)
    ortho_pen = _disc_subspace_ortho_penalty(Z_batch, model.K)
    # Normalise by N_pairs and (2K)^2 to stay roughly scale-invariant
    # in the disc count and batch size.
    M = Z_batch.shape[0]
    ortho_pen_norm = ortho_pen / (M ** 2 * model.K)
    total = mse + ortho_weight * ortho_pen_norm
    metrics = dict(
        loss=total, mse=mse, skill_std=skill_term_std,
        disc_std=jnp.std(disc),
        z_norm=jnp.sqrt(jnp.mean(z_i ** 2)),
        ortho_pen=ortho_pen_norm,
    )
    return total, metrics


def _eval_pair_mse(
    model: DiscFPTAModel, agent_data: Array, agent_token_mask: Array,
    agent_game_mask: Array, idx_i: Array, idx_j: Array, f_pairs: Array,
    use_skill: bool,
) -> float:
    """MSE on the full (idx_i, idx_j) pair list using a fixed game tensor."""
    traits = model.encoder.encode_batch(agent_data, agent_token_mask, agent_game_mask)
    Z = jax.vmap(model.disc_head)(traits)  # (N, 2K)
    K = Z.shape[-1] // 2
    Z = Z.reshape(Z.shape[0], K, 2)
    # Full F_hat: bracket sum across K discs.
    # disc(i,j) = sum_k u_k(i) v_k(j) - v_k(i) u_k(j)
    # = u v^T - v u^T  summed across K, so we vectorise per-disc:
    F_disc = (Z[..., 0] @ Z[..., 1].T) - (Z[..., 1] @ Z[..., 0].T)
    if use_skill:
        s = jax.vmap(model.skill_head)(traits)
        F_hat = (s[:, None] - s[None, :]) + F_disc
    else:
        F_hat = F_disc
    return float(jnp.mean((f_pairs - F_hat[idx_i, idx_j]) ** 2))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class DiscResult:
    model: DiscFPTAModel
    disc_magnitudes: np.ndarray  # (K,) per-disc rms magnitude across agents
    train_history: list = field(default_factory=list)
    chosen_step: int = -1
    best_test_mse: float = float("inf")


def _disc_magnitudes(traits: np.ndarray, model: DiscFPTAModel) -> np.ndarray:
    """Per-disc rms magnitude across agents, sorted descending.

    For each disc k, we compute sqrt(<u_k v_k> energy) ~= std(u_k) * std(v_k),
    then sort. This is the analog of the eigenvalue spectrum from Schur.
    """
    Z = np.asarray(jax.vmap(model.disc_head)(jnp.asarray(traits)))
    K = Z.shape[-1] // 2
    Z = Z.reshape(Z.shape[0], K, 2)
    # Per-disc total Frobenius energy of u v^T - v u^T:
    #   ||u v^T - v u^T||_F^2  = 2 (||u||^2 ||v||^2 - <u,v>^2)
    u_norm_sq = np.sum(Z[..., 0] ** 2, axis=0)  # (K,)
    v_norm_sq = np.sum(Z[..., 1] ** 2, axis=0)
    uv = np.sum(Z[..., 0] * Z[..., 1], axis=0)
    energy = 2.0 * (u_norm_sq * v_norm_sq - uv ** 2)
    mag = np.sqrt(np.maximum(energy, 0.0)) / max(Z.shape[0], 1)
    return np.sort(mag)[::-1]


def train_disc_direct(
    ds, train_pairs, test_pairs, K: int = 10, use_skill: bool = True,
    ortho_weight: float = 0.0,
    trait_dim: int = 24, d_model: int = 32, n_layers: int = 1, n_heads: int = 2,
    disc_hidden: int = 64,
    n_steps: int = 20000, batch_size: int = 32, lr: float = 5e-4,
    eval_every: int = 500,
    G_sample: int = 4, G_sample_eval: int | None = None,
    seed: int = 0, verbose: bool = True,
) -> DiscResult:
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)
    if G_sample_eval is None:
        G_sample_eval = min(8, ds.G_max)

    N = ds.policies.shape[0]
    sa_dim = ds.sa_dim
    L_max = ds.L_max
    agent_games = jnp.asarray(ds.agent_data)
    agent_tmask = jnp.asarray(ds.agent_token_mask)
    agent_gmask = jnp.asarray(ds.agent_game_mask)

    train_pairs_np = np.asarray(train_pairs)
    test_pairs_np  = np.asarray(test_pairs)
    train_idx_i = train_pairs_np[:, 0]; train_idx_j = train_pairs_np[:, 1]
    test_idx_i  = test_pairs_np[:, 0];  test_idx_j  = test_pairs_np[:, 1]
    train_f = jnp.asarray(ds.F[train_idx_i, train_idx_j])
    test_f  = jnp.asarray(ds.F[test_idx_i,  test_idx_j])
    N_train = len(train_idx_i)

    # Fixed eval-game tensor for full-population encoder forward.
    eval_games, eval_tmask, eval_gmask = _sample_games(
        np.asarray(agent_games), np.asarray(agent_tmask),
        np.asarray(agent_gmask), np.arange(N), G_sample_eval, rng,
    )
    eval_games  = jnp.asarray(eval_games)
    eval_tmask  = jnp.asarray(eval_tmask)
    eval_gmask  = jnp.asarray(eval_gmask)

    key, mkey = jax.random.split(key)
    model = _build_model(sa_dim, L_max, trait_dim, d_model, n_layers, n_heads,
                         disc_hidden, K, mkey)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _disc_loss(m, gi, tmi, gmi, gj, tmj, gmj,
                                 f_batch, use_skill, ortho_weight),
            has_aux=True,
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

    history: list[dict] = []
    best = dict(test_mse=float("inf"), model=None, step=-1)

    for step in range(n_steps):
        pair_sel = rng.randint(0, N_train, size=batch_size)
        i_idx = train_idx_i[pair_sel]; j_idx = train_idx_j[pair_sel]
        f_batch_np = np.asarray(train_f)[pair_sel]
        gi, tmi, gmi = _sample_games(np.asarray(agent_games),
                                     np.asarray(agent_tmask),
                                     np.asarray(agent_gmask),
                                     i_idx, G_sample, rng)
        gj, tmj, gmj = _sample_games(np.asarray(agent_games),
                                     np.asarray(agent_tmask),
                                     np.asarray(agent_gmask),
                                     j_idx, G_sample, rng)

        model, opt_state, metrics = train_step(
            model, opt_state,
            jnp.asarray(gi), jnp.asarray(tmi), jnp.asarray(gmi),
            jnp.asarray(gj), jnp.asarray(tmj), jnp.asarray(gmj),
            jnp.asarray(f_batch_np),
        )

        if (step % eval_every == 0) or (step == n_steps - 1):
            train_mse = _eval_pair_mse(model, eval_games, eval_tmask, eval_gmask,
                                       jnp.asarray(train_idx_i),
                                       jnp.asarray(train_idx_j),
                                       train_f, use_skill)
            test_mse  = _eval_pair_mse(model, eval_games, eval_tmask, eval_gmask,
                                       jnp.asarray(test_idx_i),
                                       jnp.asarray(test_idx_j),
                                       test_f, use_skill)
            record = dict(step=step, train_mse=train_mse, test_mse=test_mse,
                          **{k: float(v) for k, v in metrics.items()})
            history.append(record)
            if verbose:
                print(f"  step {step:5d} | mse={float(metrics['mse']):.4f} "
                      f"| train={train_mse:.4f} | test={test_mse:.4f}")
            if test_mse < best["test_mse"]:
                best = dict(test_mse=test_mse, model=model, step=step)

    if best["model"] is not None:
        model = best["model"]
        if verbose:
            print(f"\n  Early stop: best step={best['step']} "
                  f"test_mse={best['test_mse']:.4f}")

    # Final disc-magnitude spectrum
    traits_final = np.asarray(model.encoder.encode_batch(
        eval_games, eval_tmask, eval_gmask))
    mags = _disc_magnitudes(traits_final, model)
    if verbose:
        print(f"\n  Top-K disc magnitudes (sorted): {mags.round(3)}")

    return DiscResult(
        model=model, disc_magnitudes=mags,
        train_history=history,
        chosen_step=best["step"], best_test_mse=float(best["test_mse"]),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(bundle_path: Path, out_dir: Path, output_json: Path | None,
         K: int, use_skill: bool, ortho_weight: float, n_steps: int,
         seed: int, frac_train: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  Direct-disc FPTA: K={K}, use_skill={use_skill}, "
          f"ortho_weight={ortho_weight}")
    print("=" * 72)

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"  N={N}, sa_dim={ds.sa_dim}, G_max={ds.G_max}, L_max={ds.L_max}")

    # Train/test split — same convention as benchmark.py / RBF ablation.
    rng = np.random.RandomState(seed)
    iu = np.argwhere(np.triu(ds.observed_mask, 1))
    rng.shuffle(iu)
    n_train = int(frac_train * len(iu))
    train_un, test_un = iu[:n_train], iu[n_train:]
    train_pairs = np.concatenate([train_un, train_un[:, [1, 0]]], axis=0)
    test_pairs  = np.concatenate([test_un,  test_un[:, [1, 0]]],  axis=0)
    print(f"  train pairs (both dirs): {len(train_pairs)}, test: {len(test_pairs)}")

    t0 = time.time()
    res = train_disc_direct(
        ds, train_pairs, test_pairs,
        K=K, use_skill=use_skill, ortho_weight=ortho_weight,
        n_steps=n_steps, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"\n  Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    f2_test = float(np.mean(ds.F[test_pairs[:, 0], test_pairs[:, 1]] ** 2))
    norm_test = res.best_test_mse / max(f2_test, 1e-12)
    print(f"\n  HEADLINE: best test MSE = {res.best_test_mse:.4f}  "
          f"norm_test = {norm_test:.4f}  (@ step {res.chosen_step})")

    # --- Save artefacts ---
    eqx.tree_serialise_leaves(str(out_dir / "encoder.eqx"), res.model.encoder)
    eqx.tree_serialise_leaves(str(out_dir / "skill_head.eqx"), res.model.skill_head)
    eqx.tree_serialise_leaves(str(out_dir / "disc_head.eqx"), res.model.disc_head)
    meta = dict(
        sa_dim=int(ds.sa_dim), L_max=int(ds.L_max),
        trait_dim=24,
        d_model=32, n_layers=1, n_heads=2,
        mlp_ratio=4, rho_hidden=(64,),
        basis_kind="disc_direct",
        K=int(K),
        use_skill=bool(use_skill),
        ortho_weight=float(ortho_weight),
        disc_magnitudes=res.disc_magnitudes,
        seed=int(seed), n_steps=int(n_steps),
        chosen_step=int(res.chosen_step),
    )
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  Saved checkpoint to {out_dir}/")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            method_tag = "disc_direct_with_skill" if use_skill else "disc_direct_no_skill"
            if ortho_weight > 0:
                method_tag += "_ortho"
            json.dump(dict(
                method=method_tag,
                K=int(K), use_skill=bool(use_skill),
                ortho_weight=float(ortho_weight),
                n_steps=int(n_steps),
                seed=int(seed),
                chosen_step=int(res.chosen_step),
                test_mse=float(res.best_test_mse),
                norm_test=float(norm_test),
                disc_magnitudes=res.disc_magnitudes.tolist(),
                wall_time_sec=float(elapsed),
            ), f, indent=2)
        print(f"  Saved JSON → {output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--K", type=int, default=10,
                   help="number of disc games (max effective rank K of F)")
    p.add_argument("--use_skill", type=int, default=1,
                   help="1 to include skill head, 0 for pure-disc model")
    p.add_argument("--ortho_weight", type=float, default=0.0,
                   help="weight on the disc-subspace orthogonality penalty "
                        "(off-block-diagonal entries of Z^T Z); 0 disables")
    p.add_argument("--n_steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac_train", type=float, default=0.8)
    args = p.parse_args()
    main(args.bundle, args.out_dir, args.output_json,
         args.K, bool(args.use_skill), args.ortho_weight,
         args.n_steps, args.seed, args.frac_train)
