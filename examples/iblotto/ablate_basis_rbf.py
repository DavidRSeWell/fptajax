"""Path-A ablation: joint training of encoder + skill + C with a k-means RBF basis.

Replaces the trained ``NeuralBasis`` with a non-parametric RBF basis whose
centres are k-means cluster centroids of the *current* encoder embeddings.
Centres are refreshed every ``--recluster_every`` steps so they track the
encoder as training proceeds. Bandwidth ``sigma`` is set by the median
pairwise distance heuristic at each refresh. ``C`` and the encoder/skill
weights still receive gradients through the F-loss; only the centres and
the bandwidth are non-differentiable (treated as constants between
refreshes).

Outputs match the ``hierarchical_skill_fpta`` checkpoint format
``{encoder.eqx, basis.eqx, meta.pkl}`` so the existing
``trait_recovery.py`` and ``compare_pta_vs_bfpta.py`` analyses can be run
against the result. ``basis.eqx`` here is a small frozen module holding
the cluster centres and bandwidth (so trait_recovery.py needs to know
how to reconstruct it; see meta.pkl["basis_kind"] = "rbf_kmeans").

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -m examples.iblotto.ablate_basis_rbf \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --out_dir bfpta_rbf_checkpoints/main_v1_seed0 \
        --m_centres 50 --n_steps 10000 --recluster_every 500 \
        --c_correct_every 1000 --seed 0 \
        --output_json results/ablate_rbf_main_v1_seed0.json
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
from sklearn.cluster import KMeans

from examples.iblotto.behavioral import drop_dead_agents
from fptajax.decomposition import skew_symmetric_schur
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games
from fptajax.hierarchical_skill import SkillHead
from fptajax.neural import SkewParam, TrainConfig


# ---------------------------------------------------------------------------
# RBF basis helper
# ---------------------------------------------------------------------------


def rbf_evaluate(traits: Array, centres: Array, sigma: float) -> Array:
    """Compute (N, m) RBF basis at given trait points.

    ``traits``  shape (N, T), ``centres`` shape (m, T), ``sigma`` scalar.
    """
    diff = traits[:, None, :] - centres[None, :, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)
    return jnp.exp(-sq_dist / (2.0 * sigma * sigma))


def median_pairwise_distance(traits: np.ndarray) -> float:
    """Median Euclidean distance among row pairs of ``traits`` (n, T)."""
    n = traits.shape[0]
    if n < 2:
        return 1.0
    diff = traits[:, None, :] - traits[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(np.median(sq_dist[iu])))


# ---------------------------------------------------------------------------
# Custom model wrapper: encoder + skill + skew C
# ---------------------------------------------------------------------------


class _AblationModel(eqx.Module):
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    skew: SkewParam


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _ablation_loss(
    model: _AblationModel,
    games_i: Array, token_mask_i: Array, game_mask_i: Array,
    games_j: Array, token_mask_j: Array, game_mask_j: Array,
    f_batch: Array, centres: Array, sigma: Array,
) -> tuple[Array, dict]:
    """F-MSE loss with skill + RBF disc-game predictor.

    ``centres`` and ``sigma`` are passed in as constants — gradients do not
    flow into them (they're refreshed via k-means outside the gradient
    update).
    """
    C = model.skew.C
    traits_i = model.encoder.encode_batch(games_i, token_mask_i, game_mask_i)
    traits_j = model.encoder.encode_batch(games_j, token_mask_j, game_mask_j)
    s_i = jax.vmap(model.skill_head)(traits_i)
    s_j = jax.vmap(model.skill_head)(traits_j)
    B_i = rbf_evaluate(traits_i, centres, sigma)
    B_j = rbf_evaluate(traits_j, centres, sigma)
    disc = jnp.sum(B_i @ C * B_j, axis=-1)
    f_hat = (s_i - s_j) + disc
    mse = jnp.mean((f_batch - f_hat) ** 2)
    metrics = dict(loss=mse, mse=mse,
                   skill_std=jnp.std(jnp.concatenate([s_i, s_j])),
                   disc_std=jnp.std(disc),
                   C_norm=jnp.linalg.norm(C))
    return mse, metrics


def _eval_pair_mse(
    model: _AblationModel, agent_data: Array, agent_token_mask: Array,
    agent_game_mask: Array, idx_i: Array, idx_j: Array, f_pairs: Array,
    centres: Array, sigma: Array,
) -> float:
    """MSE on the given (idx_i, idx_j) pair list using all agents' games."""
    traits = model.encoder.encode_batch(agent_data, agent_token_mask, agent_game_mask)
    B = rbf_evaluate(traits, centres, sigma)
    s = jax.vmap(model.skill_head)(traits)
    C = model.skew.C
    F_hat = (s[:, None] - s[None, :]) + B @ C @ B.T
    return float(jnp.mean((f_pairs - F_hat[idx_i, idx_j]) ** 2))


def _closed_form_C_ortho(
    B: np.ndarray, F_train: np.ndarray, train_idx_i: np.ndarray,
    train_idx_j: np.ndarray, ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form skew-symmetric C in orthonormal-basis space.

    Orthonormalises B via eigen-inv-sqrt of its empirical Gram, solves
    least-squares for skew C against train pairs, and returns
    ``(C_orthonormal, L_inv_sqrt)`` so the caller can map back to raw-basis
    space if needed.
    """
    N, m = B.shape
    G = (B.T @ B) / N + ridge * np.eye(m)
    w, V = np.linalg.eigh(G)
    L_inv_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    B_ortho = B @ L_inv_sqrt
    F_full = np.zeros((N, N))
    F_full[train_idx_i, train_idx_j] = F_train
    C = (B_ortho.T @ F_full @ B_ortho) / max(len(train_idx_i), 1)
    C = 0.5 * (C - C.T)
    return C, L_inv_sqrt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    skew: SkewParam
    centres: np.ndarray
    sigma: float
    eigenvalues: np.ndarray
    schur_vectors: np.ndarray
    n_components: int
    train_history: list = field(default_factory=list)
    chosen_step: int = -1
    best_test_mse: float = float("inf")


def train_rbf_ablation(
    ds, train_pairs, test_pairs, m_centres: int = 50,
    trait_dim: int = 24, d_model: int = 32, n_layers: int = 1, n_heads: int = 2,
    n_steps: int = 10000, batch_size: int = 32, lr: float = 5e-4,
    recluster_every: int = 500, c_correct_every: int = 1000,
    eval_every: int = 500, ridge: float = 1e-4,
    G_sample: int = 4, G_sample_eval: int | None = None,
    seed: int = 0, verbose: bool = True,
) -> AblationResult:
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

    # Train pair data
    train_pairs_np = np.asarray(train_pairs)
    test_pairs_np  = np.asarray(test_pairs)
    train_idx_i = train_pairs_np[:, 0]; train_idx_j = train_pairs_np[:, 1]
    test_idx_i  = test_pairs_np[:, 0];  test_idx_j  = test_pairs_np[:, 1]
    train_f = jnp.asarray(ds.F[train_idx_i, train_idx_j])
    test_f  = jnp.asarray(ds.F[test_idx_i,  test_idx_j])
    N_train = len(train_idx_i)

    # Eval-tensor stash for full-population encoder forward (used at C-correction
    # and re-clustering and final eval). Use a fixed game sub-sample for stability.
    eval_games, eval_tmask, eval_gmask = _sample_games(
        np.asarray(agent_games), np.asarray(agent_tmask),
        np.asarray(agent_gmask), np.arange(N), G_sample_eval, rng,
    )
    eval_games  = jnp.asarray(eval_games)
    eval_tmask  = jnp.asarray(eval_tmask)
    eval_gmask  = jnp.asarray(eval_gmask)

    # Build initial model
    key, k1, k2, k3 = jax.random.split(key, 4)
    encoder = HierarchicalSetEncoder(
        token_dim=sa_dim, L_max=L_max, trait_dim=trait_dim,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        mlp_ratio=4, rho_hidden=(64,), key=k1,
    )
    skill_head = SkillHead(trait_dim, hidden=(64,), key=k2)
    skew = SkewParam(m_centres, key=k3)
    model = _AblationModel(encoder=encoder, skill_head=skill_head, skew=skew)

    # Initial centres from a k-means on the freshly-initialised encoder
    traits_init = np.asarray(model.encoder.encode_batch(eval_games, eval_tmask, eval_gmask))
    km = KMeans(n_clusters=m_centres, random_state=seed, n_init=5).fit(traits_init)
    centres = jnp.asarray(km.cluster_centers_, dtype=jnp.float32)
    sigma   = jnp.asarray(median_pairwise_distance(traits_init), dtype=jnp.float32)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch, centres, sigma):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _ablation_loss(m, gi, tmi, gmi, gj, tmj, gmj,
                                     f_batch, centres, sigma),
            has_aux=True,
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

    history: list[dict] = []
    best = dict(test_mse=float("inf"), model=None, centres=None,
                sigma=None, step=-1)

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
            jnp.asarray(f_batch_np), centres, sigma,
        )

        # ---- Periodic re-clustering ----
        if (step + 1) % recluster_every == 0:
            traits_now = np.asarray(model.encoder.encode_batch(
                eval_games, eval_tmask, eval_gmask))
            km = KMeans(n_clusters=m_centres, random_state=seed,
                        n_init=3).fit(traits_now)
            centres = jnp.asarray(km.cluster_centers_, dtype=jnp.float32)
            sigma   = jnp.asarray(median_pairwise_distance(traits_now),
                                  dtype=jnp.float32)
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            if verbose:
                print(f"  [step {step+1}] re-clustered: |c|={m_centres}, "
                      f"σ={float(sigma):.4f}")

        # ---- Periodic closed-form C correction ----
        if (step + 1) % c_correct_every == 0:
            traits_now = np.asarray(model.encoder.encode_batch(
                eval_games, eval_tmask, eval_gmask))
            B_now = np.asarray(rbf_evaluate(jnp.asarray(traits_now),
                                            centres, sigma))
            train_f_np = np.asarray(train_f)
            C_ortho, L_inv = _closed_form_C_ortho(
                B_now, train_f_np, train_idx_i, train_idx_j, ridge,
            )
            # Map orthonormal C back to raw-basis space:
            #   pred_ij = b(x_i)^T C_raw b(x_j) where b is raw RBF.
            #   Equivalent in orthonormal: b_o = b L_inv, so b_o^T C_o b_o
            #   = b^T (L_inv C_o L_inv^T) b. Hence C_raw = L_inv C_o L_inv^T.
            C_raw = L_inv @ C_ortho @ L_inv.T
            C_raw = 0.5 * (C_raw - C_raw.T)
            model = eqx.tree_at(lambda m: m.skew.A,
                                model, jnp.asarray(C_raw / 2.0))
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            if verbose:
                print(f"  [step {step+1}] C-correction (||C||="
                      f"{float(jnp.linalg.norm(C_raw)):.3f})")

        # ---- Eval + best-step tracking ----
        if (step % eval_every == 0) or (step == n_steps - 1):
            train_mse = _eval_pair_mse(model, eval_games, eval_tmask, eval_gmask,
                                       jnp.asarray(train_idx_i),
                                       jnp.asarray(train_idx_j),
                                       train_f, centres, sigma)
            test_mse  = _eval_pair_mse(model, eval_games, eval_tmask, eval_gmask,
                                       jnp.asarray(test_idx_i),
                                       jnp.asarray(test_idx_j),
                                       test_f, centres, sigma)
            record = dict(step=step, train_mse=train_mse, test_mse=test_mse,
                          **{k: float(v) for k, v in metrics.items()})
            history.append(record)
            if verbose:
                print(f"  step {step:5d} | mse={float(metrics['mse']):.4f} "
                      f"| train={train_mse:.4f} | test={test_mse:.4f}")
            if test_mse < best["test_mse"]:
                best = dict(test_mse=test_mse, model=model,
                            centres=centres, sigma=sigma, step=step)

    # Restore best
    if best["model"] is not None:
        model = best["model"]
        centres = best["centres"]
        sigma = best["sigma"]
        if verbose:
            print(f"\n  Early stop: best step={best['step']} "
                  f"test_mse={best['test_mse']:.4f}")

    # Final closed-form C in orthonormal basis (this is the "official" model)
    traits_final = np.asarray(model.encoder.encode_batch(
        eval_games, eval_tmask, eval_gmask))
    B_final = np.asarray(rbf_evaluate(jnp.asarray(traits_final), centres, sigma))
    C_ortho_final, L_inv_final = _closed_form_C_ortho(
        B_final, np.asarray(train_f), train_idx_i, train_idx_j, ridge,
    )
    sch = skew_symmetric_schur(np.asarray(C_ortho_final, dtype=np.float64))
    nc = int(sch.n_components)
    if verbose:
        print(f"\n  Final (orthonormal) Schur: K={nc}, "
              f"top-6 ω = {np.asarray(sch.eigenvalues[:6]).round(2)}")

    return AblationResult(
        encoder=model.encoder, skill_head=model.skill_head, skew=model.skew,
        centres=np.asarray(centres), sigma=float(sigma),
        eigenvalues=np.asarray(sch.eigenvalues),
        schur_vectors=np.asarray(sch.Q),
        n_components=nc,
        train_history=history,
        chosen_step=best["step"],
        best_test_mse=float(best["test_mse"]),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(bundle_path: Path, out_dir: Path, output_json: Path | None,
         m_centres: int, n_steps: int, recluster_every: int,
         c_correct_every: int, seed: int, frac_train: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  Path-A ablation: encoder + k-means RBF basis (m={m_centres})")
    print("=" * 72)

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"  N={N}, sa_dim={ds.sa_dim}, G_max={ds.G_max}, L_max={ds.L_max}")

    # Train/test split — match benchmark.py convention (only observed pairs)
    rng = np.random.RandomState(seed)
    iu = np.argwhere(np.triu(ds.observed_mask, 1))
    rng.shuffle(iu)
    n_train = int(frac_train * len(iu))
    train_un, test_un = iu[:n_train], iu[n_train:]
    train_pairs = np.concatenate([train_un, train_un[:, [1, 0]]], axis=0)
    test_pairs  = np.concatenate([test_un,  test_un[:, [1, 0]]],  axis=0)
    print(f"  train pairs (both dirs): {len(train_pairs)}, test: {len(test_pairs)}")

    t0 = time.time()
    res = train_rbf_ablation(
        ds, train_pairs, test_pairs,
        m_centres=m_centres, n_steps=n_steps,
        recluster_every=recluster_every,
        c_correct_every=c_correct_every,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"\n  Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- norm_test ---
    f2_test = float(np.mean(ds.F[test_pairs[:, 0], test_pairs[:, 1]] ** 2))
    norm_test_best = res.best_test_mse / max(f2_test, 1e-12)
    print(f"\n  HEADLINE: best test MSE = {res.best_test_mse:.4f}  "
          f"norm_test = {norm_test_best:.4f}  (@ step {res.chosen_step})")

    # --- Save artefacts ---
    eqx.tree_serialise_leaves(str(out_dir / "encoder.eqx"), res.encoder)
    eqx.tree_serialise_leaves(str(out_dir / "skill_head.eqx"), res.skill_head)
    meta = dict(
        sa_dim=int(ds.sa_dim), L_max=int(ds.L_max),
        trait_dim=24, d=int(m_centres),
        d_model=32, n_layers=1, n_heads=2,
        mlp_ratio=4, rho_hidden=(64,),
        basis_kind="rbf_kmeans",
        rbf_centres=res.centres, rbf_sigma=res.sigma,
        coefficient_matrix=np.asarray(res.skew.C),
        eigenvalues=res.eigenvalues, schur_vectors=res.schur_vectors,
        n_components=res.n_components,
        f_norm_sq=None,
        seed=int(seed), n_steps=int(n_steps),
        m_centres=int(m_centres),
        recluster_every=int(recluster_every),
        c_correct_every=int(c_correct_every),
        chosen_step=int(res.chosen_step),
    )
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  Saved checkpoint to {out_dir}/")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(dict(
                method="behavioural_rbf_kmeans",
                m_centres=int(m_centres),
                n_steps=int(n_steps),
                recluster_every=int(recluster_every),
                seed=int(seed),
                chosen_step=int(res.chosen_step),
                test_mse=float(res.best_test_mse),
                norm_test=float(norm_test_best),
                eigenvalues=res.eigenvalues.tolist(),
                wall_time_sec=float(elapsed),
            ), f, indent=2)
        print(f"  Saved JSON → {output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--m_centres", type=int, default=50)
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--recluster_every", type=int, default=500)
    p.add_argument("--c_correct_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac_train", type=float, default=0.8)
    args = p.parse_args()
    main(args.bundle, args.out_dir, args.output_json,
         args.m_centres, args.n_steps,
         args.recluster_every, args.c_correct_every,
         args.seed, args.frac_train)
