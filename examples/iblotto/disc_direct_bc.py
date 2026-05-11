"""Direct-disc FPTA + behavior-cloning head: encoder-decoder model.

Forks ``disc_direct.py`` (left untouched). Same skill + disc-game structure
plus a small ``behavior_head`` that predicts a Dirichlet over actions
``p(a | s, trait)``. The encoder is now driven by two objectives:

  * F-prediction (skill + disc), as in ``disc_direct.py``.
  * Action-prediction (Dirichlet NLL of ``own_share`` per token).

So the trait becomes a sufficient statistic for both *performance vs. the
population* and *online identification of an opponent from observed play*.

Pairs with ``online_id.py``, which uses ``p(a | s, z)`` to maintain a
particle posterior over opponent traits during play and computes a
Thompson-sampled best response in disc-game space.

Output checkpoint mirrors ``disc_direct.py`` plus one extra leaf
(``behavior_head.eqx``); ``meta.pkl["basis_kind"] = "disc_direct_bc"``.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. python -m examples.iblotto.disc_direct_bc \
        --bundle examples/iblotto/results/behavioral_main_v1_N200_k20_nr50.pkl \
        --out_dir disc_direct_bc_checkpoints/main_v1_seed0 \
        --K 10 --n_steps 20000 --bc_weight 0.1 --seed 0 \
        --output_json results/disc_direct_bc_seed0.json
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
from examples.iblotto.disc_direct import (
    _disc_predict,
    _disc_subspace_ortho_penalty,
    _disc_magnitudes,
)
from fptajax.hierarchical import HierarchicalSetEncoder, _sample_games
from fptajax.hierarchical_skill import SkillHead


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DiscFPTABCModel(eqx.Module):
    """Encoder + skill + disc + behavior heads.

    The behavior head consumes ``(trait, state_context)`` and outputs raw
    pre-softplus values that are turned into a Dirichlet concentration
    vector over the ``n_zones`` action simplex. ``state_context`` is the
    leading ``sa_dim - n_zones`` dims of a token (everything except the
    ``own_share`` action target which lives at the tail; see
    ``examples/iblotto/behavioral.py:history_to_tokens``).
    """
    encoder: HierarchicalSetEncoder
    skill_head: SkillHead
    disc_head: eqx.nn.MLP
    behavior_head: eqx.nn.MLP
    K: int = eqx.field(static=True)
    n_zones: int = eqx.field(static=True)
    sa_dim: int = eqx.field(static=True)
    alpha_floor: float = eqx.field(static=True)


def _build_model(sa_dim, n_zones, L_max, trait_dim, d_model, n_layers, n_heads,
                 disc_hidden, behavior_hidden, K, alpha_floor, key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
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
    state_ctx_dim = sa_dim - n_zones
    behavior_head = eqx.nn.MLP(
        in_size=trait_dim + state_ctx_dim, out_size=n_zones,
        width_size=behavior_hidden, depth=2,
        activation=jax.nn.gelu, key=k4,
    )
    return DiscFPTABCModel(
        encoder=encoder, skill_head=skill_head,
        disc_head=disc_head, behavior_head=behavior_head,
        K=K, n_zones=n_zones, sa_dim=sa_dim, alpha_floor=alpha_floor,
    )


# ---------------------------------------------------------------------------
# Behavior-cloning loss (Dirichlet NLL)
# ---------------------------------------------------------------------------


def _dirichlet_logpdf(a: Array, alpha: Array) -> Array:
    """Log density of a Dirichlet at point ``a``.

    Both ``a`` and ``alpha`` have shape ``(..., n_zones)`` with positive
    entries; ``a`` should sum to 1 along the last axis.
    """
    log_norm = (jax.scipy.special.gammaln(jnp.sum(alpha, axis=-1))
                - jnp.sum(jax.scipy.special.gammaln(alpha), axis=-1))
    log_kernel = jnp.sum((alpha - 1.0) * jnp.log(a), axis=-1)
    return log_norm + log_kernel


def _bc_nll(model: DiscFPTABCModel, traits: Array, games: Array,
            token_mask: Array, eps: float = 1e-3) -> Array:
    """Mean Dirichlet NLL of ``own_share`` actions under ``p(a | s, trait)``.

    Tokens where ``token_mask`` is False (padding) are excluded from the mean.
    The same encoded games are used for both trait extraction and BC targets;
    the trait is bottlenecked through ``trait_dim`` and a set pool, so
    leakage is limited and the train-time setup matches deployment (encode
    observed history, predict next action).
    """
    B, G, L, sa = games.shape
    n_zones = model.n_zones
    state_ctx = games[..., :sa - n_zones]                 # (B, G, L, sctx)
    action = games[..., sa - n_zones:]                    # (B, G, L, n_zones)

    # eps-smooth toward the simplex interior so log(a) is finite for
    # zero-share zones. Sum is preserved.
    action = (action + eps) / (1.0 + n_zones * eps)

    trait_b = jnp.broadcast_to(traits[:, None, None, :],
                               (B, G, L, traits.shape[-1]))
    inp = jnp.concatenate([trait_b, state_ctx], axis=-1)  # (B, G, L, in)

    flat = inp.reshape(-1, inp.shape[-1])
    raw = jax.vmap(model.behavior_head)(flat)             # (B*G*L, n_zones)
    alpha = jax.nn.softplus(raw) + model.alpha_floor
    alpha = alpha.reshape(B, G, L, n_zones)

    logp = _dirichlet_logpdf(action, alpha)               # (B, G, L)
    mask = token_mask.astype(logp.dtype)
    return -(logp * mask).sum() / jnp.maximum(mask.sum(), 1.0)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def _disc_bc_loss(
    model: DiscFPTABCModel,
    games_i, token_mask_i, game_mask_i,
    games_j, token_mask_j, game_mask_j,
    f_batch, use_skill, ortho_weight, bc_weight,
):
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

    Z_batch = jnp.concatenate([z_i, z_j], axis=0)
    ortho_pen = _disc_subspace_ortho_penalty(Z_batch, model.K)
    ortho_pen_norm = ortho_pen / max(4 * model.K * (model.K - 1), 1)

    bc_i = _bc_nll(model, traits_i, games_i, token_mask_i)
    bc_j = _bc_nll(model, traits_j, games_j, token_mask_j)
    bc = 0.5 * (bc_i + bc_j)

    total = mse + ortho_weight * ortho_pen_norm + bc_weight * bc
    metrics = dict(
        loss=total, mse=mse, skill_std=skill_term_std,
        disc_std=jnp.std(disc),
        z_norm=jnp.sqrt(jnp.mean(z_i ** 2)),
        ortho_pen=ortho_pen_norm,
        bc_nll=bc,
    )
    return total, metrics


def _eval_pair_mse(model, agent_data, agent_token_mask, agent_game_mask,
                   idx_i, idx_j, f_pairs, use_skill):
    traits = model.encoder.encode_batch(agent_data, agent_token_mask, agent_game_mask)
    Z = jax.vmap(model.disc_head)(traits)
    K = Z.shape[-1] // 2
    Z = Z.reshape(Z.shape[0], K, 2)
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
class DiscBCResult:
    model: DiscFPTABCModel
    disc_magnitudes: np.ndarray
    train_history: list = field(default_factory=list)
    chosen_step: int = -1
    best_test_mse: float = float("inf")
    best_bc_nll: float = float("nan")


def train_disc_direct_bc(
    ds, train_pairs, test_pairs,
    K: int = 10, use_skill: bool = True,
    ortho_weight: float = 0.0, bc_weight: float = 0.1, alpha_floor: float = 0.5,
    trait_dim: int = 24, d_model: int = 32, n_layers: int = 1, n_heads: int = 2,
    disc_hidden: int = 64, behavior_hidden: int = 64,
    n_steps: int = 20000, batch_size: int = 32, lr: float = 5e-4,
    eval_every: int = 500,
    G_sample: int = 4, G_sample_eval: int | None = None,
    seed: int = 0, verbose: bool = True,
) -> DiscBCResult:
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)
    if G_sample_eval is None:
        G_sample_eval = min(8, ds.G_max)

    N = ds.policies.shape[0]
    sa_dim = ds.sa_dim
    n_zones = ds.n_zones
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

    eval_games, eval_tmask, eval_gmask = _sample_games(
        np.asarray(agent_games), np.asarray(agent_tmask),
        np.asarray(agent_gmask), np.arange(N), G_sample_eval, rng,
    )
    eval_games  = jnp.asarray(eval_games)
    eval_tmask  = jnp.asarray(eval_tmask)
    eval_gmask  = jnp.asarray(eval_gmask)

    key, mkey = jax.random.split(key)
    model = _build_model(sa_dim, n_zones, L_max, trait_dim, d_model, n_layers,
                         n_heads, disc_hidden, behavior_hidden,
                         K, alpha_floor, mkey)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, gi, tmi, gmi, gj, tmj, gmj, f_batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            lambda m: _disc_bc_loss(m, gi, tmi, gmi, gj, tmj, gmj,
                                    f_batch, use_skill, ortho_weight, bc_weight),
            has_aux=True,
        )(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, metrics

    history: list[dict] = []
    best = dict(test_mse=float("inf"), model=None, step=-1, bc_nll=float("nan"))

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
                      f"| ortho={float(metrics['ortho_pen']):.4f} "
                      f"| bc={float(metrics['bc_nll']):.4f} "
                      f"| train={train_mse:.4f} | test={test_mse:.4f}")
            if test_mse < best["test_mse"]:
                best = dict(test_mse=test_mse, model=model, step=step,
                            bc_nll=float(metrics["bc_nll"]))

    if best["model"] is not None:
        model = best["model"]
        if verbose:
            print(f"\n  Early stop: best step={best['step']} "
                  f"test_mse={best['test_mse']:.4f}  "
                  f"bc_nll={best['bc_nll']:.4f}")

    traits_final = np.asarray(model.encoder.encode_batch(
        eval_games, eval_tmask, eval_gmask))
    mags = _disc_magnitudes(traits_final, model)
    if verbose:
        print(f"\n  Top-K disc magnitudes (sorted): {mags.round(3)}")

    return DiscBCResult(
        model=model, disc_magnitudes=mags,
        train_history=history,
        chosen_step=best["step"], best_test_mse=float(best["test_mse"]),
        best_bc_nll=float(best["bc_nll"]),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(bundle_path: Path, out_dir: Path, output_json: Path | None,
         K: int, use_skill: bool, ortho_weight: float, bc_weight: float,
         alpha_floor: float, n_steps: int, seed: int, frac_train: float,
         trait_dim: int, d_model: int, n_layers: int, n_heads: int,
         disc_hidden: int, behavior_hidden: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  disc_direct_bc: K={K}, use_skill={use_skill}, "
          f"ortho_weight={ortho_weight}, bc_weight={bc_weight}")
    print("=" * 72)

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, dropped = drop_dead_agents(ds, verbose=True)
    N = ds.policies.shape[0]
    print(f"  N={N}, sa_dim={ds.sa_dim}, n_zones={ds.n_zones}, "
          f"G_max={ds.G_max}, L_max={ds.L_max}")

    rng = np.random.RandomState(seed)
    iu = np.argwhere(np.triu(ds.observed_mask, 1))
    rng.shuffle(iu)
    n_train = int(frac_train * len(iu))
    train_un, test_un = iu[:n_train], iu[n_train:]
    train_pairs = np.concatenate([train_un, train_un[:, [1, 0]]], axis=0)
    test_pairs  = np.concatenate([test_un,  test_un[:, [1, 0]]],  axis=0)
    print(f"  train pairs (both dirs): {len(train_pairs)}, test: {len(test_pairs)}")

    t0 = time.time()
    res = train_disc_direct_bc(
        ds, train_pairs, test_pairs,
        K=K, use_skill=use_skill,
        ortho_weight=ortho_weight, bc_weight=bc_weight, alpha_floor=alpha_floor,
        trait_dim=trait_dim, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        disc_hidden=disc_hidden, behavior_hidden=behavior_hidden,
        n_steps=n_steps, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"\n  Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    f2_test = float(np.mean(ds.F[test_pairs[:, 0], test_pairs[:, 1]] ** 2))
    norm_test = res.best_test_mse / max(f2_test, 1e-12)
    print(f"\n  HEADLINE: best test MSE = {res.best_test_mse:.4f}  "
          f"norm_test = {norm_test:.4f}  "
          f"bc_nll = {res.best_bc_nll:.4f}  (@ step {res.chosen_step})")

    eqx.tree_serialise_leaves(str(out_dir / "encoder.eqx"), res.model.encoder)
    eqx.tree_serialise_leaves(str(out_dir / "skill_head.eqx"), res.model.skill_head)
    eqx.tree_serialise_leaves(str(out_dir / "disc_head.eqx"), res.model.disc_head)
    eqx.tree_serialise_leaves(str(out_dir / "behavior_head.eqx"),
                              res.model.behavior_head)
    meta = dict(
        sa_dim=int(ds.sa_dim), L_max=int(ds.L_max), n_zones=int(ds.n_zones),
        trait_dim=int(trait_dim),
        d_model=int(d_model), n_layers=int(n_layers), n_heads=int(n_heads),
        mlp_ratio=4, rho_hidden=(64,),
        basis_kind="disc_direct_bc",
        K=int(K),
        use_skill=bool(use_skill),
        ortho_weight=float(ortho_weight),
        bc_weight=float(bc_weight),
        alpha_floor=float(alpha_floor),
        disc_hidden=int(disc_hidden), behavior_hidden=int(behavior_hidden),
        disc_magnitudes=res.disc_magnitudes,
        seed=int(seed), n_steps=int(n_steps),
        chosen_step=int(res.chosen_step),
        best_bc_nll=float(res.best_bc_nll),
    )
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"  Saved checkpoint to {out_dir}/")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            method_tag = "disc_direct_bc"
            if not use_skill:
                method_tag += "_no_skill"
            if ortho_weight > 0:
                method_tag += "_ortho"
            json.dump(dict(
                method=method_tag,
                K=int(K), use_skill=bool(use_skill),
                ortho_weight=float(ortho_weight),
                bc_weight=float(bc_weight),
                alpha_floor=float(alpha_floor),
                n_steps=int(n_steps),
                seed=int(seed),
                chosen_step=int(res.chosen_step),
                test_mse=float(res.best_test_mse),
                norm_test=float(norm_test),
                bc_nll=float(res.best_bc_nll),
                disc_magnitudes=res.disc_magnitudes.tolist(),
                wall_time_sec=float(elapsed),
            ), f, indent=2)
        print(f"  Saved JSON → {output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--use_skill", type=int, default=1)
    p.add_argument("--ortho_weight", type=float, default=0.0)
    p.add_argument("--bc_weight", type=float, default=0.1,
                   help="weight on Dirichlet BC NLL term; 0 disables behavior head")
    p.add_argument("--alpha_floor", type=float, default=0.5,
                   help="lower bound on Dirichlet concentrations (numerical "
                        "safety; also a soft prior toward more diffuse predictions)")
    p.add_argument("--trait_dim", type=int, default=24)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--disc_hidden", type=int, default=64)
    p.add_argument("--behavior_hidden", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac_train", type=float, default=0.8)
    args = p.parse_args()
    main(args.bundle, args.out_dir, args.output_json,
         args.K, bool(args.use_skill), args.ortho_weight,
         args.bc_weight, args.alpha_floor,
         args.n_steps, args.seed, args.frac_train,
         args.trait_dim, args.d_model, args.n_layers, args.n_heads,
         args.disc_hidden, args.behavior_hidden)
