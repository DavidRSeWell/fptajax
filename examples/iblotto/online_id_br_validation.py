"""End-to-end validation: does ID-based best response actually win more?

For a sample of held-out opponents, this script:

  1. **Warm-up.** Replays a fixed number of tokens from one of the
     opponent's recorded games into the particle filter, building a
     posterior over their trait.

  2. **Pick a population BR** under four strategies:

       * ``filter``  — posterior mean from warm-up, then disc-space BR.
       * ``uniform`` — no warm-up, uniform prior, then disc-space BR
                       (this is the "opponent-blind" baseline).
       * ``oracle``  — uses the true opponent's encoded trait directly
                       (upper bound on what trait-based BR can buy).
       * ``random``  — random population agent (lower bound).

  3. **Play.** Runs ``--n_eval_games`` simulated games of each chosen
     BR's policy vs the true opponent's policy, measuring realised
     ``F = P1 - P2`` (BR is P1, opponent is P2).

  4. **Summarise.** Mean realised F per strategy, plus per-opponent
     uplift of ``filter`` over ``uniform``.

Note: the BR is a *static* population policy; it doesn't adapt during
the eval game. This validates the trait-identification + BR-selection
loop, not the harder question of state-conditioned best-response
policies.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m \
        examples.iblotto.online_id_br_validation \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --n_opponents 20 --n_warmup 50 --n_eval_games 10
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.game import (
    CSF_AUCTION, GameOptions, INFO_ALL_INVESTMENTS,
    REALLOC_STAY_IN_ZONE, RR_NONE,
)
from examples.iblotto.online_id import (
    OpponentPosterior, load_disc_direct_bc, population_best_response,
    split_token,
)
from examples.iblotto.simulate import simulate_iblotto


def _build_opts(n_zones: int, n_rounds: int) -> GameOptions:
    """Reproduce the GameOptions used by the data generators."""
    return GameOptions(
        n_zones=n_zones, zone_values=jnp.ones(n_zones),
        csf_mode=CSF_AUCTION, resource_return_mode=RR_NONE,
        reallocation_mode=REALLOC_STAY_IN_ZONE, depreciation=0.5,
        info_mode=INFO_ALL_INVESTMENTS, info_noise=0.0, n_rounds=n_rounds,
    )


def _make_simulate_fn(opts: GameOptions, p1_budget: float, p2_budget: float,
                      p1_inv_frac: float, p2_inv_frac: float):
    """JIT+vmap over keys for fast batched payouts of (p1, p2)."""
    @jax.jit
    def vmap_pair(p1, p2, keys):
        return jax.vmap(
            lambda k: simulate_iblotto(
                p1, p2, opts, k,
                p1_budget=p1_budget, p2_budget=p2_budget,
                p1_inv_frac=p1_inv_frac, p2_inv_frac=p2_inv_frac,
            )
        )(keys)
    return vmap_pair


def _run_warmup(posterior: OpponentPosterior, ds, true_idx: int,
                game_idx: int, n_warmup: int,
                rng: np.random.RandomState) -> int:
    """Feed up to ``n_warmup`` valid tokens from a single recorded game.

    Returns the number of tokens actually consumed (may be < n_warmup if
    the game is short).
    """
    tokens = np.asarray(ds.agent_data[true_idx, game_idx])
    mask   = np.asarray(ds.agent_token_mask[true_idx, game_idx])
    consumed = 0
    for t in range(tokens.shape[0]):
        if not mask[t]:
            continue
        if consumed >= n_warmup:
            break
        s, a = split_token(tokens[t], posterior.n_zones)
        posterior.update(s, a, rng=rng)
        consumed += 1
    return consumed


@dataclass
class StrategyResult:
    name: str
    br_idx: int
    realised_F: float          # mean across eval games
    realised_F_std: float      # std across eval games (one-sided sample)


def _evaluate_pair(simulate_vmap, policies: np.ndarray,
                   br_idx: int, true_idx: int, n_eval_games: int,
                   key_seed: int) -> tuple[float, float]:
    keys = jax.random.split(jax.random.PRNGKey(key_seed), n_eval_games)
    pays = np.asarray(simulate_vmap(
        jnp.asarray(policies[br_idx]),
        jnp.asarray(policies[true_idx]),
        keys,
    ))
    diffs = pays[:, 0] - pays[:, 1]
    return float(np.mean(diffs)), float(np.std(diffs))


def main(bundle_path: Path, ckpt_dir: Path, n_opponents: int,
         n_warmup: int, n_eval_games: int, seed: int,
         use_skill_in_br: bool) -> None:
    print("=" * 72)
    print(f"  online_id BR validation  ckpt={ckpt_dir.name}")
    print(f"  n_opponents={n_opponents}, n_warmup={n_warmup}, "
          f"n_eval_games={n_eval_games}")
    print("=" * 72)

    ckpt = load_disc_direct_bc(ckpt_dir, bundle_path, seed=seed)
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, _ = drop_dead_agents(ds, verbose=False)
    N = ds.policies.shape[0]
    n_zones = ds.n_zones
    n_rounds = ds.n_rounds

    print(f"  N={N}, n_zones={n_zones}, n_rounds={n_rounds}")

    opts = _build_opts(n_zones, n_rounds)
    md = ds.metadata
    simulate_vmap = _make_simulate_fn(
        opts,
        p1_budget=float(md.get("p1_budget", 1000.0)),
        p2_budget=float(md.get("p2_budget", 1000.0)),
        p1_inv_frac=float(md.get("p1_inv_frac", 0.1)),
        p2_inv_frac=float(md.get("p2_inv_frac", 0.1)),
    )

    rng = np.random.RandomState(seed)
    opp_indices = rng.choice(N, size=n_opponents, replace=False)
    print(f"  evaluating against opponents: {sorted(opp_indices.tolist())}\n")

    header = (f"  {'opp':>4} {'warm':>5} | "
              f"{'filter':>8} {'uniform':>8} {'oracle':>8} {'random':>8} | "
              f"{'Δ filter-uniform':>17}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: list[dict] = []
    for k_opp, true_idx in enumerate(opp_indices):
        true_idx = int(true_idx)
        # Pick a random game for warmup.
        valid_games = np.where(ds.agent_game_mask[true_idx])[0]
        warmup_game = int(rng.choice(valid_games))

        # 1) Filter strategy
        post_filter = OpponentPosterior.from_population(
            ckpt, ess_resample_threshold=0.0,
        )
        n_used = _run_warmup(post_filter, ds, true_idx, warmup_game,
                             n_warmup, rng)
        z_filter = post_filter.posterior_mean()
        br_filter, _ = population_best_response(
            ckpt, z_filter, use_skill=use_skill_in_br)

        # 2) Uniform-prior baseline (no warmup)
        post_uniform = OpponentPosterior.from_population(
            ckpt, ess_resample_threshold=0.0,
        )
        z_uniform = post_uniform.posterior_mean()
        br_uniform, _ = population_best_response(
            ckpt, z_uniform, use_skill=use_skill_in_br)

        # 3) Oracle: use the true encoded trait directly
        z_oracle = ckpt.population_traits[true_idx]
        br_oracle, _ = population_best_response(
            ckpt, z_oracle, use_skill=use_skill_in_br)

        # 4) Random
        br_random = int(rng.choice(N))

        # Evaluate each by simulating BR-vs-opponent
        F_filter, _ = _evaluate_pair(simulate_vmap, ds.policies,
                                     br_filter, true_idx, n_eval_games,
                                     seed + 10_000 + k_opp)
        F_uniform, _ = _evaluate_pair(simulate_vmap, ds.policies,
                                      br_uniform, true_idx, n_eval_games,
                                      seed + 20_000 + k_opp)
        F_oracle, _ = _evaluate_pair(simulate_vmap, ds.policies,
                                     br_oracle, true_idx, n_eval_games,
                                     seed + 30_000 + k_opp)
        F_random, _ = _evaluate_pair(simulate_vmap, ds.policies,
                                     br_random, true_idx, n_eval_games,
                                     seed + 40_000 + k_opp)

        delta = F_filter - F_uniform
        row = dict(
            true_idx=true_idx, n_used=n_used,
            br_filter=br_filter, F_filter=F_filter,
            br_uniform=br_uniform, F_uniform=F_uniform,
            br_oracle=br_oracle, F_oracle=F_oracle,
            br_random=br_random, F_random=F_random,
            delta=delta,
        )
        rows.append(row)
        print(f"  {true_idx:>4d} {n_used:>5d} | "
              f"{F_filter:>+8.2f} {F_uniform:>+8.2f} "
              f"{F_oracle:>+8.2f} {F_random:>+8.2f} | "
              f"{delta:>+17.2f}")

    F_filter_arr  = np.array([r["F_filter"]  for r in rows])
    F_uniform_arr = np.array([r["F_uniform"] for r in rows])
    F_oracle_arr  = np.array([r["F_oracle"]  for r in rows])
    F_random_arr  = np.array([r["F_random"]  for r in rows])
    deltas = F_filter_arr - F_uniform_arr

    print("\n  " + "=" * 70)
    print(f"  Summary over {len(rows)} opponents (mean ± SE):")
    se = lambda x: np.std(x) / np.sqrt(len(x))
    print(f"    filter  : {F_filter_arr.mean():+8.2f}  ± {se(F_filter_arr):.2f}")
    print(f"    uniform : {F_uniform_arr.mean():+8.2f}  ± {se(F_uniform_arr):.2f}")
    print(f"    oracle  : {F_oracle_arr.mean():+8.2f}  ± {se(F_oracle_arr):.2f}")
    print(f"    random  : {F_random_arr.mean():+8.2f}  ± {se(F_random_arr):.2f}")
    print(f"\n  Δ filter − uniform: mean {deltas.mean():+.2f}  "
          f"± SE {se(deltas):.2f}  "
          f"(positive ⇒ ID is helping)")
    print(f"  Δ oracle − uniform: mean "
          f"{(F_oracle_arr - F_uniform_arr).mean():+.2f}  "
          f"(upper bound on uplift the BC head can buy)")
    print(f"  filter beats uniform on {(deltas > 0).sum()}/{len(rows)} "
          f"opponents")
    print(f"  filter matches oracle BR on "
          f"{int(sum(r['br_filter']==r['br_oracle'] for r in rows))}/{len(rows)} "
          f"opponents")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--n_opponents", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=50,
                   help="number of warmup tokens fed into the filter "
                        "from one recorded game of the true opponent")
    p.add_argument("--n_eval_games", type=int, default=10,
                   help="simulated games per (BR, opponent) pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_skill_in_br", action="store_true",
                   help="ignore the skill term when scoring the population BR "
                        "(disc-only ranking)")
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.n_opponents,
         args.n_warmup, args.n_eval_games, args.seed,
         use_skill_in_br=not args.no_skill_in_br)
