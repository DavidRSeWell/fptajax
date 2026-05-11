"""Self-consistency demo + sanity check for ``online_id.py``.

Loads a ``disc_direct_bc`` checkpoint and a behavioural bundle, picks a
"true" opponent from the population, replays their recorded tokens
through the particle filter, and reports:

  * how quickly the posterior concentrates on the true agent,
  * the population best-response chosen at the posterior mean,
  * whether that BR is plausible against the true opponent (predicted F
    and, if available, the dataset's ground-truth F entry).

This is a closed-loop test: the "opponent" is just a stored trajectory,
so there's no live simulator. It validates that
``encode → BC head → particle filter → disc-space BR`` works end-to-end
before wiring into ``simulate.py`` / ``run_tournament.py``.

Usage:
    PYTHONPATH=src:. python -m examples.iblotto.online_id_demo \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke_seed0 \
        --seed 0
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.online_id import (
    OpponentPosterior, load_disc_direct_bc, population_best_response,
    split_token,
)


def _print_header(title: str) -> None:
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_demo(bundle_path: Path, ckpt_dir: Path, seed: int,
             true_idx: int | None, n_games: int, log_every: int,
             ess_threshold: float) -> None:
    _print_header(f"online_id demo  (ckpt={ckpt_dir.name})")

    ckpt = load_disc_direct_bc(ckpt_dir, bundle_path, seed=seed)
    print(f"  loaded ckpt: K={ckpt.meta['K']}, "
          f"trait_dim={ckpt.meta['trait_dim']}, "
          f"n_zones={ckpt.n_zones}, "
          f"alpha_floor={ckpt.model.alpha_floor}")

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, _ = drop_dead_agents(ds, verbose=False)
    N = ds.policies.shape[0]
    print(f"  loaded bundle: N={N} agents, n_zones={ds.n_zones}, "
          f"L_max={ds.L_max}")

    if ckpt.population_traits.shape[0] != N:
        raise ValueError(
            f"checkpoint encoded {ckpt.population_traits.shape[0]} agents "
            f"but bundle has {N}; bundle / drop-dead mismatch?"
        )

    rng = np.random.RandomState(seed)
    if true_idx is None:
        true_idx = int(rng.randint(N))
    elif not (0 <= true_idx < N):
        raise ValueError(f"true_idx must be in [0, {N})")
    print(f"  true opponent index: {true_idx}")

    posterior = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=ess_threshold)
    print(f"  prior particles: M={posterior.particles.shape[0]} "
          f"(uniform over training population)")
    print(f"  ess_resample_threshold = {ess_threshold} "
          f"({'no resampling' if ess_threshold <= 0 else 'multinomial when ESS < threshold*M'})")

    # ------------------------------------------------------------------
    # Replay games from the true opponent through the particle filter
    # ------------------------------------------------------------------
    valid_games = np.where(ds.agent_game_mask[true_idx])[0]
    if n_games > len(valid_games):
        n_games = len(valid_games)
    chosen_games = rng.choice(valid_games, size=n_games, replace=False)

    print(f"\n  replaying {n_games} game(s) "
          f"({list(map(int, chosen_games))}) for agent {true_idx}\n")
    print(f"  {'step':>5} {'w[true]':>10} {'rank':>5} "
          f"{'top-1 (w)':>16} {'ESS':>6} {'logp[true]-med':>14} "
          f"{'spread':>7}")
    print(f"  {'-'*5} {'-'*10} {'-'*5} {'-'*16} {'-'*6} {'-'*14} {'-'*7}")

    step = 0
    for g in chosen_games:
        tokens = np.asarray(ds.agent_data[true_idx, g])
        mask   = np.asarray(ds.agent_token_mask[true_idx, g])
        for t in range(tokens.shape[0]):
            if not mask[t]:
                continue
            s, a = split_token(tokens[t], ckpt.n_zones)
            # Diagnostic: per-token log-likelihood spread across particles
            # (computed from current particles, BEFORE the update mutates them).
            logp = posterior._logp_per_particle(s, a)
            logp_med = float(np.median(logp))
            logp_spread = float(np.percentile(logp, 95) - np.percentile(logp, 5))
            logp_true_dev = float(logp[true_idx] - logp_med)

            posterior.update(s, a, rng=rng)
            step += 1
            if step % log_every == 0 or step == 1:
                w = posterior.weights()
                rank_true = int((w > w[true_idx]).sum())
                top = int(np.argmax(w))
                ess = posterior.effective_sample_size()
                print(f"  {step:>5d} {w[true_idx]:>+10.4f} "
                      f"{rank_true:>5d} "
                      f"{top:>3d} ({w[top]:>5.3f})    "
                      f"{ess:>6.1f} "
                      f"{logp_true_dev:>+14.3f} "
                      f"{logp_spread:>7.3f}")

    w = posterior.weights()
    rank_true = int((w > w[true_idx]).sum())
    order = np.argsort(w)[::-1]
    top1_w = float(w[order[0]])
    print(f"\n  Final: w[true]={w[true_idx]:.4f}, "
          f"rank(true)={rank_true}, "
          f"ESS={posterior.effective_sample_size():.1f}")
    print(f"  Top-5 particles: "
          + ", ".join(f"{int(i)}({w[i]:.3f})" for i in order[:5]))
    print(f"  Weight ratio true/top-1: {w[true_idx]/max(top1_w, 1e-12):.3f}")

    # ------------------------------------------------------------------
    # Best response in disc-game space
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  Best-response check")
    print("-" * 72)

    z_mean = posterior.posterior_mean()
    z_thompson = posterior.thompson_sample(rng)
    z_map = posterior.map_particle()

    for tag, z_opp in [("posterior_mean", z_mean),
                       ("thompson_sample", z_thompson),
                       ("MAP_particle",    z_map)]:
        br_idx, scores = population_best_response(ckpt, z_opp)
        true_F = (float(ds.F[br_idx, true_idx])
                  if ds.observed_mask[br_idx, true_idx] else float("nan"))
        observed_str = (f"{true_F:+.2f}" if not np.isnan(true_F)
                        else "(unobserved in dataset)")
        print(f"  {tag:>16s}: pick agent {br_idx:3d}, "
              f"predicted score {scores[br_idx]:+.2f}, "
              f"true F[{br_idx},{true_idx}] = {observed_str}")

    # Reference: how a *random* opponent-side response would look.
    rand_idx = int(rng.randint(N))
    rand_F = (float(ds.F[rand_idx, true_idx])
              if ds.observed_mask[rand_idx, true_idx] else float("nan"))
    rand_str = (f"{rand_F:+.2f}" if not np.isnan(rand_F)
                else "(unobserved in dataset)")
    print(f"  {'(random)':>16s}: pick agent {rand_idx:3d}, "
          f"true F[{rand_idx},{true_idx}] = {rand_str}")

    print("\n  Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True,
                   help="path to a disc_direct_bc checkpoint dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--true_idx", type=int, default=None,
                   help="index of opponent to impersonate; random if omitted")
    p.add_argument("--n_games", type=int, default=2,
                   help="number of recorded games to replay through the filter")
    p.add_argument("--log_every", type=int, default=5,
                   help="log posterior summary every N tokens")
    p.add_argument("--ess_threshold", type=float, default=0.5,
                   help="resample when ESS < threshold*M; 0 disables "
                        "(useful for diagnostics — keeps particle identity, "
                        "so particles[true_idx] stays the true trait)")
    args = p.parse_args()
    run_demo(args.bundle, args.ckpt, args.seed,
             args.true_idx, args.n_games, args.log_every,
             args.ess_threshold)
