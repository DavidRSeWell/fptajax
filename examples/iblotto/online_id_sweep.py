"""Sweep online opponent identification across seeds / agents.

Per-agent failure characterisation for ``online_id.py``. For each
seed-selected (or all) true-opponent agents, replays their recorded
tokens through a particle-filter posterior with resampling DISABLED
(so ``particles[true_idx]`` keeps pointing at the true trait) and
records:

  * final rank of the true agent's particle (0 = top-1 = correct ID)
  * MAP particle index (whoever the filter ended up locking onto)
  * mean per-token ``logp[true] - median(logp)`` (positive ⇒ true
    trait is a better-than-median fit; negative ⇒ worse)

Reports a one-line summary plus a histogram of final ranks. Useful for
distinguishing "bad luck on one seed" from "systematic ID failure".

Usage:
    PYTHONPATH=src:. python -m examples.iblotto.online_id_sweep \
        --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
        --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
        --n_seeds 10
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.online_id import (
    OpponentPosterior, load_disc_direct_bc, split_token,
)


def run_one(ds, ckpt, true_idx: int, n_games: int,
            rng: np.random.RandomState) -> dict:
    """Replay one agent's tokens through a no-resample filter."""
    posterior = OpponentPosterior.from_population(
        ckpt, ess_resample_threshold=0.0)

    valid_games = np.where(ds.agent_game_mask[true_idx])[0]
    n_games = min(n_games, len(valid_games))
    chosen_games = rng.choice(valid_games, size=n_games, replace=False)

    n_tokens = 0
    sum_logp_dev = 0.0
    for g in chosen_games:
        tokens = np.asarray(ds.agent_data[true_idx, g])
        mask   = np.asarray(ds.agent_token_mask[true_idx, g])
        for t in range(tokens.shape[0]):
            if not mask[t]:
                continue
            s, a = split_token(tokens[t], ckpt.n_zones)
            logp = posterior._logp_per_particle(s, a)
            sum_logp_dev += float(logp[true_idx] - np.median(logp))
            posterior.update(s, a, rng=rng)
            n_tokens += 1

    w = posterior.weights()
    rank_true = int((w > w[true_idx]).sum())
    map_idx = int(np.argmax(w))
    return dict(
        true_idx=true_idx,
        n_tokens=n_tokens,
        rank_true=rank_true,
        map_idx=map_idx,
        w_true=float(w[true_idx]),
        w_top1=float(w[map_idx]),
        mean_logp_dev=sum_logp_dev / max(n_tokens, 1),
    )


def main(bundle_path: Path, ckpt_dir: Path, n_seeds: int, n_games: int,
         all_agents: bool) -> None:
    print("=" * 72)
    print(f"  online_id sweep  ckpt={ckpt_dir.name}  n_seeds={n_seeds}  "
          f"all_agents={all_agents}")
    print("=" * 72)

    ckpt = load_disc_direct_bc(ckpt_dir, bundle_path, seed=0)
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, _ = drop_dead_agents(ds, verbose=False)
    N = ds.policies.shape[0]

    if all_agents:
        agents = list(range(N))
    else:
        rng_pick = np.random.RandomState(0)
        agents = [int(rng_pick.randint(N)) for _ in range(n_seeds)]

    print(f"  {N} population agents, evaluating on {len(agents)} "
          f"true-opponent picks, {n_games} games each\n")
    header = f"  {'idx':>4} {'tokens':>6} {'rank':>5} {'map':>4} " \
             f"{'w_true':>8} {'w_top1':>8} {'mean(logp-med)':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for k, true_idx in enumerate(agents):
        rng = np.random.RandomState(k + 1)  # game-pick RNG; k=0 reserved
        r = run_one(ds, ckpt, true_idx, n_games, rng)
        rows.append(r)
        print(f"  {r['true_idx']:>4d} {r['n_tokens']:>6d} "
              f"{r['rank_true']:>5d} {r['map_idx']:>4d} "
              f"{r['w_true']:>8.4f} {r['w_top1']:>8.4f} "
              f"{r['mean_logp_dev']:>+14.3f}")

    ranks = np.array([r["rank_true"] for r in rows])
    devs  = np.array([r["mean_logp_dev"] for r in rows])
    print("\n  " + "=" * 70)
    print(f"  Summary over {len(rows)} agents:")
    print(f"    top-1 hit rate         : {(ranks == 0).mean():.2%}")
    print(f"    top-5 hit rate         : {(ranks < 5).mean():.2%}")
    print(f"    top-10 hit rate        : {(ranks < 10).mean():.2%}")
    print(f"    mean rank              : {ranks.mean():.1f}")
    print(f"    median rank            : {np.median(ranks):.1f}")
    print(f"    mean(logp[true] - med) : {devs.mean():+.3f} "
          f"(positive ⇒ true trait fits its own data better than median)")

    # Rank histogram in coarse bins.
    bins = [0, 1, 5, 10, 25, 50, N + 1]
    labels = ["0", "1-4", "5-9", "10-24", "25-49", f"50-{N-1}"]
    counts, _ = np.histogram(ranks, bins=bins)
    print("\n  Rank histogram:")
    for lab, c in zip(labels, counts):
        bar = "#" * int(40 * c / max(counts.max(), 1))
        print(f"    {lab:>8s}: {c:>3d}  {bar}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--ckpt",   type=Path, required=True)
    p.add_argument("--n_seeds", type=int, default=10,
                   help="number of randomly-chosen true opponents to sweep "
                        "(ignored if --all_agents)")
    p.add_argument("--n_games", type=int, default=2,
                   help="games per opponent to replay through the filter")
    p.add_argument("--all_agents", action="store_true",
                   help="evaluate every population agent (overrides n_seeds)")
    args = p.parse_args()
    main(args.bundle, args.ckpt, args.n_seeds, args.n_games, args.all_agents)
