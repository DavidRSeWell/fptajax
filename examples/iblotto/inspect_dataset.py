"""Load the saved behavioural dataset and dump it to inspectable formats.

Outputs alongside the pickle:

  policies.csv          — (N, 6) table of trait values
  F.csv                 — (N, N) performance matrix
  F_std.csv             — (N, N) per-entry standard errors
  feature_names.txt     — one feature name per line
  agent_data.npz        — full token tensor + masks (numpy native)
  example_game_a0_g0.csv — round-by-round tokens for agent 0, game 0, with
                          column headers from feature_names

Pass a different ``--bundle`` to inspect any other saved bundle.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main(bundle_path: Path):
    print(f"Loading {bundle_path} ...")
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)

    out_dir = bundle_path.parent / bundle_path.stem
    out_dir.mkdir(exist_ok=True)
    print(f"Writing inspectable views to {out_dir}/")

    # 1. Policies
    pol_path = out_dir / "policies.csv"
    np.savetxt(pol_path, ds.policies, delimiter=",", fmt="%.6f",
               header="learning_rate,win_reinvestment,loss_disinvestment,"
                      "opponent_allocation,innovation_noise,concentration",
               comments="")
    print(f"  policies              → {pol_path}  (shape {ds.policies.shape})")

    # 2. F matrix and stderr
    f_path = out_dir / "F.csv"
    np.savetxt(f_path, ds.F, delimiter=",", fmt="%.4f")
    print(f"  F                     → {f_path}  (shape {ds.F.shape})")

    fstd_path = out_dir / "F_std.csv"
    np.savetxt(fstd_path, ds.F_std, delimiter=",", fmt="%.4f")
    print(f"  F_std                 → {fstd_path}")

    # 3. Feature names
    feat_path = out_dir / "feature_names.txt"
    feat_path.write_text("\n".join(ds.feature_names) + "\n")
    print(f"  feature_names         → {feat_path}  ({len(ds.feature_names)} features)")

    # 4. Full token tensor as npz (preserves shape + dtype)
    npz_path = out_dir / "agent_data.npz"
    np.savez(
        npz_path,
        agent_data=ds.agent_data,
        agent_token_mask=ds.agent_token_mask,
        agent_game_mask=ds.agent_game_mask,
        policies=ds.policies,
        F=ds.F,
        F_std=ds.F_std,
        feature_names=np.array(ds.feature_names),
    )
    print(f"  agent_data.npz        → {npz_path}  (full tensor, shape {ds.agent_data.shape})")

    # 5. Example games as CSVs (one per agent, the agent's first valid game)
    for i in range(ds.policies.shape[0]):
        valid_games = np.where(ds.agent_game_mask[i])[0]
        if len(valid_games) == 0:
            continue
        g = int(valid_games[0])
        tokens = ds.agent_data[i, g]
        mask = ds.agent_token_mask[i, g]
        valid_tokens = tokens[mask]
        if valid_tokens.shape[0] == 0:
            continue
        ex_path = out_dir / f"example_game_a{i}_g{g}.csv"
        header = ",".join(ds.feature_names)
        np.savetxt(ex_path, valid_tokens, delimiter=",", fmt="%+.4f",
                   header=header, comments="")
        if i == 0:
            print(f"  example game (a0, g0) → {ex_path}  (rows = rounds, cols = features)")

    print()
    print("Browse with any of:")
    print(f"  $ open {out_dir}                   # Finder")
    print(f"  $ cat {out_dir}/policies.csv")
    print(f"  $ cat {out_dir}/F.csv")
    print(f"  $ python -c \"import numpy as np; "
          f"d = np.load('{npz_path}', allow_pickle=True); "
          f"print(d.files); print(d['agent_data'].shape)\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", type=Path,
        default=Path(__file__).resolve().parent / "results" / "behavioral_small_N8.pkl",
    )
    args = parser.parse_args()
    main(args.bundle)
