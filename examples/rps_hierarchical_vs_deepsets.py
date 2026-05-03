#!/usr/bin/env python3
"""Compare DeepSets vs Hierarchical (transformer-per-game) encoders on RPS.

Both models share the same payoff matrix, the same train/test pair split,
and similar training hyperparameters (steps, learning rate, batch size).
The two encoders differ in how they represent an agent's behavior:

  - DeepSets: treats each (state, action) pair as an independent element
    and mean-pools over all pairs from all games.
  - Hierarchical: encodes each full game via a small transformer (capturing
    within-game temporal structure), then mean-pools over games.

At the end we plot train/test MSE curves side by side to compare generalization.

Usage:
    python examples/rps_hierarchical_vs_deepsets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import (
    behavioral_fpta,
    hierarchical_behavioral_fpta,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)

HISTORY_WINDOW = 4
INCLUDE_ROUND_PROGRESS = True
INCLUDE_ACTION_COUNTS = True

# DeepSets: max (s,a) samples per bot (subsample for speed)
MAX_SAMPLES_PER_BOT = 300

# Hierarchical: max games per bot and rounds per game
# Each bot plays 246 matches of 100 rounds. Cap games for tractability on CPU.
MAX_GAMES_PER_BOT = 60
ROUNDS_PER_GAME = 100

# Training
# Defaults chosen to be CPU-friendly (~15-30 min on CPU for both models).
# Scale up N_STEPS and BATCH_SIZE for higher-quality results on GPU.
N_STEPS = 1500
BATCH_SIZE = 24
G_SAMPLE = 10            # Hierarchical: games sampled per agent per step
G_SAMPLE_EVAL = 30       # Hierarchical: games for closed-form C + eval


# ---------------------------------------------------------------------------
# Common data loading
# ---------------------------------------------------------------------------


def load_matches(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_unique_bots(matches: list[dict]) -> list[str]:
    bots = set()
    for m in matches:
        bots.add(m["player1"])
        bots.add(m["player2"])
    return sorted(bots)


def extract_state(
    my_actions, opp_actions, round_idx, total_rounds,
) -> np.ndarray:
    """State features (19-dim): action counts (6) + recent opp (12) + progress (1)."""
    features: list[float] = []
    if INCLUDE_ACTION_COUNTS:
        if round_idx > 0:
            mc = np.bincount(my_actions[:round_idx], minlength=3).astype(np.float32) / round_idx
            oc = np.bincount(opp_actions[:round_idx], minlength=3).astype(np.float32) / round_idx
        else:
            mc = np.ones(3, dtype=np.float32) / 3
            oc = np.ones(3, dtype=np.float32) / 3
        features.extend(mc.tolist())
        features.extend(oc.tolist())
    for i in range(HISTORY_WINDOW):
        idx = round_idx - HISTORY_WINDOW + i
        if 0 <= idx < len(opp_actions):
            oh = np.zeros(3, dtype=np.float32)
            oh[opp_actions[idx]] = 1.0
        else:
            oh = np.ones(3, dtype=np.float32) / 3
        features.extend(oh.tolist())
    if INCLUDE_ROUND_PROGRESS:
        features.append(round_idx / total_rounds if total_rounds > 0 else 0.0)
    return np.array(features, dtype=np.float32)


def compute_state_dim() -> int:
    dim = 0
    if INCLUDE_ACTION_COUNTS:
        dim += 6
    dim += HISTORY_WINDOW * 3
    if INCLUDE_ROUND_PROGRESS:
        dim += 1
    return dim


def build_payoff_matrix(matches: list[dict], bot_names: list[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(bot_names)}
    N = len(bot_names)
    wins = np.zeros((N, N), dtype=np.float32)
    counts = np.zeros((N, N), dtype=np.float32)
    for m in matches:
        i = name_to_idx.get(m["player1"]); j = name_to_idx.get(m["player2"])
        if i is None or j is None:
            continue
        a1 = np.array(m["player1_actions"]); a2 = np.array(m["player2_actions"])
        wins[i, j] += np.sum((a1 - a2) % 3 == 1)
        wins[j, i] += np.sum((a2 - a1) % 3 == 1)
        counts[i, j] += len(a1); counts[j, i] += len(a1)
    counts = np.maximum(counts, 1)
    return (wins - wins.T) / counts


# ---------------------------------------------------------------------------
# DeepSets data prep (flat (s,a) pairs per bot)
# ---------------------------------------------------------------------------


def extract_bot_samples(matches, bot_name) -> list[tuple[np.ndarray, int]]:
    samples = []
    for match in matches:
        if match["player1"] == bot_name:
            my_actions = match["player1_actions"]
            opp_actions = match["player2_actions"]
        elif match["player2"] == bot_name:
            my_actions = match["player2_actions"]
            opp_actions = match["player1_actions"]
        else:
            continue
        total = len(my_actions)
        for r in range(total):
            s = extract_state(my_actions, opp_actions, r, total)
            samples.append((s, my_actions[r]))
    return samples


def prepare_deepsets_data(
    matches, bot_names, max_samples=MAX_SAMPLES_PER_BOT,
) -> tuple[np.ndarray, np.ndarray, int]:
    state_dim = compute_state_dim()
    sa_dim = state_dim + 3
    N = len(bot_names)
    all_samples = {}
    for name in bot_names:
        samples = extract_bot_samples(matches, name)
        if len(samples) > max_samples:
            rng = np.random.RandomState(hash(name) % 2**31)
            idx = rng.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in idx]
        all_samples[name] = samples
    K_max = max(len(s) for s in all_samples.values())
    agent_data = np.zeros((N, K_max, sa_dim), dtype=np.float32)
    agent_mask = np.zeros((N, K_max), dtype=bool)
    for i, name in enumerate(bot_names):
        for k, (state, action) in enumerate(all_samples[name]):
            aoh = np.zeros(3, dtype=np.float32); aoh[action] = 1.0
            agent_data[i, k] = np.concatenate([state, aoh])
            agent_mask[i, k] = True
    return agent_data, agent_mask, sa_dim


# ---------------------------------------------------------------------------
# Hierarchical data prep (per-game tensors)
# ---------------------------------------------------------------------------


def prepare_hierarchical_data(
    matches, bot_names,
    max_games=MAX_GAMES_PER_BOT,
    rounds_per_game=ROUNDS_PER_GAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare per-game tensors for each bot.

    Token per round = [state(19), self_action_oh(3), opp_action_oh(3)]  -> 25 dims.

    Returns:
        games: (N, G_max, L, token_dim).
        token_mask: (N, G_max, L).
        game_mask: (N, G_max).
        token_dim: per-token dimension.
    """
    state_dim = compute_state_dim()
    n_actions = 3
    token_dim = state_dim + 2 * n_actions
    N = len(bot_names)
    name_to_idx = {n: i for i, n in enumerate(bot_names)}

    # Collect per-bot game list (each game gets encoded from THAT bot's perspective)
    bot_games: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {n: [] for n in bot_names}
    for m in matches:
        for my_name, opp_name, my_acts, opp_acts in [
            (m["player1"], m["player2"], m["player1_actions"], m["player2_actions"]),
            (m["player2"], m["player1"], m["player2_actions"], m["player1_actions"]),
        ]:
            if my_name not in name_to_idx:
                continue
            bot_games[my_name].append((np.array(my_acts), np.array(opp_acts)))

    # Subsample games per bot (deterministic per bot)
    for name in bot_names:
        g = bot_games[name]
        if len(g) > max_games:
            rng = np.random.RandomState(hash(name) % 2**31)
            idx = rng.choice(len(g), max_games, replace=False)
            bot_games[name] = [g[i] for i in idx]

    G_max = max_games
    L = rounds_per_game

    games = np.zeros((N, G_max, L, token_dim), dtype=np.float32)
    tmask = np.zeros((N, G_max, L), dtype=bool)
    gmask = np.zeros((N, G_max), dtype=bool)

    for i, name in enumerate(bot_names):
        for gi, (my_acts, opp_acts) in enumerate(bot_games[name]):
            if gi >= G_max:
                break
            T = min(len(my_acts), L)
            for t in range(T):
                state = extract_state(my_acts.tolist(), opp_acts.tolist(), t, len(my_acts))
                self_oh = np.zeros(n_actions, dtype=np.float32); self_oh[my_acts[t]] = 1.0
                opp_oh = np.zeros(n_actions, dtype=np.float32); opp_oh[opp_acts[t]] = 1.0
                games[i, gi, t] = np.concatenate([state, self_oh, opp_oh])
                tmask[i, gi, t] = True
            gmask[i, gi] = True

    return games, tmask, gmask, token_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("DeepSets vs Hierarchical Behavioral FPTA on RPS")
    print("=" * 72)

    if not TOURNAMENT_DATA.exists():
        print(f"\nError: Tournament data not found at {TOURNAMENT_DATA}")
        return

    # Load data
    print("\n[1] Loading tournament data...")
    matches = load_matches(TOURNAMENT_DATA)
    bot_names = get_unique_bots(matches)
    print(f"    {len(matches)} matches, {len(bot_names)} bots")

    print("\n[2] Building payoff matrix...")
    F = build_payoff_matrix(matches, bot_names)
    F_jax = jnp.array(F)
    N = len(bot_names)
    print(f"    F shape: {F.shape}, range: [{F.min():.3f}, {F.max():.3f}]")

    # Train/test split
    print("\n[3] Splitting pairwise indices (80/20)...")
    rng = np.random.RandomState(0)
    all_pair_idx = rng.permutation(N * N)
    split = int(0.8 * N * N)
    train_pairs = all_pair_idx[:split]
    test_pairs = all_pair_idx[split:]
    print(f"    Train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")

    # --- Prepare both data formats ---
    print("\n[4] Preparing DeepSets data...")
    ds_data, ds_mask, sa_dim = prepare_deepsets_data(matches, bot_names)
    print(f"    DeepSets shape: {ds_data.shape}, sa_dim={sa_dim}")

    print("\n[5] Preparing Hierarchical data...")
    hr_games, hr_tmask, hr_gmask, token_dim = prepare_hierarchical_data(
        matches, bot_names,
    )
    print(f"    Hierarchical shape: {hr_games.shape}, token_dim={token_dim}")
    print(f"    Avg valid games/bot: {hr_gmask.sum() / N:.1f}")

    # --- Training config ---
    common_config = TrainConfig(
        lr=5e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ortho_weight=0.1,
        ridge_lambda=1e-4,
        c_correction_every=300,
        grad_clip=1.0,
        log_every=200,
    )

    # =====================================================================
    # Train DeepSets
    # =====================================================================
    print("\n" + "=" * 72)
    print("Training DeepSets Behavioral FPTA")
    print("=" * 72)
    ds_result = behavioral_fpta(
        jnp.array(ds_data), jnp.array(ds_mask), F_jax,
        sa_dim=sa_dim,
        trait_dim=32, d=12,
        phi_hidden=(128, 128), rho_hidden=(128,), basis_hidden=(128, 128),
        config=common_config,
        key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200,
        verbose=True,
    )
    ds_history = ds_result.train_history

    # =====================================================================
    # Train Hierarchical
    # =====================================================================
    print("\n" + "=" * 72)
    print("Training Hierarchical Behavioral FPTA")
    print("=" * 72)
    hr_result = hierarchical_behavioral_fpta(
        hr_games, hr_tmask, hr_gmask, F_jax,
        token_dim=token_dim, L_max=ROUNDS_PER_GAME,
        trait_dim=32, d=12,
        d_model=64, n_heads=4, n_layers=2, mlp_ratio=4,
        rho_hidden=(128,), basis_hidden=(128, 128),
        config=common_config,
        key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200,
        G_sample=G_SAMPLE,
        G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1,
        verbose=True,
    )
    hr_history = hr_result.train_history

    # =====================================================================
    # Report
    # =====================================================================
    print("\n" + "=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None

    ds_final_train = final(ds_history, "train_mse")
    ds_final_test = final(ds_history, "test_mse")
    hr_final_train = final(hr_history, "train_mse")
    hr_final_test = final(hr_history, "test_mse")

    print(f"\n{'Model':<15} {'Train MSE':>12} {'Test MSE':>12}")
    print("-" * 42)
    print(f"{'DeepSets':<15} {ds_final_train:>12.6f} {ds_final_test:>12.6f}")
    print(f"{'Hierarchical':<15} {hr_final_train:>12.6f} {hr_final_test:>12.6f}")

    # =====================================================================
    # Plots
    # =====================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, hist, title in [
            (axes[0], ds_history, "DeepSets"),
            (axes[1], hr_history, "Hierarchical (Transformer-per-Game)"),
        ]:
            steps = [r["step"] for r in hist if "train_mse" in r]
            tr = [r["train_mse"] for r in hist if "train_mse" in r]
            te = [r["test_mse"] for r in hist if "test_mse" in r]
            ax.plot(steps, tr, label="Train MSE", linewidth=2)
            if te:
                ax.plot(steps, te, label="Test MSE", linewidth=2, linestyle="--")
            ax.set_xlabel("Training Step")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_yscale("log")
        axes[0].set_ylabel("MSE (log scale)")
        fig.suptitle("DeepSets vs Hierarchical Encoder — Train/Test MSE")
        fig.tight_layout()
        fig.savefig("rps_deepsets_vs_hierarchical_mse.png", dpi=150, bbox_inches="tight")
        print("\nSaved: rps_deepsets_vs_hierarchical_mse.png")

        # Combined comparison plot
        fig, ax = plt.subplots(figsize=(9, 5))
        for hist, name, color in [
            (ds_history, "DeepSets", "C0"),
            (hr_history, "Hierarchical", "C1"),
        ]:
            steps = [r["step"] for r in hist if "train_mse" in r]
            tr = [r["train_mse"] for r in hist if "train_mse" in r]
            te = [r["test_mse"] for r in hist if "test_mse" in r]
            ax.plot(steps, tr, label=f"{name} (train)", linewidth=2, color=color)
            if te:
                ax.plot(steps, te, label=f"{name} (test)", linewidth=2,
                        linestyle="--", color=color)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE (log scale)")
        ax.set_yscale("log")
        ax.set_title("DeepSets vs Hierarchical — Train/Test MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("rps_comparison_combined.png", dpi=150, bbox_inches="tight")
        print("Saved: rps_comparison_combined.png")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
