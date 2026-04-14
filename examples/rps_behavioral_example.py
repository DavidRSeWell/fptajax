#!/usr/bin/env python3
"""Behavioral FPTA on OpenSpiel RoShamBo tournament data.

Loads tournament match data, extracts rich state-action features for each bot,
builds a payoff matrix, and runs behavioral FPTA to discover disc game structure
from behavioral data alone — no pre-specified traits needed.

Usage:
    python examples/rps_behavioral_example.py

Requires the rps_pbt tournament data. Set TOURNAMENT_DATA below to point at
your openspiel_tournament_actions.jsonl file.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import behavioral_fpta, TrainConfig, evaluate_online, OnlinePlayer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the tournament JSONL file produced by rps_pbt
TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)

# State feature config (mirrors rps_function_encoder StateConfig defaults)
HISTORY_WINDOW = 4
INCLUDE_ROUND_PROGRESS = True
INCLUDE_ACTION_COUNTS = True

# How many (state, action) samples to keep per bot (subsample for speed)
# On CPU: use 200-500. On GPU: use 2000+.
MAX_SAMPLES_PER_BOT = 300


# ---------------------------------------------------------------------------
# Data loading: JSONL → (state, action) features
# ---------------------------------------------------------------------------


def load_matches(path: Path) -> list[dict]:
    """Load match data from JSONL."""
    matches = []
    with open(path) as f:
        for line in f:
            if line.strip():
                matches.append(json.loads(line))
    return matches


def get_unique_bots(matches: list[dict]) -> list[str]:
    """Get sorted list of unique bot names."""
    bots = set()
    for m in matches:
        bots.add(m["player1"])
        bots.add(m["player2"])
    return sorted(bots)


def extract_state(
    my_actions: list[int],
    opp_actions: list[int],
    round_idx: int,
    total_rounds: int,
) -> np.ndarray:
    """Extract state features at a given round.

    Reproduces the feature vector from rps_function_encoder:
      [my_action_counts(3), opp_action_counts(3),
       recent_opp_actions_onehot(history_window * 3),
       round_progress(1)]
    """
    features: list[float] = []

    # Action count histograms (normalized)
    if INCLUDE_ACTION_COUNTS:
        if round_idx > 0:
            my_counts = np.bincount(my_actions[:round_idx], minlength=3).astype(np.float32)
            my_counts /= round_idx
            opp_counts = np.bincount(opp_actions[:round_idx], minlength=3).astype(np.float32)
            opp_counts /= round_idx
        else:
            my_counts = np.ones(3, dtype=np.float32) / 3
            opp_counts = np.ones(3, dtype=np.float32) / 3
        features.extend(my_counts.tolist())
        features.extend(opp_counts.tolist())

    # Recent opponent actions (one-hot)
    for i in range(HISTORY_WINDOW):
        idx = round_idx - HISTORY_WINDOW + i
        if 0 <= idx < len(opp_actions):
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[opp_actions[idx]] = 1.0
        else:
            one_hot = np.ones(3, dtype=np.float32) / 3
        features.extend(one_hot.tolist())

    # Round progress
    if INCLUDE_ROUND_PROGRESS:
        features.append(round_idx / total_rounds if total_rounds > 0 else 0.0)

    return np.array(features, dtype=np.float32)


def compute_state_dim() -> int:
    """Compute state feature dimensionality."""
    dim = 0
    if INCLUDE_ACTION_COUNTS:
        dim += 6
    dim += HISTORY_WINDOW * 3
    if INCLUDE_ROUND_PROGRESS:
        dim += 1
    return dim


def extract_bot_samples(
    matches: list[dict],
    bot_name: str,
) -> list[tuple[np.ndarray, int]]:
    """Extract (state, action) samples for a bot across all its matches.

    Returns list of (state_vector, action_int) tuples.
    """
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

        total_rounds = len(my_actions)
        for r in range(total_rounds):
            state = extract_state(my_actions, opp_actions, r, total_rounds)
            samples.append((state, my_actions[r]))

    return samples


def build_payoff_matrix(matches: list[dict], bot_names: list[str]) -> np.ndarray:
    """Build payoff matrix from match results.

    F[i, j] = (wins_i - wins_j) / total_rounds across all matches of i vs j.
    Normalized to [-1, 1].
    """
    name_to_idx = {name: i for i, name in enumerate(bot_names)}
    N = len(bot_names)

    wins = np.zeros((N, N), dtype=np.float32)
    counts = np.zeros((N, N), dtype=np.float32)

    for match in matches:
        i = name_to_idx.get(match["player1"])
        j = name_to_idx.get(match["player2"])
        if i is None or j is None:
            continue

        a1 = np.array(match["player1_actions"])
        a2 = np.array(match["player2_actions"])
        n_rounds = len(a1)

        # RPS win logic: (a1 - a2) % 3 == 1 means a1 beats a2
        p1_wins = np.sum((a1 - a2) % 3 == 1)
        p2_wins = np.sum((a2 - a1) % 3 == 1)

        wins[i, j] += p1_wins
        wins[j, i] += p2_wins
        counts[i, j] += n_rounds
        counts[j, i] += n_rounds

    # Normalize: net win rate
    counts = np.maximum(counts, 1)
    F = (wins - wins.T) / counts
    return F


def prepare_behavioral_data(
    matches: list[dict],
    bot_names: list[str],
    max_samples: int = MAX_SAMPLES_PER_BOT,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare padded behavioral data for all bots.

    Returns:
        agent_data: (N, K_max, sa_dim) — padded (state, action_onehot) arrays.
        agent_mask: (N, K_max) — boolean mask for valid entries.
        sa_dim: dimensionality of each (state, action) vector.
    """
    state_dim = compute_state_dim()
    sa_dim = state_dim + 3  # state + action one-hot
    N = len(bot_names)

    # Extract samples for each bot
    all_samples = {}
    for name in bot_names:
        samples = extract_bot_samples(matches, name)
        # Subsample if too many
        if len(samples) > max_samples:
            rng = np.random.RandomState(hash(name) % 2**31)
            indices = rng.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in indices]
        all_samples[name] = samples

    # Find K_max
    K_max = max(len(s) for s in all_samples.values())

    # Build padded arrays
    agent_data = np.zeros((N, K_max, sa_dim), dtype=np.float32)
    agent_mask = np.zeros((N, K_max), dtype=bool)

    for i, name in enumerate(bot_names):
        samples = all_samples[name]
        for k, (state, action) in enumerate(samples):
            action_onehot = np.zeros(3, dtype=np.float32)
            action_onehot[action] = 1.0
            agent_data[i, k] = np.concatenate([state, action_onehot])
            agent_mask[i, k] = True

    return agent_data, agent_mask, sa_dim


def collect_opponent_sequences(
    matches: list[dict],
    bot_names: list[str],
) -> dict[str, list[list[int]]]:
    """Collect recorded action sequences for each bot.

    For each match a bot played, extract their action sequence.
    These are used for online evaluation (replay opponent actions).

    Returns:
        {bot_name: [action_sequence_1, action_sequence_2, ...]}.
    """
    sequences: dict[str, list[list[int]]] = {name: [] for name in bot_names}
    for match in matches:
        p1, p2 = match["player1"], match["player2"]
        if p1 in sequences:
            sequences[p1].append(match["player1_actions"])
        if p2 in sequences:
            sequences[p2].append(match["player2_actions"])
    return sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Behavioral FPTA on OpenSpiel RoShamBo Tournament")
    print("=" * 70)

    # Check data exists
    if not TOURNAMENT_DATA.exists():
        print(f"\nError: Tournament data not found at:\n  {TOURNAMENT_DATA}")
        print("\nRun the rps_pbt tournament first, or update TOURNAMENT_DATA path.")
        return

    # Load matches
    print("\n1. Loading tournament data...")
    matches = load_matches(TOURNAMENT_DATA)
    bot_names = get_unique_bots(matches)
    print(f"   {len(matches)} matches, {len(bot_names)} bots")

    # Build payoff matrix
    print("\n2. Building payoff matrix...")
    F = build_payoff_matrix(matches, bot_names)
    F_jax = jnp.array(F)
    print(f"   Payoff matrix shape: {F.shape}")
    print(f"   Payoff range: [{F.min():.3f}, {F.max():.3f}]")

    # Prepare behavioral data
    print(f"\n3. Extracting behavioral features (max {MAX_SAMPLES_PER_BOT} samples/bot)...")
    agent_data, agent_mask, sa_dim = prepare_behavioral_data(matches, bot_names)
    print(f"   agent_data shape: {agent_data.shape}")
    print(f"   sa_dim: {sa_dim} (state={compute_state_dim()} + action_onehot=3)")
    samples_per_bot = agent_mask.sum(axis=1)
    print(f"   Samples per bot: min={samples_per_bot.min()}, "
          f"max={samples_per_bot.max()}, mean={samples_per_bot.mean():.0f}")

    # Train/test split over pairwise indices
    print("\n4. Splitting pairwise indices into train/test...")
    N = len(bot_names)
    N_pairs = N * N
    rng = np.random.RandomState(0)
    all_pair_idx = rng.permutation(N_pairs)
    split = int(0.8 * N_pairs)
    train_pairs = all_pair_idx[:split]
    test_pairs = all_pair_idx[split:]
    print(f"   Total pairs: {N_pairs}, train: {len(train_pairs)}, test: {len(test_pairs)}")

    # Run behavioral FPTA
    print("\n5. Training behavioral FPTA...")
    print("   Pipeline: D_i → SetEncoder → traits → NeuralBasis → b(x) → disc games")
    print()

    config = TrainConfig(
        lr=5e-4,
        n_steps=10000,
        batch_size=64,
        ortho_weight=0.1,
        ridge_lambda=1e-4,
        c_correction_every=300,
        grad_clip=1.0,
        log_every=500,
    )

    result = behavioral_fpta(
        jnp.array(agent_data),
        jnp.array(agent_mask),
        F_jax,
        sa_dim=sa_dim,
        trait_dim=32,
        d=12,
        phi_hidden=(128, 128),
        rho_hidden=(128,),
        basis_hidden=(128, 128),
        config=config,
        key=jax.random.PRNGKey(42),
        n_components=None,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        eval_every=200,
        verbose=True,
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nDisc game components: {result.n_components}")
    print(f"Eigenvalues: {result.eigenvalues}")
    imp = result.get_importance()
    cum = result.get_cumulative_importance()
    for k in range(min(result.n_components, 5)):
        print(f"  Component {k+1}: ω={result.eigenvalues[k]:.4f}, "
              f"importance={float(imp[k]):.4f}, cumulative={float(cum[k]):.4f}")

    # Inferred traits
    traits = result.encode(jnp.array(agent_data), jnp.array(agent_mask))
    print(f"\nInferred traits shape: {traits.shape}")
    print(f"Trait range: [{float(traits.min()):.3f}, {float(traits.max()):.3f}]")

    # Reconstruction quality (full matrix)
    F_pred = result.predict(
        jnp.array(agent_data), jnp.array(agent_data),
        jnp.array(agent_mask), jnp.array(agent_mask),
    )
    rmse = float(jnp.sqrt(jnp.mean((F_jax - F_pred) ** 2)))
    mae = float(jnp.mean(jnp.abs(F_jax - F_pred)))
    print(f"\nReconstruction (full): RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Disc game embeddings
    Y = result.embed(jnp.array(agent_data), jnp.array(agent_mask))
    print(f"Embeddings shape: {Y.shape}")

    # Visualize
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from fptajax.viz import plot_pta_embedding, plot_importance
        from fptajax.pta import PTAResult

        # Wrap for viz compatibility
        pta_result = PTAResult(
            embeddings=Y,
            eigenvalues=result.eigenvalues,
            Q=result.schur_vectors,
            U=jnp.zeros_like(result.schur_vectors),
            n_components=result.n_components,
            f_norm_sq=result.f_norm_sq,
        )

        # --- MSE plot (train + test) ---
        history = result.train_history
        eval_steps = [r["step"] for r in history if "train_mse" in r]
        train_mses = [r["train_mse"] for r in history if "train_mse" in r]
        test_mses = [r["test_mse"] for r in history if "test_mse" in r]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eval_steps, train_mses, label="Train MSE", linewidth=2)
        if test_mses:
            ax.plot(eval_steps, test_mses, label="Test MSE", linewidth=2, linestyle="--")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE")
        ax.set_title("Behavioral FPTA — Train / Test MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig("rps_behavioral_mse.png", dpi=150, bbox_inches="tight")
        print("\nSaved: rps_behavioral_mse.png")

        # --- Disc game grid ---
        n_games = min(result.n_components, 4)
        cols = min(n_games, 2)
        rows = (n_games + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n_games == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for k in range(n_games):
            plot_pta_embedding(pta_result, k=k, labels=bot_names, ax=axes[k])
            omega_k = float(result.eigenvalues[k])
            imp_k = float(imp[k]) if k < len(imp) else 0.0
            axes[k].set_title(
                f"Disc Game {k+1}  "
                f"($\\omega$={omega_k:.3f}, imp={imp_k:.1%})"
            )
        for k in range(n_games, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("RoShamBo Bots — Disc Game Embeddings (Behavioral FPTA)",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig("rps_behavioral_disc_games.png", dpi=150, bbox_inches="tight")
        print("Saved: rps_behavioral_disc_games.png")

        # --- Importance ---
        fig, ax = plot_importance(pta_result)
        ax.set_title("Disc Game Importance (Behavioral FPTA)")
        fig.savefig("rps_behavioral_importance.png", dpi=150, bbox_inches="tight")
        print("Saved: rps_behavioral_importance.png")

        # --- Trait space PCA ---
        traits_np = np.array(traits)
        U, S, Vt = np.linalg.svd(traits_np - traits_np.mean(axis=0), full_matrices=False)
        traits_2d = U[:, :2] * S[:2]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(traits_2d[:, 0], traits_2d[:, 1], s=40)
        for i, name in enumerate(bot_names):
            ax.annotate(name, (traits_2d[i, 0], traits_2d[i, 1]),
                        fontsize=6, textcoords="offset points", xytext=(3, 3))
        ax.set_title("Inferred Trait Space (PCA of learned traits)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        fig.savefig("rps_behavioral_traits.png", dpi=150, bbox_inches="tight")
        print("Saved: rps_behavioral_traits.png")

        plt.close("all")

    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    # --- Online evaluation ---
    print("\n" + "=" * 70)
    print("ONLINE EVALUATION")
    print("=" * 70)

    print("\nCollecting opponent action sequences...")
    opp_sequences = collect_opponent_sequences(matches, bot_names)
    seq_counts = {n: len(s) for n, s in opp_sequences.items()}
    print(f"  Sequences per bot: min={min(seq_counts.values())}, "
          f"max={max(seq_counts.values())}")

    # Sample a subset of train bots + all eval bots for online play
    n_eval_opponents = min(10, len(bot_names))
    rng_eval = np.random.RandomState(123)
    eval_opponent_names = list(rng_eval.choice(
        bot_names, size=n_eval_opponents, replace=False,
    ))
    eval_sequences = {n: opp_sequences[n] for n in eval_opponent_names}

    n_games_per_opponent = 5
    print(f"\nPlaying {n_games_per_opponent} games against "
          f"{n_eval_opponents} opponents...")
    print(f"  Opponents: {eval_opponent_names}\n")

    game_results = evaluate_online(
        result,
        eval_sequences,
        state_fn=extract_state,
        n_actions=3,
        sa_dim=sa_dim,
        n_games=n_games_per_opponent,
        verbose=True,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
