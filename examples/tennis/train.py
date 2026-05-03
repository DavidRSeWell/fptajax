#!/usr/bin/env python3
"""Train Hierarchical Behavioral FPTA on Jeff Sackmann's Match Charting data.

Each player is represented by the set of matches they played; each match is
encoded shot-by-shot by a transformer and mean-pooled across a player's
matches to produce a trait vector. FPTA then decomposes the learned
coefficient matrix into disc game components, revealing cyclic matchup
structure among professional tennis players.

Usage:
    python examples/tennis/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the examples/tennis folder importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import (
    hierarchical_behavioral_fpta,
    hierarchical_mlp_baseline,
    TrainConfig,
)

from loader import build_tennis_dataset  # type: ignore
from baseline_rf import train_rf_baseline, top_feature_importance  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MCP_DIR = Path("/tmp/mcp")
OUTPUT_DIR = _HERE

# Data
MIN_MATCHES_PER_PLAYER = 30
MAX_PLAYERS = 40
MAX_GAMES_PER_PLAYER = 50     # G_max
MAX_SHOTS_PER_GAME = 200      # L_max — first ~40-60 points of each match
SURFACE_FILTER = None          # or "Hard" / "Clay" / "Grass" to restrict

# Hierarchical model
TRAIT_DIM = 24
BASIS_DIM = 12
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 1
MLP_RATIO = 2

# Training
N_STEPS = 2000
BATCH_SIZE = 8
G_SAMPLE = 4                   # games per agent per training step
G_SAMPLE_EVAL = 20             # games per agent for closed-form C + MSE eval
LR = 5e-4
ORTHO_WEIGHT = 0.1
C_CORRECTION_EVERY = 300       # longer runway for the encoder between resets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Hierarchical Behavioral FPTA on Tennis (MCP)")
    print("=" * 72)

    # ---- Load dataset ----
    print("\n[1] Building tennis dataset...")
    ds = build_tennis_dataset(
        matches_csv=MCP_DIR / "charting-m-matches.csv",
        points_csvs=[
            MCP_DIR / "charting-m-points-2020s.csv",
            MCP_DIR / "charting-m-points-2010s.csv",
        ],
        min_matches_per_player=MIN_MATCHES_PER_PLAYER,
        max_players=MAX_PLAYERS,
        max_games_per_player=MAX_GAMES_PER_PLAYER,
        max_shots_per_game=MAX_SHOTS_PER_GAME,
        surface_filter=SURFACE_FILTER,
        verbose=True,
    )
    N = len(ds.player_names)

    # ---- Train/test split over observed pairs only ----
    print("\n[2] Splitting observed pairs into train/test...")
    observed = np.array(ds.observed_pairs)  # (P, 2)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pair_tuples = observed[perm[:split]]
    test_pair_tuples = observed[perm[split:]]

    def to_flat(pairs):
        return np.array([i * N + j for i, j in pairs], dtype=np.int64)

    train_pairs = to_flat(train_pair_tuples)
    test_pairs = to_flat(test_pair_tuples)
    print(f"  Observed pairs: {len(observed)}")
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # ---- Train ----
    print("\n[3] Training hierarchical FPTA...")
    print("    Pipeline: shots → Transformer → mean-pool over matches → "
          "NeuralBasis → disc games")
    config = TrainConfig(
        lr=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ortho_weight=ORTHO_WEIGHT,
        ridge_lambda=1e-4,
        c_correction_every=C_CORRECTION_EVERY,
        grad_clip=1.0,
        log_every=100,
    )
    result = hierarchical_behavioral_fpta(
        ds.agent_games, ds.agent_token_mask, ds.agent_game_mask,
        jnp.array(ds.F),
        token_dim=ds.token_dim,
        L_max=ds.L_max,
        trait_dim=TRAIT_DIM, d=BASIS_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO,
        rho_hidden=(D_MODEL,), basis_hidden=(128, 128),
        config=config,
        key=jax.random.PRNGKey(42),
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        eval_every=200,
        G_sample=G_SAMPLE,
        G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1,
        verbose=True,
    )

    # ---- Report ----
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(f"\nDisc game components: {result.n_components}")
    print(f"Eigenvalues: {np.asarray(result.eigenvalues)}")
    imp = np.asarray(result.get_importance())
    cum = np.asarray(result.get_cumulative_importance())
    for k in range(min(result.n_components, 6)):
        print(f"  Component {k+1}: ω={float(result.eigenvalues[k]):.4f}, "
              f"importance={imp[k]:.4f}, cumulative={cum[k]:.4f}")

    # Final MSEs
    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None
    fpta_train_mse = final(result.train_history, 'train_mse')
    fpta_test_mse = final(result.train_history, 'test_mse')
    print(f"\nFinal train MSE: {fpta_train_mse:.6f}")
    print(f"Final test  MSE: {fpta_test_mse:.6f}")

    # ---- Ablation: same encoder, antisymmetric MLP head (no disc game) ----
    print("\n" + "=" * 72)
    print("ABLATION: HierarchicalSetEncoder + antisymmetric MLP head")
    print("=" * 72)
    print("    f_hat(x, y) = g(phi(x) || phi(y)) - g(phi(y) || phi(x))")
    mlp_config = TrainConfig(
        lr=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        grad_clip=1.0,
        log_every=100,
    )
    mlp_result = hierarchical_mlp_baseline(
        ds.agent_games, ds.agent_token_mask, ds.agent_game_mask,
        jnp.array(ds.F),
        token_dim=ds.token_dim, L_max=ds.L_max,
        trait_dim=TRAIT_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO,
        rho_hidden=(D_MODEL,), head_hidden=(128, 128),
        config=mlp_config,
        key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200,
        G_sample=G_SAMPLE, G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1, verbose=True,
    )
    mlp_train_mse = final(mlp_result.train_history, 'train_mse')
    mlp_test_mse = final(mlp_result.train_history, 'test_mse')
    print(f"\nMLP ablation final train MSE: {mlp_train_mse:.6f}")
    print(f"MLP ablation final test  MSE: {mlp_test_mse:.6f}")

    # ---- Baseline A: Random Forest on hand-crafted per-player stats ----
    print("\n" + "=" * 72)
    print("BASELINE: Random Forest on hand-crafted features")
    print("=" * 72)
    rf_result = train_rf_baseline(
        ds, train_pair_tuples, test_pair_tuples, verbose=True,
    )
    print("\nTop 10 features by RF importance:")
    for _name, _imp in top_feature_importance(rf_result, k=10):
        print(f"  {_imp:.4f}  {_name}")

    # ---- Comparison table ----
    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"\n{'Model':<42} {'Train MSE':>12} {'Test MSE':>12}")
    print("-" * 68)
    print(f"{'Null (predict train-pair mean)':<42} "
          f"{rf_result.null_train_mse:>12.6f} {rf_result.null_test_mse:>12.6f}")
    print(f"{'Random Forest (handcrafted features)':<42} "
          f"{rf_result.train_mse:>12.6f} {rf_result.test_mse:>12.6f}")
    print(f"{'Hierarchical Behavioral FPTA':<42} "
          f"{fpta_train_mse:>12.6f} {fpta_test_mse:>12.6f}")
    print(f"{'Hierarchical encoder + antisym MLP head':<42} "
          f"{mlp_train_mse:>12.6f} {mlp_test_mse:>12.6f}")

    # Relative improvement over null (test)
    null_t = rf_result.null_test_mse
    rf_gain = (1 - rf_result.test_mse / null_t) * 100
    fpta_gain = (1 - fpta_test_mse / null_t) * 100
    mlp_gain = (1 - mlp_test_mse / null_t) * 100
    print(f"\nTest-MSE improvement over null baseline:")
    print(f"  Random Forest:  {rf_gain:+.2f}%")
    print(f"  FPTA:           {fpta_gain:+.2f}%")
    print(f"  Encoder + MLP:  {mlp_gain:+.2f}%")

    # ---- Predictions for diagnostics (test + reliable subset) ----
    # FPTA and MLP test predictions: call predict() with all agents once.
    agents_games_jnp = jnp.array(ds.agent_games)
    agents_tmask_jnp = jnp.array(ds.agent_token_mask)
    agents_gmask_jnp = jnp.array(ds.agent_game_mask)
    F_pred_fpta = np.asarray(result.predict(
        agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
    ))
    F_pred_mlp = np.asarray(mlp_result.predict(
        agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
    ))
    test_true = np.array([ds.F[i, j] for i, j in test_pair_tuples], dtype=np.float32)
    test_pred_fpta = np.array(
        [F_pred_fpta[i, j] for i, j in test_pair_tuples], dtype=np.float32,
    )
    test_pred_mlp = np.array(
        [F_pred_mlp[i, j] for i, j in test_pair_tuples], dtype=np.float32,
    )
    # Per-pair match counts for plotting + filtering
    test_match_counts = np.array(
        [ds.pair_counts[i, j] for i, j in test_pair_tuples], dtype=np.int32,
    )
    y_mean_train = float(np.mean(rf_result.y_train))

    # ---- Reliable-pairs comparison (>=3 matches) ----
    reliable_thr = 3
    mask = test_match_counts >= reliable_thr
    n_rel = int(mask.sum())
    print(f"\n{'-' * 72}")
    print(f"Restricted to test pairs with >= {reliable_thr} charted meetings "
          f"({n_rel}/{len(mask)} pairs)")
    print(f"{'-' * 72}")
    if n_rel > 0:
        null_rel = float(np.mean((test_true[mask] - y_mean_train) ** 2))
        rf_rel = float(np.mean((test_true[mask] - rf_result.pred_test[mask]) ** 2))
        fpta_rel = float(np.mean((test_true[mask] - test_pred_fpta[mask]) ** 2))
        mlp_rel = float(np.mean((test_true[mask] - test_pred_mlp[mask]) ** 2))
        print(f"\n{'Model':<42} {'Test MSE (reliable)':>20}")
        print("-" * 62)
        print(f"{'Null (predict train mean)':<42} {null_rel:>20.6f}")
        print(f"{'Random Forest':<42} {rf_rel:>20.6f}")
        print(f"{'Hierarchical Behavioral FPTA':<42} {fpta_rel:>20.6f}")
        print(f"{'Hierarchical encoder + antisym MLP head':<42} {mlp_rel:>20.6f}")
        print(f"\nTest-MSE improvement over null (reliable subset):")
        if null_rel > 0:
            print(f"  Random Forest:  {(1 - rf_rel / null_rel) * 100:+.2f}%")
            print(f"  FPTA:           {(1 - fpta_rel / null_rel) * 100:+.2f}%")
            print(f"  Encoder + MLP:  {(1 - mlp_rel / null_rel) * 100:+.2f}%")
    else:
        print("(no test pairs meet the threshold)")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from fptajax.viz import plot_pta_embedding, plot_importance
        from fptajax.pta import PTAResult

        # Disc game embeddings — compute from full data (all games per player)
        print("\nComputing disc game embeddings from full dataset...")
        Y = result.embed(
            jnp.array(ds.agent_games),
            jnp.array(ds.agent_token_mask),
            jnp.array(ds.agent_game_mask),
        )

        # Print trait spread diagnostic
        traits = result.encode(
            jnp.array(ds.agent_games),
            jnp.array(ds.agent_token_mask),
            jnp.array(ds.agent_game_mask),
        )
        traits_np = np.asarray(traits)
        print(f"  Trait vectors: mean={traits_np.mean(axis=0)[:4].round(3)}, "
              f"std per dim (first 4): {traits_np.std(axis=0)[:4].round(3)}")
        print(f"  Inter-player L2 spread: {np.linalg.norm(traits_np - traits_np.mean(0), axis=1).mean():.4f}")

        pta_like = PTAResult(
            embeddings=Y,
            eigenvalues=result.eigenvalues,
            Q=result.schur_vectors,
            U=jnp.zeros_like(result.schur_vectors),
            n_components=result.n_components,
            f_norm_sq=result.f_norm_sq,
        )

        # MSE curve
        hist = result.train_history
        steps = [r["step"] for r in hist if "train_mse" in r]
        tr = [r["train_mse"] for r in hist if "train_mse" in r]
        te = [r["test_mse"] for r in hist if "test_mse" in r]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, tr, label="Train MSE", linewidth=2)
        if te:
            ax.plot(steps, te, label="Test MSE", linewidth=2, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE (log scale)")
        ax.set_title("Tennis Hierarchical FPTA — Train/Test MSE")
        ax.legend(); ax.grid(True, alpha=0.3)
        out = OUTPUT_DIR / "tennis_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # Disc games grid — centered so inter-player spread is visible
        short_names = [n.split()[-1] for n in ds.player_names]
        n_games = min(result.n_components, 4)
        cols = min(n_games, 2)
        rows = (n_games + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
        if n_games == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for k in range(n_games):
            # center=True subtracts the mean so spread is visible
            plot_pta_embedding(pta_like, k=k, labels=short_names, ax=axes[k],
                               center=True)
            omega_k = float(result.eigenvalues[k])
            imp_k = float(imp[k]) if k < len(imp) else 0.0
            axes[k].set_title(
                f"Disc Game {k+1}  ($\\omega$={omega_k:.3f}, imp={imp_k:.1%})"
            )
        for k in range(n_games, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("Tennis Disc Game Embeddings — centered (Behavioral FPTA)",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        out = OUTPUT_DIR / "tennis_disc_games.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # Importance (now properly normalised to sum to 1)
        fig, ax = plot_importance(pta_like)
        ax.set_title("Disc Game Importance (Tennis Behavioral FPTA)")
        out = OUTPUT_DIR / "tennis_importance.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # Predicted vs. actual scatter (test set, colored by match count)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        lim = float(max(np.abs(test_true).max(),
                        np.abs(rf_result.pred_test).max(),
                        np.abs(test_pred_fpta).max(),
                        np.abs(test_pred_mlp).max())) * 1.05
        lim = max(lim, 0.1)

        # Color scale by match count (clip at 10 for visibility)
        mc_clip = np.clip(test_match_counts, 1, 10)

        panels = [
            (rf_result.pred_test, "Random Forest",
             float(np.mean((test_true - rf_result.pred_test) ** 2))),
            (test_pred_fpta, "Hierarchical FPTA",
             float(np.mean((test_true - test_pred_fpta) ** 2))),
            (test_pred_mlp, "Encoder + antisym MLP head",
             float(np.mean((test_true - test_pred_mlp) ** 2))),
        ]
        for ax_i, (pred, title, mse) in enumerate(panels):
            ax = axes[ax_i]
            sc = ax.scatter(test_true, pred, c=mc_clip, cmap="viridis",
                            s=40, alpha=0.75, edgecolors="k", linewidths=0.3)
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.5,
                    label="y = x")
            ax.axhline(y_mean_train, color="red", linewidth=1, alpha=0.4,
                       linestyle=":", label=f"null (train mean={y_mean_train:+.3f})")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("True F[i, j]")
            if ax_i == 0:
                ax.set_ylabel("Predicted F[i, j]")
            ax.set_title(f"{title}  (test MSE = {mse:.5f})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            if ax_i == len(panels) - 1:
                cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
                cb.set_label("match count (clipped at 10)")

        fig.suptitle("Predicted vs. actual F on test pairs  "
                     f"(n={len(test_true)}, reliable ≥3 matches: {n_rel})",
                     fontsize=13)
        out = OUTPUT_DIR / "tennis_pred_vs_actual.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
