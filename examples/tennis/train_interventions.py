#!/usr/bin/env python3
"""Intervention study: which fix addresses the trait-collapse problem?

Runs three Hierarchical Behavioral FPTA variants on the same tennis dataset
and split, keeping the encoder architecture fixed at the small size
(d_model=32, n_layers=1, trait_dim=24). Baseline numbers are taken from
the prior full run (tennis_run5.log) to save compute.

  (B) FPTA with spread regularisation (VICReg-style per-dim std hinge)
  (C) Contrastive pretraining (InfoNCE) + FPTA fine-tune

Each variant is trained with the same seed, optimizer, N_STEPS, batch size,
and game-sampling strategy as the baseline in train.py.

Usage:
    python examples/tennis/train_interventions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import (
    hierarchical_behavioral_fpta,
    contrastive_pretrain,
    TrainConfig,
)

from loader import build_tennis_dataset  # type: ignore


# ---------------------------------------------------------------------------
# Config (matches train.py)
# ---------------------------------------------------------------------------

MCP_DIR = Path("/tmp/mcp")
OUTPUT_DIR = _HERE

MIN_MATCHES_PER_PLAYER = 30
MAX_PLAYERS = 40
MAX_GAMES_PER_PLAYER = 50
MAX_SHOTS_PER_GAME = 200
SURFACE_FILTER = None

TRAIT_DIM = 24
BASIS_DIM = 12
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 1
MLP_RATIO = 2

N_STEPS = 2000
BATCH_SIZE = 8
G_SAMPLE = 4
G_SAMPLE_EVAL = 20
LR = 5e-4
ORTHO_WEIGHT = 0.1
C_CORRECTION_EVERY = 300

# Intervention knobs
SPREAD_WEIGHT = 0.1             # VICReg-style per-dim std hinge weight
SPREAD_TARGET = 1.0             # target per-dim std

CONTRASTIVE_STEPS = 500
CONTRASTIVE_BATCH = 32          # nearly all agents -> more negatives
CONTRASTIVE_G_SAMPLE = 4
CONTRASTIVE_LR = 3e-4
CONTRASTIVE_TEMP = 0.2

# Baseline numbers (from /tmp/tennis_run5.log, same seed/split/config):
BASELINE = {
    "null_train": 0.015918,
    "null_test":  0.015348,
    "rf_train":   0.002753,
    "rf_test":    0.015117,
    "fpta_train": 0.014711,
    "fpta_test":  0.014651,
    "mlp_train":  0.014594,
    "mlp_test":   0.014405,
    # Reliable subset (>= 3 matches, 46/168 pairs)
    "null_rel":   0.007353,
    "rf_rel":     0.005692,
    "fpta_rel":   0.006050,
    "mlp_rel":    0.005899,
}


def main():
    print("=" * 72)
    print("Intervention study: fixing encoder collapse")
    print("=" * 72)

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

    # Same train/test split as train.py (seed=0, 80/20 of observed pairs)
    observed = np.array(ds.observed_pairs)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pair_tuples = observed[perm[:split]]
    test_pair_tuples = observed[perm[split:]]
    train_pairs = np.array([i * N + j for i, j in train_pair_tuples],
                           dtype=np.int64)
    test_pairs = np.array([i * N + j for i, j in test_pair_tuples],
                          dtype=np.int64)
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Common FPTA config (sans spread_weight).
    def fpta_config(spread_weight=0.0, spread_target=1.0):
        return TrainConfig(
            lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
            ortho_weight=ORTHO_WEIGHT, ridge_lambda=1e-4,
            c_correction_every=C_CORRECTION_EVERY,
            grad_clip=1.0, log_every=100,
            spread_weight=spread_weight, spread_target=spread_target,
        )

    def run_fpta(label, config, pretrained_encoder=None):
        print("\n" + "=" * 72)
        print(f"[{label}]")
        print("=" * 72)
        return hierarchical_behavioral_fpta(
            ds.agent_games, ds.agent_token_mask, ds.agent_game_mask,
            jnp.array(ds.F),
            token_dim=ds.token_dim, L_max=ds.L_max,
            trait_dim=TRAIT_DIM, d=BASIS_DIM,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            mlp_ratio=MLP_RATIO,
            rho_hidden=(D_MODEL,), basis_hidden=(128, 128),
            config=config,
            key=jax.random.PRNGKey(42),
            train_pairs=train_pairs, test_pairs=test_pairs,
            eval_every=200,
            G_sample=G_SAMPLE, G_sample_eval=G_SAMPLE_EVAL,
            numpy_seed=1,
            pretrained_encoder=pretrained_encoder,
            verbose=True,
        )

    # -----------------------------------------------------------------
    # Intervention (B): Spread regularisation
    # -----------------------------------------------------------------
    result_spread = run_fpta(
        f"Intervention B: FPTA with spread regulariser "
        f"(w={SPREAD_WEIGHT}, target={SPREAD_TARGET})",
        fpta_config(spread_weight=SPREAD_WEIGHT, spread_target=SPREAD_TARGET),
    )

    # -----------------------------------------------------------------
    # Intervention (C): Contrastive pretraining + FPTA fine-tune
    # -----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[Intervention C: Contrastive pretraining -> FPTA fine-tune]")
    print("=" * 72)
    pre = contrastive_pretrain(
        ds.agent_games, ds.agent_token_mask, ds.agent_game_mask,
        token_dim=ds.token_dim, L_max=ds.L_max, trait_dim=TRAIT_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        mlp_ratio=MLP_RATIO, rho_hidden=(D_MODEL,),
        n_steps=CONTRASTIVE_STEPS, batch_size=CONTRASTIVE_BATCH,
        G_sample=CONTRASTIVE_G_SAMPLE,
        lr=CONTRASTIVE_LR, temperature=CONTRASTIVE_TEMP,
        log_every=50, key=jax.random.PRNGKey(42), numpy_seed=7,
        verbose=True,
    )
    # FPTA fine-tune from pretrained encoder (no spread reg, just to isolate)
    result_pretrain = run_fpta(
        "FPTA fine-tune (warm-started from contrastive)",
        fpta_config(spread_weight=0.0),
        pretrained_encoder=pre.encoder,
    )

    # -----------------------------------------------------------------
    # Compute test predictions + reliable-subset MSE for each variant
    # -----------------------------------------------------------------
    agents_games_jnp = jnp.array(ds.agent_games)
    agents_tmask_jnp = jnp.array(ds.agent_token_mask)
    agents_gmask_jnp = jnp.array(ds.agent_game_mask)
    test_true = np.array([ds.F[i, j] for i, j in test_pair_tuples],
                         dtype=np.float32)
    test_match_counts = np.array(
        [ds.pair_counts[i, j] for i, j in test_pair_tuples], dtype=np.int32,
    )

    def test_preds(result):
        F_pred = np.asarray(result.predict(
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        ))
        return np.array([F_pred[i, j] for i, j in test_pair_tuples],
                        dtype=np.float32)

    preds_spread = test_preds(result_spread)
    preds_pretrain = test_preds(result_pretrain)

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None

    spread_train_mse = final(result_spread.train_history, 'train_mse')
    spread_test_mse = final(result_spread.train_history, 'test_mse')
    pretrain_train_mse = final(result_pretrain.train_history, 'train_mse')
    pretrain_test_mse = final(result_pretrain.train_history, 'test_mse')

    y_mean_train = float(np.mean([ds.F[i, j] for i, j in train_pair_tuples]))
    mask_rel = test_match_counts >= 3
    n_rel = int(mask_rel.sum())
    null_rel = float(np.mean((test_true[mask_rel] - y_mean_train) ** 2))
    spread_rel = float(np.mean((test_true[mask_rel] - preds_spread[mask_rel]) ** 2))
    pretrain_rel = float(np.mean((test_true[mask_rel] - preds_pretrain[mask_rel]) ** 2))

    # Trait spread diagnostics for both variants
    def trait_spread(result):
        traits = np.asarray(result.encode(
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        ))
        m = traits.mean(axis=0)
        return {
            "std_per_dim_mean": float(traits.std(axis=0).mean()),
            "l2_spread": float(np.linalg.norm(traits - m, axis=1).mean()),
            "pred_std": float(preds_spread.std()) if result is result_spread
                          else float(preds_pretrain.std()),
        }
    sdiag = trait_spread(result_spread)
    pdiag = trait_spread(result_pretrain)

    # -----------------------------------------------------------------
    # Master comparison table
    # -----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MASTER COMPARISON (vs baseline from tennis_run5.log)")
    print("=" * 72)
    print(f"\n{'Model':<48} {'Train MSE':>12} {'Test MSE':>12} {'Δ null':>8}")
    print("-" * 82)
    rows = [
        ("Null (predict train mean)",             BASELINE["null_train"], BASELINE["null_test"]),
        ("Random Forest (handcrafted)",           BASELINE["rf_train"],   BASELINE["rf_test"]),
        ("FPTA — baseline (no intervention)",     BASELINE["fpta_train"], BASELINE["fpta_test"]),
        ("FPTA — antisym MLP head (no bilinear)", BASELINE["mlp_train"],  BASELINE["mlp_test"]),
        ("FPTA + spread reg (w=%.2f)" % SPREAD_WEIGHT,
                                                  spread_train_mse,       spread_test_mse),
        ("FPTA from contrastive pretrain",        pretrain_train_mse,     pretrain_test_mse),
    ]
    null_t = BASELINE["null_test"]
    for name, tr, te in rows:
        gain = (1 - te / null_t) * 100
        print(f"{name:<48} {tr:>12.6f} {te:>12.6f} {gain:>+7.2f}%")

    print(f"\n--- Reliable subset (>= 3 meetings, {n_rel}/{len(test_true)} pairs) ---")
    print(f"{'Model':<48} {'Test MSE':>12} {'Δ null':>8}")
    print("-" * 70)
    rel_rows = [
        ("Null",                                 BASELINE["null_rel"]),
        ("Random Forest",                        BASELINE["rf_rel"]),
        ("FPTA — baseline",                      BASELINE["fpta_rel"]),
        ("FPTA — antisym MLP head",              BASELINE["mlp_rel"]),
        ("FPTA + spread reg (w=%.2f)" % SPREAD_WEIGHT, spread_rel),
        ("FPTA from contrastive pretrain",       pretrain_rel),
    ]
    null_r = BASELINE["null_rel"]
    for name, te in rel_rows:
        gain = (1 - te / null_r) * 100
        print(f"{name:<48} {te:>12.6f} {gain:>+7.2f}%")

    print("\n--- Trait / prediction diagnostics ---")
    print(f"{'Variant':<36} {'trait std/dim':>14} {'trait L2 spread':>16} {'pred std':>10}")
    print("-" * 80)
    # baseline diagnostics from the prior log (hard-coded for comparison)
    print(f"{'FPTA baseline':<36} {'0.00223':>14} {'0.00890':>16} {'0.00700':>10}"
          "   <- from tennis_run5.log")
    print(f"{'FPTA + spread reg':<36} "
          f"{sdiag['std_per_dim_mean']:>14.5f} "
          f"{sdiag['l2_spread']:>16.5f} "
          f"{sdiag['pred_std']:>10.5f}")
    print(f"{'FPTA from contrastive pretrain':<36} "
          f"{pdiag['std_per_dim_mean']:>14.5f} "
          f"{pdiag['l2_spread']:>16.5f} "
          f"{pdiag['pred_std']:>10.5f}")

    # -----------------------------------------------------------------
    # Plots: MSE curves + predicted-vs-actual for the two interventions
    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # MSE curves: 2-panel (spread vs. pretrain), train + test together
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, result, title in zip(axes,
                                     [result_spread, result_pretrain],
                                     ["FPTA + spread reg",
                                      "FPTA from contrastive pretrain"]):
            hist = result.train_history
            steps = [r["step"] for r in hist if "train_mse" in r]
            tr = [r["train_mse"] for r in hist if "train_mse" in r]
            te = [r["test_mse"] for r in hist if "test_mse" in r]
            ax.plot(steps, tr, label="Train MSE", linewidth=2)
            ax.plot(steps, te, label="Test MSE", linewidth=2, linestyle="--")
            ax.axhline(BASELINE["fpta_test"], color="red", linestyle=":",
                       alpha=0.6, label=f"Baseline FPTA test ({BASELINE['fpta_test']:.5f})")
            ax.axhline(BASELINE["null_test"], color="gray", linestyle=":",
                       alpha=0.6, label=f"Null test ({BASELINE['null_test']:.5f})")
            ax.set_yscale("log")
            ax.set_xlabel("Training Step"); ax.set_ylabel("MSE (log scale)")
            ax.set_title(title)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle("Intervention study — Train/Test MSE", fontsize=13)
        out = OUTPUT_DIR / "tennis_interventions_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # Pred vs actual scatter for the two interventions
        lim = float(max(np.abs(test_true).max(),
                        np.abs(preds_spread).max(),
                        np.abs(preds_pretrain).max())) * 1.05
        lim = max(lim, 0.1)
        mc_clip = np.clip(test_match_counts, 1, 10)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for ax_i, (pred, title, mse) in enumerate([
            (preds_spread, f"FPTA + spread reg (w={SPREAD_WEIGHT})", spread_test_mse),
            (preds_pretrain, "FPTA from contrastive pretrain", pretrain_test_mse),
        ]):
            ax = axes[ax_i]
            sc = ax.scatter(test_true, pred, c=mc_clip, cmap="viridis",
                            s=40, alpha=0.75, edgecolors="k", linewidths=0.3)
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.5,
                    label="y = x")
            ax.axhline(y_mean_train, color="red", linewidth=1, alpha=0.4,
                       linestyle=":", label=f"null ({y_mean_train:+.3f})")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("True F[i, j]")
            if ax_i == 0:
                ax.set_ylabel("Predicted F[i, j]")
            ax.set_title(f"{title}  (test MSE = {mse:.5f})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            if ax_i == 1:
                cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
                cb.set_label("match count (clipped at 10)")
        fig.suptitle("Intervention study — Predicted vs. actual F (test pairs)",
                     fontsize=13)
        out = OUTPUT_DIR / "tennis_interventions_pred_vs_actual.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
