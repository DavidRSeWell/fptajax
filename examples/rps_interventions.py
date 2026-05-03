#!/usr/bin/env python3
"""Same intervention study as examples/tennis/train_interventions.py, but on
the iterative-RPS tournament data.

Sanity-check: if the trait-collapse problem we saw on tennis persists at the
same encoder size on RPS (where F is noise-free and every pair has ~6
deterministic-policy meetings of 100 rounds), then the collapse is a
methodology issue. If RPS works cleanly at the same encoder size, the
problem is tennis-data specific (label noise + few meetings per pair).

Keeps the encoder architecture and training hyperparameters identical to
the tennis interventions (d_model=32, n_layers=1, trait_dim=24, N_STEPS=2000,
batch_size=8, G_sample=4, spread_weight=0.1, contrastive 500 steps, etc.).

Models compared:
  - Null (predict train-pair mean)
  - Random Forest on hand-crafted features (mean of self/opp tokens)
  - FPTA (baseline)
  - FPTA + antisymmetric MLP head
  - FPTA + spread regulariser
  - FPTA warm-started from contrastive pretraining

Usage:
    python examples/rps_interventions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import (
    hierarchical_behavioral_fpta,
    hierarchical_mlp_baseline,
    contrastive_pretrain,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Config (matched to tennis interventions for apples-to-apples comparison)
# ---------------------------------------------------------------------------

TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)
OUTPUT_DIR = _HERE

HISTORY_WINDOW = 4

# Hierarchical data caps
MAX_GAMES_PER_BOT = 60
ROUNDS_PER_GAME = 100

# Same encoder arch as tennis interventions
TRAIT_DIM = 24
BASIS_DIM = 12
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 1
MLP_RATIO = 2

# Same training config as tennis interventions
N_STEPS = 2000
BATCH_SIZE = 8
G_SAMPLE = 4
G_SAMPLE_EVAL = 20
LR = 5e-4
ORTHO_WEIGHT = 0.1
C_CORRECTION_EVERY = 300

SPREAD_WEIGHT = 0.1
SPREAD_TARGET = 1.0

CONTRASTIVE_STEPS = 500
CONTRASTIVE_BATCH = 32
CONTRASTIVE_G_SAMPLE = 4
CONTRASTIVE_LR = 3e-4
CONTRASTIVE_TEMP = 0.2


# ---------------------------------------------------------------------------
# Data loading (adapted from rps_hierarchical_vs_deepsets.py)
# ---------------------------------------------------------------------------


def _extract_state(my_actions, opp_actions, round_idx, total_rounds) -> np.ndarray:
    """State features (19-dim): action counts (6) + recent opp (12) + progress (1)."""
    features: list[float] = []
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
            oh = np.zeros(3, dtype=np.float32); oh[opp_actions[idx]] = 1.0
        else:
            oh = np.ones(3, dtype=np.float32) / 3
        features.extend(oh.tolist())
    features.append(round_idx / total_rounds if total_rounds > 0 else 0.0)
    return np.array(features, dtype=np.float32)


STATE_DIM = 6 + HISTORY_WINDOW * 3 + 1  # = 19
TOKEN_DIM = STATE_DIM + 2 * 3            # state + self_oh + opp_oh = 25


def load_rps_dataset():
    """Returns a TennisDataset-like bundle for RPS."""
    assert TOURNAMENT_DATA.exists(), f"Missing {TOURNAMENT_DATA}"
    matches = [json.loads(l) for l in open(TOURNAMENT_DATA)]
    bots = sorted({b for m in matches for b in (m["player1"], m["player2"])})
    N = len(bots)
    name_to_idx = {n: i for i, n in enumerate(bots)}
    print(f"  {len(matches):,} matches, {N} bots")

    # Build F
    wins = np.zeros((N, N), dtype=np.float64)
    counts = np.zeros((N, N), dtype=np.float64)
    pair_counts = np.zeros((N, N), dtype=np.int64)
    for m in matches:
        i = name_to_idx[m["player1"]]; j = name_to_idx[m["player2"]]
        a1 = np.array(m["player1_actions"]); a2 = np.array(m["player2_actions"])
        wins[i, j] += np.sum((a1 - a2) % 3 == 1)
        wins[j, i] += np.sum((a2 - a1) % 3 == 1)
        counts[i, j] += len(a1); counts[j, i] += len(a1)
        pair_counts[i, j] += 1; pair_counts[j, i] += 1
    denom = np.maximum(counts, 1)
    F = ((wins - wins.T) / denom).astype(np.float32)
    F = 0.5 * (F - F.T)  # enforce skew-sym exactly

    # Collect per-bot games
    bot_games: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {n: [] for n in bots}
    for m in matches:
        for my_name, opp_name, my_acts, opp_acts in [
            (m["player1"], m["player2"], m["player1_actions"], m["player2_actions"]),
            (m["player2"], m["player1"], m["player2_actions"], m["player1_actions"]),
        ]:
            bot_games[my_name].append((np.array(my_acts), np.array(opp_acts)))

    # Subsample games per bot
    for name in bots:
        g = bot_games[name]
        if len(g) > MAX_GAMES_PER_BOT:
            rng = np.random.RandomState(hash(name) % 2**31)
            idx = rng.choice(len(g), MAX_GAMES_PER_BOT, replace=False)
            bot_games[name] = [g[k] for k in idx]

    G_max = MAX_GAMES_PER_BOT
    L = ROUNDS_PER_GAME
    games = np.zeros((N, G_max, L, TOKEN_DIM), dtype=np.float32)
    tmask = np.zeros((N, G_max, L), dtype=bool)
    gmask = np.zeros((N, G_max), dtype=bool)

    for i, name in enumerate(bots):
        for gi, (my_acts, opp_acts) in enumerate(bot_games[name]):
            T = min(len(my_acts), L)
            for t in range(T):
                state = _extract_state(my_acts.tolist(), opp_acts.tolist(), t, len(my_acts))
                self_oh = np.zeros(3, dtype=np.float32); self_oh[my_acts[t]] = 1.0
                opp_oh = np.zeros(3, dtype=np.float32); opp_oh[opp_acts[t]] = 1.0
                games[i, gi, t] = np.concatenate([state, self_oh, opp_oh])
                tmask[i, gi, t] = True
            gmask[i, gi] = True

    observed = np.argwhere(pair_counts > 0)
    observed = np.array([(int(a), int(b)) for a, b in observed if a != b])

    return {
        "player_names": bots,
        "agent_games": games,
        "agent_token_mask": tmask,
        "agent_game_mask": gmask,
        "F": F,
        "pair_counts": pair_counts,
        "observed_pairs": observed,
        "token_dim": TOKEN_DIM,
        "L_max": L,
    }


# ---------------------------------------------------------------------------
# Random Forest baseline (mean of self/opp tokens, same recipe as tennis)
# ---------------------------------------------------------------------------


def run_rf_baseline(ds, train_pair_tuples, test_pair_tuples):
    from sklearn.ensemble import RandomForestRegressor
    games = ds["agent_games"]
    tmask = ds["agent_token_mask"]
    # is_self bit: last 3 of token = self_oh, so "is this a self action" = sum of self slots
    # Actually we need a single is_self bit. Each token has both self_oh and opp_oh filled in.
    # Since every token represents a single round, both self AND opp action are in the token.
    # So there's no self/opp distinction per-token — just take mean over all valid tokens.
    N = games.shape[0]
    flat = games.reshape(N, -1, games.shape[-1])
    mflat = tmask.reshape(N, -1)
    cnt = np.maximum(mflat.sum(axis=-1), 1)[:, None]
    stats = (flat * mflat[..., None]).sum(axis=1) / cnt  # (N, token_dim)
    # Pair features: concat(i, j, i - j)
    ti = stats[train_pair_tuples[:, 0]]
    tj = stats[train_pair_tuples[:, 1]]
    X_tr = np.concatenate([ti, tj, ti - tj], axis=1)
    y_tr = np.array([ds["F"][i, j] for i, j in train_pair_tuples], dtype=np.float32)

    ei = stats[test_pair_tuples[:, 0]]
    ej = stats[test_pair_tuples[:, 1]]
    X_te = np.concatenate([ei, ej, ei - ej], axis=1)
    y_te = np.array([ds["F"][i, j] for i, j in test_pair_tuples], dtype=np.float32)

    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
    )
    rf.fit(X_tr, y_tr)
    p_tr = rf.predict(X_tr)
    p_te = rf.predict(X_te)
    y_mean_train = float(y_tr.mean())
    return {
        "train_mse": float(np.mean((y_tr - p_tr) ** 2)),
        "test_mse": float(np.mean((y_te - p_te) ** 2)),
        "null_train_mse": float(np.mean((y_tr - y_mean_train) ** 2)),
        "null_test_mse": float(np.mean((y_te - y_mean_train) ** 2)),
        "pred_test": p_te,
        "y_test": y_te,
        "y_train": y_tr,
        "y_mean_train": y_mean_train,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("RPS Intervention Study (sanity-check vs tennis)")
    print("=" * 72)

    print("\n[1] Loading RPS tournament data...")
    ds = load_rps_dataset()
    N = len(ds["player_names"])
    print(f"  Dataset: N={N} bots, G_max={ds['agent_games'].shape[1]}, "
          f"L={ds['L_max']}, token_dim={ds['token_dim']}")
    print(f"  F range: [{ds['F'].min():.3f}, {ds['F'].max():.3f}]")
    print(f"  Observed ordered pairs (i,j), i!=j: {len(ds['observed_pairs'])}")

    # Train/test split (80/20, seed 0) — same recipe as tennis
    print("\n[2] Splitting observed pairs 80/20 (seed 0)...")
    observed = np.array(ds["observed_pairs"])
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pair_tuples = observed[perm[:split]]
    test_pair_tuples = observed[perm[split:]]
    train_pairs = np.array([i * N + j for i, j in train_pair_tuples], dtype=np.int64)
    test_pairs = np.array([i * N + j for i, j in test_pair_tuples], dtype=np.int64)
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Helpers
    def fpta_config(spread_weight=0.0):
        return TrainConfig(
            lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
            ortho_weight=ORTHO_WEIGHT, ridge_lambda=1e-4,
            c_correction_every=C_CORRECTION_EVERY,
            grad_clip=1.0, log_every=100,
            spread_weight=spread_weight, spread_target=SPREAD_TARGET,
        )

    def run_fpta(label, config, pretrained_encoder=None):
        print("\n" + "=" * 72)
        print(f"[{label}]")
        print("=" * 72)
        return hierarchical_behavioral_fpta(
            ds["agent_games"], ds["agent_token_mask"], ds["agent_game_mask"],
            jnp.array(ds["F"]),
            token_dim=ds["token_dim"], L_max=ds["L_max"],
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

    # (A) Baseline FPTA
    result_baseline = run_fpta("A: FPTA baseline (no intervention)", fpta_config())

    # (B) Spread reg
    result_spread = run_fpta(
        f"B: FPTA + spread regulariser (w={SPREAD_WEIGHT}, target={SPREAD_TARGET})",
        fpta_config(spread_weight=SPREAD_WEIGHT),
    )

    # (C) Contrastive pretrain + FPTA fine-tune
    print("\n" + "=" * 72)
    print("[C: Contrastive pretraining -> FPTA fine-tune]")
    print("=" * 72)
    pre = contrastive_pretrain(
        ds["agent_games"], ds["agent_token_mask"], ds["agent_game_mask"],
        token_dim=ds["token_dim"], L_max=ds["L_max"], trait_dim=TRAIT_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        mlp_ratio=MLP_RATIO, rho_hidden=(D_MODEL,),
        n_steps=CONTRASTIVE_STEPS, batch_size=CONTRASTIVE_BATCH,
        G_sample=CONTRASTIVE_G_SAMPLE,
        lr=CONTRASTIVE_LR, temperature=CONTRASTIVE_TEMP,
        log_every=50, key=jax.random.PRNGKey(42), numpy_seed=7,
        verbose=True,
    )
    result_pretrain = run_fpta(
        "FPTA fine-tune (warm-started from contrastive)",
        fpta_config(), pretrained_encoder=pre.encoder,
    )

    # (D) Antisym MLP head (no FPTA bilinear)
    print("\n" + "=" * 72)
    print("[D: Hierarchical encoder + antisym MLP head]")
    print("=" * 72)
    mlp_config = TrainConfig(
        lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        grad_clip=1.0, log_every=100,
    )
    result_mlp = hierarchical_mlp_baseline(
        ds["agent_games"], ds["agent_token_mask"], ds["agent_game_mask"],
        jnp.array(ds["F"]),
        token_dim=ds["token_dim"], L_max=ds["L_max"], trait_dim=TRAIT_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO,
        rho_hidden=(D_MODEL,), head_hidden=(128, 128),
        config=mlp_config, key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200,
        G_sample=G_SAMPLE, G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1, verbose=True,
    )

    # (E) RF baseline
    print("\n" + "=" * 72)
    print("[E: Random Forest on hand-crafted features]")
    print("=" * 72)
    rf = run_rf_baseline(ds, train_pair_tuples, test_pair_tuples)

    # -----------------------------------------------------------------
    # Collect predictions + diagnostics
    # -----------------------------------------------------------------
    agents_games_jnp = jnp.array(ds["agent_games"])
    agents_tmask_jnp = jnp.array(ds["agent_token_mask"])
    agents_gmask_jnp = jnp.array(ds["agent_game_mask"])
    test_true = rf["y_test"]

    def test_preds(result):
        Fp = np.asarray(result.predict(
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        ))
        return np.array([Fp[i, j] for i, j in test_pair_tuples], dtype=np.float32)

    p_base = test_preds(result_baseline)
    p_spread = test_preds(result_spread)
    p_pretrain = test_preds(result_pretrain)
    p_mlp = test_preds(result_mlp)
    p_rf = rf["pred_test"]

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None

    def trait_diag(result):
        traits = np.asarray(result.encode(
            agents_games_jnp, agents_tmask_jnp, agents_gmask_jnp,
        ))
        m = traits.mean(axis=0)
        return float(traits.std(axis=0).mean()), float(np.linalg.norm(traits - m, axis=1).mean())

    base_std, base_l2 = trait_diag(result_baseline)
    spr_std, spr_l2 = trait_diag(result_spread)
    pre_std, pre_l2 = trait_diag(result_pretrain)
    mlp_std, mlp_l2 = trait_diag(result_mlp)

    # -----------------------------------------------------------------
    # Master comparison table
    # -----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS — RPS")
    print("=" * 72)
    rows = [
        ("Null (predict train mean)", rf["null_train_mse"], rf["null_test_mse"], None, None, None),
        ("Random Forest (handcrafted features)", rf["train_mse"], rf["test_mse"], None, None, float(p_rf.std())),
        ("FPTA baseline", final(result_baseline.train_history, "train_mse"), final(result_baseline.train_history, "test_mse"),
         base_std, base_l2, float(p_base.std())),
        ("FPTA + spread reg (w=%.2f)" % SPREAD_WEIGHT,
         final(result_spread.train_history, "train_mse"), final(result_spread.train_history, "test_mse"),
         spr_std, spr_l2, float(p_spread.std())),
        ("FPTA from contrastive pretrain",
         final(result_pretrain.train_history, "train_mse"), final(result_pretrain.train_history, "test_mse"),
         pre_std, pre_l2, float(p_pretrain.std())),
        ("Encoder + antisym MLP head",
         final(result_mlp.train_history, "train_mse"), final(result_mlp.train_history, "test_mse"),
         mlp_std, mlp_l2, float(p_mlp.std())),
    ]
    null_t = rf["null_test_mse"]
    print(f"\n{'Model':<40} {'Train MSE':>12} {'Test MSE':>12} {'Δ null':>8}")
    print("-" * 76)
    for name, tr, te, _, _, _ in rows:
        gain = (1 - te / null_t) * 100 if null_t > 0 else 0
        print(f"{name:<40} {tr:>12.6f} {te:>12.6f} {gain:>+7.2f}%")

    print("\n--- Trait / prediction diagnostics ---")
    print(f"{'Variant':<40} {'trait std/dim':>14} {'trait L2':>10} {'pred std':>10}")
    print("-" * 76)
    for name, tr, te, s, l, p in rows:
        if s is None:
            continue
        print(f"{name:<40} {s:>14.5f} {l:>10.5f} {p:>10.5f}")

    # -----------------------------------------------------------------
    # Plots: 5-panel pred-vs-actual scatter
    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharex=True, sharey=True)
        lim = float(max(
            np.abs(test_true).max(), np.abs(p_rf).max(),
            np.abs(p_base).max(), np.abs(p_spread).max(),
            np.abs(p_pretrain).max(), np.abs(p_mlp).max(),
        )) * 1.05
        lim = max(lim, 0.1)

        panels = [
            (p_rf, "Random Forest", float(np.mean((test_true - p_rf) ** 2))),
            (p_base, "FPTA baseline", float(np.mean((test_true - p_base) ** 2))),
            (p_spread, f"FPTA + spread reg (w={SPREAD_WEIGHT})", float(np.mean((test_true - p_spread) ** 2))),
            (p_pretrain, "FPTA + contrastive pretrain", float(np.mean((test_true - p_pretrain) ** 2))),
            (p_mlp, "Encoder + antisym MLP head", float(np.mean((test_true - p_mlp) ** 2))),
        ]
        for ax_i, (pred, title, mse) in enumerate(panels):
            ax = axes[ax_i]
            ax.scatter(test_true, pred, s=25, alpha=0.55, edgecolors="k", linewidths=0.2)
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.5)
            ax.axhline(rf["y_mean_train"], color="red", linewidth=1, alpha=0.4, linestyle=":")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("True F[i, j]")
            if ax_i == 0:
                ax.set_ylabel("Predicted F[i, j]")
            ax.set_title(f"{title}\n(test MSE = {mse:.5f})", fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"RPS — Predicted vs. actual F on test pairs (n={len(test_true)})",
                     fontsize=13)
        out = OUTPUT_DIR / "rps_interventions_pred_vs_actual.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # MSE curves, all 4 neural variants
        fig, ax = plt.subplots(figsize=(10, 6))
        for hist, name, colour, style in [
            (result_baseline.train_history, "FPTA baseline", "C0", "-"),
            (result_spread.train_history, "FPTA + spread reg", "C1", "-"),
            (result_pretrain.train_history, "FPTA + contrastive pretrain", "C2", "-"),
            (result_mlp.train_history, "Encoder + MLP head", "C3", "-"),
        ]:
            steps = [r["step"] for r in hist if "test_mse" in r]
            te = [r["test_mse"] for r in hist if "test_mse" in r]
            ax.plot(steps, te, label=name, linewidth=2, color=colour, linestyle=style)
        ax.axhline(rf["null_test_mse"], color="gray", linestyle=":", alpha=0.7,
                   label=f"Null ({rf['null_test_mse']:.5f})")
        ax.axhline(rf["test_mse"], color="purple", linestyle=":", alpha=0.7,
                   label=f"Random Forest ({rf['test_mse']:.5f})")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step"); ax.set_ylabel("Test MSE (log scale)")
        ax.set_title("RPS — Test MSE across variants")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        out = OUTPUT_DIR / "rps_interventions_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
