#!/usr/bin/env python3
"""Hierarchical skill + disc-game FPTA on RPS tournament data.

Same encoder/data/split as rps_interventions.py, but swaps the model for
the skill+disc variant. Reports variance decomposition, per-bot skill
ranking, and disc-game embeddings (which should now be purely cyclic).

Usage:
    python examples/rps_skill.py
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

from fptajax import hierarchical_skill_fpta, TrainConfig


# ---------------------------------------------------------------------------
# Config (matched to rps_interventions.py)
# ---------------------------------------------------------------------------

TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)
OUTPUT_DIR = _HERE

HISTORY_WINDOW = 4
MAX_GAMES_PER_BOT = 60
ROUNDS_PER_GAME = 100

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

SKILL_CENTERING_WEIGHT = 1.0
SKILL_HIDDEN = (D_MODEL,)

# Prior run numbers from /tmp/rps_interventions.log
BASELINE = {
    "null_train":  0.125318, "null_test":  0.118562,
    "rf_train":    0.008067, "rf_test":    0.023655,
    "fpta_train":  0.052141, "fpta_test":  0.051439,
    "spread_train": 0.041438, "spread_test": 0.043424,
    "contrastive_train": 0.051931, "contrastive_test": 0.049987,
    "mlp_train":   0.049378, "mlp_test":   0.047537,
}


# ---------------------------------------------------------------------------
# Data loading (copied from rps_interventions.py for self-containedness)
# ---------------------------------------------------------------------------


def _extract_state(my_actions, opp_actions, round_idx, total_rounds) -> np.ndarray:
    features: list[float] = []
    if round_idx > 0:
        mc = np.bincount(my_actions[:round_idx], minlength=3).astype(np.float32) / round_idx
        oc = np.bincount(opp_actions[:round_idx], minlength=3).astype(np.float32) / round_idx
    else:
        mc = np.ones(3, dtype=np.float32) / 3
        oc = np.ones(3, dtype=np.float32) / 3
    features.extend(mc.tolist()); features.extend(oc.tolist())
    for i in range(HISTORY_WINDOW):
        idx = round_idx - HISTORY_WINDOW + i
        if 0 <= idx < len(opp_actions):
            oh = np.zeros(3, dtype=np.float32); oh[opp_actions[idx]] = 1.0
        else:
            oh = np.ones(3, dtype=np.float32) / 3
        features.extend(oh.tolist())
    features.append(round_idx / total_rounds if total_rounds > 0 else 0.0)
    return np.array(features, dtype=np.float32)


STATE_DIM = 6 + HISTORY_WINDOW * 3 + 1
TOKEN_DIM = STATE_DIM + 2 * 3


def load_rps_dataset():
    assert TOURNAMENT_DATA.exists(), f"Missing {TOURNAMENT_DATA}"
    matches = [json.loads(l) for l in open(TOURNAMENT_DATA)]
    bots = sorted({b for m in matches for b in (m["player1"], m["player2"])})
    name_to_idx = {n: i for i, n in enumerate(bots)}
    N = len(bots)

    wins = np.zeros((N, N))
    counts = np.zeros((N, N))
    pair_counts = np.zeros((N, N), dtype=np.int64)
    for m in matches:
        i, j = name_to_idx[m["player1"]], name_to_idx[m["player2"]]
        a1 = np.array(m["player1_actions"]); a2 = np.array(m["player2_actions"])
        wins[i, j] += np.sum((a1 - a2) % 3 == 1)
        wins[j, i] += np.sum((a2 - a1) % 3 == 1)
        counts[i, j] += len(a1); counts[j, i] += len(a1)
        pair_counts[i, j] += 1; pair_counts[j, i] += 1
    denom = np.maximum(counts, 1)
    F = ((wins - wins.T) / denom).astype(np.float32)
    F = 0.5 * (F - F.T)

    bot_games: dict[str, list] = {n: [] for n in bots}
    for m in matches:
        for name, opp, acts, opp_acts in [
            (m["player1"], m["player2"], m["player1_actions"], m["player2_actions"]),
            (m["player2"], m["player1"], m["player2_actions"], m["player1_actions"]),
        ]:
            bot_games[name].append((np.array(acts), np.array(opp_acts)))
    for name in bots:
        g = bot_games[name]
        if len(g) > MAX_GAMES_PER_BOT:
            rng = np.random.RandomState(hash(name) % 2**31)
            idx = rng.choice(len(g), MAX_GAMES_PER_BOT, replace=False)
            bot_games[name] = [g[k] for k in idx]

    G_max = MAX_GAMES_PER_BOT; L = ROUNDS_PER_GAME
    games = np.zeros((N, G_max, L, TOKEN_DIM), dtype=np.float32)
    tmask = np.zeros((N, G_max, L), dtype=bool)
    gmask = np.zeros((N, G_max), dtype=bool)
    for i, name in enumerate(bots):
        for gi, (my, opp) in enumerate(bot_games[name]):
            T = min(len(my), L)
            for t in range(T):
                state = _extract_state(my.tolist(), opp.tolist(), t, len(my))
                self_oh = np.zeros(3, dtype=np.float32); self_oh[my[t]] = 1.0
                opp_oh = np.zeros(3, dtype=np.float32); opp_oh[opp[t]] = 1.0
                games[i, gi, t] = np.concatenate([state, self_oh, opp_oh])
                tmask[i, gi, t] = True
            gmask[i, gi] = True

    observed = np.argwhere(pair_counts > 0)
    observed = np.array([(int(a), int(b)) for a, b in observed if a != b])
    return {
        "bot_names": bots, "agent_games": games,
        "agent_token_mask": tmask, "agent_game_mask": gmask,
        "F": F, "pair_counts": pair_counts,
        "observed_pairs": observed,
        "token_dim": TOKEN_DIM, "L_max": L,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Hierarchical Skill+Disc FPTA on RPS")
    print("=" * 72)

    print("\n[1] Loading RPS tournament data...")
    ds = load_rps_dataset()
    N = len(ds["bot_names"])
    print(f"  Dataset: N={N} bots, G_max={ds['agent_games'].shape[1]}, "
          f"L={ds['L_max']}, token_dim={ds['token_dim']}")
    print(f"  F range: [{ds['F'].min():.3f}, {ds['F'].max():.3f}]")
    print(f"  Observed ordered pairs (i,j), i!=j: {len(ds['observed_pairs'])}")

    # Same train/test split as rps_interventions.py
    observed = np.array(ds["observed_pairs"])
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pair_tuples = observed[perm[:split]]
    test_pair_tuples = observed[perm[split:]]
    train_pairs = np.array([i * N + j for i, j in train_pair_tuples], dtype=np.int64)
    test_pairs = np.array([i * N + j for i, j in test_pair_tuples], dtype=np.int64)
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # ---- Train ----
    print("\n[2] Training hierarchical skill + disc-game FPTA...")
    config = TrainConfig(
        lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        ortho_weight=ORTHO_WEIGHT, ridge_lambda=1e-4,
        c_correction_every=C_CORRECTION_EVERY,
        grad_clip=1.0, log_every=100,
        skill_centering_weight=SKILL_CENTERING_WEIGHT,
    )
    result = hierarchical_skill_fpta(
        ds["agent_games"], ds["agent_token_mask"], ds["agent_game_mask"],
        jnp.array(ds["F"]),
        token_dim=ds["token_dim"], L_max=ds["L_max"],
        trait_dim=TRAIT_DIM, d=BASIS_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO,
        rho_hidden=(D_MODEL,), basis_hidden=(128, 128),
        skill_hidden=SKILL_HIDDEN,
        config=config, key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200, G_sample=G_SAMPLE, G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1, verbose=True,
    )

    # ---- Report ----
    print("\n" + "=" * 72)
    print("RESULTS — RPS skill+disc FPTA")
    print("=" * 72)

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None
    skill_train_mse = final(result.train_history, "train_mse")
    skill_test_mse = final(result.train_history, "test_mse")
    print(f"\nFinal train MSE: {skill_train_mse:.6f}")
    print(f"Final test  MSE: {skill_test_mse:.6f}")

    print(f"\nDisc game components: {result.n_components}")
    imp = np.asarray(result.get_importance())
    cum = np.asarray(result.get_cumulative_importance())
    for k in range(min(result.n_components, 6)):
        print(f"  Component {k+1}: ω={float(result.eigenvalues[k]):.5f}, "
              f"importance={imp[k]:.4f}, cumulative={cum[k]:.4f}")

    games_j = jnp.array(ds["agent_games"])
    tmask_j = jnp.array(ds["agent_token_mask"])
    gmask_j = jnp.array(ds["agent_game_mask"])
    skills = np.asarray(result.skills(games_j, tmask_j, gmask_j))

    # Skill ranking
    print("\n[3] Bot skill ranking (top 15, bottom 10):")
    order = np.argsort(-skills)
    for rank, i in enumerate(order[:15], 1):
        print(f"  {rank:>3}. {ds['bot_names'][i]:<32} s = {skills[i]:+.4f}")
    print("  ...")
    for rank, i in enumerate(order[-10:], len(order) - 9):
        print(f"  {rank:>3}. {ds['bot_names'][i]:<32} s = {skills[i]:+.4f}")

    # Variance decomposition
    vd_observed = result.variance_decomposition(games_j, tmask_j, gmask_j, pair_tuples=observed)
    vd_train = result.variance_decomposition(games_j, tmask_j, gmask_j, pair_tuples=train_pair_tuples)
    vd_test = result.variance_decomposition(games_j, tmask_j, gmask_j, pair_tuples=test_pair_tuples)
    print("\n[4] Variance decomposition of predicted F:")
    print(f"  {'scope':<22} {'skill_var':>10} {'disc_var':>10} {'total_var':>10} "
          f"{'skill%':>7} {'disc%':>7}")
    for scope, vd in [("all observed pairs", vd_observed),
                      ("train pairs", vd_train),
                      ("test pairs", vd_test)]:
        print(f"  {scope:<22} {vd['skill_var']:>10.5f} {vd['disc_var']:>10.5f} "
              f"{vd['total_var']:>10.5f} {vd['skill_frac']*100:>6.2f}% "
              f"{vd['disc_frac']*100:>6.2f}%")

    # Contribution of each part to test MSE
    F_skill, F_disc, F_total = result.decompose_F(games_j, tmask_j, gmask_j)
    F_skill = np.asarray(F_skill); F_disc = np.asarray(F_disc); F_total = np.asarray(F_total)
    def pair_mse(F_pred, pairs):
        i = pairs[:, 0]; j = pairs[:, 1]
        y = np.array([ds["F"][a, b] for a, b in pairs], dtype=np.float32)
        return float(np.mean((y - F_pred[i, j]) ** 2))
    print("\n[5] Test MSE of each contribution on test pairs:")
    print(f"  skill only (F̂ = s_i - s_j):             {pair_mse(F_skill, test_pair_tuples):.6f}")
    print(f"  disc only (F̂ = bᵢᵀ C bⱼ):              {pair_mse(F_disc, test_pair_tuples):.6f}")
    print(f"  combined (F̂ = skill + disc):            {pair_mse(F_total, test_pair_tuples):.6f}")

    # Comparison table
    print("\n" + "=" * 72)
    print("COMPARISON — RPS")
    print("=" * 72)
    null_t = BASELINE["null_test"]
    print(f"\n{'Model':<42} {'Train MSE':>12} {'Test MSE':>12} {'Δ null':>8}")
    print("-" * 80)
    rows = [
        ("Null (predict train mean)",     BASELINE["null_train"], BASELINE["null_test"]),
        ("Random Forest (handcrafted)",   BASELINE["rf_train"],   BASELINE["rf_test"]),
        ("FPTA baseline",                 BASELINE["fpta_train"], BASELINE["fpta_test"]),
        ("FPTA + spread reg",             BASELINE["spread_train"], BASELINE["spread_test"]),
        ("FPTA + contrastive pretrain",   BASELINE["contrastive_train"], BASELINE["contrastive_test"]),
        ("Encoder + antisym MLP head",    BASELINE["mlp_train"],  BASELINE["mlp_test"]),
        ("Hierarchical skill + disc FPTA", skill_train_mse,       skill_test_mse),
    ]
    for name, tr, te in rows:
        gain = (1 - te / null_t) * 100
        print(f"{name:<42} {tr:>12.6f} {te:>12.6f} {gain:>+7.2f}%")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from fptajax.viz import plot_pta_embedding
        from fptajax.pta import PTAResult

        # (1) MSE curves
        hist = result.train_history
        steps = [r["step"] for r in hist if "train_mse" in r]
        tr = [r["train_mse"] for r in hist if "train_mse" in r]
        te = [r["test_mse"] for r in hist if "test_mse" in r]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, tr, label="Train MSE", linewidth=2)
        ax.plot(steps, te, label="Test MSE", linewidth=2, linestyle="--")
        ax.axhline(BASELINE["null_test"], color="gray", linestyle=":",
                   alpha=0.7, label=f"Null ({BASELINE['null_test']:.5f})")
        ax.axhline(BASELINE["fpta_test"], color="red", linestyle=":",
                   alpha=0.7, label=f"FPTA baseline ({BASELINE['fpta_test']:.5f})")
        ax.axhline(BASELINE["rf_test"], color="purple", linestyle=":",
                   alpha=0.7, label=f"Random Forest ({BASELINE['rf_test']:.5f})")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step"); ax.set_ylabel("MSE (log scale)")
        ax.set_title("RPS Skill+Disc FPTA — Train/Test MSE")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        out = OUTPUT_DIR / "rps_skill_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # (2) Disc-game embeddings (centered)
        short = [n.split("_")[-1] if "_" in n else n for n in ds["bot_names"]]
        Y = result.embed(games_j, tmask_j, gmask_j)
        pta_like = PTAResult(
            embeddings=Y, eigenvalues=result.eigenvalues, Q=result.schur_vectors,
            U=jnp.zeros_like(result.schur_vectors),
            n_components=result.n_components, f_norm_sq=result.f_norm_sq,
        )
        n_games = min(result.n_components, 4)
        cols = min(n_games, 2); rows = (n_games + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
        axes = [axes] if n_games == 1 else np.array(axes).flatten()
        for k in range(n_games):
            plot_pta_embedding(pta_like, k=k, labels=short, ax=axes[k], center=True)
            omega_k = float(result.eigenvalues[k])
            imp_k = float(imp[k]) if k < len(imp) else 0.0
            axes[k].set_title(
                f"Disc Game {k+1}  ($\\omega$={omega_k:.4f}, imp={imp_k:.1%})"
            )
        for k in range(n_games, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("RPS Disc-game Embeddings (skill+disc FPTA) — centered",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_skill_disc_games.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (3) Skill ranking bar chart
        fig, ax = plt.subplots(figsize=(11, 10))
        order_plot = np.argsort(skills)
        y_pos = np.arange(len(order_plot))
        colours = ["C2" if skills[i] > 0 else "C3" for i in order_plot]
        ax.barh(y_pos, skills[order_plot], color=colours, alpha=0.85, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([ds["bot_names"][i] for i in order_plot], fontsize=8)
        ax.set_xlabel("Learned skill s(x)")
        ax.set_title("RPS — Per-bot skill (learned by skill head)")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_skill_ranking.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (4) F decomposition heatmap, bots sorted by learned skill
        order_by_skill = np.argsort(-skills)
        F_true_re = ds["F"][np.ix_(order_by_skill, order_by_skill)]
        F_skill_re = F_skill[np.ix_(order_by_skill, order_by_skill)]
        F_disc_re = F_disc[np.ix_(order_by_skill, order_by_skill)]
        F_total_re = F_total[np.ix_(order_by_skill, order_by_skill)]
        vmax = float(max(abs(F_true_re).max(), abs(F_total_re).max(),
                         abs(F_skill_re).max(), abs(F_disc_re).max()))
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
        for ax, mat, title in zip(
            axes,
            [F_true_re, F_total_re, F_skill_re, F_disc_re],
            ["True F", "Predicted F (skill+disc)", "F_skill only", "F_disc only"],
        ):
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        fig.suptitle("RPS — F decomposition (bots sorted by learned skill)", fontsize=12)
        out = OUTPUT_DIR / "rps_skill_decomposition.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (5) Predicted vs actual scatter, colored by |skill_i - skill_j|
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        y_true = np.array([ds["F"][i, j] for i, j in test_pair_tuples], dtype=np.float32)
        y_pred = np.array([F_total[i, j] for i, j in test_pair_tuples], dtype=np.float32)
        skill_diff_mag = np.array(
            [abs(skills[i] - skills[j]) for i, j in test_pair_tuples],
            dtype=np.float32,
        )
        sc = ax.scatter(y_true, y_pred, c=skill_diff_mag, cmap="viridis",
                        s=40, alpha=0.7, edgecolors="black", linewidths=0.3)
        lim = float(max(abs(y_true).max(), abs(y_pred).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y = x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("True F[i, j]"); ax.set_ylabel("Predicted F[i, j]")
        ax.set_title(f"RPS skill+disc FPTA — test set  (MSE = {skill_test_mse:.5f})")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.colorbar(sc, ax=ax, label="|skill_i - skill_j|")
        out = OUTPUT_DIR / "rps_skill_pred_vs_actual.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
