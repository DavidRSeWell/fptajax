#!/usr/bin/env python3
"""Direct-g-prediction hierarchical FPTA on RPS tournament data.

Label: g(x, y) = total rounds x won against y across all matches.
This is NOT normalised — pairs that played more matches contribute larger
g values, which is exactly the asymmetric-symmetric decomposition we want
the model to capture.

f(x, y) is recovered via two passes: f_hat = g_hat(x, y) - g_hat(y, x),
then normalised by total rounds-played for comparison against prior F-
based RPS models.

Usage:
    python examples/rps_g.py
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

from fptajax import hierarchical_g_fpta, TrainConfig


# ---------------------------------------------------------------------------
# Config (matched to rps_interventions.py / rps_skill.py)
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
SKILL_DIM = 4

N_STEPS = 2000
BATCH_SIZE = 8
G_SAMPLE = 4
G_SAMPLE_EVAL = 20
LR = 5e-4
ORTHO_WEIGHT = 0.1
C_CORRECTION_EVERY = 300

# Prior F-based baselines from /tmp/rps_interventions.log + /tmp/rps_skill.log.
F_BASELINE = {
    "null_train":       0.125318, "null_test":       0.118562,
    "rf_train":         0.008067, "rf_test":         0.023655,
    "fpta_train":       0.052141, "fpta_test":       0.051439,
    "spread_train":     0.041438, "spread_test":     0.043424,
    "contrastive_train": 0.051931, "contrastive_test": 0.049987,
    "mlp_train":        0.049378, "mlp_test":        0.047537,
    "skill_train":      0.040466, "skill_test":      0.037710,
}


# ---------------------------------------------------------------------------
# Data loading (produces F, G labels)
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
    G = wins.astype(np.float32)  # raw total wins by i against j

    bot_games: dict[str, list] = {n: [] for n in bots}
    for m in matches:
        for name, _, acts, opp_acts in [
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
        "F": F, "G": G, "pair_total_rounds": counts.astype(np.float32),
        "pair_counts": pair_counts,
        "observed_pairs": observed,
        "token_dim": TOKEN_DIM, "L_max": L,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("Direct-g-prediction Hierarchical FPTA on RPS")
    print("=" * 72)

    print("\n[1] Loading RPS tournament data...")
    ds = load_rps_dataset()
    N = len(ds["bot_names"])
    print(f"  Dataset: N={N} bots, G_max={ds['agent_games'].shape[1]}, "
          f"L={ds['L_max']}, token_dim={ds['token_dim']}")
    print(f"  F range: [{ds['F'].min():.3f}, {ds['F'].max():.3f}]")
    print(f"  G range: [{ds['G'].min():.0f}, {ds['G'].max():.0f}]")
    print(f"  Observed ordered pairs (i,j), i!=j: {len(ds['observed_pairs'])}")

    # Same 80/20 split as prior RPS runs
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
    print("\n[2] Training hierarchical g-FPTA...")
    config = TrainConfig(
        lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        ortho_weight=ORTHO_WEIGHT, ridge_lambda=1e-4,
        c_correction_every=C_CORRECTION_EVERY,
        grad_clip=1.0, log_every=100,
    )
    result = hierarchical_g_fpta(
        ds["agent_games"], ds["agent_token_mask"], ds["agent_game_mask"],
        jnp.array(ds["G"]),
        token_dim=ds["token_dim"], L_max=ds["L_max"],
        trait_dim=TRAIT_DIM, d=BASIS_DIM, skill_dim=SKILL_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mlp_ratio=MLP_RATIO,
        rho_hidden=(D_MODEL,), basis_hidden=(128, 128),
        skill_hidden=(D_MODEL,),
        config=config, key=jax.random.PRNGKey(42),
        train_pairs=train_pairs, test_pairs=test_pairs,
        eval_every=200, G_sample=G_SAMPLE, G_sample_eval=G_SAMPLE_EVAL,
        numpy_seed=1, verbose=True,
    )

    # ---- Report ----
    print("\n" + "=" * 72)
    print("RESULTS — RPS g-FPTA")
    print("=" * 72)

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None
    g_train_mse = final(result.train_history, "train_mse")
    g_test_mse = final(result.train_history, "test_mse")
    print(f"\n(g-space) Final train MSE: {g_train_mse:.3f}")
    print(f"(g-space) Final test  MSE: {g_test_mse:.3f}")
    g_test_std = float(np.std([ds['G'][i, j] for i, j in test_pair_tuples]))
    print(f"(g-space) Test label std:  {g_test_std:.3f}  (so R² ≈ "
          f"{1 - g_test_mse / g_test_std**2:.3f})")

    c0 = np.asarray(result.c_0); c1 = np.asarray(result.c_1)
    print(f"\nLearned coefficients:")
    print(f"  c_0:       {c0}")
    print(f"  c_1:       {c1}")
    print(f"  c_0 - c_1: {c0 - c1}")
    print(f"  c_0 + c_1: {c0 + c1}")
    print(f"  ||c_0-c_1||={np.linalg.norm(c0-c1):.3f},  "
          f"||c_0+c_1||={np.linalg.norm(c0+c1):.3f}")

    # Predict F via two passes, normalise by total rounds per pair
    games_j = jnp.array(ds["agent_games"])
    tmask_j = jnp.array(ds["agent_token_mask"])
    gmask_j = jnp.array(ds["agent_game_mask"])
    F_pred_raw = np.asarray(result.predict_f(
        games_j, tmask_j, gmask_j, games_j, tmask_j, gmask_j,
    ))
    pts_total = ds["pair_total_rounds"]
    F_pred = np.zeros_like(F_pred_raw)
    mask = pts_total > 0
    F_pred[mask] = F_pred_raw[mask] / pts_total[mask]

    def pair_mse(F_pred_mat, pairs):
        i = pairs[:, 0]; j = pairs[:, 1]
        y = np.array([ds["F"][a, b] for a, b in pairs], dtype=np.float32)
        return float(np.mean((y - F_pred_mat[i, j]) ** 2))

    f_train_mse = pair_mse(F_pred, train_pair_tuples)
    f_test_mse = pair_mse(F_pred, test_pair_tuples)
    print(f"\n(F-space)  Train F-MSE: {f_train_mse:.6f}")
    print(f"(F-space)  Test  F-MSE: {f_test_mse:.6f}")

    # Effective scalar skill ranking
    eff_skill = np.asarray(result.effective_scalar_skill(games_j, tmask_j, gmask_j))
    print(f"\n[3] Effective scalar skill (c_0-c_1)^T s(x) — top 15 / bottom 10:")
    order = np.argsort(-eff_skill)
    for rank, i in enumerate(order[:15], 1):
        print(f"  {rank:>3}. {ds['bot_names'][i]:<32} s_eff = {eff_skill[i]:+.4f}")
    print("  ...")
    for rank, i in enumerate(order[-10:], len(order) - 9):
        print(f"  {rank:>3}. {ds['bot_names'][i]:<32} s_eff = {eff_skill[i]:+.4f}")

    # Variance decomposition
    vd_g = result.variance_decomposition_g(games_j, tmask_j, gmask_j, pair_tuples=observed)
    vd_f = result.variance_decomposition_f(games_j, tmask_j, gmask_j, pair_tuples=observed)
    print(f"\n[4] Variance decomposition of predictions (observed pairs):")
    print(f"  g-space:")
    print(f"    skill_var = {vd_g['skill_var']:.2f}, bilinear_var = {vd_g['bilinear_var']:.2f}, "
          f"total_var = {vd_g['total_var']:.2f}")
    print(f"    skill_frac = {vd_g['skill_frac']*100:.1f}%, "
          f"bilinear_frac = {vd_g['bilinear_frac']*100:.1f}%")
    print(f"  f-space (antisymmetric):")
    print(f"    skill_var = {vd_f['skill_var']:.4f}, disc_var = {vd_f['disc_var']:.4f}")
    print(f"    skill_frac = {vd_f['skill_frac']*100:.1f}%, "
          f"disc_frac = {vd_f['disc_frac']*100:.1f}%")

    # Per-contribution test F-MSE
    f_sk, f_di, f_tot = result.decompose_f(games_j, tmask_j, gmask_j)
    def ratio(mat):
        out = np.zeros_like(np.asarray(mat))
        m = pts_total > 0
        out[m] = np.asarray(mat)[m] / pts_total[m]
        return out
    print(f"\n[5] Test F-MSE of each f-contribution:")
    print(f"  skill only: {pair_mse(ratio(f_sk), test_pair_tuples):.6f}")
    print(f"  disc only:  {pair_mse(ratio(f_di), test_pair_tuples):.6f}")
    print(f"  combined:   {pair_mse(ratio(f_tot), test_pair_tuples):.6f}")

    # Comparison
    print("\n" + "=" * 72)
    print("COMPARISON — RPS F-MSE")
    print("=" * 72)
    null_t = F_BASELINE["null_test"]
    print(f"\n{'Model':<42} {'Train F-MSE':>12} {'Test F-MSE':>12} {'Δ null':>8}")
    print("-" * 80)
    rows = [
        ("Null (predict train mean)",     F_BASELINE["null_train"], F_BASELINE["null_test"]),
        ("Random Forest (handcrafted)",   F_BASELINE["rf_train"],   F_BASELINE["rf_test"]),
        ("FPTA baseline",                 F_BASELINE["fpta_train"], F_BASELINE["fpta_test"]),
        ("FPTA + spread reg",             F_BASELINE["spread_train"], F_BASELINE["spread_test"]),
        ("FPTA + contrastive pretrain",   F_BASELINE["contrastive_train"], F_BASELINE["contrastive_test"]),
        ("Encoder + antisym MLP head",    F_BASELINE["mlp_train"],  F_BASELINE["mlp_test"]),
        ("Hierarchical skill + disc",     F_BASELINE["skill_train"], F_BASELINE["skill_test"]),
        ("Direct g-prediction FPTA",      f_train_mse,              f_test_mse),
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

        # (1) g-MSE curve
        hist = result.train_history
        steps = [r["step"] for r in hist if "train_mse" in r]
        tr = [r["train_mse"] for r in hist if "train_mse" in r]
        te = [r["test_mse"] for r in hist if "test_mse" in r]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, tr, label="Train MSE (g)", linewidth=2)
        ax.plot(steps, te, label="Test MSE (g)", linewidth=2, linestyle="--")
        ax.set_yscale("log"); ax.set_xlabel("Training Step"); ax.set_ylabel("g-MSE")
        ax.set_title("RPS g-FPTA — g-space train/test MSE (wins units²)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        out = OUTPUT_DIR / "rps_g_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # (2) Disc-game embeddings
        short = [n.split("_")[-1] if "_" in n else n for n in ds["bot_names"]]
        Y = result.embed(games_j, tmask_j, gmask_j)
        pta_like = PTAResult(
            embeddings=Y, eigenvalues=result.eigenvalues, Q=result.schur_vectors,
            U=jnp.zeros_like(result.schur_vectors),
            n_components=result.n_components, f_norm_sq=result.f_norm_sq,
        )
        imp = np.asarray(result.get_importance())
        n_games = min(result.n_components, 4)
        cols = min(n_games, 2); rows = (n_games + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
        axes = [axes] if n_games == 1 else np.array(axes).flatten()
        for k in range(n_games):
            plot_pta_embedding(pta_like, k=k, labels=short, ax=axes[k], center=True)
            omega_k = float(result.eigenvalues[k])
            imp_k = float(imp[k]) if k < len(imp) else 0.0
            axes[k].set_title(
                f"Disc Game {k+1}  ($\\omega$={omega_k:.3f}, imp={imp_k:.1%})"
            )
        for k in range(n_games, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle("RPS Disc-game Embeddings (g-FPTA) — centered",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_g_disc_games.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (3) Effective skill ranking
        fig, ax = plt.subplots(figsize=(11, 10))
        order_plot = np.argsort(eff_skill)
        y_pos = np.arange(len(order_plot))
        colours = ["C2" if eff_skill[i] > 0 else "C3" for i in order_plot]
        ax.barh(y_pos, eff_skill[order_plot], color=colours, alpha=0.85,
                edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([ds["bot_names"][i] for i in order_plot], fontsize=8)
        ax.set_xlabel("Effective scalar skill  $(c_0-c_1)^\\top s(x)$")
        ax.set_title("RPS g-FPTA — effective scalar skill controlling f")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_g_effective_skill.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (4) g-decomposition heatmap
        g_sx, g_sy, g_bl, g_b, g_tot = result.decompose_g(games_j, tmask_j, gmask_j)
        g_sx = np.asarray(g_sx); g_sy = np.asarray(g_sy)
        g_bl = np.asarray(g_bl); g_tot = np.asarray(g_tot)
        order_by_skill = np.argsort(-eff_skill)
        G_re = ds["G"][np.ix_(order_by_skill, order_by_skill)]
        tot_re = g_tot[np.ix_(order_by_skill, order_by_skill)]
        sx_re = g_sx[np.ix_(order_by_skill, order_by_skill)]
        sy_re = g_sy[np.ix_(order_by_skill, order_by_skill)]
        bl_re = g_bl[np.ix_(order_by_skill, order_by_skill)]
        vmax = float(max(abs(G_re).max(), abs(tot_re).max()))
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
        for ax, mat, title in zip(
            axes,
            [G_re, tot_re, sx_re + sy_re, bl_re],
            ["True g (total wins)", "Predicted g",
             "g skill part (c_0·s(x) + c_1·s(y))",
             "g bilinear part (B^T C B)"],
        ):
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        fig.suptitle("RPS — g (raw wins) decomposition, bots sorted by effective skill",
                     fontsize=12)
        out = OUTPUT_DIR / "rps_g_decomposition.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (5) F-space scatter: predicted vs actual
        F_true_test = np.array([ds["F"][i, j] for i, j in test_pair_tuples],
                               dtype=np.float32)
        F_pred_test = np.array([F_pred[i, j] for i, j in test_pair_tuples],
                               dtype=np.float32)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.scatter(F_true_test, F_pred_test, s=30, alpha=0.6,
                   edgecolors="black", linewidths=0.3)
        lim = float(max(abs(F_true_test).max(), abs(F_pred_test).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y = x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("True F[i, j]"); ax.set_ylabel("Predicted F[i, j]")
        ax.set_title(f"RPS g-FPTA — test F (antisymmetric recovery)  "
                     f"MSE = {f_test_mse:.5f}")
        ax.grid(True, alpha=0.3); ax.legend()
        out = OUTPUT_DIR / "rps_g_pred_vs_actual.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
