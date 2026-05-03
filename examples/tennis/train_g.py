#!/usr/bin/env python3
"""Direct-g-prediction hierarchical FPTA on tennis.

Label: g(x, y) = total points player x scored against player y, summed
across all charted matches they played. This is a per-pair scalar that is
NOT required to be skew-symmetric — its symmetric part captures
rally-length / total-points-played structure, and its antisymmetric part
recovers the F-based signal we've been training on before.

f(x, y) is recovered via two passes: f_hat = g_hat(x, y) - g_hat(y, x),
and compared against prior neural-F baselines.

Usage:
    python examples/tennis/train_g.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from fptajax import hierarchical_g_fpta, TrainConfig

from loader import build_tennis_dataset  # type: ignore


# ---------------------------------------------------------------------------
# Config (matches prior tennis runs — same encoder, same split, same seed)
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
SKILL_DIM = 4

N_STEPS = 2000
BATCH_SIZE = 8
G_SAMPLE = 4
G_SAMPLE_EVAL = 20
LR = 5e-4
ORTHO_WEIGHT = 0.1
C_CORRECTION_EVERY = 300

# Prior F-based baselines from /tmp/tennis_run5.log for comparison.
# IMPORTANT: these are MSE on F = (pts_i - pts_j) / total_pts (range [-0.5, 0.5]),
# not on g. We'll compute an equivalent F-MSE for the g-model via two passes
# through the trained network.
F_BASELINE = {
    "null_train": 0.015918, "null_test": 0.015348,
    "rf_train":   0.002753, "rf_test":   0.015117,
    "fpta_train": 0.014711, "fpta_test": 0.014651,
    "mlp_train":  0.014594, "mlp_test":  0.014405,
    "skill_train": 0.014371, "skill_test": 0.014533,
    # Reliable subset (>= 3 matches, 46/168 test pairs)
    "null_rel":   0.007353,
    "rf_rel":     0.005692,
    "fpta_rel":   0.006050,
    "mlp_rel":    0.005899,
    "skill_rel":  0.005860,
}


def main():
    print("=" * 72)
    print("Direct-g-prediction Hierarchical FPTA on Tennis")
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

    # ---- Train/test split — identical to train.py ----
    print("\n[2] Splitting observed pairs into train/test (seed 0, 80/20)...")
    observed = np.array(ds.observed_pairs)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pair_tuples = observed[perm[:split]]
    test_pair_tuples = observed[perm[split:]]
    train_pairs = np.array([i * N + j for i, j in train_pair_tuples], dtype=np.int64)
    test_pairs = np.array([i * N + j for i, j in test_pair_tuples], dtype=np.int64)
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # ---- Build G label matrix ----
    # G[i, j] = total points i scored vs j across all their matches.
    # Unobserved pairs are zero and will not be used (only train/test pairs
    # from observed_pairs are sampled during training).
    G = ds.pair_points_scored.astype(np.float32)
    print(f"\n[3] G label statistics (observed pairs):")
    obs_mask = ds.pair_counts > 0
    g_obs = G[obs_mask]
    print(f"  shape: {G.shape}")
    print(f"  observed entries: {obs_mask.sum()}")
    print(f"  mean/std/min/max: "
          f"{g_obs.mean():.1f} / {g_obs.std():.1f} / {g_obs.min():.0f} / {g_obs.max():.0f}")

    # ---- Train ----
    print("\n[4] Training hierarchical g-FPTA...")
    print("    g_hat(x, y) = c_0^T s(x) + c_1^T s(y) + B(x)^T C B(y) + bias")
    config = TrainConfig(
        lr=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        ortho_weight=ORTHO_WEIGHT, ridge_lambda=1e-4,
        c_correction_every=C_CORRECTION_EVERY,
        grad_clip=1.0, log_every=100,
    )
    result = hierarchical_g_fpta(
        ds.agent_games, ds.agent_token_mask, ds.agent_game_mask,
        jnp.array(G),
        token_dim=ds.token_dim, L_max=ds.L_max,
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
    print("RESULTS — tennis g-FPTA")
    print("=" * 72)

    def final(hist, key):
        for r in reversed(hist):
            if key in r:
                return r[key]
        return None
    g_train_mse = final(result.train_history, "train_mse")
    g_test_mse = final(result.train_history, "test_mse")
    print(f"\n(g-space) Final train MSE: {g_train_mse:.4f}")
    print(f"(g-space) Final test  MSE: {g_test_mse:.4f}")
    g_test_std = float(np.std([G[i, j] for i, j in test_pair_tuples]))
    print(f"(g-space) Test label std:  {g_test_std:.4f}  (so R² ≈ "
          f"{1 - g_test_mse / g_test_std**2:.3f})")

    # Effective c_0, c_1, and the antisymmetric coefficient
    c0 = np.asarray(result.c_0); c1 = np.asarray(result.c_1)
    print(f"\nLearned coefficients:")
    print(f"  c_0:        {c0}")
    print(f"  c_1:        {c1}")
    print(f"  c_0 - c_1:  {c0 - c1}    (antisymmetric skill direction)")
    print(f"  c_0 + c_1:  {c0 + c1}    (symmetric skill direction)")
    print(f"  ||c_0-c_1||={np.linalg.norm(c0-c1):.3f},  "
          f"||c_0+c_1||={np.linalg.norm(c0+c1):.3f}")

    # Compute F on the full agent grid via two passes
    games_j = jnp.array(ds.agent_games)
    tmask_j = jnp.array(ds.agent_token_mask)
    gmask_j = jnp.array(ds.agent_game_mask)
    F_pred = np.asarray(result.predict_f(
        games_j, tmask_j, gmask_j, games_j, tmask_j, gmask_j,
    ))
    # Scale: g is in "points" units; F in tennis is (pts_i - pts_j) / total_pts.
    # predict_f returns (pts_i - pts_j) predicted counts. To compare to F we
    # divide each pair by the observed total points played between them.
    pts_total = ds.pair_points_total
    F_pred_ratio = np.zeros_like(F_pred)
    mask = pts_total > 0
    F_pred_ratio[mask] = F_pred[mask] / pts_total[mask]

    def pair_mse(F_pred_mat, pairs):
        i = pairs[:, 0]; j = pairs[:, 1]
        y = np.array([ds.F[a, b] for a, b in pairs], dtype=np.float32)
        return float(np.mean((y - F_pred_mat[i, j]) ** 2))

    f_train_mse = pair_mse(F_pred_ratio, train_pair_tuples)
    f_test_mse = pair_mse(F_pred_ratio, test_pair_tuples)
    print(f"\n(F-space, for comparison against prior F-models)")
    print(f"  Train F MSE: {f_train_mse:.6f}")
    print(f"  Test  F MSE: {f_test_mse:.6f}")

    # Effective-scalar-skill ranking (the (c_0 - c_1)^T s(x) axis that
    # controls f — the "Elo-equivalent" of the g-model).
    eff_skill = np.asarray(result.effective_scalar_skill(games_j, tmask_j, gmask_j))
    s_vec = np.asarray(result.skills(games_j, tmask_j, gmask_j))
    print(f"\n[5] Effective scalar skill (c_0-c_1)^T s(x) — top 15 / bottom 10:")
    order = np.argsort(-eff_skill)
    for rank, i in enumerate(order[:15], 1):
        print(f"  {rank:>3}. {ds.player_names[i]:<32} s_eff = {eff_skill[i]:+.4f}")
    print("  ...")
    for rank, i in enumerate(order[-10:], len(order) - 9):
        print(f"  {rank:>3}. {ds.player_names[i]:<32} s_eff = {eff_skill[i]:+.4f}")

    # Variance decomposition
    vd_g = result.variance_decomposition_g(
        games_j, tmask_j, gmask_j, pair_tuples=observed,
    )
    vd_f = result.variance_decomposition_f(
        games_j, tmask_j, gmask_j, pair_tuples=observed,
    )
    print(f"\n[6] Variance decomposition of predictions on observed pairs:")
    print(f"  g-space (observed pairs):")
    print(f"    skill_var = {vd_g['skill_var']:.3f}  "
          f"(c_0·s(x) + c_1·s(y))")
    print(f"    bilinear_var = {vd_g['bilinear_var']:.3f}  "
          f"(B^T C B)")
    print(f"    total_var    = {vd_g['total_var']:.3f}")
    print(f"    skill_frac = {vd_g['skill_frac']*100:.1f}%  "
          f"bilinear_frac = {vd_g['bilinear_frac']*100:.1f}%")
    print(f"  f-space (antisymmetric recovery):")
    print(f"    skill_var = {vd_f['skill_var']:.4f}")
    print(f"    disc_var  = {vd_f['disc_var']:.4f}")
    print(f"    skill_frac = {vd_f['skill_frac']*100:.1f}%  "
          f"disc_frac = {vd_f['disc_frac']*100:.1f}%")

    # Per-contribution MSE on test pairs (f-space)
    f_skill, f_disc, f_total = result.decompose_f(games_j, tmask_j, gmask_j)
    f_skill = np.asarray(f_skill); f_disc = np.asarray(f_disc); f_total = np.asarray(f_total)
    # Convert to ratio form (divide by pts_total)
    def ratio(mat):
        out = np.zeros_like(mat)
        out[mask] = mat[mask] / pts_total[mask]
        return out
    f_skill_ratio = ratio(f_skill)
    f_disc_ratio = ratio(f_disc)
    f_total_ratio = ratio(f_total)
    print(f"\n[7] Test F-MSE of each contribution (f-space, ratio normalisation):")
    print(f"  skill only: {pair_mse(f_skill_ratio, test_pair_tuples):.6f}")
    print(f"  disc only:  {pair_mse(f_disc_ratio, test_pair_tuples):.6f}")
    print(f"  combined:   {pair_mse(f_total_ratio, test_pair_tuples):.6f}")

    # ---- Comparison with prior F-based baselines ----
    print("\n" + "=" * 72)
    print("COMPARISON — tennis F-MSE across all tennis models")
    print("=" * 72)
    null_t = F_BASELINE["null_test"]
    print(f"\n{'Model':<42} {'Train F-MSE':>12} {'Test F-MSE':>12} {'Δ null':>8}")
    print("-" * 80)
    rows = [
        ("Null (predict train mean)",         F_BASELINE["null_train"], F_BASELINE["null_test"]),
        ("Random Forest (handcrafted)",       F_BASELINE["rf_train"],   F_BASELINE["rf_test"]),
        ("FPTA baseline",                      F_BASELINE["fpta_train"], F_BASELINE["fpta_test"]),
        ("Encoder + antisym MLP head",        F_BASELINE["mlp_train"],  F_BASELINE["mlp_test"]),
        ("Hierarchical skill + disc (f-sup)", F_BASELINE["skill_train"], F_BASELINE["skill_test"]),
        ("Direct g-prediction FPTA (g-sup)",  f_train_mse,               f_test_mse),
    ]
    for name, tr, te in rows:
        gain = (1 - te / null_t) * 100
        print(f"{name:<42} {tr:>12.6f} {te:>12.6f} {gain:>+7.2f}%")

    # Reliable subset (>= 3 matches)
    test_mc = np.array([ds.pair_counts[i, j] for i, j in test_pair_tuples],
                       dtype=np.int32)
    mask_rel = test_mc >= 3
    n_rel = int(mask_rel.sum())
    if n_rel > 0:
        f_rel = pair_mse(F_pred_ratio, test_pair_tuples[mask_rel])
        print(f"\n--- Reliable test subset (>= 3 matches, {n_rel}/{len(mask_rel)} pairs) ---")
        null_r = F_BASELINE["null_rel"]
        rel_rows = [
            ("Null",                         F_BASELINE["null_rel"]),
            ("Random Forest",                F_BASELINE["rf_rel"]),
            ("FPTA baseline",                F_BASELINE["fpta_rel"]),
            ("Encoder + antisym MLP head",   F_BASELINE["mlp_rel"]),
            ("Hierarchical skill + disc",    F_BASELINE["skill_rel"]),
            ("Direct g-prediction FPTA",     f_rel),
        ]
        for name, te in rel_rows:
            gain = (1 - te / null_r) * 100
            print(f"  {name:<42} {te:>12.6f}  ({gain:+.2f}% vs null)")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from fptajax.viz import plot_pta_embedding
        from fptajax.pta import PTAResult

        # (1) MSE curves (g-space)
        hist = result.train_history
        steps = [r["step"] for r in hist if "train_mse" in r]
        tr = [r["train_mse"] for r in hist if "train_mse" in r]
        te = [r["test_mse"] for r in hist if "test_mse" in r]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, tr, label="Train MSE (g)", linewidth=2)
        ax.plot(steps, te, label="Test MSE (g)", linewidth=2, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step"); ax.set_ylabel("g-MSE (log scale)")
        ax.set_title("Tennis g-FPTA — Train/Test MSE (g-space, in point-count units)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        out = OUTPUT_DIR / "tennis_g_mse.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # (2) Disc-game embeddings (centered) from the skew-sym part of C
        short = [n.split()[-1] for n in ds.player_names]
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
        fig.suptitle("Tennis Disc-game Embeddings (g-FPTA) — centered",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        out = OUTPUT_DIR / "tennis_g_disc_games.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (3) Effective skill ranking
        fig, ax = plt.subplots(figsize=(11, 9))
        order_plot = np.argsort(eff_skill)
        y_pos = np.arange(len(order_plot))
        colours = ["C2" if eff_skill[i] > 0 else "C3" for i in order_plot]
        ax.barh(y_pos, eff_skill[order_plot], color=colours, alpha=0.85,
                edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([ds.player_names[i] for i in order_plot], fontsize=8)
        ax.set_xlabel("Effective scalar skill  $(c_0-c_1)^\\top s(x)$")
        ax.set_title("Tennis g-FPTA — effective scalar skill that controls f")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        out = OUTPUT_DIR / "tennis_g_effective_skill.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # (4) g-decomposition heatmap on full grid, rows/cols sorted by
        # effective skill
        g_sx, g_sy, g_bl, g_b, g_tot = result.decompose_g(games_j, tmask_j, gmask_j)
        g_sx = np.asarray(g_sx); g_sy = np.asarray(g_sy)
        g_bl = np.asarray(g_bl); g_b = np.asarray(g_b); g_tot = np.asarray(g_tot)
        order_by_skill = np.argsort(-eff_skill)
        G_re  = G[np.ix_(order_by_skill, order_by_skill)]
        tot_re = g_tot[np.ix_(order_by_skill, order_by_skill)]
        sx_re  = g_sx[np.ix_(order_by_skill, order_by_skill)]
        sy_re  = g_sy[np.ix_(order_by_skill, order_by_skill)]
        bl_re  = g_bl[np.ix_(order_by_skill, order_by_skill)]
        sym_re = 0.5 * (G_re + G_re.T)  # observed symmetric part (where observed)
        asym_re = 0.5 * (G_re - G_re.T)
        vmax = float(max(abs(G_re).max(), abs(tot_re).max()))
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
        for ax, mat, title in zip(
            axes,
            [G_re, tot_re, sx_re + sy_re, bl_re],
            ["True g (points scored)", "Predicted g", "g skill part  (c_0·s(x) + c_1·s(y))",
             "g bilinear part  (B^T C B)"],
        ):
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        fig.suptitle("Tennis — g (raw points) decomposition, "
                     "rows/cols sorted by effective skill", fontsize=12)
        out = OUTPUT_DIR / "tennis_g_decomposition.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    main()
