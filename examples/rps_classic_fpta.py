#!/usr/bin/env python3
"""Classical (non-neural) FPTA on RPS using hand-crafted basis functions.

Since RPS's expected per-round payoff is literally a bilinear form in the
two bots' action distributions,

    E[F(i, j)]  =  p_i^T M p_j,    M = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],

a tiny linear basis in (p_R, p_P, p_S) should be able to recover F with very
high fidelity — if bots played exactly-mixed strategies. In practice bots are
reactive, so we also test richer bases:

  1. Self action probs                      (3-d)    canonical RPS basis
  2. Self + opp action probs                (6-d)    adds opponent context
  3. Degree-2 monomials on self probs       (10-d)   captures within-bot nonlinearity
  4. Degree-2 monomials on self + opp       (28-d)   full product basis

For each:
  * Gram-Schmidt-orthonormalise the basis w.r.t. the uniform measure over bots
  * Fit skew-symmetric C via ridge least-squares on TRAIN pairs
  * Evaluate train and test MSE
  * Schur-decompose C to pull out disc games; plot top-k embeddings

Usage:
    python examples/rps_classic_fpta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np

from fptajax.decomposition import skew_symmetric_schur


# ---------------------------------------------------------------------------
# Config / paths (match rps_interventions.py)
# ---------------------------------------------------------------------------

TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)
OUTPUT_DIR = _HERE

RIDGE = 1e-4                 # ridge for skew-symmetric LS
GRAM_RIDGE = 1e-8            # tiny ridge for Gram-Schmidt stability
RESULTS_FOR_PLOTS = 3        # plot up to this many disc games per basis


# ---------------------------------------------------------------------------
# Data: F matrix and per-bot action distributions
# ---------------------------------------------------------------------------


def load_rps_F_and_probs():
    """Load F and per-bot feature tables (self probs, opp probs, and temporal
    statistics: self-transition matrix, response-to-opp matrix, self entropy).

    Returns a dict with keys:
        bot_names, F, self_probs, opp_probs, self_trans, resp_to_opp,
        self_entropy, pair_counts.
    """
    assert TOURNAMENT_DATA.exists()
    matches = [json.loads(l) for l in open(TOURNAMENT_DATA)]
    bots = sorted({b for m in matches for b in (m["player1"], m["player2"])})
    name_to_idx = {n: i for i, n in enumerate(bots)}
    N = len(bots)

    wins = np.zeros((N, N))
    counts = np.zeros((N, N))
    pair_counts = np.zeros((N, N), dtype=np.int64)

    self_counts = np.zeros((N, 3))                # action counts per bot
    opp_counts = np.zeros((N, 3))                  # opp-vs-bot action counts
    self_trans_counts = np.zeros((N, 3, 3))        # [i, a_{t-1}, a_t]
    resp_to_opp_counts = np.zeros((N, 3, 3))       # [i, opp_{t-1}, my_t]

    for m in matches:
        i, j = name_to_idx[m["player1"]], name_to_idx[m["player2"]]
        a1 = np.array(m["player1_actions"])
        a2 = np.array(m["player2_actions"])

        wins[i, j] += np.sum((a1 - a2) % 3 == 1)
        wins[j, i] += np.sum((a2 - a1) % 3 == 1)
        counts[i, j] += len(a1); counts[j, i] += len(a1)
        pair_counts[i, j] += 1; pair_counts[j, i] += 1

        for a in range(3):
            self_counts[i, a] += np.sum(a1 == a)
            self_counts[j, a] += np.sum(a2 == a)
            opp_counts[i, a] += np.sum(a2 == a)
            opp_counts[j, a] += np.sum(a1 == a)

        # Temporal counts — one transition per t>=1
        for t in range(1, len(a1)):
            self_trans_counts[i, a1[t-1], a1[t]] += 1
            self_trans_counts[j, a2[t-1], a2[t]] += 1
            resp_to_opp_counts[i, a2[t-1], a1[t]] += 1
            resp_to_opp_counts[j, a1[t-1], a2[t]] += 1

    denom = np.maximum(counts, 1)
    F = ((wins - wins.T) / denom).astype(np.float64)
    F = 0.5 * (F - F.T)

    self_probs = self_counts / np.maximum(self_counts.sum(axis=1, keepdims=True), 1)
    opp_probs = opp_counts / np.maximum(opp_counts.sum(axis=1, keepdims=True), 1)

    # Row-normalise transition / response matrices to conditional probabilities.
    def _row_normalise(mats):
        out = mats.copy()
        s = out.sum(axis=-1, keepdims=True)
        out = np.where(s > 0, out / np.maximum(s, 1), 1.0 / 3)  # fallback uniform
        return out
    self_trans = _row_normalise(self_trans_counts)            # (N, 3, 3)
    resp_to_opp = _row_normalise(resp_to_opp_counts)          # (N, 3, 3)

    # Shannon entropy of each bot's marginal action distribution (nats)
    eps = 1e-12
    self_entropy = -np.sum(self_probs * np.log(self_probs + eps), axis=1)  # (N,)

    return {
        "bot_names": bots,
        "F": F,
        "self_probs": self_probs,
        "opp_probs": opp_probs,
        "self_trans": self_trans,
        "resp_to_opp": resp_to_opp,
        "self_entropy": self_entropy,
        "pair_counts": pair_counts,
    }


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------


def basis_self_probs(stats):
    """3-d basis: (p_R, p_P, p_S)."""
    return stats["self_probs"].copy()


def basis_self_and_opp(stats):
    """6-d basis: concat(self, opp) action probs."""
    return np.concatenate([stats["self_probs"], stats["opp_probs"]], axis=1)


def basis_deg2_self(stats):
    """10-d basis: all monomials in self probs up to degree 2.

    {1, p_R, p_P, p_S, p_R^2, p_P^2, p_S^2, p_Rp_P, p_Rp_S, p_Pp_S}
    """
    p_self = stats["self_probs"]
    N = p_self.shape[0]
    pR, pP, pS = p_self[:, 0], p_self[:, 1], p_self[:, 2]
    cols = [
        np.ones(N),
        pR, pP, pS,
        pR ** 2, pP ** 2, pS ** 2,
        pR * pP, pR * pS, pP * pS,
    ]
    return np.stack(cols, axis=1)


def basis_deg2_self_and_opp(stats):
    """28-d basis: all monomials in (self, opp) probs up to degree 2.

    Let x = [pR, pP, pS, qR, qP, qS]. Features = {1} ∪ {x_a} ∪ {x_a * x_b : a <= b}.
    """
    p_self, p_opp = stats["self_probs"], stats["opp_probs"]
    N = p_self.shape[0]
    x = np.concatenate([p_self, p_opp], axis=1)  # (N, 6)
    cols = [np.ones(N)]
    cols.extend(x[:, a] for a in range(6))
    for a in range(6):
        for b in range(a, 6):
            cols.append(x[:, a] * x[:, b])
    return np.stack(cols, axis=1)


def basis_temporal(stats):
    """22-d basis: temporal reactivity features per bot.

      - self marginal action distribution:   (p_R, p_P, p_S)          (3)
      - self-transition matrix P(a_t | a_{t-1}) flattened              (9)
      - response-to-opp matrix P(my_t | opp_{t-1}) flattened           (9)
      - self Shannon entropy                                           (1)

    The transition / response matrices capture REACTIVITY, which is the
    signal that aggregate action distributions miss when bots play
    state-dependent policies that average near uniform.
    """
    p_self = stats["self_probs"]
    self_trans = stats["self_trans"]       # (N, 3, 3)
    resp_to_opp = stats["resp_to_opp"]     # (N, 3, 3)
    ent = stats["self_entropy"][:, None]   # (N, 1)
    N = p_self.shape[0]
    cols = [
        p_self,                             # 3
        self_trans.reshape(N, 9),           # 9
        resp_to_opp.reshape(N, 9),          # 9
        ent,                                # 1
    ]
    return np.concatenate(cols, axis=1)


BASES = [
    ("self_probs (3-d)", basis_self_probs),
    ("self + opp probs (6-d)", basis_self_and_opp),
    ("deg-2 monomials on self (10-d)", basis_deg2_self),
    ("deg-2 monomials on self+opp (28-d)", basis_deg2_self_and_opp),
    ("temporal reactivity (22-d)", basis_temporal),
]


# ---------------------------------------------------------------------------
# Orthogonalise + fit skew-symmetric C
# ---------------------------------------------------------------------------


def orthonormalise(B, ridge=GRAM_RIDGE):
    """Return B' with (1/N) B'^T B' ≈ I, by whitening through Gram inverse-sqrt.

    If B has linearly-dependent columns (rank deficient), the small ridge in
    the inverse-sqrt keeps things well-conditioned; the effective rank of B'
    stays the same but no column blows up.
    """
    N = B.shape[0]
    gram = B.T @ B / N
    # Symmetric inverse sqrt via eigen-decomp
    eigvals, eigvecs = np.linalg.eigh(gram + ridge * np.eye(gram.shape[0]))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
    return B @ inv_sqrt


def fit_skew_C(B, F, train_pairs, ridge=RIDGE):
    """Least-squares fit of skew-symmetric C minimising MSE on train pairs.

    prediction: f_hat(i, j) = B[i] @ C @ B[j]
    unknowns:   vec(C) of size d*d, with C = -C^T enforced by symmetrisation
                of the solution (ridge LS on the full d*d space, then skew-proj).
    """
    d = B.shape[1]
    bi = B[train_pairs[:, 0]]            # (P, d)
    bj = B[train_pairs[:, 1]]            # (P, d)
    f_train = F[train_pairs[:, 0], train_pairs[:, 1]]
    # Design matrix: row for pair p is bi[p] ⊗ bj[p]
    M = (bi[:, :, None] * bj[:, None, :]).reshape(-1, d * d)
    MtM = M.T @ M + ridge * np.eye(d * d)
    Mtf = M.T @ f_train
    c_vec = np.linalg.solve(MtM, Mtf)
    C = c_vec.reshape(d, d)
    return 0.5 * (C - C.T)                # project to skew-symmetric cone


def predict(B, C, pairs):
    bi = B[pairs[:, 0]]
    bj = B[pairs[:, 1]]
    return np.sum((bi @ C) * bj, axis=-1)


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class BasisResult:
    name: str
    d: int                  # orthonormal dim
    train_mse: float
    test_mse: float
    n_components: int
    eigenvalues: np.ndarray
    embeddings: np.ndarray  # (N, n_components, 2)
    C: np.ndarray


def run():
    print("=" * 72)
    print("Classical FPTA on RPS — multiple basis choices")
    print("=" * 72)

    print("\n[1] Loading RPS data + computing action distributions + temporal stats...")
    stats = load_rps_F_and_probs()
    bots = stats["bot_names"]
    F = stats["F"]
    p_self = stats["self_probs"]
    p_opp = stats["opp_probs"]
    pair_counts = stats["pair_counts"]
    N = len(bots)
    print(f"  N = {N} bots")
    print(f"  F range: [{F.min():+.3f}, {F.max():+.3f}], std = {F.std():.3f}")
    print(f"  Sample action distributions (first 5 bots):")
    print(f"  {'bot':<40} {'p_R':>6} {'p_P':>6} {'p_S':>6} {'entropy':>8}")
    for i in range(5):
        print(f"  {bots[i]:<40} {p_self[i,0]:>6.3f} {p_self[i,1]:>6.3f} "
              f"{p_self[i,2]:>6.3f} {stats['self_entropy'][i]:>8.3f}")
    print(f"  Entropy range across bots: "
          f"[{stats['self_entropy'].min():.3f}, {stats['self_entropy'].max():.3f}] nats "
          f"(uniform = {np.log(3):.3f})")

    # Same 80/20 split as all other RPS experiments
    print("\n[2] Building 80/20 train/test split of observed pairs (seed 0)...")
    observed = np.argwhere(pair_counts > 0)
    observed = np.array([[i, j] for i, j in observed if i != j])
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(observed))
    split = int(0.8 * len(observed))
    train_pairs = observed[perm[:split]]
    test_pairs = observed[perm[split:]]
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    y_train = F[train_pairs[:, 0], train_pairs[:, 1]]
    y_test = F[test_pairs[:, 0], test_pairs[:, 1]]
    y_train_mean = float(np.mean(y_train))
    null_train_mse = float(np.mean((y_train - y_train_mean) ** 2))
    null_test_mse = float(np.mean((y_test - y_train_mean) ** 2))
    print(f"  Null MSE (predict train mean={y_train_mean:+.4f}):  "
          f"train={null_train_mse:.5f}, test={null_test_mse:.5f}")

    print("\n[3] Fitting classical FPTA for each basis...\n")
    results: list[BasisResult] = []
    for name, builder in BASES:
        B_raw = builder(stats)
        d_raw = B_raw.shape[1]
        B = orthonormalise(B_raw)
        # Effective rank — drop basis columns that are numerically zero after whitening
        eff = int(np.sum(np.linalg.norm(B, axis=0) > 1e-6))
        C = fit_skew_C(B, F, train_pairs, ridge=RIDGE)
        pred_train = predict(B, C, train_pairs)
        pred_test = predict(B, C, test_pairs)
        tr = mse(pred_train, y_train)
        te = mse(pred_test, y_test)

        # Schur decomposition for disc games
        schur = skew_symmetric_schur(np.asarray(C))
        nc = int(schur.n_components)
        eigs = np.asarray(schur.eigenvalues)
        Q = np.asarray(schur.Q)

        # Disc game embeddings: Y^(k)(i) = sqrt(omega_k) * B[i] @ [q_{2k-1}, q_{2k}]
        embeddings = np.zeros((N, nc, 2))
        for k in range(nc):
            q1 = Q[:, 2 * k]
            q2 = Q[:, 2 * k + 1]
            omega_k = float(eigs[k])
            scale = np.sqrt(max(omega_k, 0.0))
            embeddings[:, k, 0] = scale * (B @ q1)
            embeddings[:, k, 1] = scale * (B @ q2)

        results.append(BasisResult(
            name=name, d=B.shape[1],
            train_mse=tr, test_mse=te,
            n_components=nc,
            eigenvalues=eigs,
            embeddings=embeddings,
            C=C,
        ))

        print(f"  {name}:")
        print(f"     dim={d_raw} (effective {eff})  train MSE={tr:.5f}  test MSE={te:.5f}")
        print(f"     {nc} disc games, top eigenvalues: "
              f"{np.array2string(eigs[:min(5, nc)], precision=4)}")
        # Importance fraction: each disc game contributes 2 ω_k^2 to ||C||_F^2
        total = float(2.0 * np.sum(eigs ** 2))
        imps = 2.0 * eigs ** 2 / max(total, 1e-12)
        print(f"     disc-game importance: "
              f"{np.array2string(imps[:min(5, nc)] * 100, precision=2, suppress_small=True)}%")
        print()

    # ---------------------------------------------------------------------
    # Comparison table with Null, RF, neural-FPTA (RPS baseline)
    # ---------------------------------------------------------------------
    print("=" * 72)
    print("COMPARISON — RPS")
    print("=" * 72)
    # Reference numbers from rps_interventions.log
    REF = {
        "RF (handcrafted)":       (0.008067, 0.023655),
        "Neural FPTA baseline":   (0.052141, 0.051439),
        "Neural encoder + MLP":   (0.049378, 0.047537),
    }
    print(f"\n{'Model':<44} {'Train MSE':>12} {'Test MSE':>12} {'Δ null':>8}")
    print("-" * 80)
    print(f"{'Null (predict train mean)':<44} "
          f"{null_train_mse:>12.5f} {null_test_mse:>12.5f} {0:>+7.2f}%")
    for r in results:
        gain = (1 - r.test_mse / null_test_mse) * 100
        print(f"{'Classical FPTA — ' + r.name:<44} "
              f"{r.train_mse:>12.5f} {r.test_mse:>12.5f} {gain:>+7.2f}%")
    for name, (tr, te) in REF.items():
        gain = (1 - te / null_test_mse) * 100
        print(f"{name:<44} {tr:>12.5f} {te:>12.5f} {gain:>+7.2f}%")

    # ---------------------------------------------------------------------
    # Disc-game plots
    # ---------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot: for each basis, the top disc games as scatter plots
        # Use short labels so the scatter is readable
        short_names = [b.split("_")[-1] if "_" in b else b for b in bots]

        # Figure 1: 4 rows (bases) x RESULTS_FOR_PLOTS cols (disc games), all centered
        n_cols = RESULTS_FOR_PLOTS
        fig, axes = plt.subplots(
            len(results), n_cols, figsize=(5 * n_cols, 4.2 * len(results)),
            squeeze=False,
        )
        for row_i, r in enumerate(results):
            Y = r.embeddings
            n_use = min(n_cols, r.n_components)
            for k in range(n_cols):
                ax = axes[row_i, k]
                if k >= n_use:
                    ax.axis("off")
                    ax.set_title(f"(no component {k+1})")
                    continue
                Yk = Y[:, k, :] - Y[:, k, :].mean(axis=0, keepdims=True)
                # Colour points by dominant action probability, just for interpretability
                colours = p_self  # (N, 3) → RGB
                ax.scatter(Yk[:, 0], Yk[:, 1], c=colours, s=65,
                           edgecolors="black", linewidths=0.4, alpha=0.9)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
                omega_k = float(r.eigenvalues[k])
                total_omega2 = 2.0 * float(np.sum(r.eigenvalues ** 2))
                imp = (2.0 * omega_k ** 2 / max(total_omega2, 1e-12)) * 100
                ax.set_title(f"{r.name}\nDisc game {k+1}   ω={omega_k:.4f}   imp={imp:.1f}%",
                             fontsize=10)
                ax.set_xlabel(f"$Y^{{({k+1})}}_1$")
                ax.set_ylabel(f"$Y^{{({k+1})}}_2$")
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal")
        fig.suptitle("Classical FPTA on RPS — disc-game embeddings (points coloured by RGB = (p_R, p_P, p_S))",
                     fontsize=12, y=1.005)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_classic_disc_games.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")

        # Figure 2: overlay of top disc game from each basis, with bot labels
        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        for ax, r in zip(axes.flatten(), results):
            Y0 = r.embeddings[:, 0, :] - r.embeddings[:, 0, :].mean(axis=0)
            ax.scatter(Y0[:, 0], Y0[:, 1], c=p_self, s=90,
                       edgecolors="black", linewidths=0.5, alpha=0.9)
            for i, name in enumerate(bots):
                ax.annotate(short_names[i], Y0[i],
                            fontsize=6.5,
                            textcoords="offset points", xytext=(3, 2))
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
            omega0 = float(r.eigenvalues[0])
            total_omega2 = 2.0 * float(np.sum(r.eigenvalues ** 2))
            imp = (2.0 * omega0 ** 2 / max(total_omega2, 1e-12)) * 100
            ax.set_title(f"{r.name}\nDisc game 1   ω={omega0:.4f}   imp={imp:.1f}%   "
                         f"test MSE={r.test_mse:.5f}",
                         fontsize=10)
            ax.set_xlabel("$Y^{(1)}_1$")
            ax.set_ylabel("$Y^{(1)}_2$")
        fig.suptitle(
            "RPS Classical FPTA — disc game 1 by basis choice (RGB = (p_R, p_P, p_S))",
            fontsize=12, y=1.005,
        )
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_classic_disc_game1_labeled.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        # Figure 3: MSE bar chart comparing all methods
        fig, ax = plt.subplots(figsize=(11, 5))
        names = ["Null"]
        tests = [null_test_mse]
        for r in results:
            names.append("Classical FPTA: " + r.name)
            tests.append(r.test_mse)
        for nm, (_, te) in REF.items():
            names.append(nm); tests.append(te)
        x = np.arange(len(names))
        bars = ax.bar(x, tests, color=["gray"] + ["C0"] * len(results) + ["C2"] * len(REF))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Test MSE")
        ax.set_title("RPS — Test MSE across methods")
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, tests):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out = OUTPUT_DIR / "rps_classic_vs_neural_bar.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

        plt.close("all")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    print("\nDone!")


if __name__ == "__main__":
    run()
