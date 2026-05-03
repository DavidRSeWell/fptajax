"""RPS bot dataset for the classical FPTA suite.

Wraps the data-loading + per-bot statistics from
``examples/rps_classic_fpta.py`` into the unified ``FPTADataset`` interface.

Trait = the per-bot summary feature vector the chosen basis exposes.

Bases (carry over from rps_classic_fpta.py):

  - "self_probs"        :  3-d   {p_R, p_P, p_S}
  - "self_and_opp"      :  6-d   self + opp action probabilities
  - "deg2_self"         : 10-d   degree-≤2 monomials in self probs
  - "deg2_self_and_opp" : 28-d   degree-≤2 monomials in (self, opp) probs
  - "temporal"          : 22-d   self probs + transition matrix + response-to-opp + entropy

These bases differ in dimensionality so they're useful for H1/H9/H10
(approximation-rate experiments). For H7 (basis independence), we'd compare
``deg2_self`` with a same-span basis rotated to e.g. Chebyshev features —
the rps_classic_fpta.py file didn't include such a comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples.classical_fpta_suite.protocol import FPTADataset, random_pair_split


_TOURNAMENT_DATA = Path(
    "/Users/davidsewell/Projects/rps_pbt/tournament_results"
    "/20260119_085535/openspiel_tournament_actions.jsonl"
)


# ---------------------------------------------------------------------------
# Loader (mirrors examples/rps_classic_fpta.py:load_rps_F_and_probs)
# ---------------------------------------------------------------------------


def _load_F_and_stats(path: Path = _TOURNAMENT_DATA) -> dict:
    assert path.exists(), f"Tournament data missing at {path}"
    matches = [json.loads(l) for l in open(path)]
    bots = sorted({b for m in matches for b in (m["player1"], m["player2"])})
    name_to_idx = {n: i for i, n in enumerate(bots)}
    N = len(bots)

    wins = np.zeros((N, N))
    counts = np.zeros((N, N))
    self_counts = np.zeros((N, 3))
    opp_counts = np.zeros((N, 3))
    self_trans = np.zeros((N, 3, 3))
    resp_to_opp = np.zeros((N, 3, 3))

    for m in matches:
        i, j = name_to_idx[m["player1"]], name_to_idx[m["player2"]]
        a1 = np.array(m["player1_actions"]); a2 = np.array(m["player2_actions"])
        wins[i, j] += np.sum((a1 - a2) % 3 == 1)
        wins[j, i] += np.sum((a2 - a1) % 3 == 1)
        counts[i, j] += len(a1); counts[j, i] += len(a1)
        for a in range(3):
            self_counts[i, a] += np.sum(a1 == a); self_counts[j, a] += np.sum(a2 == a)
            opp_counts[i, a]  += np.sum(a2 == a); opp_counts[j, a]  += np.sum(a1 == a)
        for t in range(1, len(a1)):
            self_trans[i, a1[t-1], a1[t]] += 1
            self_trans[j, a2[t-1], a2[t]] += 1
            resp_to_opp[i, a2[t-1], a1[t]] += 1
            resp_to_opp[j, a1[t-1], a2[t]] += 1

    denom = np.maximum(counts, 1)
    F = ((wins - wins.T) / denom).astype(np.float64)
    F = 0.5 * (F - F.T)

    self_probs = self_counts / np.maximum(self_counts.sum(axis=1, keepdims=True), 1)
    opp_probs  = opp_counts  / np.maximum(opp_counts.sum (axis=1, keepdims=True), 1)

    def _row_normalise(M):
        s = M.sum(axis=-1, keepdims=True)
        return np.where(s > 0, M / np.maximum(s, 1), 1.0 / 3)
    self_trans  = _row_normalise(self_trans)
    resp_to_opp = _row_normalise(resp_to_opp)

    eps = 1e-12
    self_entropy = -np.sum(self_probs * np.log(self_probs + eps), axis=1)

    return dict(
        bot_names=bots, F=F,
        self_probs=self_probs, opp_probs=opp_probs,
        self_trans=self_trans, resp_to_opp=resp_to_opp,
        self_entropy=self_entropy,
    )


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


def _basis_self_probs(s):     return s["self_probs"].copy()
def _basis_self_and_opp(s):   return np.concatenate([s["self_probs"], s["opp_probs"]], axis=1)


def _basis_deg2_self(s):
    p = s["self_probs"]; N = p.shape[0]
    cols = [
        np.ones(N), p[:, 0], p[:, 1], p[:, 2],
        p[:, 0] ** 2, p[:, 1] ** 2, p[:, 2] ** 2,
        p[:, 0] * p[:, 1], p[:, 0] * p[:, 2], p[:, 1] * p[:, 2],
    ]
    return np.stack(cols, axis=1)


def _basis_deg2_self_and_opp(s):
    p, q = s["self_probs"], s["opp_probs"]; N = p.shape[0]
    x = np.concatenate([p, q], axis=1)
    cols = [np.ones(N)]; cols.extend(x[:, a] for a in range(6))
    for a in range(6):
        for b in range(a, 6):
            cols.append(x[:, a] * x[:, b])
    return np.stack(cols, axis=1)


def _basis_temporal(s):
    p = s["self_probs"]; N = p.shape[0]
    return np.concatenate([
        p,
        s["self_trans"].reshape(N, 9),
        s["resp_to_opp"].reshape(N, 9),
        s["self_entropy"][:, None],
    ], axis=1)


_BASIS_FNS = {
    "self_probs":         _basis_self_probs,
    "self_and_opp":       _basis_self_and_opp,
    "deg2_self":          _basis_deg2_self,
    "deg2_self_and_opp":  _basis_deg2_self_and_opp,
    "temporal":           _basis_temporal,
}


# ---------------------------------------------------------------------------
# Public dataset constructor
# ---------------------------------------------------------------------------


def build_rps(frac_train: float = 0.8, seed: int = 0) -> FPTADataset:
    stats = _load_F_and_stats()
    F = stats["F"]
    N = F.shape[0]
    # The "trait" we expose is the richest feature set (temporal); per-basis
    # eval just slices/rebuilds from the underlying stats anyway.
    traits = _basis_temporal(stats)
    train_pairs, test_pairs = random_pair_split(N, frac_train=frac_train, seed=seed)

    def _basis(name: str, m_target: int) -> np.ndarray:
        if name not in _BASIS_FNS:
            raise KeyError(f"Unknown basis: {name}")
        B_ = _BASIS_FNS[name](stats)
        if 0 < m_target < B_.shape[1]:
            B_ = B_[:, :m_target]
        return B_

    return FPTADataset(
        name="rps",
        traits=traits,
        F=F,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        available_bases=tuple(_BASIS_FNS.keys()),
        basis_fn=_basis,
        eval_f=None,           # f is empirical; no closed form
        metadata=dict(bot_names=stats["bot_names"], **{k: stats[k] for k in
            ("self_probs", "opp_probs", "self_trans", "resp_to_opp", "self_entropy")}),
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ds = build_rps()
    print(
        f"  {ds.name:8s}  N={ds.N}  ||F||²/N²={ds.f_norm_sq:.4f}  "
        f"skew err={np.max(np.abs(ds.F + ds.F.T)):.2e}  "
        f"train/test={len(ds.train_pairs)}/{len(ds.test_pairs)}"
    )
    for bn in ds.available_bases:
        B_ = ds.basis(bn, 1000)
        print(f"      basis {bn:22s}: shape={B_.shape}")
