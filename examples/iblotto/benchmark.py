"""Benchmark: classical FPTA on ground-truth traits vs. behavioural FPTA on trajectories.

Both methods are trained and evaluated on the same train/test pair split of
the *observed* edges in a saved :class:`BlottoBehavioralDataset` bundle.
Classical FPTA uses the agents' ground-truth 5-dim policy parameters
(``learning_rate, win_reinvestment, loss_disinvestment, opponent_allocation,
innovation_noise``) standardised to ``[-1, 1]^5`` with a total-degree
polynomial basis. Behavioural FPTA learns its own trait representation from
``agent_data`` token sequences via the hierarchical skill+disc encoder.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto.benchmark \
        --bundle examples/iblotto/results/behavioral_bench_N50_k15_nr20.pkl \
        --frac_train 0.8 --seed 0
"""

from __future__ import annotations

import argparse
import pickle
import time
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from examples.classical_fpta_suite.protocol import (
    fit_skew_C_train, orthonormalise, predict, pair_mse, truncate_C,
)
from examples.iblotto.behavioral import drop_dead_agents


# ---------------------------------------------------------------------------
# Trait standardisation + polynomial basis
# ---------------------------------------------------------------------------

# Per-axis sampling ranges used by generate_behavioral_data.py
_TRAIT_RANGES = (
    ("learning_rate",        0.10, 0.70),
    ("win_reinvestment",    -2.00, 2.00),
    ("loss_disinvestment",  -2.00, 2.00),
    ("opponent_allocation", -2.00, 2.00),
    ("innovation_noise",     0.01, 0.30),
)


def standardise_traits(policies: np.ndarray) -> np.ndarray:
    """Map the first 5 columns of ``policies`` from their sampling ranges to ``[-1, 1]``."""
    out = np.zeros((policies.shape[0], 5), dtype=np.float64)
    for j, (_, lo, hi) in enumerate(_TRAIT_RANGES):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out[:, j] = (policies[:, j] - mid) / half
    return out


def total_degree_monomials(traits: np.ndarray, max_deg: int) -> tuple[np.ndarray, list[str]]:
    """Total-degree-≤``max_deg`` monomials in ``traits`` (T variables).

    Number of basis functions = C(T + max_deg, max_deg). For T=5 that's
    21 (deg 2), 56 (deg 3), 126 (deg 4).
    """
    N, T = traits.shape
    cols, labels = [], []
    for d in range(max_deg + 1):
        if d == 0:
            cols.append(np.ones(N))
            labels.append("1")
            continue
        for combo in combinations_with_replacement(range(T), d):
            counts = [0] * T
            for k in combo:
                counts[k] += 1
            col = np.ones(N)
            for k, p in enumerate(counts):
                if p > 0:
                    col = col * (traits[:, k] ** p)
            cols.append(col)
            labels.append(" * ".join(f"x{k}^{p}" for k, p in enumerate(counts) if p > 0))
    return np.stack(cols, axis=1), labels


# ---------------------------------------------------------------------------
# Pair splitting on the OBSERVED edge set
# ---------------------------------------------------------------------------


def split_observed_pairs(
    observed_mask: np.ndarray, frac_train: float = 0.8, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed unordered pairs (i<j) into train/test, then symmetrise.

    Returns ``(train_pairs, test_pairs)`` as ``(P, 2)`` arrays containing
    BOTH ordered directions: skew-symmetry means (j, i) carries the same
    info as (i, j) so we always keep them on the same side.
    """
    rng = np.random.RandomState(seed)
    iu = np.argwhere(np.triu(observed_mask, 1))                 # unordered, i<j
    rng.shuffle(iu)
    n_train = int(frac_train * len(iu))
    train_un, test_un = iu[:n_train], iu[n_train:]
    train_pairs = np.concatenate([train_un, train_un[:, [1, 0]]], axis=0)
    test_pairs  = np.concatenate([test_un,  test_un[:, [1, 0]]],  axis=0)
    return train_pairs, test_pairs


def to_flat_indices(pairs: np.ndarray, N: int) -> np.ndarray:
    """Flatten (P, 2) ordered pairs to flat row-major indices into the N×N grid."""
    return pairs[:, 0] * N + pairs[:, 1]


# ---------------------------------------------------------------------------
# Classical FPTA on ground-truth traits
# ---------------------------------------------------------------------------


def run_classical_fpta(
    ds, train_pairs: np.ndarray, test_pairs: np.ndarray,
    max_degrees: Sequence[int] = (2, 3, 4),
    k_truncs: Sequence[int | None] = (None,),
    verbose: bool = True,
) -> list[dict]:
    """Sweep total-degree polynomial bases and report train/test MSE.

    Normalisation is the mean-square F value over the relevant pair set
    (so ``norm_test = test_mse / mean(F[test]^2)``). This keeps the ratio
    interpretable for sparse F matrices where unobserved entries are 0.
    """
    traits = standardise_traits(ds.policies)
    N = ds.policies.shape[0]
    f2_train = float(np.mean(ds.F[train_pairs[:, 0], train_pairs[:, 1]] ** 2))
    f2_test  = float(np.mean(ds.F[test_pairs[:, 0],  test_pairs[:, 1]]  ** 2))

    rows: list[dict] = []
    for d in max_degrees:
        B_raw, labels = total_degree_monomials(traits, max_deg=d)
        m = B_raw.shape[1]
        B = orthonormalise(B_raw)
        # Use the suite's MC projection (= paper's Algorithm 3.1 on observed edges)
        C = fit_skew_C_train(B, ds.F, train_pairs, ridge=0.0)
        F_hat_full = predict(B, C)
        train_mse = pair_mse(ds.F, F_hat_full, train_pairs)
        test_mse  = pair_mse(ds.F, F_hat_full, test_pairs)
        for k_trunc in k_truncs:
            if k_trunc is None:
                C_used = C
                k_used = "full"
                F_hat_k = F_hat_full
            else:
                C_used, _, _ = truncate_C(C, k_trunc)
                k_used = str(k_trunc)
                F_hat_k = predict(B, C_used)
            tr = pair_mse(ds.F, F_hat_k, train_pairs)
            te = pair_mse(ds.F, F_hat_k, test_pairs)
            rows.append(dict(
                method="classical", basis=f"monomial_d{d}", m=int(m),
                k_trunc=k_used,
                train_mse=tr, test_mse=te,
                norm_train=tr / max(f2_train, 1e-12),
                norm_test=te  / max(f2_test,  1e-12),
            ))
            if verbose:
                print(f"  classical d={d:1d}  m={m:3d}  k={k_used:>4s}  "
                      f"train={tr:.4f}  test={te:.4f}  "
                      f"norm_test={te/max(f2_test,1e-12):.4f}")
    return rows


# ---------------------------------------------------------------------------
# Behavioural FPTA on trajectories
# ---------------------------------------------------------------------------


def run_behavioural_fpta(
    ds, train_pairs: np.ndarray, test_pairs: np.ndarray,
    n_steps: int = 8000, batch_size: int = 32, lr: float = 5e-4,
    trait_dim: int = 24, d: int = 12,
    d_model: int = 32, n_layers: int = 1, n_heads: int = 2,
    c_correction_every: int = 1000,
    seed: int = 0, verbose: bool = True,
    save_bfpta_dir: Path | None = None,
) -> dict:
    """Train hierarchical skill + disc-game FPTA on the trajectory tensor."""
    from fptajax.hierarchical_skill import hierarchical_skill_fpta
    from fptajax.neural import TrainConfig

    N = ds.policies.shape[0]
    train_flat = to_flat_indices(train_pairs, N)
    test_flat  = to_flat_indices(test_pairs,  N)

    # We need at minimum 1 valid game per agent — guard
    games_per_agent = ds.agent_game_mask.sum(axis=1)
    if (games_per_agent < 1).any():
        bad = np.where(games_per_agent < 1)[0].tolist()
        raise RuntimeError(f"agents {bad} have 0 games; can't train BFPTA")

    config = TrainConfig(
        lr=lr, n_steps=n_steps, batch_size=batch_size,
        ortho_weight=0.1, ridge_lambda=1e-4,
        c_correction_every=c_correction_every,
        grad_clip=1.0, log_every=500,
        skill_centering_weight=1.0,
    )

    t0 = time.time()
    result = hierarchical_skill_fpta(
        agent_games=ds.agent_data,
        agent_token_mask=ds.agent_token_mask,
        agent_game_mask=ds.agent_game_mask,
        F=ds.F,
        token_dim=ds.sa_dim, L_max=ds.L_max,
        trait_dim=trait_dim, d=d,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        config=config, key=jax.random.PRNGKey(seed),
        train_pairs=train_flat, test_pairs=test_flat,
        eval_every=500, G_sample=4, G_sample_eval=min(8, ds.G_max),
        numpy_seed=seed, verbose=verbose,
    )
    elapsed = time.time() - t0

    # Save trained model artefacts so trait_recovery.py can encode + decompose later.
    if save_bfpta_dir is not None:
        save_bfpta_dir.mkdir(parents=True, exist_ok=True)
        import equinox as eqx
        import pickle as _pickle
        eqx.tree_serialise_leaves(str(save_bfpta_dir / "encoder.eqx"), result.encoder)
        eqx.tree_serialise_leaves(str(save_bfpta_dir / "basis.eqx"),   result.basis)
        meta = dict(
            sa_dim=int(ds.sa_dim), L_max=int(ds.L_max),
            trait_dim=int(trait_dim), d=int(d),
            d_model=int(d_model), n_layers=int(n_layers), n_heads=int(n_heads),
            mlp_ratio=4, rho_hidden=(64,), basis_hidden=(128, 128),
            coefficient_matrix=np.asarray(result.coefficient_matrix),
            eigenvalues=np.asarray(result.eigenvalues),
            schur_vectors=np.asarray(result.schur_vectors),
            n_components=int(result.n_components),
            f_norm_sq=(float(result.f_norm_sq) if result.f_norm_sq is not None else None),
            seed=int(seed), n_steps=int(n_steps),
        )
        with open(save_bfpta_dir / "meta.pkl", "wb") as f:
            _pickle.dump(meta, f)
        if verbose:
            print(f"  saved BFPTA artefacts → {save_bfpta_dir}/")

    # The training history contains per-eval-step train/test MSE. C-corrections
    # can transiently spike the MSE, so we report the BEST (early-stopping)
    # test pair MSE as well as the final-step value, so the user can see both.
    history = result.train_history
    eval_history = [r for r in history if "test_mse" in r]
    final = eval_history[-1] if eval_history else history[-1]
    best  = min(eval_history, key=lambda r: r["test_mse"]) if eval_history else final
    f2_train = float(np.mean(ds.F[train_pairs[:, 0], train_pairs[:, 1]] ** 2))
    f2_test  = float(np.mean(ds.F[test_pairs[:, 0],  test_pairs[:, 1]]  ** 2))
    return dict(
        method="behavioural",
        basis="hierarchical skill+disc",
        m=trait_dim, k_trunc=int(result.n_components),
        train_mse_final=final.get("train_mse", float("nan")),
        test_mse_final=final.get("test_mse",  float("nan")),
        norm_test_final=final.get("test_mse", float("nan")) / max(f2_test, 1e-12),
        train_mse_best=best.get("train_mse", float("nan")),
        test_mse_best=best.get("test_mse",  float("nan")),
        norm_test_best=best.get("test_mse", float("nan")) / max(f2_test, 1e-12),
        best_step=int(best.get("step", -1)),
        eigenvalues=np.asarray(result.eigenvalues).tolist(),
        wall_time_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(
    bundle_path: Path, frac_train: float = 0.8, seed: int = 0,
    bfpta_steps: int = 8000, c_correction_every: int = 1000,
    run_bfpta: bool = True, output_json: Path | None = None,
    save_bfpta_dir: Path | None = None,
):
    print(f"=== Benchmark: classical vs. behavioural FPTA on iblotto ===")
    print(f"  bundle: {bundle_path}")
    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)

    # Defensive: older bundles may still have NaN in agent_data even though
    # their token mask is False. Zero those out so downstream attention
    # doesn't NaN-poison.
    n_nan = int(np.sum(~np.all(np.isfinite(ds.agent_data), axis=-1)))
    if n_nan > 0:
        print(f"  sanitising {n_nan} NaN tokens in agent_data (mask preserved)")
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    # Defensive: older bundles may have agents with zero valid games (the
    # stability filter wasn't strong enough at generation time). Drop them
    # and remap pair indices so the BFPTA training has data to fit.
    ds, dropped = drop_dead_agents(ds, verbose=True)
    if dropped:
        print(f"  agents dropped: {dropped} (original numbering); "
              f"benchmark will run on {ds.policies.shape[0]} survivors")

    N = ds.policies.shape[0]
    n_pairs_obs = int(ds.observed_mask.sum() // 2)
    print(f"  N={N}  observed pairs={n_pairs_obs}/{N*(N-1)//2}  "
          f"sa_dim={ds.sa_dim}  G_max={ds.G_max}  L_max={ds.L_max}")
    print(f"  ‖F‖²/N² = {np.sum(ds.F**2)/(N*N):.4f}")

    # Train/test split on the observed edges
    train_pairs, test_pairs = split_observed_pairs(
        ds.observed_mask, frac_train=frac_train, seed=seed,
    )
    print(f"  train pairs (ordered, both dirs): {len(train_pairs)}  "
          f"test pairs: {len(test_pairs)}")

    print("\n--- Classical FPTA (ground-truth traits + polynomial basis) ---")
    classical_rows = run_classical_fpta(
        ds, train_pairs, test_pairs,
        max_degrees=(2, 3, 4),
        k_truncs=(2, 4, 6, 10, None),
    )

    bfpta_row = None
    if run_bfpta:
        print("\n--- Behavioural FPTA (hierarchical skill+disc on trajectories) ---")
        bfpta_row = run_behavioural_fpta(
            ds, train_pairs, test_pairs,
            n_steps=bfpta_steps, c_correction_every=c_correction_every,
            seed=seed, save_bfpta_dir=save_bfpta_dir,
        )
        r = bfpta_row
        print(f"\n  behavioural  final  step=last  test={r['test_mse_final']:.4f}  "
              f"norm_test={r['norm_test_final']:.4f}")
        print(f"  behavioural  BEST   step={r['best_step']}  test={r['test_mse_best']:.4f}  "
              f"norm_test={r['norm_test_best']:.4f}  "
              f"({r['wall_time_sec']:.1f}s)")

    print("\n=== Summary ===")
    print(f"  {'method':>14s}  {'basis':>22s}  {'m':>4s}  {'k':>5s}  "
          f"{'train':>8s}  {'test':>8s}  {'norm_test':>10s}")
    for r in classical_rows:
        print(f"  {r['method']:>14s}  {r['basis']:>22s}  {r['m']:>4d}  "
              f"{str(r['k_trunc']):>5s}  {r['train_mse']:>8.4f}  "
              f"{r['test_mse']:>8.4f}  {r['norm_test']:>10.4f}")
    if bfpta_row is not None:
        r = bfpta_row
        print(f"  {r['method']+'_final':>14s}  {r['basis']:>22s}  {r['m']:>4d}  "
              f"{str(r['k_trunc']):>5s}  {r['train_mse_final']:>8.4f}  "
              f"{r['test_mse_final']:>8.4f}  {r['norm_test_final']:>10.4f}")
        print(f"  {r['method']+'_best':>14s}  {r['basis']:>22s}  {r['m']:>4d}  "
              f"{str(r['k_trunc']):>5s}  {r['train_mse_best']:>8.4f}  "
              f"{r['test_mse_best']:>8.4f}  {r['norm_test_best']:>10.4f}    "
              f"(@ step {r['best_step']})")

    if output_json is not None:
        import json
        out = dict(
            bundle=str(bundle_path),
            N=int(N), n_pairs_observed=int(n_pairs_obs),
            frac_train=float(frac_train), seed=int(seed),
            f_norm_sq=float(np.sum(ds.F ** 2) / (N * N)),
            classical_rows=classical_rows,
            behavioural=bfpta_row,
        )
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"\nSaved JSON results → {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frac_train", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bfpta_steps", type=int, default=8000)
    parser.add_argument("--c_correction_every", type=int, default=1000)
    parser.add_argument("--no_bfpta", action="store_true",
                        help="skip the behavioural-FPTA run (classical only)")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="optional path to dump the full comparison as JSON")
    parser.add_argument("--save_bfpta", type=Path, default=None,
                        help="optional directory to save the trained BFPTA "
                             "encoder/basis/Schur artefacts (consumed by "
                             "trait_recovery.py)")
    args = parser.parse_args()
    main(args.bundle, args.frac_train, args.seed, args.bfpta_steps,
         c_correction_every=args.c_correction_every,
         run_bfpta=not args.no_bfpta, output_json=args.output_json,
         save_bfpta_dir=args.save_bfpta)
