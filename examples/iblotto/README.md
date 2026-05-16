# Iterated Blotto with autoregressive policies — JAX port

A direct JAX port of the MATLAB code under `Alex_IBlotto/`. The original
code generates iterated Colonel Blotto tournament data among autoregressive
agents and runs PTA on the resulting performance matrix; this port
reproduces the MATLAB outputs to within sampling error and adds JAX-native
vectorisation so the inner loop is JIT-compiled.

## Game

Two players, each with a budget, simultaneously allocate non-negative
real-valued bids across `n_zones` battlefields per round. A **contest
success function (CSF)** decides who wins each zone; payouts equal the sum
of zone-values won (with ties splitting). Players have memory of the past
round and update an **autoregressive policy** that is biased to:

1. reinvest in zones they won (or disinvest, if `win_reinvestment` is negative),
2. disinvest from zones they lost (or reinvest, if `loss_disinvestment` is negative),
3. pursue or avoid the opponent's last-round allocation,
4. plus Gaussian innovation noise.

Policies are 6-dimensional parameter vectors

```
[learning_rate, win_reinvestment, loss_disinvestment,
 opponent_allocation, innovation_noise, concentration]
```

and the per-agent policy state is an `(n_zones + 1)`-vector summing to
`concentration`: the first `n_zones` entries are zone weights and the last
is a "no-invest" weight that controls the per-round investment fraction
(held fixed across rounds in this code).

After 100 rounds, the cumulative payouts to each player are recorded; the
**performance matrix** `F[i, j] = E[payout_i − payout_j]` averaged over
many independent realisations is then fed to PTA.

## File map

| MATLAB | JAX port |
|---|---|
| `Single_Blotto_Round.m` | `game.py::single_round` |
| `Autoregressive_Policy.m` | `policy.py::autoregressive_update` |
| `Iterated_Blotto_Autoregressive.m` | `simulate.py::simulate_iblotto` |
| `Iterated_Blotto_Tournament.m` | `tournament.py::run_tournament` + `run_tournament.py` driver |
| `Perform_Principal_Tradeoff_Analysis.m` | `pta_compat.py::perform_pta` (wraps `fptajax.pta` + Bandeira–van Handel bound) |
| `Test_Iterative_Blotto_With_Autoregressive_Policies.m` | `test_pair.py` |

## Model variants (training and analysis scripts)

These scripts train and analyse different FPTA-style models on a behavioural bundle (the `.pkl` files produced by `generate_behavioral_data.py`). All share the same train/test split convention on observed pairs.

| Script | What it trains | Notes |
|---|---|---|
| `ablate_basis_rbf.py` | BFPTA with k-means RBF basis over the trait space | `basis_kind = "rbf_kmeans"`; periodic re-clustering of centres |
| `disc_direct.py` | Encoder → MLP → disc-game coordinates (no basis) | `basis_kind = "disc_direct"`; the strongest baseline |
| `disc_direct_bc.py` | `disc_direct` + Dirichlet behaviour-cloning head `p(a \| s, z)` | `basis_kind = "disc_direct_bc"`; unlocks online opponent ID |
| `online_id.py` | (library) particle filter + disc-space best response | Consumes a `disc_direct_bc` checkpoint |
| `online_id_demo.py` | Single-opponent walk-through of the filter + BR | Useful for debugging the inference loop |
| `online_id_sweep.py` | Per-agent ID hit-rate characterisation across the population | Produces the data behind Figure 1 of the online-ID note |
| `online_id_br_validation.py` | End-to-end BR vs. simulated opponents at varying warmup | Produces the data behind Figure 2 of the online-ID note |
| `render_online_id_figures.py` | Renders the three publication-ready PDFs | Caches sweep data to `figures/online_id_sweep_cache.npz` |
| `render_match_dynamics.py` | Multi-game adaptive BR with cumulative-payoff plot | Produces `fig_online_id_match.pdf` |
| `render_match_animation.py` | Animated per-token posterior + cumulative payoff GIF | Single-opponent illustrative animation |

Higher-level results write-up: [`RESULTS_online_id.md`](RESULTS_online_id.md).

## Game options encoding

`GameOptions` is a frozen dataclass; mode strings are encoded as small
integers so the struct is jit-friendly. The mappings:

```
csf_mode:                 0 = auction
                          1 = win by threshold auction
                          2 = lottery (sigmoid in allocation difference)
resource_return_mode:     0 = none
                          1 = keep investment
                          2 = keep investment difference
                          3 = win opponent bid
                          4 = keep losing bids
reallocation_mode:        0 = stay in zone (returned resources locked there)
                          1 = reallocate (returned resources go back to free budget)
info_mode:                0 = all investments
                          1 = total invested
```

## Running

The drivers run on CPU at modest speed (no GPU needed for these sizes):

```bash
# Single-pair smoke test (mirrors the MATLAB Test_... driver):
JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto.test_pair

# Full 40-agent × 800-realisation tournament + PTA:
JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto.run_tournament

# Validation against the saved MATLAB tournament (subset):
JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m examples.iblotto._validate_against_matlab
```

The results pickle is dropped at `examples/iblotto/results/tournament_*.pkl`.

## Validation against `IB_test.mat`

The validation script loads the 40-agent policies from the saved MATLAB
tournament, runs the JAX tournament on a 6-agent subset (`n_real=400`,
matched game options), and compares per-pair `F` values. With the
NaN-propagation fix described below, all 15 pairs agree within 2 standard
errors:

```
mean |z|: 0.95,  max |z|: 1.77,  fraction within 2σ: 1.00
```

This is consistent with independent runs differing only by sampling noise.

## Implementation notes / gotchas

### `JAX_ENABLE_X64=1` is required.

The original simulations use double-precision throughout. Single precision
shifts the autoregressive policy enough to bias `F` outside the per-pair
standard error.

### NaN propagation matches MATLAB's collapse semantics.

For high learning-rate agents (`lr` close to 1), the policy can land in a
state where all zone weights clip to zero. The MATLAB then divides 0/0 in
the renormalisation step, producing NaN policy parameters; subsequent
rounds' allocations are NaN, all CSF comparisons return False, and
**both players receive 0 payouts** for the rest of the game. The
cumulative payout difference therefore freezes at whatever it was when
the agent collapsed.

The JAX port mirrors this exactly — we deliberately do **not** guard the
`(conc − past_no_inv) * new_zone / sum(new_zone)` divisions. JAX's NaN
propagation through arithmetic + comparisons (`NaN > 0` → `False`) gives
bit-identical post-collapse behaviour to MATLAB. An earlier draft of the
port had a `safe_zsum` clamp here that produced consistent zero
allocations forever instead, biasing `F` for unstable agents by a factor
of ~25×; that's been removed.

### Memory > 1 is not implemented.

The MATLAB `Autoregressive_Policy.m` references `memory > 1` history but
the per-zone indexing in the innovation step (lines 28–29 of the MATLAB)
breaks for memory > 1; the driver scripts always set `memory = 1`. This
JAX port assumes memory = 1 throughout.

### Initial allocation uses `jax.random.dirichlet`.

The MATLAB samples `gamrnd(policy + 1, 1)` and renormalises to a Dirichlet
sample by hand; we use `jax.random.dirichlet(key, alpha=policy + 1)` which
is the same distribution.

### Tournament parallelism.

`run_tournament` `vmap`s + `jit`s over realisations within a pair, with
the outer pair loop in Python. For 40 agents × 780 pairs × 800 realisations
× 100 rounds this is ~25M-round-equivalents of work; on an Apple Silicon
CPU it completes in roughly 3–5 minutes. The dominant cost is the JIT
re-compile per pair (the inner `simulate_iblotto` is parameterised on
`opts` which is captured in closure as a constant). If running many
tournaments back-to-back, the cache hit dominates after the first pair.

### PTA + stability bounds.

`pta_compat.perform_pta` wraps `fptajax.pta.pta()` (the project's own
classical-PTA engine) with three additional pieces from
`Perform_Principal_Tradeoff_Analysis.m`:

1. The per-disc-plane rotation that puts the centroid on the +x axis.
2. The eigengap-based stability ratios.
3. The Bandeira–van Handel (2016) operator-norm expectation bound for
   `‖E‖₂` from per-entry standard errors of `F`, with the same 90%-failure
   probability constant.

The output is a `PTAReport` dataclass exposing the same fields as the
MATLAB function returned (in tuple form).
