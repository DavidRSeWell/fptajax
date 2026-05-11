# Online opponent identification + best-response in disc-game space

Short results note for the encoder-decoder extension to `disc_direct.py`.
The full pipeline (training, ID, BR, figures) is implemented in:

* [`disc_direct_bc.py`](disc_direct_bc.py) — encoder + skill + disc heads,
  plus a Dirichlet behaviour-cloning head trained jointly.
* [`online_id.py`](online_id.py) — particle filter posterior over opponent
  trait, scored by `p(action | state, trait)` from the BC head; population
  best-response in disc-game space.
* [`online_id_demo.py`](online_id_demo.py) — single-opponent walkthrough.
* [`online_id_sweep.py`](online_id_sweep.py) — per-agent ID failure
  characterisation.
* [`online_id_br_validation.py`](online_id_br_validation.py) — end-to-end
  BR vs. simulated opponents.
* [`render_online_id_figures.py`](render_online_id_figures.py) — produces
  the three figures referenced below.

## Setup

* Bundle: `behavioral_bc_smoke_N100_k10_nr20.pkl` — N=100 agents, 5 zones,
  50 rounds/game, ~315 valid tokens/agent, sparse F (957/4950 pairs).
* Model: `disc_direct_bc`, K=6 disc games, trait_dim=24, `bc_weight=0.1`,
  20k training steps. Best test MSE 147 (norm 0.073), per-token Dirichlet
  log-density +8.1.
* Particle prior: 100 particles, one per encoded population trait.
  Resampling disabled for the figures (preserves particle identity for
  ranking).

## What the figures show

### Figure 1 — identification quality vs. evidence
**[figures/fig_online_id_hitrate.pdf](../../figures/fig_online_id_hitrate.pdf)**

For all 100 population agents, replay every valid token through the
filter and record the rank of the true particle at intermediate
warm-up checkpoints. Top-1 / top-5 / top-10 hit rates as a function of
tokens observed:

| tokens | top-1 | top-5 | top-10 | median rank |
|-------:|------:|------:|-------:|------------:|
|     5  | 16%   | 47%   | 63%    | 6           |
|    25  | 28%   | 64%   | 83%    | 2           |
|   100  | 49%   | 85%   | 92%    | 1           |
|   392  | 56%   | 90%   | 99%    | 0           |

Within ~25 tokens (≈half a game) the median rank is 2; by ~100 tokens
the true agent is in the top-5 for 85% of the population.

### Figure 2 — best-response payoff vs. evidence
**[figures/fig_online_id_br.pdf](../../figures/fig_online_id_br.pdf)**

For 20 randomly-chosen test opponents, run four BR strategies and
simulate 10 evaluation games of each against the opponent's actual
policy. Disc-only BR scoring (skill term suppressed; see below for why
this isolation is the right comparison). Mean realised
`F = P_BR − P_opp` ± SE across opponents:

| tokens | filter | uniform | oracle | random |
|-------:|-------:|--------:|-------:|-------:|
|     0  | +38.7  | +44.1   | +68.4  | +13.9  |
|    10  | +60.1  | +41.2   | +70.6  | +16.9  |
|    25  | +69.7  | +41.9   | +72.6  | +15.5  |
|   100  | +63.7  | +40.3   | +68.4  | +14.8  |
|   392  | +62.8  | +38.7   | +70.5  | +11.9  |

Filter uplift over the opponent-blind uniform baseline (paired across
opponents):

| tokens | Δ filter − uniform | wins/total |
|-------:|-------------------:|-----------:|
|     0  |  −5.4 ± 2.3        | 7/20       |
|    10  | +19.0 ± 10.2       | 14/20      |
|    25  | +27.8 ± 7.4        | 16/20      |
|   100  | +23.4 ± 7.0        | 16/20      |
|   392  | +24.1 ± 6.4        | 16/20      |

By 25 warm-up tokens the filter strategy captures **27.8 / (27.8 + 4.5)
≈ 86%** of the uplift available between the uniform-prior baseline and
the trait-oracle upper bound.

### Figure 3 — posterior trajectory in disc-game space
**[figures/fig_online_id_disc.pdf](../../figures/fig_online_id_disc.pdf)**

Population disc-1 embedding `(u_1, v_1)` for every agent (grey dots);
posterior-mean disc-1 of one example test opponent (#55) at warm-up
levels n ∈ {0, 5, 10, 25, 50, 100, 200, 392}, colour-coded; true
opponent location as a red star. The trajectory starts at the
population centroid (uniform prior) and migrates toward the true
location as evidence accumulates — this is the geometric content of
"locate the opponent in disc space."

## Why disc-only BR is the right comparison

The full BR scoring `score = skill + disc_term` reduces to "pick the
strongest agent" under a uniform prior, because the skill term
dominates and is opponent-independent. With skill in the score, oracle
beats uniform by only +1.2 — there is almost no opponent-specific BR
signal to extract. Stripping the skill term isolates the *cyclic*
structure of the disc-game decomposition, which is the part where
opponent identification can buy you anything. Filter uplift in this
isolated regime is +22–28 across opponents.

In a real deployment one would likely use `score = skill + λ ·
disc_term` with λ chosen by cross-validation; the disc-only result
upper-bounds the contribution of the ID-aware term.

## Caveats

1. **Smoke bundle.** N=100 agents from a 6-d policy class is small;
   ~10–15% of agents end up in clusters where ID is fundamentally
   ambiguous, capping the upside for those opponents. A
   `--N 200 --n_rounds 50` bundle would push top-1 ID higher.
2. **Static BR.** The "filter" strategy picks a static population
   policy and plays it; it does not adapt within the eval game. A
   conditional BR head `π(a | s, z_opp)` would push closer to oracle.
3. **bc_weight tuning.** Trained with `bc_weight=0.1`; the F-prediction
   objective dominates the encoder. Bumping to 0.3–1.0 would likely
   improve top-1 ID at the cost of slightly higher F-MSE.

## Reproducing

```bash
# 1. Generate (or copy) the bundle.
PYTHONPATH=src:. python -m examples.iblotto.generate_behavioral_data \
  --N 100 --k 10 --n_real 20 --n_rounds 50 --G_max 8 --tag bc_smoke

# 2. Train.
JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m examples.iblotto.disc_direct_bc \
  --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
  --out_dir disc_direct_bc_checkpoints/smoke20k_seed0 \
  --K 6 --bc_weight 0.1 --n_steps 20000 --seed 0

# 3. Render figures (caches sweep data after first run).
JAX_ENABLE_X64=1 PYTHONPATH=src:. python -m examples.iblotto.render_online_id_figures \
  --bundle examples/iblotto/results/behavioral_bc_smoke_N100_k10_nr20.pkl \
  --ckpt   disc_direct_bc_checkpoints/smoke20k_seed0 \
  --out_dir figures
```
