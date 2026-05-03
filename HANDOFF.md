# Handoff Note — 2026-04-23

Session ended because accumulated wide plots (>2000px) tripped the API's
image-dimension limit. Everything on disk is intact; this note captures
where the project is and where it's going next.

**Summary in one line**: Phase 1–2 empirical work on tennis + RPS is
complete and written up. A design for compel/deter planning on top of
FPTA is documented as reference/background but shelved. **The next
research direction is NBA data.**

---

## Repo state

### Library (`src/fptajax/`) — complete, 21/21 tests pass

| Module | Purpose |
|---|---|
| `core.py`, `pta.py`, `basis.py`, `quad.py`, `decomposition.py` | Classical FPTA (non-neural) |
| `neural.py`, `behavioral.py` | Flat-DeepSets F-supervised neural FPTA |
| `hierarchical.py` | Hierarchical transformer-per-game + DeepSets encoder |
| `hierarchical_skill.py` | **Best F-supervised variant**: skill + disc-game decomposition with mean-zero-basis gauge-breaking |
| `hierarchical_g.py` | Asymmetric variant: unconstrained `C`, vector skill, non-skew labels |
| `mlp_baseline.py` | Ablation: antisymmetric MLP head instead of bilinear form |
| `contrastive.py` | InfoNCE pretraining for the encoder |
| `online.py` | Incremental opponent-trait encoder (online/streaming) |
| `utils.py`, `viz.py` | Utilities + matplotlib plots |

All exported from `fptajax.__init__`. Package installs via `pip install -e ".[neural,viz]"`.

### Experiment scripts (already run; results captured in papers)

- `examples/tennis/` — MCP data pipeline (`loader.py`, `parser.py`) + training scripts
  (`train.py`, `train_skill.py`, `train_g.py`, `train_interventions.py`) + RF baseline
- `examples/rps_*.py` — RPS drivers (`rps_interventions.py`, `rps_skill.py`, `rps_g.py`,
  `rps_classic_fpta.py`)

### Data paths used so far

- **MCP tennis**: `/tmp/mcp/{charting-m-matches.csv, charting-m-points-2020s.csv, charting-m-points-2010s.csv}` — re-download from `https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/` if `/tmp/mcp/` got cleared.
- **RPS tournament**: `/Users/davidsewell/Projects/rps_pbt/tournament_results/20260119_085535/openspiel_tournament_actions.jsonl`

### Papers (all compile cleanly)

1. `paper/behavioral_fpta.tex` → `behavioral_fpta.pdf` — foundational: what Behavioral FPTA is, encoder architecture, training.
2. `paper/neural_fpta_experiments.tex` → `neural_fpta_experiments.pdf` — 11 pages, empirical
   study of all neural variants on tennis + RPS. Trait collapse diagnosis, label-noise-floor
   argument for tennis, encoder-capacity argument for RPS.
3. `paper/compel_deter_planning.tex` → `compel_deter_planning.pdf` — 11 pages, design doc
   for trait-conditioned planning (**see "Context/background" section below — this direction
   is shelved for now**).

---

## Research narrative so far

### Phase 1 — Empirical characterisation of neural Behavioral FPTA

At moderate encoder size (`d_model=32, n_layers=1`, ~18.5K params), **seven
architectural variants all plateau at broadly similar test MSE**. The encoder
converges to trait vectors with very small inter-agent spread (std/dim ≈ 0.005
on tennis), a stable fixed point of the F-loss landscape. Spread regularisation,
contrastive pretraining, alternative heads — none break out of this.

| Best numbers | Tennis test F-MSE (Δ null) | RPS test F-MSE (Δ null) |
|---|---|---|
| Random Forest (hand-crafted features) | 0.01512 (+1.5%) | 0.02366 (**+80.1%**) |
| FPTA baseline | 0.01465 (+4.5%) | 0.05144 (+56.6%) |
| **Skill + disc (f-sup)** | **0.01453 (+5.3%)** | **0.03771 (+68.2%)** |
| g-FPTA (asymmetric) | 0.01520 (+0.95%) | 0.04452 (+62.4%) |

### Phase 2 — Diagnose the two datasets

- **Tennis is label-noise-limited.** Half of observed pairs have a single
  charted meeting; per-label sampling σ ≈ 0.04; recoverable-variance ceiling
  ≈ 0.002. No neural variant gets close. RF at +1.5% is also near ceiling.
- **RPS is encoder-capacity-limited.** Prior work at `d_model=64, n_layers=2`
  reached RF-parity (~0.023 test MSE). At `d_model=32`, neural is stuck at
  +62–68% while RF is at +80%. The encoder is the bottleneck, not the head.

### Phase 3 — Design direction (shelved): compel/deter planning

Explored combining FPTA trait inference with AlphaZero-style MCTS planning
for **compel/deter subgoals** in competitive settings — inspired by
HOP (Huang et al. 2024) but stripped of mixed-motive components. Designed
a minimal testbed ("Lure v0") to isolate the core claim: trait inference
enables type-specific exploit strategies that a type-agnostic planner
cannot find.

**Status**: design complete and documented in `paper/compel_deter_planning.tex`;
minimal environment scaffold (`envs/lure_v0/env.py`) started but **not continued**.
Kept for reference — see "Context/background" below.

---

## Next direction: NBA data

The tennis/RPS pair bracketed a useful region of parameter space but both are
small (~40 agents, ~800–1800 observed pairs). We want to test Behavioral FPTA
on a richer, more structurally-interesting dataset. **NBA play-by-play data**
is the target.

### Why NBA is interesting for FPTA

- **Scale.** 30 teams × 82 regular-season games per team = ~1200 games per
  season, ~100 possessions each = ~120K per-possession records per season.
  Many decades of data available.
- **Multiple granularities.** Player-level, lineup-level (5-man units), and
  team-level traits are all meaningful and each has its own matchup structure.
- **Clear competitive framing.** Possessions end with a scalar outcome
  (points scored or turnover), which aggregates cleanly to pair-level
  matchup statistics — analogous to tennis points but with more per-pair data.
- **Documented "styles".** Basketball has a rich shared vocabulary for player
  archetypes (rim protectors, floor spacers, primary initiators, 3-and-D
  wings, etc.). A well-trained FPTA trait encoder should recover clusters
  aligned with these archetypes — a natural interpretability check.
- **Plausibly higher-rank cyclic structure than RPS/tennis.** Basketball
  matchups involve speed vs size, shooting vs defence, pace preferences —
  rock-paper-scissors-like dynamics at the team level (small-ball teams beat
  traditional bigs who beat lob-threats who beat small-ball, etc.). If true,
  the disc-game spectrum should show multiple non-trivial modes rather than
  the one-dominant-mode patterns we've seen.

### Open questions to scope in early exploration

1. **What is an "agent"?** Three natural choices:
   - **Player**: trait per NBA player. Matchups = on-court player pairings.
     Rich but fragmented — most player pairs never directly face off.
   - **Team**: trait per franchise-season (so the 2016 Warriors ≠ 2020
     Warriors). Cleaner dense matchup matrix. Fewer total agents but more
     observations per pair.
   - **Lineup**: trait per 5-man unit. Most granular, enormous sparsity.
   The right first experiment is probably team-season — matches the
   RPS/tennis sample size regime we've validated against.

2. **What is a "game" (in the FPTA tokenisation sense)?** For team-level,
   a single NBA game has ~200 possessions, each with a structured outcome.
   A per-possession token could encode: possession-start context (score
   differential, time remaining), play type (pick-and-roll, iso, transition),
   shot location / outcome, rebound disposition. ~10–20 channels.

3. **What is `F` (the scalar label)?** Candidates:
   - Net points per possession differential (analogous to tennis point-share
     differential)
   - Win probability added (matchup-level, regular season only vs including
     playoffs)
   - Offensive efficiency differential (points-per-100-possessions)

4. **Data sources.** NBA Stats API via the `nba_api` Python package is the
   obvious starting point; Basketball-Reference has cleaner historical box
   scores; the 2016 NBA Hackathon dataset on Kaggle has detailed
   play-by-play plus win probability.

### Suggested first experiment

Start with **team-season level** on **one modern regular season** (say 2023-24):

1. Pull play-by-play for all 1230 regular-season games.
2. Build per-game tensors in the existing `hierarchical` token format —
   one game per team-season per-game perspective, so ~82 "games" per
   team-season.
3. Compute pairwise `F` as net-points-per-possession differential on
   direct matchups (each team plays ~2–4 games per opponent).
4. Train `hierarchical_skill_fpta` (the best F-supervised variant from
   Phase 1) and inspect:
   - Does the skill ranking match known end-of-season team quality?
   - Does the disc-game spectrum show multiple non-trivial modes?
   - Do the disc-game embeddings cluster teams by stylistic archetype?

This should be ~1–2 days of data engineering + a few hours of training
on existing infrastructure. No new library code needed — `hierarchical_skill`
is ready to go. The main work is the data pipeline.

### What to read first in the new session

1. `HANDOFF.md` (this file) — state + direction.
2. `paper/neural_fpta_experiments.tex` §1–5 — what we learned from the
   tennis/RPS runs; informs hyperparameters and expectations for NBA.
3. `src/fptajax/hierarchical_skill.py` — the head to use.
4. `examples/tennis/train_skill.py` — template for a new dataset driver.

Then start sketching the NBA data pipeline.

---

## Context/background: compel-deter planning and Lure (shelved)

For future reference: we designed (but did not fully build) a research
direction combining FPTA trait inference with AlphaZero-style planning
for competitive settings where the focal agent has compel/deter
**subgoals** — steering the opponent toward or away from specific states
or actions, not just winning the game directly. This is motivated by
real adversarial settings (Diplomacy, international relations).

Key pieces of the design:

- **HOP (Huang et al. 2024) is the architectural template** — hierarchical
  opponent model + MCTS planner — but its discrete-goal inference layer
  collapses in competitive settings (opponent's goal is trivially "win").
  What remains informative is *style/trait* inference, which is exactly
  what FPTA provides.
- **Asymmetric-`C` formulation.** When the focal agent has a compel reward
  that the opponent lacks, `f(focal, opp) ≠ -f(opp, focal)` — the
  skew-symmetric assumption fails. Use the `hierarchical_g` module's
  unconstrained `C` with vector skill and learned `c_0, c_1` coefficient
  vectors. Disc games are extracted post-hoc from `C − Cᵀ`.
- **Three-head extension.** Encoder + `g`-head (pair-level reward) +
  `π`-head (trait-conditioned opponent policy) + `Q`-head (trait-conditioned
  state-action value). All three train jointly sharing the encoder.
- **Lure v0 testbed.** 7×7 gridworld, 1v1, pressure cell `P = (3,3)`, walls
  forcing any N-S travel through a central corridor containing `P`. Three
  heuristic opponent types (`greedy`, `chaser`, `defender`), each
  compellable by a different focal strategy. Research question: does
  FPTA trait inference enable selection of the right type-specific
  exploit?

Files that exist from this direction:

- `paper/compel_deter_planning.tex` — full design document (11 pages,
  motivation, HOP mapping, asymmetric-`f` derivation, Lure v0 spec,
  three-stage experimental protocol).
- `envs/lure_v0/__init__.py` + `envs/lure_v0/env.py` — the Lure v0
  environment itself (~300 lines, complete and working).
- **NOT built**: `heuristics.py`, `focal_policies.py`, `tournament.py`,
  `test_env.py`. Roughly 300–400 lines of additional work if this
  direction is ever resumed.

**Why shelved for now**: NBA is a richer and more immediately interpretable
dataset; the compel/deter planning direction is more exploratory and has
higher implementation cost. Nothing wrong with the design — it's just not
the next thing.

---

## Housekeeping

- **Tests**: `.venv/bin/python -m pytest tests/ -v` → 21/21 pass. Run this
  in the new session as a baseline.
- **Python env**: `.venv/bin/python` has equinox, optax, sklearn, matplotlib,
  jaxtyping, nba_api is **NOT yet installed** — the new session will
  need `pip install nba_api` (or whichever data-access package we settle on).
- **Plot sizing**: future plot generation should cap `figsize × dpi` below
  ~1800px in either dimension to stay under the image-dimension API limit.
  `dpi=100` or `figsize=(12, 6)` at `dpi=150` are safe.
- **Minor outstanding**: `compel_deter_planning.tex` has 15 "Undefined
  control sequence" LaTeX warnings on build (PDF still renders fine).
  Low priority since that direction is shelved.

---

Good luck with NBA. Everything from the previous phases is in the repo and
the papers tell the full story.
