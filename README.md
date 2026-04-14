# fptajax

A JAX library for **Functional Principal Tradeoff Analysis** (FPTA) — a spectral decomposition method that breaks a game into a sequence of simple "disc games", embedding agents into 2D planes where each plane captures a distinct strategic tradeoff.

Also implements the original **Principal Tradeoff Analysis** (PTA) for finite payoff matrices.

## Background

Many games have no single dominant strategy. Instead, agents face tradeoffs: being strong in one dimension means being weak in another (think rock-paper-scissors). FPTA decomposes a game's performance function into a sequence of these cyclic tradeoffs, ordered by importance, analogous to how PCA decomposes variance into principal components.

Given a skew-symmetric performance function `f(x, y) = -f(y, x)` measuring the advantage of agent `x` over agent `y`, FPTA finds embeddings `Y(x)` such that:

```
f(x, y) = sum_k  Y^(k)(x) x Y^(k)(y)
```

where `x` denotes the 2D cross product (disc game) and each component `k` represents a distinct strategic tradeoff.

## Installation

```bash
# Core (JAX required, install jaxlib separately for your hardware)
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With neural basis learning (equinox + optax)
pip install -e ".[neural]"

# Everything
pip install -e ".[all]"

# Dev/test
pip install -e ".[dev]"
```

JAX must be installed separately with the appropriate backend:

```bash
# CPU only
pip install jax jaxlib

# GPU (CUDA)
pip install jax[cuda12]
```

## Quick Start

### PTA on a Payoff Matrix

```python
import jax.numpy as jnp
from fptajax import pta, disc

# Rock-Paper-Scissors payoff matrix
F = jnp.array([
    [ 0.,  1., -1.],
    [-1.,  0.,  1.],
    [ 1., -1.,  0.],
])

result = pta(F)

print(result.n_components)   # 1 disc game
print(result.eigenvalues)    # [1.732...]  (sqrt(3))
print(result.embeddings)     # (3, 1, 2) — 3 agents, 1 disc game, 2D each

# Verify: disc game reconstruction recovers F
for i in range(3):
    for j in range(3):
        print(disc(result.embeddings[i, 0], result.embeddings[j, 0]))
```

### FPTA on a Continuous Game

```python
import jax.numpy as jnp
from fptajax import fpta, FourierBasis

# Performance function: advantage depends on trait difference
def f(x, y):
    return jnp.sin(x - y) + 0.3 * jnp.sin(3 * (x - y))

basis = FourierBasis()
result = fpta(f, basis, n_basis=9, n_quad=150)

print(result.n_components)          # 2 significant disc games
print(result.eigenvalues)           # [0.5, 0.15, ...]
print(result.get_importance())      # [0.917, 0.083, ...]

# Evaluate embeddings at new trait values
x = jnp.linspace(0, 2 * jnp.pi, 100)
Y = result.embed_from_basis(basis, x)  # (100, d, 2)

# Reconstruct the performance function
f_hat = result.reconstruct(x, x, basis, n_components=2)
```

### Visualization

```python
from fptajax.viz import (
    plot_disc_game,
    plot_disc_games_grid,
    plot_importance,
    plot_performance_heatmap,
    plot_pta_embedding,
)

# FPTA: show how traits map into disc game planes
plot_disc_game(result, basis, k=0, trait_range=(0, 2 * jnp.pi))

# FPTA: side-by-side true vs reconstructed performance
plot_performance_heatmap(result, basis, (0, 2*jnp.pi), f=f, n_components=2)

# PTA: scatter agents in disc game plane
plot_pta_embedding(pta_result, k=0, labels=["Rock", "Paper", "Scissors"])

# Eigenvalue importance bar chart
plot_importance(result)
```

### Neural Basis Learning

When the right basis family is unknown, learn it from data. A neural network `b_theta: X -> R^d` maps traits to basis values, jointly trained with a skew-symmetric coefficient matrix `C`:

```python
from fptajax import neural_fpta, TrainConfig

# From pairwise data: (x_i, y_i, f(x_i, y_i))
result = neural_fpta(
    x_data, y_data, f_data,
    d=16,                     # basis dimension
    trait_dim=1,              # input dimensionality
    hidden_dims=(64, 64),     # MLP architecture
    config=TrainConfig(
        n_steps=2000,
        lr=1e-3,
        ortho_weight=0.1,     # orthogonality regularization
        c_correction_every=200,  # periodic closed-form C solve
    ),
)

# Evaluate embeddings at any trait value (continuous interpolation)
Y = result.embed(x_new)            # (N, d, 2)
F_hat = result.predict(x_new, y_new)  # direct prediction
F_disc = result.reconstruct(x_new, y_new, n_components=3)  # truncated

# Or from a payoff matrix with agent traits
from fptajax import neural_fpta_from_matrix
result = neural_fpta_from_matrix(F, traits=agent_traits, d=16)
```

**Training approach** (hybrid end-to-end + periodic correction):
1. Joint Adam optimization of NN weights and `C = A - A^T` (automatic skew-symmetry)
2. Every K steps, recompute the optimal `C` in closed form via ridge-regularized least squares
3. Orthogonality penalty `||B^TB/N - I||^2` prevents dimensional collapse
4. After training, Schur decomposition of learned `C` extracts disc game embeddings

This is inspired by [Function Encoders](https://arxiv.org/abs/2401.17173) (Ingebrand et al., 2024), adapted for the bilinear skew-symmetric structure of game performance functions.

### Behavioral FPTA — Learning Traits from Play Data

When agent traits are not observed directly, learn them from behavioral data. Each agent is described by a set of state-action pairs `D_i = {(s, a)_1, ..., (s, a)_K}` showing how they play. A [DeepSets](https://arxiv.org/abs/1703.06114) encoder infers latent traits, which feed into the FPTA pipeline:

```
D_i → SetEncoder(φ, ρ) → x_i (traits) → NeuralBasis(b) → b(x_i)
                                                              ↓
                             f̂(i,j) = b(x_i)^T C b(x_j)
```

```python
from fptajax import behavioral_fpta, TrainConfig

# agent_data: (N, K_max, sa_dim) — padded behavior data per agent
# agent_mask: (N, K_max) — True where data is valid (handles variable-length)
# F: (N, N) — observed payoff matrix

result = behavioral_fpta(
    agent_data, agent_mask, F,
    sa_dim=6,             # dimensionality of (state, action) vectors
    trait_dim=8,          # latent trait dimensionality
    d=16,                 # basis dimension
    phi_hidden=(64, 64),  # DeepSets per-element MLP
    rho_hidden=(64,),     # DeepSets aggregation MLP
    basis_hidden=(64, 64),
    config=TrainConfig(n_steps=3000, lr=1e-3),
)

# Inspect inferred traits
traits = result.encode(agent_data, agent_mask)  # (N, trait_dim)

# Embed agents in disc game planes
Y = result.embed(agent_data, agent_mask)  # (N, n_components, 2)

# Predict payoffs for new agents from their behavior
F_pred = result.predict(new_agent_data, new_agent_data, new_mask, new_mask)
```

The encoder, basis, and coefficient matrix are all trained jointly end-to-end. The DeepSets architecture ensures permutation invariance over the set of (state, action) pairs.

## Basis Functions

FPTA works with any basis. The library provides:

| Basis | Domain | Weight | Use Case |
|-------|--------|--------|----------|
| `FourierBasis` | `[0, 2pi]` | uniform | Periodic games, allocation games |
| `ChebyshevBasis` | `[-1, 1]` | `1/sqrt(1-x^2)` | Smooth functions on bounded domains |
| `LegendreBasis` | `[-1, 1]` | uniform | General polynomial approximation |
| `JacobiBasis(a, b)` | `(-1, 1)` | `(1-x)^a(1+x)^b` | Flexible endpoint behavior |
| `HermiteBasis` | `(-inf, inf)` | `exp(-x^2)` | Gaussian-weighted traits |
| `LaguerreBasis` | `[0, inf)` | `exp(-x)` | Non-negative traits |
| `MonomialBasis` | user-defined | user-defined | Raw monomials (auto-orthogonalized) |
| `CustomBasis` | user-defined | user-defined | Any user-provided functions |
| `NeuralBasis` | any | learned | Learned MLP basis (requires `fptajax[neural]`) |

## API Reference

### Core Functions

- **`fpta(f, basis, n_basis, ...)`** — Functional PTA. Takes a performance function and basis, returns `FPTAResult` with embeddings, eigenvalues, and reconstruction methods.
- **`pta(F)`** — Pointwise PTA on a payoff matrix. Returns `PTAResult` with agent embeddings.
- **`fpta_empirical(F_X, B_X)`** — FPTA with empirical measure. Combines sample data with a basis for interpolation to new agents.
- **`neural_fpta(x_data, y_data, f_data, d, ...)`** — Neural FPTA. Learns basis functions from data via an MLP, returns `NeuralFPTAResult`.
- **`neural_fpta_from_matrix(F, traits, d, ...)`** — Convenience wrapper for neural FPTA from a payoff matrix.
- **`behavioral_fpta(agent_data, agent_mask, F, ...)`** — Behavioral FPTA. Learns traits from play data via DeepSets + FPTA pipeline.

### Result Objects

**`FPTAResult`**:
- `.eigenvalues` — importance magnitudes, sorted decreasing
- `.n_components` — number of disc game components
- `.embed_at_nodes()` — evaluate embeddings at quadrature nodes
- `.embed_from_basis(basis, x)` — evaluate at arbitrary trait values
- `.reconstruct(x, y, basis, n_components=None)` — reconstruct `f_hat(x, y)`
- `.get_importance()` — relative importance of each component
- `.get_cumulative_importance()` — cumulative explained variance

**`PTAResult`**:
- `.embeddings` — shape `(N, d, 2)`, agent embeddings in each disc game
- `.eigenvalues` — importance magnitudes
- `.reconstruct(n_components=None)` — reconstruct the payoff matrix

**`NeuralFPTAResult`**:
- `.embed(x)` — evaluate disc game embeddings at any trait values
- `.predict(x, y)` — predict `f(x, y)` via `b(x)^T C b(y)`
- `.reconstruct(x, y, n_components=None)` — truncated disc game reconstruction
- `.basis` — trained `NeuralBasis` (Equinox module)
- `.coefficient_matrix` — learned skew-symmetric `C`
- `.train_history` — training metrics log

**`BehavioralFPTAResult`**:
- `.encode(sa_data, mask)` — infer trait vectors from behavior data
- `.embed(sa_data, mask)` — full pipeline: behavior → disc game embeddings
- `.embed_from_traits(traits)` — embeddings from pre-computed traits
- `.predict(sa_i, sa_j, mask_i, mask_j)` — predict payoff from behavior data
- `.encoder` — trained `SetEncoder` (DeepSets module)
- `.basis` — trained `NeuralBasis`

### Visualization (`fptajax.viz`)

- `plot_disc_game(result, basis, k, trait_range)` — agents in disc game plane, colored by trait
- `plot_disc_games_grid(result, basis, trait_range, n_games)` — multi-panel grid
- `plot_embedding_trajectory(result, basis, trait_range, k)` — parametric embedding curve
- `plot_importance(result)` — eigenvalue bar chart with cumulative line
- `plot_reconstruction_error(result, basis, f, trait_range)` — error vs. components
- `plot_performance_heatmap(result, basis, trait_range, f)` — true vs. reconstructed heatmap
- `plot_pta_embedding(result, k, labels)` — PTA scatter plot
- `plot_pta_spinning_top(result)` — 3D spinning top (skill vs. intransitivity)

## Algorithm

FPTA implements Algorithm 3.1 from the paper:

1. **Basis Formulation** — Orthogonalize basis functions `{b_j}` via Gram-Schmidt w.r.t. the trait distribution
2. **Projection** — Compute the skew-symmetric coefficient matrix `C_ij = <f, [b_i, b_j]>`
3. **Decomposition** — Real Schur decomposition of `C = Q U Q^T`
4. **Embedding** — Extract disc game embeddings `Y^(k)(x) = sqrt(omega_k) * b(x)^T [q_{2k-1}, q_{2k}]`

The GPU-friendly implementation uses `jnp.linalg.eig` (which runs on GPU) rather than `jax.scipy.linalg.schur` (CPU-only), reconstructing the real Schur form from eigenpairs of the skew-symmetric matrix.

## References

- Ingebrand, Doty, Topcu. [*Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces.*](https://arxiv.org/abs/2401.17173) (2024).

## License

MIT
