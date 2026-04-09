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

# With dev/test dependencies
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

## API Reference

### Core Functions

- **`fpta(f, basis, n_basis, ...)`** — Functional PTA. Takes a performance function and basis, returns `FPTAResult` with embeddings, eigenvalues, and reconstruction methods.
- **`pta(F)`** — Pointwise PTA on a payoff matrix. Returns `PTAResult` with agent embeddings.
- **`fpta_empirical(F_X, B_X)`** — FPTA with empirical measure. Combines sample data with a basis for interpolation to new agents.

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

- Strang, Sewell, Kim, Alcedo, Rosenbluth. [*Principal Trade-off Analysis.*](https://journals.sagepub.com/doi/10.1177/14738716241239018) Information Visualization (2024).

## License

MIT
