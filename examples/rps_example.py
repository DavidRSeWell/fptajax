"""Rock-Paper-Scissors example demonstrating PTA and FPTA.

This example shows:
1. PTA on the classic RPS payoff matrix
2. Visualization of the circular embedding
3. FPTA on a continuous version of RPS-like game
"""

import jax.numpy as jnp

import fptajax
from fptajax import pta, fpta, FourierBasis, disc
from fptajax.viz import (
    plot_pta_embedding,
    plot_importance,
    plot_disc_game,
    plot_performance_heatmap,
)


def main():
    print("=" * 60)
    print("fptajax Example: Rock-Paper-Scissors")
    print("=" * 60)

    # =====================================================================
    # Part 1: Pointwise PTA on the RPS payoff matrix
    # =====================================================================
    print("\n--- Part 1: PTA on RPS ---")

    F = jnp.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    labels = ["Rock", "Paper", "Scissors"]

    result = pta(F)

    print(f"Number of disc game components: {result.n_components}")
    print(f"Eigenvalues (omega_k): {result.eigenvalues}")
    print(f"Importance: {result.get_importance()}")

    # Print embeddings
    for i, label in enumerate(labels):
        y = result.embeddings[i, 0, :]
        print(f"  {label}: Y = ({y[0]:.4f}, {y[1]:.4f})")

    # Verify disc game reconstruction
    print("\nReconstruction check:")
    for i in range(3):
        for j in range(3):
            f_ij = disc(result.embeddings[i, 0, :], result.embeddings[j, 0, :])
            print(f"  disc(Y({labels[i]}), Y({labels[j]})) = {f_ij:.4f}  "
                  f"(true: {F[i, j]:.1f})")

    # Visualize
    try:
        fig, ax = plot_pta_embedding(result, k=0, labels=labels)
        fig.savefig("rps_pta_embedding.png", dpi=150, bbox_inches="tight")
        print("\nSaved: rps_pta_embedding.png")

        fig, ax = plot_importance(result)
        fig.savefig("rps_importance.png", dpi=150, bbox_inches="tight")
        print("Saved: rps_importance.png")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    # =====================================================================
    # Part 2: FPTA on a continuous sine game
    # =====================================================================
    print("\n--- Part 2: FPTA on f(x,y) = sin(x-y) ---")

    def f_sine(x, y):
        return jnp.sin(x - y)

    basis = FourierBasis()
    result_fpta = fpta(f_sine, basis, n_basis=7, n_quad=100)

    print(f"Number of disc game components: {result_fpta.n_components}")
    print(f"Eigenvalues: {result_fpta.eigenvalues}")
    print(f"Importance: {result_fpta.get_importance()}")
    print(f"Cumulative importance: {result_fpta.get_cumulative_importance()}")

    # Visualize
    try:
        trait_range = (0.0, 2 * jnp.pi)

        fig, ax = plot_disc_game(result_fpta, basis, k=0, trait_range=trait_range)
        fig.savefig("sine_disc_game_0.png", dpi=150, bbox_inches="tight")
        print("\nSaved: sine_disc_game_0.png")

        fig, axes = plot_performance_heatmap(
            result_fpta, basis, trait_range,
            f=f_sine, n_components=1,
        )
        fig.savefig("sine_heatmap.png", dpi=150, bbox_inches="tight")
        print("Saved: sine_heatmap.png")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")

    # =====================================================================
    # Part 3: FPTA on a multi-component game
    # =====================================================================
    print("\n--- Part 3: FPTA on f(x,y) = sin(x-y) + 0.3*sin(3(x-y)) ---")

    def f_multi(x, y):
        return jnp.sin(x - y) + 0.3 * jnp.sin(3 * (x - y))

    result_multi = fpta(f_multi, basis, n_basis=9, n_quad=150)

    print(f"Number of disc game components: {result_multi.n_components}")
    for k in range(min(result_multi.n_components, 4)):
        omega = result_multi.eigenvalues[k]
        imp = result_multi.get_importance()[k]
        print(f"  Component {k+1}: omega = {omega:.4f}, importance = {imp:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
