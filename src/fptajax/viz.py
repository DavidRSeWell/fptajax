"""Visualization for FPTA and PTA results.

All functions return (fig, ax) tuples for composability.
Requires matplotlib (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from fptajax.utils import importance, cumulative_importance


def _import_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install fptajax[viz]"
        )


# ---------------------------------------------------------------------------
# FPTA visualizations
# ---------------------------------------------------------------------------

def plot_disc_game(
    result,
    basis,
    k: int,
    trait_range: tuple[float, float],
    n_points: int = 200,
    ax=None,
    cmap: str = "viridis",
    label: str | None = None,
):
    """Plot agents embedded in disc game k across a range of traits.

    Evaluates Y^(k)(x) for x in trait_range and plots the 2D embedding,
    colored by trait value.

    Args:
        result: FPTAResult.
        basis: BasisFamily used in the FPTA.
        k: disc game index (0-based).
        trait_range: (min, max) of trait values to plot.
        n_points: number of trait values to sample.
        ax: matplotlib Axes. If None, creates new figure.
        cmap: colormap name.
        label: optional label for the colorbar.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()

    x = jnp.linspace(trait_range[0], trait_range[1], n_points)
    Y = result.embed_from_basis(basis, x)  # (n_points, d, 2)
    Y_k = Y[:, k, :]  # (n_points, 2)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    sc = ax.scatter(
        Y_k[:, 0], Y_k[:, 1],
        c=jnp.array(x), cmap=cmap, s=10, zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label or "Trait value")

    omega = result.eigenvalues[k]
    imp = float(2 * omega**2 / result.f_norm_sq) if result.f_norm_sq else None
    title = f"Disc Game {k + 1} ($\\omega_{k + 1}$ = {omega:.4f})"
    if imp is not None:
        title += f" [{imp:.1%} variance]"
    ax.set_title(title)
    ax.set_xlabel(f"$Y_1^{{({k + 1})}}(x)$")
    ax.set_ylabel(f"$Y_2^{{({k + 1})}}(x)$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_disc_games_grid(
    result,
    basis,
    trait_range: tuple[float, float],
    n_games: int = 4,
    n_points: int = 200,
    cmap: str = "viridis",
):
    """Multi-panel grid showing the first n disc game planes.

    Args:
        result: FPTAResult.
        basis: BasisFamily.
        trait_range: (min, max) trait range.
        n_games: number of disc games to show.
        n_points: trait samples per game.
        cmap: colormap.

    Returns:
        (fig, axes) tuple.
    """
    plt = _import_mpl()

    n_games = min(n_games, result.n_components)
    cols = min(n_games, 3)
    rows = (n_games + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_games == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for k in range(n_games):
        plot_disc_game(result, basis, k, trait_range, n_points, ax=axes[k], cmap=cmap)

    # Hide unused axes
    for k in range(n_games, len(axes)):
        axes[k].set_visible(False)

    fig.tight_layout()
    return fig, axes[:n_games]


def plot_embedding_trajectory(
    result,
    basis,
    trait_range: tuple[float, float],
    k: int = 0,
    n_points: int = 200,
    ax=None,
    cmap: str = "viridis",
):
    """Parametric curve of Y^(k)(x) as x varies, colored by trait value.

    Shows how the embedding traces through the disc game plane.

    Args:
        result: FPTAResult.
        basis: BasisFamily.
        trait_range: (min, max) trait range.
        k: disc game index.
        n_points: number of points.
        ax: optional Axes.
        cmap: colormap.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()

    x = jnp.linspace(trait_range[0], trait_range[1], n_points)
    Y = result.embed_from_basis(basis, x)
    Y_k = Y[:, k, :]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Plot as connected line segments colored by trait value
    from matplotlib.collections import LineCollection
    points = jnp.stack([Y_k[:, 0], Y_k[:, 1]], axis=-1)
    points_np = jnp.array(points)
    segments = jnp.stack([points_np[:-1], points_np[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, linewidths=2)
    lc.set_array(jnp.array(x[:-1]))
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Trait value")

    # Mark start and end
    ax.plot(Y_k[0, 0], Y_k[0, 1], 'o', color='green', markersize=8, label='Start', zorder=3)
    ax.plot(Y_k[-1, 0], Y_k[-1, 1], 's', color='red', markersize=8, label='End', zorder=3)
    ax.legend()

    ax.set_title(f"Embedding Trajectory — Disc Game {k + 1}")
    ax.set_xlabel(f"$Y_1^{{({k + 1})}}(x)$")
    ax.set_ylabel(f"$Y_2^{{({k + 1})}}(x)$")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_importance(result, ax=None):
    """Bar chart of eigenvalue importance and cumulative explained variance.

    Args:
        result: FPTAResult or PTAResult.
        ax: optional Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()

    omegas = result.eigenvalues
    d = len(omegas)
    f_norm_sq = getattr(result, 'f_norm_sq', None)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    imp = importance(omegas, f_norm_sq)
    cum_imp = cumulative_importance(omegas, f_norm_sq)

    x_pos = jnp.arange(d) + 1
    ax.bar(x_pos, imp, alpha=0.7, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(x_pos, cum_imp, 'ro-', label="Cumulative")
    ax2.set_ylabel("Cumulative")

    ax.set_xlabel("Disc Game Component")
    ax.set_ylabel("Importance ($2\\omega_k^2 / \\|f\\|^2$)" if f_norm_sq else "$2\\omega_k^2$")
    ax.set_title("Disc Game Importance")
    ax.set_xticks(x_pos)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    return fig, ax


def plot_reconstruction_error(
    result,
    basis,
    f,
    trait_range: tuple[float, float],
    max_components: int | None = None,
    n_points: int = 50,
    ax=None,
):
    """Show reconstruction error as a function of number of disc games kept.

    Args:
        result: FPTAResult.
        basis: BasisFamily.
        f: true performance function f(x, y).
        trait_range: trait range for evaluation.
        max_components: max components to test.
        n_points: grid points for error evaluation.
        ax: optional Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()

    nc = max_components or result.n_components
    x = jnp.linspace(trait_range[0], trait_range[1], n_points)
    xx, yy = jnp.meshgrid(x, x, indexing='ij')
    f_true = f(xx, yy)

    errors = []
    for k in range(1, nc + 1):
        f_hat = result.reconstruct(x, x, basis, n_components=k)
        err = float(jnp.mean((f_true - f_hat) ** 2))
        errors.append(err)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.semilogy(range(1, nc + 1), errors, 'bo-')
    ax.set_xlabel("Number of Disc Games")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Reconstruction Error vs. Components")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, nc + 1))

    fig.tight_layout()
    return fig, ax


def plot_performance_heatmap(
    result,
    basis,
    trait_range: tuple[float, float],
    f=None,
    n_components: int | None = None,
    n_points: int = 50,
    ax=None,
    cmap: str = "RdBu_r",
):
    """Heatmap of reconstructed f_hat(x,y), optionally side-by-side with true f.

    Args:
        result: FPTAResult.
        basis: BasisFamily.
        trait_range: (min, max) for both axes.
        f: true performance function. If provided, shows side-by-side.
        n_components: disc games to use in reconstruction.
        n_points: grid resolution.
        ax: optional Axes (ignored if f is provided, since 2 panels are made).
        cmap: colormap.

    Returns:
        (fig, axes) tuple.
    """
    plt = _import_mpl()

    x = jnp.linspace(trait_range[0], trait_range[1], n_points)
    f_hat = result.reconstruct(x, x, basis, n_components=n_components)

    if f is not None:
        xx, yy = jnp.meshgrid(x, x, indexing='ij')
        f_true = f(xx, yy)
        vmax = max(float(jnp.max(jnp.abs(f_true))), float(jnp.max(jnp.abs(f_hat))))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(
            f_true, extent=[trait_range[0], trait_range[1]] * 2,
            origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax,
        )
        axes[0].set_title("True $f(x, y)$")
        fig.colorbar(im0, ax=axes[0])

        nc = n_components or result.n_components
        im1 = axes[1].imshow(
            f_hat, extent=[trait_range[0], trait_range[1]] * 2,
            origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax,
        )
        axes[1].set_title(f"Reconstructed $\\hat{{f}}$ ({nc} disc games)")
        fig.colorbar(im1, ax=axes[1])

        error = jnp.abs(f_true - f_hat)
        im2 = axes[2].imshow(
            error, extent=[trait_range[0], trait_range[1]] * 2,
            origin='lower', cmap='hot',
        )
        axes[2].set_title("$|f - \\hat{f}|$")
        fig.colorbar(im2, ax=axes[2])

        for a in axes:
            a.set_xlabel("$x$")
            a.set_ylabel("$y$")

        fig.tight_layout()
        return fig, axes
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = ax.get_figure()

        vmax = float(jnp.max(jnp.abs(f_hat)))
        im = ax.imshow(
            f_hat, extent=[trait_range[0], trait_range[1]] * 2,
            origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax,
        )
        nc = n_components or result.n_components
        ax.set_title(f"Reconstructed $\\hat{{f}}$ ({nc} disc games)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# PTA-specific visualizations
# ---------------------------------------------------------------------------

def plot_pta_embedding(
    result,
    k: int = 0,
    labels: list[str] | None = None,
    ax=None,
    annotate: bool = True,
):
    """Scatter plot of pointwise agent embeddings in disc game k.

    Args:
        result: PTAResult.
        k: disc game index (0-based).
        labels: optional agent labels.
        ax: optional Axes.
        annotate: if True and labels provided, annotate points.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()

    Y_k = result.embeddings[:, k, :]  # (N, 2)
    N = Y_k.shape[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(Y_k[:, 0], Y_k[:, 1], s=60, zorder=2)

    if labels is not None and annotate:
        for i, lbl in enumerate(labels):
            ax.annotate(
                lbl, (Y_k[i, 0], Y_k[i, 1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=9,
            )

    omega = result.eigenvalues[k]
    ax.set_title(f"PTA Disc Game {k + 1} ($\\omega_{k + 1}$ = {omega:.4f})")
    ax.set_xlabel(f"$Y_1^{{({k + 1})}}$")
    ax.set_ylabel(f"$Y_2^{{({k + 1})}}$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Draw origin
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    return fig, ax


def plot_pta_spinning_top(
    result,
    transitive_skill: Array | None = None,
    k: int = 0,
    labels: list[str] | None = None,
    ax=None,
):
    """3D 'spinning top' plot: transitive axis (skill) vs first disc game.

    The vertical axis represents transitive strength (e.g., Elo rating or
    row sum of F), while the horizontal plane is disc game k.

    Args:
        result: PTAResult.
        transitive_skill: skill values, shape (N,). If None, uses row sums of
            the reconstructed payoff matrix.
        k: disc game index for the horizontal plane.
        labels: optional agent labels.
        ax: optional 3D Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _import_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Y_k = result.embeddings[:, k, :]  # (N, 2)

    if transitive_skill is None:
        F_hat = result.reconstruct()
        transitive_skill = jnp.sum(F_hat, axis=1)  # row sums as proxy for skill

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    ax.scatter(Y_k[:, 0], Y_k[:, 1], transitive_skill, s=60, zorder=2)

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.text(Y_k[i, 0], Y_k[i, 1], transitive_skill[i], lbl, fontsize=8)

    ax.set_xlabel(f"$Y_1^{{({k + 1})}}$")
    ax.set_ylabel(f"$Y_2^{{({k + 1})}}$")
    ax.set_zlabel("Transitive Strength")
    ax.set_title("Spinning Top Visualization")

    return fig, ax
