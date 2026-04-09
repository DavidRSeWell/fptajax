"""Basis function families and Gram-Schmidt orthogonalization.

Each basis class provides:
- evaluate(x, n): evaluate the first n basis functions at points x
- The associated weight function and domain for natural quadrature
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from fptajax.quad import QuadRule, inner_product_1d


@dataclass
class BasisFamily:
    """A family of basis functions on a domain with associated weight."""

    name: str
    domain: tuple[float, float]

    def evaluate(self, x: Array, n: int) -> Array:
        """Evaluate the first n basis functions at points x.

        Args:
            x: evaluation points, shape (N,) or scalar.
            n: number of basis functions.

        Returns:
            Array of shape (N, n) where result[i, j] = b_j(x_i).
        """
        raise NotImplementedError

    def quad_rule(self, n_nodes: int) -> QuadRule:
        """Return a natural quadrature rule for this basis."""
        raise NotImplementedError


class FourierBasis(BasisFamily):
    """Fourier basis {1, cos(x), sin(x), cos(2x), sin(2x), ...} on [0, 2pi].

    Orthonormal with respect to the uniform measure (1/2pi) on [0, 2pi].
    Ordering: [1/sqrt(2pi), cos(x)/sqrt(pi), sin(x)/sqrt(pi),
               cos(2x)/sqrt(pi), sin(2x)/sqrt(pi), ...]
    """

    def __init__(self):
        super().__init__(name="fourier", domain=(0.0, 2 * jnp.pi))

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        result = []
        # Orthonormal w.r.t. uniform measure dx/(2*pi) on [0, 2*pi]:
        # <1, 1> = 1, so constant = 1
        # <cos(kx), cos(kx)> = 1/2, so normalize by sqrt(2)
        # <sin(kx), sin(kx)> = 1/2, so normalize by sqrt(2)
        result.append(jnp.ones_like(x))
        freq = 1
        while len(result) < n:
            result.append(jnp.sqrt(2.0) * jnp.cos(freq * x))
            if len(result) < n:
                result.append(jnp.sqrt(2.0) * jnp.sin(freq * x))
            freq += 1
        return jnp.stack(result[:n], axis=-1)

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import trapezoidal
        rule = trapezoidal(n_nodes, 0.0, 2 * jnp.pi)
        # Normalize weights for the uniform measure on [0, 2pi]
        # Trapezoidal gives weights summing to 2*pi; we want measure 1/(2pi)
        return QuadRule(rule.nodes, rule.weights / (2 * jnp.pi))


class ChebyshevBasis(BasisFamily):
    """Chebyshev polynomials of the first kind on [-1, 1].

    Orthogonal w.r.t. weight 1/sqrt(1 - x^2).
    Normalized so that <T_i, T_j>_w = delta_ij.
    """

    def __init__(self):
        super().__init__(name="chebyshev", domain=(-1.0, 1.0))

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        if n == 0:
            return jnp.empty((x.shape[0], 0))

        # Three-term recurrence: T_0=1, T_1=x, T_{n+1} = 2x T_n - T_{n-1}
        T = [jnp.ones_like(x)]
        if n > 1:
            T.append(x)
        for k in range(2, n):
            T.append(2 * x * T[-1] - T[-2])

        result = jnp.stack(T, axis=-1)

        # Normalize: ||T_0||^2 = pi, ||T_k||^2 = pi/2 for k >= 1
        norms = jnp.ones(n) * jnp.sqrt(2.0 / jnp.pi)
        norms = norms.at[0].set(1.0 / jnp.sqrt(jnp.pi))
        return result * norms[None, :]

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_chebyshev
        return gauss_chebyshev(n_nodes)


class LegendreBasis(BasisFamily):
    """Legendre polynomials on [-1, 1].

    Orthogonal w.r.t. uniform weight on [-1, 1].
    Normalized so that <P_i, P_j> = delta_ij w.r.t. measure dx/2.
    """

    def __init__(self):
        super().__init__(name="legendre", domain=(-1.0, 1.0))

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        if n == 0:
            return jnp.empty((x.shape[0], 0))

        # Three-term recurrence
        P = [jnp.ones_like(x)]
        if n > 1:
            P.append(x)
        for k in range(2, n):
            kf = float(k - 1)
            P.append(((2 * kf + 1) * x * P[-1] - kf * P[-2]) / (kf + 1))

        result = jnp.stack(P, axis=-1)

        # Normalize: ||P_k||^2 = 2/(2k+1) w.r.t. dx on [-1,1]
        # We use measure dx/2, so ||P_k||^2 = 1/(2k+1)
        ks = jnp.arange(n, dtype=jnp.float32)
        norms = jnp.sqrt(2 * ks + 1)
        return result * norms[None, :]

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_legendre
        rule = gauss_legendre(n_nodes, -1.0, 1.0)
        # Normalize for measure dx/2 on [-1,1]
        return QuadRule(rule.nodes, rule.weights / 2.0)


class JacobiBasis(BasisFamily):
    """Jacobi polynomials P_n^{(alpha, beta)} on (-1, 1).

    Orthogonal w.r.t. weight (1-x)^alpha (1+x)^beta.
    """

    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(name="jacobi", domain=(-1.0, 1.0))
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        a, b = self.alpha, self.beta
        if n == 0:
            return jnp.empty((x.shape[0], 0))

        P = [jnp.ones_like(x)]
        if n > 1:
            P.append(0.5 * ((a - b) + (a + b + 2) * x))

        for k in range(2, n):
            kf = float(k - 1)
            ab = a + b
            c1_num = (2 * kf + ab + 1) * (2 * kf + ab + 2)
            c1_den = 2 * (kf + 1) * (kf + ab + 1)
            c2_num = (a**2 - b**2) * (2 * kf + ab + 1)
            c2_den = 2 * (kf + 1) * (kf + ab + 1) * (2 * kf + ab)
            c3_num = (kf + a) * (kf + b) * (2 * kf + ab + 2)
            c3_den = (kf + 1) * (kf + ab + 1) * (2 * kf + ab)
            P.append(
                (c1_num / c1_den * x + c2_num / c2_den) * P[-1]
                - c3_num / c3_den * P[-2]
            )

        result = jnp.stack(P, axis=-1)

        # Normalize using the known L2 norm
        import scipy.special as sp
        norms = []
        for k in range(n):
            h_k = (2**(a + b + 1) / (2 * k + a + b + 1)
                   * sp.gamma(k + a + 1) * sp.gamma(k + b + 1)
                   / (sp.gamma(k + a + b + 1) * sp.factorial(k)))
            norms.append(1.0 / jnp.sqrt(h_k))
        norms = jnp.array(norms)
        return result * norms[None, :]

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_jacobi
        return gauss_jacobi(n_nodes, self.alpha, self.beta)


class HermiteBasis(BasisFamily):
    """Probabilist's Hermite polynomials on (-inf, inf).

    Orthogonal w.r.t. weight exp(-x^2/2) / sqrt(2*pi).
    Uses the physicist's convention internally then normalizes.
    """

    def __init__(self):
        super().__init__(name="hermite", domain=(-jnp.inf, jnp.inf))

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        if n == 0:
            return jnp.empty((x.shape[0], 0))

        # Physicist's Hermite: H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
        H = [jnp.ones_like(x)]
        if n > 1:
            H.append(2 * x)
        for k in range(2, n):
            H.append(2 * x * H[-1] - 2 * (k - 1) * H[-2])

        result = jnp.stack(H, axis=-1)

        # Normalize: ||H_n||^2 = 2^n n! sqrt(pi) w.r.t. exp(-x^2)
        import math
        norms = jnp.array([
            1.0 / jnp.sqrt(2**k * math.factorial(k) * jnp.sqrt(jnp.pi))
            for k in range(n)
        ])
        return result * norms[None, :]

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_hermite
        return gauss_hermite(n_nodes)


class LaguerreBasis(BasisFamily):
    """Laguerre polynomials on [0, inf).

    Orthogonal w.r.t. weight exp(-x).
    """

    def __init__(self):
        super().__init__(name="laguerre", domain=(0.0, jnp.inf))

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        if n == 0:
            return jnp.empty((x.shape[0], 0))

        # L_0=1, L_1=1-x, (k+1)L_{k+1} = (2k+1-x)L_k - k L_{k-1}
        L = [jnp.ones_like(x)]
        if n > 1:
            L.append(1.0 - x)
        for k in range(2, n):
            kf = float(k - 1)
            L.append(((2 * kf + 1 - x) * L[-1] - kf * L[-2]) / (kf + 1))

        result = jnp.stack(L, axis=-1)
        # Already orthonormal w.r.t. exp(-x): <L_i, L_j> = delta_ij
        return result

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_laguerre
        return gauss_laguerre(n_nodes)


class MonomialBasis(BasisFamily):
    """Monomial basis {1, x, x^2, ..., x^{n-1}}.

    NOT orthogonal — must be orthogonalized via Gram-Schmidt before use in FPTA.
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        super().__init__(name="monomial", domain=domain)

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        powers = jnp.arange(n)
        return x[:, None] ** powers[None, :]

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_legendre
        rule = gauss_legendre(n_nodes, self.domain[0], self.domain[1])
        length = self.domain[1] - self.domain[0]
        return QuadRule(rule.nodes, rule.weights / length)


class CustomBasis(BasisFamily):
    """User-provided basis functions.

    Each function should accept a 1D array x and return a 1D array of the same shape.
    """

    def __init__(
        self,
        functions: list[Callable[[Array], Array]],
        domain: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(name="custom", domain=domain)
        self._functions = functions

    def evaluate(self, x: Array, n: int) -> Array:
        x = jnp.atleast_1d(x)
        n = min(n, len(self._functions))
        return jnp.stack([fn(x) for fn in self._functions[:n]], axis=-1)

    def quad_rule(self, n_nodes: int) -> QuadRule:
        from fptajax.quad import gauss_legendre
        rule = gauss_legendre(n_nodes, self.domain[0], self.domain[1])
        length = self.domain[1] - self.domain[0]
        return QuadRule(rule.nodes, rule.weights / length)


def gram_schmidt(
    basis: BasisFamily | Array,
    quad: QuadRule,
    n: int | None = None,
) -> Array:
    """Modified Gram-Schmidt orthogonalization w.r.t. weighted inner product.

    Given basis functions evaluated at quadrature nodes, returns orthonormal
    basis function values at the same nodes.

    Args:
        basis: either a BasisFamily (will be evaluated at quad.nodes) or
            a pre-evaluated array of shape (N, m) where N = len(quad.nodes).
        quad: quadrature rule providing nodes and weights.
        n: number of basis functions to orthogonalize. If None, uses all.

    Returns:
        Array of shape (N, n) — orthonormal basis evaluated at quad nodes.
        Satisfies: sum_k w_k result[k, i] * result[k, j] = delta_ij
    """
    if isinstance(basis, BasisFamily):
        n = n or 10  # default
        V = basis.evaluate(quad.nodes, n)
    else:
        V = basis
        n = n or V.shape[1]
        V = V[:, :n]

    V = jnp.array(V, dtype=jnp.float32)
    w = quad.weights

    Q = jnp.zeros_like(V)
    for j in range(n):
        v = V[:, j]
        for i in range(j):
            qi = Q[:, i]
            # <qi, v>_w
            proj = jnp.sum(w * qi * v)
            v = v - proj * qi
        # Normalize
        norm = jnp.sqrt(jnp.sum(w * v * v))
        v = v / jnp.maximum(norm, 1e-15)
        Q = Q.at[:, j].set(v)

    return Q
