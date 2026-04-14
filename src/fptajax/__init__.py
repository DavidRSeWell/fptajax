"""fptajax: Functional Principal Tradeoff Analysis in JAX.

A library for decomposing functional form games into sequences of disc games
via spectral embedding. Implements both FPTA (functional, continuous) and
PTA (pointwise, empirical) approaches.

References:
    Bai, Liu, Sun, Sewell, Strang. "Functional Principal Trade-off Analysis:
    Universal Approximation via Disc Game Embedding." (2024)

    Strang, Sewell, Kim, Alcedo, Rosenbluth. "Principal Trade-off Analysis."
    Information Visualization (2024).
"""

__version__ = "0.1.0"

# Core algorithms
from fptajax.core import fpta, FPTAResult
from fptajax.pta import pta, fpta_empirical, PTAResult, FPTAEmpiricalResult

# Basis function families
from fptajax.basis import (
    BasisFamily,
    FourierBasis,
    ChebyshevBasis,
    LegendreBasis,
    JacobiBasis,
    HermiteBasis,
    LaguerreBasis,
    MonomialBasis,
    CustomBasis,
    gram_schmidt,
)

# Quadrature
from fptajax.quad import (
    QuadRule,
    gauss_legendre,
    gauss_chebyshev,
    gauss_hermite,
    gauss_laguerre,
    gauss_jacobi,
    trapezoidal,
    empirical,
)

# Utilities
from fptajax.utils import disc, disc_embedding, make_skew_symmetric

# Decomposition
from fptajax.decomposition import skew_symmetric_schur, general_real_schur

# Neural basis (optional — requires equinox + optax)
try:
    from fptajax.neural import (
        NeuralBasis,
        NeuralFPTAResult,
        TrainConfig,
        neural_fpta,
        neural_fpta_from_matrix,
    )
    from fptajax.behavioral import (
        SetEncoder,
        BehavioralFPTAResult,
        behavioral_fpta,
    )
    from fptajax.online import (
        OnlinePlayer,
        play_game,
        evaluate_online,
    )
    _HAS_NEURAL = True
except ImportError:
    _HAS_NEURAL = False

__all__ = [
    # Core
    "fpta",
    "FPTAResult",
    "pta",
    "fpta_empirical",
    "PTAResult",
    "FPTAEmpiricalResult",
    # Basis
    "BasisFamily",
    "FourierBasis",
    "ChebyshevBasis",
    "LegendreBasis",
    "JacobiBasis",
    "HermiteBasis",
    "LaguerreBasis",
    "MonomialBasis",
    "CustomBasis",
    "gram_schmidt",
    # Quadrature
    "QuadRule",
    "gauss_legendre",
    "gauss_chebyshev",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_jacobi",
    "trapezoidal",
    "empirical",
    # Utilities
    "disc",
    "disc_embedding",
    "make_skew_symmetric",
    # Decomposition
    "skew_symmetric_schur",
    "general_real_schur",
    # Neural (optional)
    "NeuralBasis",
    "NeuralFPTAResult",
    "TrainConfig",
    "neural_fpta",
    "neural_fpta_from_matrix",
    # Behavioral (optional)
    "SetEncoder",
    "BehavioralFPTAResult",
    "behavioral_fpta",
    # Online (optional)
    "OnlinePlayer",
    "play_game",
    "evaluate_online",
]
