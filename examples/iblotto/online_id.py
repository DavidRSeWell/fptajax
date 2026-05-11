"""Online opponent identification + Thompson best-response in disc-game space.

Loads a ``disc_direct_bc`` checkpoint and exposes:

  * ``DiscBCCheckpoint`` — the loaded model plus per-population trait,
    disc, and skill summaries used to score best responses.
  * ``OpponentPosterior`` — particle filter over opponent trait, scored by
    ``p(action | state, trait)`` from the BC head, with Thompson sampling
    and ESS-triggered resampling.
  * ``population_best_response`` — given a sampled opponent trait (via
    Thompson or posterior mean), pick the population agent whose disc
    embedding maximises predicted ``F`` against it.

This module provides primitives only. Plugging the loop into
``simulate.py`` / ``run_tournament.py`` to actually drive play is left to
the caller.

Typical use:

    ckpt = load_disc_direct_bc("disc_direct_bc_checkpoints/run0",
                               "examples/iblotto/results/...pkl")
    posterior = OpponentPosterior.from_population(ckpt)
    rng = np.random.RandomState(0)
    for state_ctx, action in observed_stream:        # numpy arrays
        posterior.update(state_ctx, action)
    z_opp = posterior.thompson_sample(rng)            # one trait sample
    agent_idx, scores = population_best_response(ckpt, z_opp)
    # then play the policy of training-population agent ``agent_idx``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from examples.iblotto.behavioral import drop_dead_agents
from examples.iblotto.disc_direct_bc import (
    DiscFPTABCModel, _build_model, _dirichlet_logpdf,
)
from fptajax.hierarchical import _sample_games


# ---------------------------------------------------------------------------
# Checkpoint container
# ---------------------------------------------------------------------------


@dataclass
class DiscBCCheckpoint:
    """Loaded ``disc_direct_bc`` model plus pre-computed population summary."""
    model: DiscFPTABCModel
    meta: dict
    population_traits: np.ndarray   # (N, trait_dim)
    population_disc: np.ndarray     # (N, K, 2)
    population_skill: np.ndarray    # (N,)
    n_zones: int
    state_ctx_dim: int


def load_disc_direct_bc(
    ckpt_dir: str | Path,
    bundle_path: str | Path,
    seed: int = 0,
    G_sample_eval: int | None = None,
) -> DiscBCCheckpoint:
    """Load a ``disc_direct_bc`` checkpoint and encode the training population.

    The bundle is needed because particles in the prior are the training
    agents' encoded traits — the encoder needs games to run against. We
    drop dead agents the same way training did, so indices match the
    in-memory ``ds.policies`` order after ``drop_dead_agents``.
    """
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    if meta.get("basis_kind") != "disc_direct_bc":
        raise ValueError(
            f"checkpoint at {ckpt_dir} has basis_kind={meta.get('basis_kind')!r}, "
            f"expected 'disc_direct_bc'"
        )

    with open(bundle_path, "rb") as f:
        ds = pickle.load(f)
    if np.any(~np.all(np.isfinite(ds.agent_data), axis=-1)):
        ds.agent_data = np.where(np.isfinite(ds.agent_data),
                                 ds.agent_data, 0.0).astype(np.float32)
    ds, _ = drop_dead_agents(ds, verbose=False)

    key = jax.random.PRNGKey(0)  # arbitrary; weights overwritten on load.
    skel = _build_model(
        sa_dim=meta["sa_dim"], n_zones=meta["n_zones"], L_max=meta["L_max"],
        trait_dim=meta["trait_dim"], d_model=meta["d_model"],
        n_layers=meta["n_layers"], n_heads=meta["n_heads"],
        disc_hidden=meta["disc_hidden"], behavior_hidden=meta["behavior_hidden"],
        K=meta["K"], alpha_floor=meta["alpha_floor"], key=key,
    )
    encoder = eqx.tree_deserialise_leaves(str(ckpt_dir / "encoder.eqx"),
                                          skel.encoder)
    skill_head = eqx.tree_deserialise_leaves(str(ckpt_dir / "skill_head.eqx"),
                                             skel.skill_head)
    disc_head = eqx.tree_deserialise_leaves(str(ckpt_dir / "disc_head.eqx"),
                                            skel.disc_head)
    behavior_head = eqx.tree_deserialise_leaves(
        str(ckpt_dir / "behavior_head.eqx"), skel.behavior_head)
    model = DiscFPTABCModel(
        encoder=encoder, skill_head=skill_head,
        disc_head=disc_head, behavior_head=behavior_head,
        K=skel.K, n_zones=skel.n_zones,
        sa_dim=skel.sa_dim, alpha_floor=skel.alpha_floor,
    )

    if G_sample_eval is None:
        G_sample_eval = min(8, ds.G_max)
    rng = np.random.RandomState(seed)
    eval_games, eval_tmask, eval_gmask = _sample_games(
        ds.agent_data, ds.agent_token_mask, ds.agent_game_mask,
        np.arange(ds.policies.shape[0]), G_sample_eval, rng,
    )
    traits = np.asarray(model.encoder.encode_batch(
        jnp.asarray(eval_games), jnp.asarray(eval_tmask), jnp.asarray(eval_gmask),
    ))
    Z = np.asarray(jax.vmap(model.disc_head)(jnp.asarray(traits)))
    Z = Z.reshape(traits.shape[0], meta["K"], 2)
    s = np.asarray(jax.vmap(model.skill_head)(jnp.asarray(traits)))

    return DiscBCCheckpoint(
        model=model, meta=meta,
        population_traits=traits, population_disc=Z, population_skill=s,
        n_zones=meta["n_zones"],
        state_ctx_dim=meta["sa_dim"] - meta["n_zones"],
    )


# ---------------------------------------------------------------------------
# Best response in disc-game space
# ---------------------------------------------------------------------------


def disc_embedding_from_trait(ckpt: DiscBCCheckpoint,
                              z_trait: np.ndarray) -> np.ndarray:
    """Map a (single) trait vector to its disc embedding ``(K, 2)``."""
    z = np.asarray(jax.vmap(ckpt.model.disc_head)(
        jnp.asarray(z_trait)[None, :]))[0]
    return z.reshape(ckpt.meta["K"], 2)


def population_best_response(
    ckpt: DiscBCCheckpoint,
    z_opp_trait: np.ndarray,
    use_skill: bool | None = None,
) -> tuple[int, np.ndarray]:
    """Pick the population agent maximising predicted F vs ``z_opp_trait``.

    Returns ``(agent_idx, scores)`` where ``scores`` is the predicted F
    against the opponent for every population agent.
    """
    if use_skill is None:
        use_skill = bool(ckpt.meta.get("use_skill", True))
    opp_disc = disc_embedding_from_trait(ckpt, z_opp_trait)        # (K, 2)
    pop = ckpt.population_disc                                     # (N, K, 2)
    disc = (pop[..., 0] * opp_disc[..., 1]
            - pop[..., 1] * opp_disc[..., 0]).sum(axis=-1)         # (N,)
    if use_skill:
        # Opponent's skill is a constant across candidates → drops out
        # of the argmax. Keep the population skill term so ``scores`` is
        # the actual predicted F up to that constant.
        scores = ckpt.population_skill + disc
    else:
        scores = disc
    return int(np.argmax(scores)), np.asarray(scores)


# ---------------------------------------------------------------------------
# Particle filter posterior
# ---------------------------------------------------------------------------


@dataclass
class OpponentPosterior:
    """Particle posterior over opponent trait, scored by the BC head.

    Particles live in trait space (``trait_dim``). The prior is built from
    the encoded training population (one particle per population agent),
    which is a coarse but useful kernel when the unknown opponent
    plausibly resembles a training-time policy. For broader priors,
    perturb the particles or supply your own at construction.
    """
    particles: np.ndarray          # (M, trait_dim)
    log_weights: np.ndarray        # (M,)
    behavior_head: eqx.nn.MLP
    n_zones: int
    state_ctx_dim: int
    alpha_floor: float
    eps_smooth: float = 1e-3
    ess_resample_threshold: float = 0.5

    @classmethod
    def from_population(
        cls, ckpt: DiscBCCheckpoint,
        ess_resample_threshold: float = 0.5,
    ) -> "OpponentPosterior":
        M = ckpt.population_traits.shape[0]
        return cls(
            particles=ckpt.population_traits.copy(),
            log_weights=np.full(M, -np.log(M)),
            behavior_head=ckpt.model.behavior_head,
            n_zones=ckpt.n_zones,
            state_ctx_dim=ckpt.state_ctx_dim,
            alpha_floor=float(ckpt.model.alpha_floor),
            ess_resample_threshold=ess_resample_threshold,
        )

    # ----- core update step ------------------------------------------------

    def _logp_per_particle(self, state_ctx: np.ndarray,
                           action: np.ndarray) -> np.ndarray:
        a_smooth = (action + self.eps_smooth) \
            / (1.0 + self.n_zones * self.eps_smooth)
        traits = jnp.asarray(self.particles)
        s = jnp.broadcast_to(jnp.asarray(state_ctx),
                             (traits.shape[0], self.state_ctx_dim))
        inp = jnp.concatenate([traits, s], axis=-1)
        raw = jax.vmap(self.behavior_head)(inp)
        alpha = jax.nn.softplus(raw) + self.alpha_floor
        return np.asarray(_dirichlet_logpdf(jnp.asarray(a_smooth), alpha))

    def update(self, state_ctx: np.ndarray, action: np.ndarray,
               rng: np.random.RandomState | None = None) -> None:
        """Bayes update: ``w_m ∝ w_m · p(action | state_ctx, particle_m)``.

        Resamples (multinomial) if the effective sample size drops below
        ``ess_resample_threshold * M``. Pass an ``rng`` to make the
        resampling reproducible.
        """
        logp = self._logp_per_particle(np.asarray(state_ctx),
                                       np.asarray(action))
        self.log_weights = self.log_weights + logp
        m = float(np.max(self.log_weights))
        log_z = m + np.log(np.sum(np.exp(self.log_weights - m)))
        self.log_weights = self.log_weights - log_z

        w = np.exp(self.log_weights)
        ess = 1.0 / np.sum(w ** 2)
        if ess < self.ess_resample_threshold * len(w):
            self._resample(w, rng)

    def _resample(self, w: np.ndarray,
                  rng: np.random.RandomState | None) -> None:
        choose = (rng if rng is not None else np.random).choice
        idx = choose(len(w), size=len(w), replace=True, p=w)
        self.particles = self.particles[idx]
        self.log_weights = np.full(len(w), -np.log(len(w)))

    # ----- queries ---------------------------------------------------------

    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    def effective_sample_size(self) -> float:
        w = self.weights()
        return 1.0 / np.sum(w ** 2)

    def posterior_mean(self) -> np.ndarray:
        w = self.weights()
        return (w[:, None] * self.particles).sum(axis=0)

    def thompson_sample(self, rng: np.random.RandomState) -> np.ndarray:
        w = self.weights()
        idx = rng.choice(len(w), p=w)
        return self.particles[idx]

    def map_particle(self) -> np.ndarray:
        return self.particles[int(np.argmax(self.log_weights))]

    def top_k(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, weights)`` of the ``k`` highest-weight particles."""
        order = np.argsort(self.log_weights)[::-1][:k]
        return order, np.exp(self.log_weights[order])


# ---------------------------------------------------------------------------
# Convenience: split a token into (state_ctx, action)
# ---------------------------------------------------------------------------


def split_token(token: np.ndarray, n_zones: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Split a behavioral token into ``(state_ctx, action)`` pieces.

    Inverse of the layout in ``examples/iblotto/behavioral.py``: the last
    ``n_zones`` dims are ``own_share`` (the action), everything before is
    state context. Works for any leading shape; splits along axis -1.
    """
    return token[..., :-n_zones], token[..., -n_zones:]
