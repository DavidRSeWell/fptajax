"""Self-supervised contrastive pretraining for HierarchicalSetEncoder.

Attacks the trait-collapse problem observed in behavioral FPTA: even with
10k–40k training pairs and a properly-sized encoder, joint FPTA training
converges to a fixed point where all agents have nearly identical traits
and predictions are tiny.

The idea here is to force the encoder, before FPTA training starts, to
produce consistent traits for the same agent across different game samples
while keeping different agents separated. Implemented with an InfoNCE loss:

  - For each agent i in a batch, draw two disjoint samples of G_sample
    games. Encode both into traits; call them anchor_i and positive_i.
  - Treat the two samples from the same agent as a positive pair; treat
    samples from other agents in the batch as negatives.
  - InfoNCE(anchor, positives, all) with cosine-similarity / temperature.

After pretraining, the encoder is handed to hierarchical_behavioral_fpta
via the ``pretrained_encoder`` argument.

Requires: pip install fptajax[neural]
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

try:
    import equinox as eqx
    import optax
except ImportError:
    raise ImportError(
        "Contrastive pretraining requires equinox and optax. "
        "Install with: pip install fptajax[neural]"
    )

from fptajax.hierarchical import HierarchicalSetEncoder


# ---------------------------------------------------------------------------
# Paired sampling
# ---------------------------------------------------------------------------


def _sample_two_disjoint(
    all_games: np.ndarray,
    all_token_mask: np.ndarray,
    all_game_mask: np.ndarray,
    agent_idx: np.ndarray,
    G_sample: int,
    rng: np.random.RandomState,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
           tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each agent in ``agent_idx``, draw two disjoint subsets of G_sample
    games each. If an agent has fewer than 2*G_sample valid games, we fall
    back to random subsets with replacement.

    Returns two tuples (games, token_mask, game_mask) — anchor and positive.
    """
    B = len(agent_idx)
    L = all_games.shape[2]
    td = all_games.shape[3]

    def _empty():
        return (
            np.zeros((B, G_sample, L, td), dtype=all_games.dtype),
            np.zeros((B, G_sample, L), dtype=bool),
            np.zeros((B, G_sample), dtype=bool),
        )
    anc_g, anc_tm, anc_gm = _empty()
    pos_g, pos_tm, pos_gm = _empty()

    for b in range(B):
        a = int(agent_idx[b])
        valid_idx = np.where(all_game_mask[a])[0]
        n_valid = len(valid_idx)

        if n_valid >= 2 * G_sample:
            sel = rng.choice(valid_idx, size=2 * G_sample, replace=False)
            a_sel, p_sel = sel[:G_sample], sel[G_sample:]
        elif n_valid >= G_sample + 1:
            # Fall back: anchor takes first G_sample, positive takes a random
            # subset of the remaining (with replacement if too few).
            a_sel = rng.choice(valid_idx, size=G_sample, replace=False)
            remaining = np.setdiff1d(valid_idx, a_sel, assume_unique=False)
            if len(remaining) >= G_sample:
                p_sel = rng.choice(remaining, size=G_sample, replace=False)
            else:
                p_sel = rng.choice(remaining, size=G_sample, replace=True)
        else:
            # Tiny pool: sample with replacement from whatever is there.
            a_sel = rng.choice(valid_idx, size=G_sample, replace=True)
            p_sel = rng.choice(valid_idx, size=G_sample, replace=True)

        anc_g[b] = all_games[a, a_sel]
        anc_tm[b] = all_token_mask[a, a_sel]
        anc_gm[b, :] = True
        pos_g[b] = all_games[a, p_sel]
        pos_tm[b] = all_token_mask[a, p_sel]
        pos_gm[b, :] = True

    return (anc_g, anc_tm, anc_gm), (pos_g, pos_tm, pos_gm)


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------


def _info_nce_loss(
    encoder: HierarchicalSetEncoder,
    anchor_games: Array, anchor_tmask: Array, anchor_gmask: Array,
    pos_games: Array, pos_tmask: Array, pos_gmask: Array,
    temperature: float,
) -> tuple[Array, dict]:
    """Symmetric InfoNCE loss.

    Anchors and positives are B traits each. Build a (2B, 2B) cosine-sim
    matrix; for each of the 2B rows the positive is at the paired index
    (anchor_i <-> positive_i), all other rows are negatives.
    """
    a = encoder.encode_batch(anchor_games, anchor_tmask, anchor_gmask)  # (B, D)
    p = encoder.encode_batch(pos_games, pos_tmask, pos_gmask)            # (B, D)

    z = jnp.concatenate([a, p], axis=0)  # (2B, D)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    B = a.shape[0]
    sim = (z @ z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity on the diagonal (set to -inf before softmax).
    big_neg = jnp.finfo(sim.dtype).min
    sim = sim - jnp.eye(2 * B) * (jnp.abs(big_neg) + 1e5)

    # Target: each row's positive is at index (i + B) mod 2B.
    targets = jnp.concatenate([jnp.arange(B, 2 * B), jnp.arange(0, B)])

    log_probs = jax.nn.log_softmax(sim, axis=-1)
    nll = -log_probs[jnp.arange(2 * B), targets]
    loss = jnp.mean(nll)

    # Accuracy: how often is the top-1 non-self neighbour the true positive?
    pred = jnp.argmax(sim, axis=-1)
    acc = jnp.mean((pred == targets).astype(jnp.float32))

    # Trait spread diagnostics
    trait_std = jnp.mean(jnp.sqrt(jnp.var(jnp.concatenate([a, p], axis=0), axis=0) + 1e-6))

    return loss, {"loss": loss, "acc": acc, "trait_std": trait_std}


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------


@dataclass
class ContrastivePretrainResult:
    encoder: HierarchicalSetEncoder
    train_history: list


def contrastive_pretrain(
    agent_games: np.ndarray,
    agent_token_mask: np.ndarray,
    agent_game_mask: np.ndarray,
    token_dim: int,
    L_max: int,
    trait_dim: int = 24,
    d_model: int = 32,
    n_heads: int = 2,
    n_layers: int = 1,
    mlp_ratio: int = 2,
    rho_hidden: tuple[int, ...] = (32,),
    n_steps: int = 500,
    batch_size: int = 16,
    G_sample: int = 4,
    lr: float = 3e-4,
    temperature: float = 0.2,
    grad_clip: float = 1.0,
    log_every: int = 50,
    key: Array | None = None,
    numpy_seed: int = 0,
    verbose: bool = True,
    encoder: HierarchicalSetEncoder | None = None,
) -> ContrastivePretrainResult:
    """Contrastively pretrain HierarchicalSetEncoder on (agent_games, ...).

    Args:
        agent_games, agent_token_mask, agent_game_mask: same layout as for
            hierarchical_behavioral_fpta. (N, G_max, L_max, token_dim) etc.
        token_dim, L_max, trait_dim, d_model, n_heads, n_layers, mlp_ratio,
            rho_hidden: encoder architecture. Ignored if ``encoder`` is given.
        n_steps: training iterations.
        batch_size: number of agents per InfoNCE batch.
        G_sample: games per sample (each step draws 2*G_sample games/agent).
        lr: Adam learning rate.
        temperature: InfoNCE temperature. Lower = sharper separations.
        grad_clip: global-norm gradient clip.
        log_every: print interval.
        key: PRNG key for encoder init.
        numpy_seed: seed for numpy game-sampling.
        verbose: print diagnostics.
        encoder: optional pre-initialised encoder. If None, creates fresh.

    Returns:
        ContrastivePretrainResult with the pretrained encoder + history.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    agent_games = np.asarray(agent_games, dtype=np.float32)
    agent_token_mask = np.asarray(agent_token_mask, dtype=bool)
    agent_game_mask = np.asarray(agent_game_mask, dtype=bool)
    N = agent_games.shape[0]

    if encoder is None:
        encoder = HierarchicalSetEncoder(
            token_dim=token_dim, L_max=L_max, trait_dim=trait_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            mlp_ratio=mlp_ratio, rho_hidden=rho_hidden, key=key,
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr),
    )
    opt_state = optimizer.init(eqx.filter(encoder, eqx.is_array))

    @eqx.filter_jit
    def train_step(encoder, opt_state, ag, atm, agm, pg, ptm, pgm):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            _info_nce_loss, has_aux=True,
        )(encoder, ag, atm, agm, pg, ptm, pgm, temperature)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(encoder, eqx.is_array),
        )
        return eqx.apply_updates(encoder, updates), new_opt_state, metrics

    rng = np.random.RandomState(numpy_seed)
    history: list[dict] = []

    if verbose:
        print(f"  Contrastive pretraining: {n_steps} steps, batch={batch_size}, "
              f"G_sample={G_sample}, temperature={temperature}")

    for step in range(n_steps):
        # Sample B distinct agents if possible (batch_size <= N)
        size = min(batch_size, N)
        agent_batch = rng.choice(N, size=size, replace=False)

        (ag, atm, agm), (pg, ptm, pgm) = _sample_two_disjoint(
            agent_games, agent_token_mask, agent_game_mask,
            agent_batch, G_sample, rng,
        )
        encoder, opt_state, metrics = train_step(
            encoder, opt_state,
            jnp.array(ag), jnp.array(atm), jnp.array(agm),
            jnp.array(pg), jnp.array(ptm), jnp.array(pgm),
        )

        if step % log_every == 0 or step == n_steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            record = {"step": step, **m}
            history.append(record)
            if verbose:
                print(f"  step {step:5d} | nce_loss={m['loss']:.4f} | "
                      f"top1_acc={m['acc']:.3f} | trait_std={m['trait_std']:.4f}")

    if verbose:
        print("  Contrastive pretraining complete.")

    return ContrastivePretrainResult(encoder=encoder, train_history=history)
