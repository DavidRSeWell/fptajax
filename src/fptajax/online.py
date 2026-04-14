"""Online action selection using a trained behavioral FPTA model.

Given a trained BehavioralFPTAResult, the OnlinePlayer accumulates opponent
observations during a game and selects actions to maximize predicted payoff.

Key efficiency trick: the DeepSets encoder computes trait = rho(mean_k phi(sa_k)).
Rather than re-encoding the full history every round, we maintain a running sum
of phi(sa_k) outputs. Each observe() call costs one phi() forward pass, and
each select_action() call costs n_actions phi() calls + (n_actions+1) rho() calls
+ (n_actions+1) basis() calls — all O(1) regardless of history length.

Requires: pip install fptajax[neural]  (equinox, optax)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

try:
    import equinox as eqx
except ImportError:
    raise ImportError(
        "Online player requires equinox. Install with: pip install fptajax[neural]"
    )

from fptajax.behavioral import BehavioralFPTAResult


class OnlinePlayer:
    """Select actions online using a trained behavioral FPTA model.

    Accumulates observations of both players during a game and uses the
    learned model to predict which action maximizes payoff against the
    opponent.

    Internally maintains running sums of phi(sa) for incremental DeepSets
    encoding, making per-round cost O(1) rather than O(history_length).

    Args:
        result: trained BehavioralFPTAResult.
        n_actions: number of possible discrete actions.
        state_fn: maps (player_actions, opponent_actions, round_idx,
            total_rounds) → state feature vector (np.ndarray). Must use
            only history *before* round_idx (i.e., actions[:round_idx]).
        sa_dim: dimensionality of each (state, action_onehot) vector.
    """

    def __init__(
        self,
        result: BehavioralFPTAResult,
        n_actions: int,
        state_fn: Callable[[list[int], list[int], int, int], np.ndarray],
        sa_dim: int,
    ):
        self.result = result
        self.n_actions = n_actions
        self.state_fn = state_fn
        self.sa_dim = sa_dim

        # Infer phi output dim from the encoder
        self._phi_dim = result.encoder.phi.out_size

        self.reset()

    def reset(self):
        """Reset state for a new game."""
        self.my_actions: list[int] = []
        self.opp_actions: list[int] = []
        # Running sums for incremental DeepSets encoding
        self._my_phi_sum = np.zeros(self._phi_dim, dtype=np.float32)
        self._opp_phi_sum = np.zeros(self._phi_dim, dtype=np.float32)
        self._n_me = 0
        self._n_opp = 0

    def select_action(self, round_idx: int, total_rounds: int) -> int:
        """Select the best action given history so far.

        On the first round (no opponent data), returns a random action.

        Args:
            round_idx: current round number (0-based).
            total_rounds: total rounds in the game.

        Returns:
            Action index to play.
        """
        if self._n_opp < 1:
            return int(np.random.randint(self.n_actions))

        # Opponent trait from running phi average
        opp_pooled = jnp.array(self._opp_phi_sum / self._n_opp)
        opp_trait = self.result.encoder.rho(opp_pooled)  # (trait_dim,)
        b_opp = self.result.basis(opp_trait)  # (d,)

        # Current state from our perspective
        state = self.state_fn(
            self.my_actions, self.opp_actions, round_idx, total_rounds,
        )

        C = self.result.coefficient_matrix  # (d, d)
        Cb_opp = C @ b_opp  # (d,) — precompute once

        # For each candidate action, compute predicted payoff
        best_a = 0
        best_payoff = -jnp.inf

        for a in range(self.n_actions):
            # Build (state, action_onehot)
            action_oh = np.zeros(self.n_actions, dtype=np.float32)
            action_oh[a] = 1.0
            sa = jnp.array(np.concatenate([state, action_oh]))

            # One phi call for the candidate
            phi_a = self.result.encoder.phi(sa)  # (phi_dim,)

            # My trait if I append this action
            my_pooled = (jnp.array(self._my_phi_sum) + phi_a) / (self._n_me + 1)
            my_trait = self.result.encoder.rho(my_pooled)  # (trait_dim,)
            b_me = self.result.basis(my_trait)  # (d,)

            payoff = b_me @ Cb_opp  # scalar
            if payoff > best_payoff:
                best_payoff = payoff
                best_a = a

        return int(best_a)

    def observe(self, my_action: int, opp_action: int,
                round_idx: int, total_rounds: int):
        """Record actions from the round just played.

        Updates the running phi sums for both players. Must be called
        after select_action() each round with the actions actually taken.

        Args:
            my_action: action taken by this player.
            opp_action: action taken by the opponent.
            round_idx: the round these actions were taken in.
            total_rounds: total rounds in the game.
        """
        self.my_actions.append(my_action)
        self.opp_actions.append(opp_action)

        # My (state, action) from my perspective
        state_me = self.state_fn(
            self.my_actions, self.opp_actions, round_idx, total_rounds,
        )
        action_oh_me = np.zeros(self.n_actions, dtype=np.float32)
        action_oh_me[my_action] = 1.0
        sa_me = jnp.array(np.concatenate([state_me, action_oh_me]))
        phi_me = self.result.encoder.phi(sa_me)
        self._my_phi_sum += np.array(phi_me)
        self._n_me += 1

        # Opponent (state, action) from opponent's perspective
        state_opp = self.state_fn(
            self.opp_actions, self.my_actions, round_idx, total_rounds,
        )
        action_oh_opp = np.zeros(self.n_actions, dtype=np.float32)
        action_oh_opp[opp_action] = 1.0
        sa_opp = jnp.array(np.concatenate([state_opp, action_oh_opp]))
        phi_opp = self.result.encoder.phi(sa_opp)
        self._opp_phi_sum += np.array(phi_opp)
        self._n_opp += 1


def play_game(
    player: OnlinePlayer,
    opponent_actions: list[int],
    total_rounds: int | None = None,
) -> dict:
    """Play a game between the online player and a recorded action sequence.

    The opponent's actions are replayed from a recording. The online player
    observes each opponent action and selects its own action using the model.

    Note: since the opponent's actions are pre-recorded, the opponent does NOT
    adapt to the online player's actions. This is an offline evaluation.

    Args:
        player: OnlinePlayer instance (will be reset).
        opponent_actions: the opponent's recorded action sequence.
        total_rounds: total number of rounds. Defaults to len(opponent_actions).

    Returns:
        Dict with: wins, losses, ties, n_rounds, win_rate, net_rate,
        my_actions (list of actions taken by the online player).
    """
    if total_rounds is None:
        total_rounds = len(opponent_actions)

    n_rounds = min(total_rounds, len(opponent_actions))
    n_actions = player.n_actions
    player.reset()

    wins = 0
    losses = 0
    ties = 0
    my_actions_log = []

    for r in range(n_rounds):
        my_action = player.select_action(r, total_rounds)
        opp_action = opponent_actions[r]

        # Record before scoring
        player.observe(my_action, opp_action, r, total_rounds)
        my_actions_log.append(my_action)

        # Generalized scoring: (my - opp) % n_actions == 1 means win
        # (works for RPS where n_actions=3)
        diff = (my_action - opp_action) % n_actions
        if diff == 0:
            ties += 1
        elif diff <= n_actions // 2:
            wins += 1
        else:
            losses += 1

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_rounds": n_rounds,
        "win_rate": wins / n_rounds if n_rounds > 0 else 0.0,
        "net_rate": (wins - losses) / n_rounds if n_rounds > 0 else 0.0,
        "my_actions": my_actions_log,
    }


def evaluate_online(
    result: BehavioralFPTAResult,
    opponent_action_sequences: dict[str, list[list[int]]],
    state_fn: Callable,
    n_actions: int,
    sa_dim: int,
    n_games: int = 5,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Evaluate the online player against a set of opponents.

    For each opponent, plays n_games by randomly selecting from their
    recorded action sequences. Reports per-opponent and aggregate stats.

    Args:
        result: trained BehavioralFPTAResult.
        opponent_action_sequences: {bot_name: [action_seq_1, action_seq_2, ...]}.
            Each action_seq is a list[int] of one recorded match's actions.
        state_fn: state feature function (see OnlinePlayer).
        n_actions: number of actions (3 for RPS).
        sa_dim: dimensionality of (state, action_onehot) vectors.
        n_games: number of games to play per opponent.
        verbose: print progress.

    Returns:
        Dict mapping bot_name → list of game result dicts.
    """
    player = OnlinePlayer(result, n_actions, state_fn, sa_dim)
    rng = np.random.RandomState(42)

    all_results: dict[str, list[dict]] = {}
    total_wins = 0
    total_losses = 0
    total_rounds = 0

    for name, sequences in opponent_action_sequences.items():
        opp_results = []
        for g in range(n_games):
            # Pick a random recorded match for this opponent
            seq_idx = rng.randint(len(sequences))
            opp_actions = sequences[seq_idx]

            game_result = play_game(player, opp_actions)
            opp_results.append(game_result)

            total_wins += game_result["wins"]
            total_losses += game_result["losses"]
            total_rounds += game_result["n_rounds"]

        all_results[name] = opp_results

        if verbose:
            avg_net = np.mean([r["net_rate"] for r in opp_results])
            print(f"  vs {name:30s}: avg net_rate={avg_net:+.3f} "
                  f"({n_games} games)")

    if verbose and total_rounds > 0:
        overall = (total_wins - total_losses) / total_rounds
        print(f"\n  Overall: net_rate={overall:+.3f} "
              f"({total_wins}W / {total_losses}L / "
              f"{total_rounds - total_wins - total_losses}T, "
              f"{total_rounds} rounds)")

    return all_results
