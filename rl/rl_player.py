"""Wraps a trained `PolicyValueNet` for actual (imperfect-information)
gameplay — the "Play against RL" opponent in the web app.

The network itself was trained on full information (see `rl/env.py`): at
training time it can see every hand because a self-play game is a fully
determined world. Real gameplay isn't like that — a defender can't see
declarer's and dummy's hidden cards. So, exactly like `cardplay/dds_player
.py` does for the exact DDS solver, this samples several plausible
completions of the unseen hands (PIMC), asks the network for its policy
in each fully-known sample, and combines the results — averaging the
per-sample probability the network assigns to each legal card and picking
the card with the highest combined score. This is the standard way to
turn a full-information game-playing policy into an imperfect-information
one without retraining it.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from engine.deal import Card, RANKS, SEATS, Suit
from cardplay.trick_engine import PlayState, legal_plays
from cardplay.dds_player import _known_hands_for, _sample_unseen_hands
from rl import features
from rl.network import PolicyValueNet

_CACHED_NET: Optional[PolicyValueNet] = None
_CACHED_PATH: Optional[str] = None


def load_checkpoint(path: str) -> PolicyValueNet:
    global _CACHED_NET, _CACHED_PATH
    if _CACHED_PATH != path:
        _CACHED_NET = PolicyValueNet.load(path)
        _CACHED_PATH = path
    return _CACHED_NET


def rl_checkpoint_status(path: str) -> str:
    import os

    if os.path.exists(path):
        return f"Loaded RL checkpoint from {path}."
    return (
        f"No trained checkpoint found at {path} — 'Play against RL' will fall "
        f"back to the heuristic bot until you train one (see docs/rl_training.md, "
        f"`python -m rl.train`)."
    )


def choose_card(
    seat: str,
    play_state: PlayState,
    net: PolicyValueNet,
    seed: Optional[int] = None,
    num_samples: int = 12,
) -> Card:
    hand = play_state.hands[seat]
    legal = legal_plays(hand, play_state.current_trick.led_suit())
    if len(legal) == 1:
        return legal[0]

    rng = random.Random(seed)
    known = _known_hands_for(seat, play_state)
    unseen_seats = [s for s in SEATS if s not in known]

    combined = np.zeros(features.N_CARDS, dtype=np.float64)
    if not unseen_seats:
        # Fully known position already (e.g. declarer near the end of the
        # hand): just ask the network once.
        x = features.encode(seat, play_state)
        mask = features.legal_mask(seat, play_state)
        combined = net.policy(x, mask)
    else:
        unseen_sizes = {s: len(play_state.hands[s]) for s in unseen_seats}
        for _ in range(num_samples):
            sampled = _sample_unseen_hands(known, unseen_sizes, rng)
            sample_hands = {s: list(play_state.hands[s]) for s in SEATS}
            for s in unseen_seats:
                sample_hands[s] = sampled[s]
            sample_state = PlayState(
                hands=sample_hands,
                declarer=play_state.declarer,
                strain=play_state.strain,
                trump=play_state.trump,
                dealer=play_state.dealer,
            )
            # Line up the mechanical state (whose turn, current trick,
            # tricks so far) with the real position — only the unseen
            # hands' contents were sampled, everything else is real.
            sample_state.completed_tricks = list(play_state.completed_tricks)
            sample_state.current_trick = play_state.current_trick
            sample_state.next_to_play = play_state.next_to_play
            sample_state.opening_lead_made = play_state.opening_lead_made

            x = features.encode(seat, sample_state)
            mask = features.legal_mask(seat, sample_state)
            combined += net.policy(x, mask)
        combined /= num_samples

    best_idx = int(np.argmax(combined))
    return features.card_for_index(best_idx)
