"""Fixed-size numeric state encoding for the card-play RL agent.

Every decision the network makes is from the mover's own point of view, so
the same network can play any of the four seats (that's what makes
self-play with one shared network meaningful): slot 0 is always "my
remaining cards", slots 1-3 are the other three seats in seat-order
starting from the mover, and the current trick is encoded as "the cards
played so far this trick, in playing order" rather than by absolute seat.

This module has no dependency on DDS/endplay and is pure NumPy, since no
deep-learning framework is available in this environment (see
`rl/network.py`).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from engine.deal import Card, RANKS, SEATS, Suit
from bidding.calls import seat_after
from cardplay.trick_engine import PlayState

STRAINS = ["C", "D", "H", "S", "NT"]

ALL_CARDS: List[Card] = [Card(s, r) for s in Suit for r in RANKS]
CARD_INDEX: Dict[Card, int] = {c: i for i, c in enumerate(ALL_CARDS)}
N_CARDS = 52

# own hand (52) + 3 other seats' remaining hands, relative order (3*52)
# + up to 3 current-trick cards in play order (3*52) + trump one-hot (5)
# + tricks won so far by mover's side / other side (2) + cards already
# played in the current trick, as a count (1)
FEATURE_SIZE = 52 + 3 * 52 + 3 * 52 + 5 + 2 + 1


def encode(seat: str, play_state: PlayState) -> np.ndarray:
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)
    offset = 0

    def plane(cards):
        nonlocal offset
        for c in cards:
            x[offset + CARD_INDEX[c]] = 1.0
        offset += N_CARDS

    plane(play_state.hands[seat])
    for k in (1, 2, 3):
        other = seat_after(seat, k)
        plane(play_state.hands[other])

    trick_cards = [card for _, card in play_state.current_trick.plays]
    for i in range(3):
        if i < len(trick_cards):
            plane([trick_cards[i]])
        else:
            offset += N_CARDS

    strain_idx = STRAINS.index(play_state.strain)
    x[offset + strain_idx] = 1.0
    offset += 5

    declaring_side = {play_state.declarer, play_state.dummy}
    mover_side = declaring_side if seat in declaring_side else (set(SEATS) - declaring_side)
    other_side = set(SEATS) - mover_side

    def side_tricks(side):
        return sum(1 for t in play_state.completed_tricks if t.winner(play_state.trump) in side)

    x[offset] = side_tricks(mover_side) / 13.0
    x[offset + 1] = side_tricks(other_side) / 13.0
    offset += 2

    x[offset] = len(trick_cards) / 3.0
    offset += 1

    assert offset == FEATURE_SIZE
    return x


def legal_mask(seat: str, play_state: PlayState) -> np.ndarray:
    from cardplay.trick_engine import legal_plays

    mask = np.zeros(N_CARDS, dtype=np.float32)
    for c in legal_plays(play_state.hands[seat], play_state.current_trick.led_suit()):
        mask[CARD_INDEX[c]] = 1.0
    return mask


def card_for_index(i: int) -> Card:
    return ALL_CARDS[i]
