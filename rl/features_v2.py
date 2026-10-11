"""State encoding for the v2 card-play agent.

Two ideas make this much easier to learn from than rl/features.py (v1):

1. **Trump-canonical suits.** Suits are reordered so the trump suit is always
   suit 0 (no trump: S, H, D, C as usual). The network learns "how to use
   trumps" once instead of four separate times.

2. **Relative ranks.** A card is described by how many still-unplayed cards of
   its suit outrank it, not by its printed rank: once the A and K of spades are
   gone, the Q plays exactly like an ace, so it is encoded exactly like one.
   This is the same observation behind the relative-rank transposition-table
   keys in this project's own double-dummy solver (docs/dds_from_scratch.md).
   Index of a card = canonical_suit * 13 + relative_rank (0 = top card).

The 52 policy outputs use the same index space, so the action "play my top
trump" is the same output unit in every deal.

Two observation modes:

- `"full"`: all four hands visible. Double-dummy play; matches the oracle's
  own view. Used for analysis and as a reference.
- `"visible"`: only what the player to act can actually see at the table:
  its own hand, dummy (once the opening lead is made; declarer always sees
  it), the cards still unseen as a pool, the hidden hands' sizes, which
  players have shown out of which suits, and the cards played so far by each
  player. This is the mode the app uses, so no sampling (PIMC) is needed at
  play time. See docs/rl_training.md ("v2") for why training it on the true
  double-dummy values amounts to learning PIMC's average over the hidden
  hands, with a learned rather than uniform belief about them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.deal import Card, RANKS, Suit
from bidding.calls import partner_of, seat_after
from cardplay.trick_engine import PlayState, legal_plays, _card_strength

N_ACTIONS = 52
_BASE_ORDER = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]

# planes of 52
_FULL_PLANES = 4 + 3          # four hands (relative to mover) + up to three trick cards
_VIS_PLANES = 4 + 1 + 3 + 4   # visible hands (4 slots), unseen pool, trick cards, played-by-seat
_SCALARS = (
    1      # no trump?
    + 4    # mover's role: declarer, LHO, dummy, RHO
    + 4    # position in the current trick (1st..4th)
    + 3    # tricks: mover's side / other side / still to play (each / 13)
    + 1    # partner currently winning this trick
    + 3    # which trick card (slot 0..2) is currently winning
)
_VIS_EXTRA = 16 + 3           # shown-out flags (4 seats x 4 suits) + hidden hand sizes / 13

FULL_SIZE = _FULL_PLANES * 52 + _SCALARS
VISIBLE_SIZE = _VIS_PLANES * 52 + _SCALARS + _VIS_EXTRA


def feature_size(obs: str) -> int:
    return FULL_SIZE if obs == "full" else VISIBLE_SIZE


def suit_order(trump: Optional[Suit]) -> List[Suit]:
    if trump is None:
        return list(_BASE_ORDER)
    return [trump] + [s for s in _BASE_ORDER if s != trump]


class Indexer:
    """Maps cards <-> the canonical (suit, relative-rank) index for one position."""

    def __init__(self, ps: PlayState):
        order = suit_order(ps.trump)
        self.suit_pos = {s: i for i, s in enumerate(order)}
        unplayed = [c for h in ps.hands.values() for c in h]
        unplayed += [c for _, c in ps.current_trick.plays]
        by_suit: Dict[Suit, List[int]] = {s: [] for s in Suit}
        for c in unplayed:
            by_suit[c.suit].append(RANKS.index(c.rank))
        self._rel: Dict[Tuple[Suit, int], int] = {}
        for s, ranks in by_suit.items():
            for i, r in enumerate(sorted(ranks, reverse=True)):
                self._rel[(s, r)] = i
        self._card_at: Dict[int, Card] = {}
        for c in unplayed:
            self._card_at[self.index(c)] = c

    def index(self, c: Card) -> int:
        return self.suit_pos[c.suit] * 13 + self._rel[(c.suit, RANKS.index(c.rank))]

    def card(self, idx: int) -> Card:
        return self._card_at[idx]


def _role(seat: str, ps: PlayState) -> int:
    if seat == ps.declarer:
        return 0
    if seat == seat_after(ps.declarer, 1):
        return 1
    if seat == ps.dummy:
        return 2
    return 3


def _side_of(seat: str) -> str:
    return "NS" if seat in ("N", "S") else "EW"


def visible_seats(seat: str, ps: PlayState) -> set:
    """Hands the player choosing `seat`'s card can see. Declarer chooses for
    dummy, so both of them see declarer + dummy."""
    decl_side = {ps.declarer, ps.dummy}
    if seat in decl_side:
        return set(decl_side)
    vis = {seat}
    if ps.opening_lead_made:
        vis.add(ps.dummy)
    return vis


def encode(seat: str, ps: PlayState, obs: str = "full", idx: Optional[Indexer] = None) -> np.ndarray:
    idx = idx or Indexer(ps)
    x = np.zeros(feature_size(obs), dtype=np.float32)
    rel_seats = [seat_after(seat, k) for k in range(4)]  # me, LHO, partner, RHO
    o = 0

    def plane(cards):
        nonlocal o
        for c in cards:
            x[o + idx.index(c)] = 1.0
        o += 52

    if obs == "full":
        for s in rel_seats:
            plane(ps.hands[s])
    else:
        vis = visible_seats(seat, ps)
        for s in rel_seats:
            plane(ps.hands[s] if s in vis else [])
        hidden = [c for s in rel_seats if s not in vis for c in ps.hands[s]]
        plane(hidden)  # the pool: who holds which of these is unknown

    trick = [c for _, c in ps.current_trick.plays]
    for i in range(3):
        plane([trick[i]] if i < len(trick) else [])

    if obs != "full":
        # cards each player has played in completed tricks (relative ranks are
        # computed among unplayed cards, so mark these by their old position in
        # the suit order instead: canonical suit * 13 + (12 - printed rank))
        for s in rel_seats:
            for t in ps.completed_tricks:
                for who, c in t.plays:
                    if who == s:
                        x[o + idx.suit_pos[c.suit] * 13 + (12 - RANKS.index(c.rank))] = 1.0
            o += 52

    # scalars
    x[o] = 1.0 if ps.trump is None else 0.0
    o += 1
    x[o + _role(seat, ps)] = 1.0
    o += 4
    x[o + len(trick)] = 1.0
    o += 4
    my_side = _side_of(seat)
    won = {"NS": 0, "EW": 0}
    for t in ps.completed_tricks:
        won[_side_of(t.winner(ps.trump))] += 1
    other = "EW" if my_side == "NS" else "NS"
    x[o] = won[my_side] / 13.0
    x[o + 1] = won[other] / 13.0
    x[o + 2] = len(ps.hands[seat]) / 13.0
    o += 3
    if trick:
        led = ps.current_trick.led_suit()
        best_i = max(range(len(trick)), key=lambda i: _card_strength(trick[i], led, ps.trump))
        best_seat = ps.current_trick.plays[best_i][0]
        x[o] = 1.0 if best_seat == partner_of(seat) else 0.0
        x[o + 1 + best_i] = 1.0
    o += 4

    if obs != "full":
        for k, s in enumerate(rel_seats):
            for t in ps.completed_tricks + [ps.current_trick]:
                led = t.led_suit()
                for who, c in t.plays:
                    if who == s and led is not None and c.suit != led:
                        x[o + k * 4 + idx.suit_pos[led]] = 1.0
        o += 16
        vis = visible_seats(seat, ps)
        for k, s in enumerate(rel_seats[1:]):
            x[o + k] = (0 if s in vis else len(ps.hands[s])) / 13.0
        o += 3

    assert o == x.size, (o, x.size)
    return x


def legal_mask(seat: str, ps: PlayState, idx: Optional[Indexer] = None) -> np.ndarray:
    idx = idx or Indexer(ps)
    m = np.zeros(N_ACTIONS, dtype=np.float32)
    for c in legal_plays(ps.hands[seat], ps.current_trick.led_suit()):
        m[idx.index(c)] = 1.0
    return m
