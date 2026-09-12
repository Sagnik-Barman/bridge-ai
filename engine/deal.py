"""Card, Hand, and Deal representations, plus random deal generation.

Kept independent of any specific DDS binding so it can be unit-tested
without native dependencies installed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Suit(str, Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SEATS = ["N", "E", "S", "W"]


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: str  # one of RANKS

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.value}"


def parse_card(text: str) -> "Card":
    """Inverse of str(card): 'AH' -> Card(HEARTS, 'A'), 'TC' -> Card(CLUBS, 'T').
    Used by the web app to turn a card string from the frontend back into
    a Card object.
    """
    text = text.strip().upper()
    if len(text) < 2:
        raise ValueError(f"Not a valid card string: {text!r}")
    rank, suit_char = text[:-1], text[-1]
    if rank not in RANKS:
        raise ValueError(f"Unrecognized rank in {text!r}")
    return Card(Suit(suit_char), rank)


@dataclass
class Hand:
    cards: List[Card] = field(default_factory=list)

    def by_suit(self) -> Dict[Suit, List[Card]]:
        out: Dict[Suit, List[Card]] = {s: [] for s in Suit}
        for c in self.cards:
            out[c.suit].append(c)
        for suit_cards in out.values():
            suit_cards.sort(key=lambda c: RANKS.index(c.rank), reverse=True)
        return out

    def hcp(self) -> int:
        """High card points: A=4, K=3, Q=2, J=1."""
        points = {"A": 4, "K": 3, "Q": 2, "J": 1}
        return sum(points.get(c.rank, 0) for c in self.cards)

    def suit_lengths(self) -> Dict[Suit, int]:
        by_suit = self.by_suit()
        return {s: len(cards) for s, cards in by_suit.items()}

    def is_balanced(self) -> bool:
        """5-4-2-2, 4-3-3-3, or 4-4-3-2 shape, no singleton/void."""
        lengths = sorted(self.suit_lengths().values(), reverse=True)
        if min(self.suit_lengths().values()) < 2:
            return False
        return lengths in ([4, 3, 3, 3], [4, 4, 3, 2], [5, 4, 2, 2])

    def pbn_str(self) -> str:
        by_suit = self.by_suit()
        return ".".join(
            "".join(c.rank for c in by_suit[s])
            for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
        )

    def __str__(self) -> str:
        return self.pbn_str()


@dataclass
class Deal:
    hands: Dict[str, Hand]  # keyed by seat: "N", "E", "S", "W"
    dealer: str = "N"
    vulnerable: str = "None"  # "None", "NS", "EW", "All"

    def pbn(self) -> str:
        order = ["N", "E", "S", "W"]
        start = order.index(self.dealer)
        rotated = order[start:] + order[:start]
        hand_strs = [self.hands[seat].pbn_str() for seat in rotated]
        return f"{self.dealer}:" + " ".join(hand_strs)

    def __str__(self) -> str:
        lines = [f"Dealer: {self.dealer}  Vulnerable: {self.vulnerable}"]
        for seat in ["N", "E", "S", "W"]:
            lines.append(f"  {seat}: {self.hands[seat]}")
        return "\n".join(lines)


def parse_pbn_deal(pbn_str: str, dealer: str | None = None, vulnerable: str = "None") -> "Deal":
    """Inverse of `Deal.pbn()`: 'N:AKQ.T98.76.5432 ...' -> Deal.

    The seat letter before the colon is just where the space-separated
    hand list *starts* (per the PBN spec) — it is not necessarily this
    deal's real dealer, especially when the string came from an external
    tool (e.g. `endplay`'s `Deal.to_pbn()`, which always starts from
    North). Pass `dealer=` explicitly when you know the real dealer from
    elsewhere (e.g. a real tournament board's own dealer field); it
    defaults to the starting seat in the string if not given, matching
    how `Deal.pbn()` writes it.
    """
    pbn_str = pbn_str.strip()
    start_seat, sep, rest = pbn_str.partition(":")
    start_seat = start_seat.strip().upper()
    if not sep or start_seat not in SEATS:
        raise ValueError(f"Unrecognized starting seat in PBN deal string: {pbn_str!r}")

    hand_strs = rest.split()
    if len(hand_strs) != 4:
        raise ValueError(f"Expected 4 hands in PBN deal string, got {len(hand_strs)}: {pbn_str!r}")

    order = ["N", "E", "S", "W"]
    start_idx = order.index(start_seat)
    rotated_seats = order[start_idx:] + order[:start_idx]

    hands: Dict[str, Hand] = {}
    for seat, hand_str in zip(rotated_seats, hand_strs):
        suit_strs = hand_str.split(".")
        if len(suit_strs) != 4:
            raise ValueError(f"Expected 4 suits (S.H.D.C) in hand {hand_str!r} within {pbn_str!r}")
        cards: List[Card] = []
        for suit, ranks in zip((Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS), suit_strs):
            for r in ranks:
                if r not in RANKS:
                    raise ValueError(f"Unrecognized rank {r!r} in hand {hand_str!r} within {pbn_str!r}")
                cards.append(Card(suit, r))
        hands[seat] = Hand(cards=cards)

    return Deal(hands=hands, dealer=dealer or start_seat, vulnerable=vulnerable)


def deal_random(dealer: str = "N", vulnerable: str = "None", seed: int | None = None) -> Deal:
    """Deal a random, unconstrained 52-card deal into 4 hands of 13."""
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in RANKS]
    rng.shuffle(deck)
    hands = {
        seat: Hand(cards=deck[i * 13 : (i + 1) * 13])
        for i, seat in enumerate(["N", "E", "S", "W"])
    }
    return Deal(hands=hands, dealer=dealer, vulnerable=vulnerable)


if __name__ == "__main__":
    d = deal_random(seed=42)
    print(d)
    print("PBN:", d.pbn())
    print("North HCP:", d.hands["N"].hcp())
