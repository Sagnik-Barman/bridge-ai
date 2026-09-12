"""Trick-taking mechanics: follow-suit legality, trick-winner
determination, and the running state of a single board's card play
(13 tricks, four hands, one contract).

This is just the mechanical rules of trick-taking bridge — the same
regardless of bidding system or skill level — so, like `bidding/calls.py`,
it's drafted from first principles rather than any external source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.deal import Card, Hand, RANKS, Suit
from bidding.calls import SEATS, partner_of, seat_after


def _rank_value(card: Card) -> int:
    return RANKS.index(card.rank)


def _card_strength(card: Card, led_suit: Suit, trump: Optional[Suit]) -> Tuple[int, int]:
    """Sort key: trump beats everything, then the led suit, then cards
    that can't possibly win the trick. Higher tuple = stronger."""
    if trump is not None and card.suit == trump:
        return (2, _rank_value(card))
    if card.suit == led_suit:
        return (1, _rank_value(card))
    return (0, _rank_value(card))


@dataclass
class Trick:
    leader: str
    plays: List[Tuple[str, Card]] = field(default_factory=list)

    def led_suit(self) -> Optional[Suit]:
        return self.plays[0][1].suit if self.plays else None

    def is_complete(self) -> bool:
        return len(self.plays) == 4

    def winner(self, trump: Optional[Suit]) -> str:
        led_suit = self.led_suit()
        best_seat, _ = max(self.plays, key=lambda sc: _card_strength(sc[1], led_suit, trump))
        return best_seat


def legal_plays(hand_cards: List[Card], led_suit: Optional[Suit]) -> List[Card]:
    """Must follow suit if able; otherwise anything in hand is legal."""
    if led_suit is None:
        return list(hand_cards)
    same_suit = [c for c in hand_cards if c.suit == led_suit]
    return same_suit if same_suit else list(hand_cards)


class IllegalPlayError(ValueError):
    pass


@dataclass
class PlayState:
    hands: Dict[str, List[Card]]  # mutated as cards are played
    declarer: str
    strain: str  # "C","D","H","S","NT"
    trump: Optional[Suit]
    dealer: str  # for reference/undo bookkeeping only

    current_trick: Trick = field(init=False)
    completed_tricks: List[Trick] = field(default_factory=list)
    next_to_play: str = field(init=False)
    opening_lead_made: bool = False

    def __post_init__(self):
        self.next_to_play = seat_after(self.declarer, 1)  # LHO of declarer leads
        self.current_trick = Trick(leader=self.next_to_play)

    @property
    def dummy(self) -> str:
        return partner_of(self.declarer)

    def legal_cards_for(self, seat: str) -> List[Card]:
        if seat != self.next_to_play:
            return []
        return legal_plays(self.hands[seat], self.current_trick.led_suit())

    def play_card(self, seat: str, card: Card) -> None:
        if seat != self.next_to_play:
            raise IllegalPlayError(f"It's {self.next_to_play}'s turn, not {seat}'s.")
        legal = legal_plays(self.hands[seat], self.current_trick.led_suit())
        if card not in legal:
            raise IllegalPlayError(f"{card} is not a legal play for {seat} (must follow suit if able).")

        self.hands[seat].remove(card)
        self.current_trick.plays.append((seat, card))

        if seat == seat_after(self.declarer, 1) and len(self.completed_tricks) == 0 and len(self.current_trick.plays) == 1:
            self.opening_lead_made = True

        if self.current_trick.is_complete():
            winner = self.current_trick.winner(self.trump)
            self.completed_tricks.append(self.current_trick)
            self.next_to_play = winner
            self.current_trick = Trick(leader=winner)
        else:
            self.next_to_play = seat_after(seat, 1)

    def is_complete(self) -> bool:
        return len(self.completed_tricks) == 13

    def tricks_by_side(self) -> Dict[str, int]:
        counts = {"NS": 0, "EW": 0}
        for trick in self.completed_tricks:
            side = "NS" if trick.winner(self.trump) in ("N", "S") else "EW"
            counts[side] += 1
        return counts

    def declarer_tricks(self) -> int:
        declaring_side = "NS" if self.declarer in ("N", "S") else "EW"
        return self.tricks_by_side()[declaring_side]

    # -- snapshot/restore for undo (game/session.py uses this to let the
    # human step back one card at a time without re-dealing the board) --

    def clone_state(self) -> dict:
        return {
            "hands": {s: list(cards) for s, cards in self.hands.items()},
            "completed_tricks": [Trick(t.leader, list(t.plays)) for t in self.completed_tricks],
            "current_trick": Trick(self.current_trick.leader, list(self.current_trick.plays)),
            "next_to_play": self.next_to_play,
            "opening_lead_made": self.opening_lead_made,
        }

    def restore_state(self, snapshot: dict) -> None:
        self.hands = {s: list(cards) for s, cards in snapshot["hands"].items()}
        self.completed_tricks = [Trick(t.leader, list(t.plays)) for t in snapshot["completed_tricks"]]
        self.current_trick = Trick(snapshot["current_trick"].leader, list(snapshot["current_trick"].plays))
        self.next_to_play = snapshot["next_to_play"]
        self.opening_lead_made = snapshot["opening_lead_made"]
