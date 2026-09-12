"""Call representation, legality, and auction-outcome helpers.

Shared infrastructure used by the auction engine (to filter/validate calls),
the game session (to detect when the auction ends and derive the contract +
declarer), and the web frontend (to enable/disable bidding-box buttons).

None of this is convention-specific — it's just the mechanical rules of
how a bridge auction is structured, which are the same regardless of which
bidding system two partnerships are playing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

STRAINS = ["C", "D", "H", "S", "NT"]  # ascending rank within a level
SEATS = ["N", "E", "S", "W"]

# All 35 suit/NT bids in ascending rank order: 1C, 1D, 1H, 1S, 1NT, 2C, ...
SUIT_BID_ORDER: List[str] = [f"{level}{strain}" for level in range(1, 8) for strain in STRAINS]


def is_suit_bid(call: str) -> bool:
    return call in SUIT_BID_ORDER


def bid_rank(call: str) -> int:
    """Index into SUIT_BID_ORDER; higher = a more expensive/later bid."""
    return SUIT_BID_ORDER.index(call)


def seat_after(seat: str, n: int = 1) -> str:
    return SEATS[(SEATS.index(seat) + n) % 4]


def partner_of(seat: str) -> str:
    return seat_after(seat, 2)


@dataclass
class AuctionState:
    """Everything derivable purely from (dealer, calls) needed to decide
    what's legal next and, once the auction is over, what the contract is.
    """

    dealer: str
    calls: List[str]

    def whose_turn(self) -> str:
        return seat_after(self.dealer, len(self.calls))

    def add(self, call: str) -> None:
        self.calls.append(call)

    def last_suit_bid(self) -> Optional[str]:
        for call in reversed(self.calls):
            if is_suit_bid(call):
                return call
        return None

    def last_suit_bid_seat(self) -> Optional[str]:
        for i in range(len(self.calls) - 1, -1, -1):
            if is_suit_bid(self.calls[i]):
                return seat_after(self.dealer, i)
        return None

    def current_double_state(self) -> str:
        """'' (undoubled), 'X' (doubled), or 'XX' (redoubled) — resets to ''
        whenever a new suit bid is made.
        """
        state = ""
        for call in self.calls:
            if is_suit_bid(call):
                state = ""
            elif call == "Dbl":
                state = "X"
            elif call == "Rdbl":
                state = "XX"
        return state

    def is_complete(self) -> bool:
        if len(self.calls) < 4:
            return False
        if all(c == "Pass" for c in self.calls):
            return True  # passed out
        return len(self.calls) >= 4 and all(c == "Pass" for c in self.calls[-3:])

    def is_passed_out(self) -> bool:
        return len(self.calls) == 4 and all(c == "Pass" for c in self.calls)

    def legal_calls(self) -> List[str]:
        if self.is_complete():
            return []
        legal = ["Pass"]
        last_suit = self.last_suit_bid()
        last_suit_seat = self.last_suit_bid_seat()
        dbl_state = self.current_double_state()
        turn = self.whose_turn()

        # New suit bids: anything ranked above the last suit bid (or any
        # bid at all if none has been made yet).
        start_rank = bid_rank(last_suit) + 1 if last_suit else 0
        legal.extend(SUIT_BID_ORDER[start_rank:])

        # Double: only of an opponent's undoubled bid.
        if last_suit is not None and dbl_state == "" and last_suit_seat is not None:
            is_opponent = (SEATS.index(turn) - SEATS.index(last_suit_seat)) % 2 == 1
            if is_opponent:
                legal.append("Dbl")

        # Redouble: only of the opponents' double on our side's bid.
        if dbl_state == "X" and last_suit_seat is not None:
            is_partnership = (SEATS.index(turn) - SEATS.index(last_suit_seat)) % 2 == 0
            if is_partnership:
                legal.append("Rdbl")

        return legal

    def declarer_and_contract(self):
        """Once complete (and not passed out), return
        (declarer_seat, level, strain, double_state) — the standard rule
        that declarer is whichever partner of the winning side *first*
        bid the contract's strain, not necessarily whoever made the final
        bid.
        """
        if not self.is_complete() or self.is_passed_out():
            return None
        final_bid = self.last_suit_bid()
        final_seat = self.last_suit_bid_seat()
        level = int(final_bid[0])
        strain = final_bid[1:]
        dbl_state = self.current_double_state()

        # Find the first call of this strain made by final_seat or their
        # partner.
        partnership = {final_seat, partner_of(final_seat)}
        for i, call in enumerate(self.calls):
            if is_suit_bid(call) and call[1:] == strain:
                seat = seat_after(self.dealer, i)
                if seat in partnership:
                    return (seat, level, strain, dbl_state)
        return (final_seat, level, strain, dbl_state)  # should not happen
