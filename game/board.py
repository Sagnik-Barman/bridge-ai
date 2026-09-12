"""Standard duplicate-bridge board numbering: dealer and vulnerability
cycle on a 16-board repeat.

This is the universal convention printed on every duplicate board and
used at every bridge club/tournament — public-domain mechanics, not
sourced from any book. It repeats every 16 boards:

  Board  1: Dealer N, Vul None      Board  9: Dealer N, Vul EW
  Board  2: Dealer E, Vul NS        Board 10: Dealer E, Vul All
  Board  3: Dealer S, Vul EW        Board 11: Dealer S, Vul None
  Board  4: Dealer W, Vul All       Board 12: Dealer W, Vul NS
  Board  5: Dealer N, Vul NS        Board 13: Dealer N, Vul All
  Board  6: Dealer E, Vul EW        Board 14: Dealer E, Vul None
  Board  7: Dealer S, Vul All       Board 15: Dealer S, Vul NS
  Board  8: Dealer W, Vul None      Board 16: Dealer W, Vul EW
"""
from __future__ import annotations

from dataclasses import dataclass

SEATS = ["N", "E", "S", "W"]

# Index 0 == board 1.
VULNERABILITY_CYCLE = [
    "None", "NS", "EW", "All",
    "NS", "EW", "All", "None",
    "EW", "All", "None", "NS",
    "All", "None", "NS", "EW",
]


@dataclass(frozen=True)
class BoardInfo:
    number: int
    dealer: str
    vulnerable: str  # "None", "NS", "EW", "All"

    def is_vulnerable(self, side: str) -> bool:
        """`side` is "NS" or "EW"."""
        return self.vulnerable == "All" or self.vulnerable == side


def board_info(number: int) -> BoardInfo:
    if number < 1:
        raise ValueError("Board numbers start at 1.")
    dealer = SEATS[(number - 1) % 4]
    vulnerable = VULNERABILITY_CYCLE[(number - 1) % 16]
    return BoardInfo(number=number, dealer=dealer, vulnerable=vulnerable)


def user_seat_for_board(number: int) -> str:
    """Rotates which seat the human plays, one step per board, so a long
    session works through all four positions in turn. This is a practice
    convenience Sagnik asked for, not a real-world team-movement rule —
    real IMP teams don't literally rotate seats like this."""
    return SEATS[(number - 1) % 4]
