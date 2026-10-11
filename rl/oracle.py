"""Double-dummy oracle: the exact value of every legal card at any position.

For the player to act, `card_values(play_state)` returns, for each legal card,
how many of the remaining tricks (the current trick included) the mover's side
takes if that card is played now and everybody plays perfectly afterwards
(double dummy). The best card's value is V*(s); any other card's shortfall
V*(s) - Q*(s, a) is exactly the number of tricks that card throws away.

This is what the v2 agent learns from (see docs/rl_training.md, "v2"): a
per-card, per-decision signal, instead of one trick count at the end of a hand.

Two backends give identical answers (checked in tests/test_rl_v2.py and, at
scale, by benchmark/dds_scratch_benchmark.py's `positions` command):

  - "dds":     the DDS library through endplay. About 0.16 s to score all 52
               decisions of a game. Used for bulk label generation.
  - "scratch": this project's own solver (ddsolver/, C++). Same answers, about
               40x slower on early-hand positions (a median game took ~6 s here).

`backend="auto"` picks DDS when endplay is installed and the scratch solver
otherwise.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from engine.deal import Card, Suit
from cardplay.trick_engine import PlayState

_SUIT_ORDER = "SHDC"
_RANK_ORDER = "AKQJT98765432"
_EP_SUIT = {"spades": Suit.SPADES, "hearts": Suit.HEARTS, "diamonds": Suit.DIAMONDS, "clubs": Suit.CLUBS}


def _hand_pbn(cards) -> str:
    by = {s: [] for s in _SUIT_ORDER}
    for c in cards:
        by[c.suit.value].append(c.rank)
    return ".".join("".join(sorted(by[s], key=_RANK_ORDER.index)) for s in _SUIT_ORDER)


def endplay_available() -> bool:
    try:
        from endplay.dds import solve_board  # noqa: F401
        return True
    except Exception:
        return False


def _values_dds(ps: PlayState) -> Dict[Card, int]:
    from endplay.dds import solve_board
    from endplay.types import Deal as EDeal, Denom, Player

    # endplay wants the hands as they were at the start of the current trick,
    # then the trick's cards replayed onto it.
    hands = {s: list(ps.hands[s]) for s in "NESW"}
    for seat, c in ps.current_trick.plays:
        hands[seat].append(c)
    d = EDeal("N:" + " ".join(_hand_pbn(hands[s]) for s in "NESW"))
    d.trump = Denom.nt if ps.trump is None else {
        "S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds, "C": Denom.clubs}[ps.trump.value]
    d.first = Player("NESW".index(ps.current_trick.leader))
    for _, c in ps.current_trick.plays:
        d.play(f"{c.suit.value}{c.rank}")
    out: Dict[Card, int] = {}
    for ecard, tricks in solve_board(d):
        # endplay's solve_board reports tricks for the side on play
        out[Card(_EP_SUIT[ecard.suit.name], ecard.rank.abbr)] = int(tricks)
    return out


def _values_scratch(ps: PlayState) -> Dict[Card, int]:
    import ddsolver

    scored, _ = ddsolver.score_play_state(ps)
    left = len(ps.hands[ps.next_to_play])  # tricks still to finish, the current one included
    mover_ns = ps.next_to_play in ("N", "S")
    return {card: (ns if mover_ns else left - ns) for card, ns in scored}


def card_values(ps: PlayState, backend: str = "auto") -> Dict[Card, int]:
    """{legal card: tricks the mover's side takes from here if it is played}."""
    if backend == "auto":
        backend = "dds" if endplay_available() else "scratch"
    if backend == "dds":
        return _values_dds(ps)
    if backend == "scratch":
        return _values_scratch(ps)
    raise ValueError(f"unknown backend {backend!r}")


def best_cards(values: Dict[Card, int]) -> Tuple[int, List[Card]]:
    """(V*, every card that achieves it)."""
    v = max(values.values())
    return v, [c for c, q in values.items() if q == v]
