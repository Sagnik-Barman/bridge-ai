"""A from-scratch classical search engine: minimax with alpha-beta pruning
and memoization, solving small bridge endgames EXACTLY.

Honest scope — read this before treating it as "a DDS": a real
double-dummy solver (dds-bridge/dds, wrapped in engine/dds_wrapper.py)
handles full 13-trick deals in milliseconds because of years of
specialized optimization, the single biggest of which is suit-symmetry
reduction — recognizing that, say, the 4 and 2 of a suit are
interchangeable once no higher card of that suit separates them, which
collapses the branching factor enormously. This module does NOT implement
that (or the other classical DDS optimizations — move ordering heuristics,
quick tricks, partial-search bounds). It is plain minimax + alpha-beta +
a transposition table, which is exactly tractable for SMALL endgames (a
handful of tricks) and exponentially not tractable for a full deal. That
tradeoff is the point: this exists to demonstrate the search technique
itself (a genuine DSA/algorithms project — game trees, pruning, memoized
search, complexity) and to cross-check the real DDS solver on positions
small enough for both to solve, not to replace it. `game/session.py`'s
"Analyze" feature uses this only when few enough cards remain, and falls
back to the real DDS/PIMC path (`cardplay/dds_player.py`) otherwise.

How small is "small enough" — measured, not guessed: on ordinary
hardware, 5 cards left per hand solves in well under a second; 6 usually
finishes in a second or two; 7-8 frequently exhaust the 200,000-node
budget below without finishing at all (i.e. `SearchTooLargeError`, not a
wrong answer — this solver never returns an unverified result). That's
why `max_cards_per_hand` defaults to 6, not 8: a limit that usually can't
actually finish isn't a meaningfully higher limit, just a slower way to
hit the same error. Raise it if you want to gamble on a particular
position finishing anyway (nothing stops you from trying 8), but 6 is
the number this module can promise, not just attempt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from engine.deal import Card
from cardplay.trick_engine import PlayState, Trick

NS = {"N", "S"}


class SearchTooLargeError(RuntimeError):
    """Raised when the position has too many cards left for an exact,
    unoptimized search to be a reasonable idea (see module docstring), or
    when the search actually blows through its node budget trying anyway."""


@dataclass
class EndgameResult:
    best_card: Card
    ns_tricks_from_here: int
    nodes_explored: int


def _side(seat: str) -> str:
    return "NS" if seat in NS else "EW"


def _state_key(ps: PlayState) -> tuple:
    # Canonical enough to be a correct memoization key (exact remaining
    # cards per hand + who's played what into the current trick + whose
    # turn) — NOT suit-symmetry-reduced, which is exactly the optimization
    # this module deliberately skips (see module docstring).
    hands_key = tuple(tuple(sorted(str(c) for c in ps.hands[s])) for s in ("N", "E", "S", "W"))
    trick_key = tuple((seat, str(card)) for seat, card in ps.current_trick.plays)
    return (hands_key, trick_key, ps.next_to_play)


def _ns_gain(ps: PlayState, tricks_before: int) -> int:
    """1 if the card just played completed a trick that N-S won, else 0."""
    if len(ps.completed_tricks) > tricks_before:
        return 1 if ps.completed_tricks[-1].winner(ps.trump) in NS else 0
    return 0


def _value(ps: PlayState, alpha: float, beta: float, memo: Dict[tuple, Tuple[float, float]], nodes: list, budget: int) -> int:
    """N-S tricks from this position on (not counting tricks already completed).

    Values are relative to the position, so the memo key (which says nothing about how
    many tricks each side has already won) stays correct. The memo stores BOUNDS, not
    values: a result found with an alpha-beta cutoff anywhere below it is only a bound.
    (An earlier version cached such results as exact values, and used absolute trick
    counts with a key that ignores the tricks already won; both gave wrong answers on
    some positions. Found by tests/test_ddsolver.py, which checks this solver against
    a brute-force search and against ddsolver/.)
    """
    # Out of cards = nothing left to win. Checked on the hands, not on
    # ps.is_complete() (13 completed tricks): positions built by trimming
    # hands (tests, the Analyzer's what-ifs) end before trick 13, and
    # without this check the search hit an empty move list and returned
    # an infinite value (OverflowError in int()).
    if ps.is_complete() or not ps.hands[ps.next_to_play]:
        return 0

    nodes[0] += 1
    if nodes[0] > budget:
        raise SearchTooLargeError(
            f"Exact search exceeded its node budget ({budget}) without finishing — "
            f"this position is too large for this unoptimized solver (see module "
            f"docstring); use the DDS/PIMC path instead."
        )

    key = _state_key(ps)
    lo, hi = memo.get(key, (float("-inf"), float("inf")))
    if lo >= beta:
        return int(lo)
    if hi <= alpha:
        return int(hi)
    if lo == hi:
        return int(lo)
    a, b = max(alpha, lo), min(beta, hi)

    seat = ps.next_to_play
    legal = ps.legal_cards_for(seat)
    maximizing = _side(seat) == "NS"
    value = float("-inf") if maximizing else float("inf")

    for card in legal:
        snapshot = ps.clone_state()
        before = len(ps.completed_tricks)
        ps.play_card(seat, card)
        gained = _ns_gain(ps, before)
        child_value = gained + _value(ps, a - gained, b - gained, memo, nodes, budget)
        ps.restore_state(snapshot)

        if maximizing:
            value = max(value, child_value)
            a = max(a, value)
        else:
            value = min(value, child_value)
            b = min(b, value)
        if a >= b:
            break

    # what the search proved, given the window it ran with
    a0, b0 = max(alpha, lo), min(beta, hi)
    if value <= a0:
        hi = min(hi, value)      # failed low: the true value is at most this
    elif value >= b0:
        lo = max(lo, value)      # failed high: the true value is at least this
    else:
        lo = hi = value          # inside the window: exact
    memo[key] = (lo, hi)
    return int(value)


def solve_endgame(play_state: PlayState, node_budget: int = 200_000, max_cards_per_hand: int = 6) -> EndgameResult:
    """Exactly solve the position from `play_state.next_to_play`'s turn
    onward: the best card to play now, how many tricks NS will end up
    with from this point forward under optimal play by both sides, and
    how many search nodes it took (a genuine, honest number to quote —
    contrast it with how fast the real DDS solver does the equivalent
    full-deal problem, in `docs/endgame_solver.md`).

    Operates on a private clone internally — `play_state` itself is left
    untouched no matter what happens inside the search.
    """
    remaining = len(play_state.hands[play_state.next_to_play])
    if remaining > max_cards_per_hand:
        raise SearchTooLargeError(
            f"{remaining} cards left per hand — this unoptimized solver (no "
            f"suit-symmetry reduction, unlike real DDS) is only attempted up to "
            f"{max_cards_per_hand}; it grows far too slowly for anything bigger. "
            f"Use the DDS/PIMC path for full-size positions."
        )

    ps = PlayState(
        hands={s: list(cs) for s, cs in play_state.hands.items()},
        declarer=play_state.declarer,
        strain=play_state.strain,
        trump=play_state.trump,
        dealer=play_state.dealer,
    )
    # Deep-copy, not alias: `Trick` is a mutable dataclass, and the search
    # below calls the real `play_card()`, which appends in place to
    # `current_trick.plays`. Aliasing `play_state.current_trick` here (as
    # an earlier version of this function did) meant that very first
    # append mutated the CALLER's live trick object too -- silently
    # corrupting whatever real game state `play_state` belonged to,
    # directly contradicting this function's own "play_state itself is
    # left untouched" promise. Caught via benchmark/expert_benchmark.py's
    # replay tests, which are exactly the kind of long, realistic
    # sequence of solve_endgame() calls needed to expose it.
    ps.completed_tricks = [Trick(t.leader, list(t.plays)) for t in play_state.completed_tricks]
    ps.current_trick = Trick(play_state.current_trick.leader, list(play_state.current_trick.plays))
    ps.next_to_play = play_state.next_to_play
    ps.opening_lead_made = play_state.opening_lead_made

    memo: Dict[tuple, Tuple[float, float]] = {}
    nodes = [0]

    seat = ps.next_to_play
    side = _side(seat)
    legal = ps.legal_cards_for(seat)
    maximizing = side == "NS"
    best_value = float("-inf") if maximizing else float("inf")
    best_card: Optional[Card] = None
    alpha, beta = float("-inf"), float("inf")

    for card in legal:
        snapshot = ps.clone_state()
        before = len(ps.completed_tricks)
        ps.play_card(seat, card)
        gained = _ns_gain(ps, before)
        value = gained + _value(ps, alpha - gained, beta - gained, memo, nodes, node_budget)
        ps.restore_state(snapshot)

        improves = value > best_value if maximizing else value < best_value
        if improves or best_card is None:
            best_value = value
            best_card = card
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)

    assert best_card is not None
    return EndgameResult(
        best_card=best_card,
        ns_tricks_from_here=int(best_value),
        nodes_explored=nodes[0],
    )
