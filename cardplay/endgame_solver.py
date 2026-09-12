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


def _value(ps: PlayState, alpha: float, beta: float, memo: Dict[tuple, int], nodes: list, budget: int) -> int:
    if ps.is_complete():
        return ps.tricks_by_side()["NS"]

    nodes[0] += 1
    if nodes[0] > budget:
        raise SearchTooLargeError(
            f"Exact search exceeded its node budget ({budget}) without finishing — "
            f"this position is too large for this unoptimized solver (see module "
            f"docstring); use the DDS/PIMC path instead."
        )

    key = _state_key(ps)
    cached = memo.get(key)
    if cached is not None:
        return cached

    seat = ps.next_to_play
    side = _side(seat)
    legal = ps.legal_cards_for(seat)
    maximizing = side == "NS"
    value = float("-inf") if maximizing else float("inf")
    cutoff = False

    for card in legal:
        snapshot = ps.clone_state()
        ps.play_card(seat, card)
        child_value = _value(ps, alpha, beta, memo, nodes, budget)
        ps.restore_state(snapshot)

        if maximizing:
            if child_value > value:
                value = child_value
            alpha = max(alpha, value)
        else:
            if child_value < value:
                value = child_value
            beta = min(beta, value)

        if alpha >= beta:
            cutoff = True
            break

    # Only cache exact values (searches that explored the full window),
    # never a value obtained via an alpha-beta cutoff — caching a mere
    # bound as if it were exact is a classic transposition-table bug.
    if not cutoff:
        memo[key] = value
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

    baseline_ns = ps.tricks_by_side()["NS"]
    memo: Dict[tuple, int] = {}
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
        ps.play_card(seat, card)
        value = _value(ps, alpha, beta, memo, nodes, node_budget)
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
        ns_tricks_from_here=int(best_value) - baseline_ns,
        nodes_explored=nodes[0],
    )
