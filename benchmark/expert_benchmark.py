"""Benchmark this project's card-play engines against real, recorded play
from actual bridge tournaments (World Championship finals and similar —
see `docs/real_deals.md` for where these come from and why that's a
different situation from the uploaded copyrighted book: these are
factual tournament records — who held what, what was bid, what was
played, what happened — not anyone's copyrighted analysis or commentary).

Deliberately endplay-independent: this module only knows about
`ExpertBoard`, a plain dataclass built from this project's own `Card` /
`Deal` types. The endplay-specific work of turning a real `.pbn` file
into `ExpertBoard`s lives in `benchmark/pbn_loader.py`, kept separate for
exactly the same reason `engine/dds_wrapper.py` isolates its
endplay-specific code — so this comparison logic can be fully unit-tested
with hand-built fixtures in any environment, endplay installed or not
(see tests/test_expert_benchmark.py).

How the comparison works: rather than letting our engines branch off into
their own hypothetical lines (which would compound small differences and
stop meaning anything after a few tricks), we replay the *exact* recorded
sequence of 52 cards from the real board. At each seat's turn, before
playing the real card, we ask each available engine what *it* would have
played in that exact, real position, and record whether it matches — then
we play the real card regardless of what any engine suggested, and move
on. This "move-matching" style of evaluation is the same idea used to
benchmark chess engines against grandmaster games, and it stays
meaningful trick after trick because every engine is always being asked
about a real position that actually occurred, never a hypothetical one
built from an earlier wrong guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.deal import Card, Deal, Suit
from cardplay.trick_engine import PlayState
from cardplay.bot_player import choose_card as heuristic_choose_card
from cardplay.dds_player import choose_card as dds_choose_card, dds_available
from cardplay.endgame_solver import solve_endgame, SearchTooLargeError

SOURCES = ("heuristic", "dds", "rl", "exact_endgame")
DEFAULT_ENDGAME_CARD_LIMIT = 6  # see cardplay/endgame_solver.py's docstring: 6 reliably finishes, 8 usually doesn't
_PARTNER = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class ExpertBoard:
    """One real, completely-played board, in this project's own types."""

    deal: Deal
    declarer: str  # "N" | "E" | "S" | "W"
    strain: str  # "C" | "D" | "H" | "S" | "NT"
    trump: Optional[Suit]  # None for NT
    play_order: List[Card]  # exactly 52 Cards, in the real order they were played
    board_num: Optional[int] = None
    event: Optional[str] = None


@dataclass
class DecisionRecord:
    trick_number: int  # 1-13
    seat: str
    is_declaring_side: bool
    actual_card: str
    forced: bool  # only one legal card here -- not a real decision to score
    matches: Dict[str, Optional[bool]] = field(default_factory=dict)  # source -> True/False/None(unavailable)


@dataclass
class BoardBenchmarkResult:
    board_num: Optional[int]
    event: Optional[str]
    actual_declarer_tricks: int
    decisions: List[DecisionRecord]

    def match_rate(self, source: str, declaring_side_only: bool = False) -> Optional[float]:
        """Fraction of *real, non-forced* decisions where `source`'s
        suggestion matched the card actually played — None if that source
        was never available to judge on this board."""
        relevant = [
            d for d in self.decisions
            if not d.forced and (not declaring_side_only or d.is_declaring_side)
        ]
        judged = [d for d in relevant if d.matches.get(source) is not None]
        if not judged:
            return None
        return sum(1 for d in judged if d.matches[source]) / len(judged)


def _matches(chosen: Optional[Card], actual: Card) -> Optional[bool]:
    return None if chosen is None else chosen == actual


def benchmark_board(
    board: ExpertBoard,
    rl_net=None,
    endgame_card_limit: int = DEFAULT_ENDGAME_CARD_LIMIT,
) -> BoardBenchmarkResult:
    """Replay `board.play_order` exactly as it really happened, asking
    each available engine what it would have played at every real,
    non-forced decision point along the way. `rl_net`, if given, is a
    loaded `rl.network.PolicyValueNet` (pass None to skip the RL
    comparison, e.g. when no checkpoint/torch is available)."""
    if len(board.play_order) != 52:
        raise ValueError(f"expected a complete 52-card play record, got {len(board.play_order)}")

    hands = {s: list(cs.cards) for s, cs in board.deal.hands.items()}
    ps = PlayState(hands=hands, declarer=board.declarer, strain=board.strain, trump=board.trump, dealer=board.declarer)
    declaring_side = {board.declarer, _PARTNER[board.declarer]}

    decisions: List[DecisionRecord] = []
    for i, actual_card in enumerate(board.play_order):
        seat = ps.next_to_play
        legal = ps.legal_cards_for(seat)
        forced = len(legal) <= 1

        matches: Dict[str, Optional[bool]] = {}
        if not forced:
            matches["heuristic"] = _matches(_safe(heuristic_choose_card, seat, ps, seed=0), actual_card)

            if dds_available():
                matches["dds"] = _matches(_safe(dds_choose_card, seat, ps, seed=0, num_samples=8), actual_card)
            else:
                matches["dds"] = None

            if rl_net is not None:
                from rl.rl_player import choose_card as rl_choose_card

                matches["rl"] = _matches(_safe(rl_choose_card, seat, ps, rl_net, seed=0, num_samples=12), actual_card)
            else:
                matches["rl"] = None

            remaining = len(ps.hands[seat])
            if remaining <= endgame_card_limit:
                try:
                    result = solve_endgame(ps, max_cards_per_hand=endgame_card_limit)
                    matches["exact_endgame"] = _matches(result.best_card, actual_card)
                except SearchTooLargeError:
                    matches["exact_endgame"] = None
            else:
                matches["exact_endgame"] = None

        decisions.append(DecisionRecord(
            trick_number=i // 4 + 1,
            seat=seat,
            is_declaring_side=seat in declaring_side,
            actual_card=str(actual_card),
            forced=forced,
            matches=matches,
        ))
        ps.play_card(seat, actual_card)

    return BoardBenchmarkResult(
        board_num=board.board_num,
        event=board.event,
        actual_declarer_tricks=ps.declarer_tricks(),
        decisions=decisions,
    )


def _safe(fn, *args, **kwargs):
    """Any single engine hiccuping on a real (sometimes unusual) tournament
    position shouldn't take down the whole benchmark run -- treat it as
    'unavailable for this decision' instead."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def summarize(results: List[BoardBenchmarkResult]) -> dict:
    """Aggregate match rates (all seats, and declaring-side only) across
    many boards, plus the actual declarer trick counts seen."""
    summary: dict = {"num_boards": len(results)}
    for source in SOURCES:
        all_rates = [r.match_rate(source) for r in results]
        all_rates = [r for r in all_rates if r is not None]
        decl_rates = [r.match_rate(source, declaring_side_only=True) for r in results]
        decl_rates = [r for r in decl_rates if r is not None]
        summary[source] = {
            "boards_judged": len(all_rates),
            "avg_match_rate": sum(all_rates) / len(all_rates) if all_rates else None,
            "avg_match_rate_declaring_side": sum(decl_rates) / len(decl_rates) if decl_rates else None,
        }
    tricks = [r.actual_declarer_tricks for r in results]
    summary["actual_declarer_tricks_avg"] = sum(tricks) / len(tricks) if tricks else None
    return summary
