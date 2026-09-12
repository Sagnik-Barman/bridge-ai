"""Turn real PBN tournament files into `benchmark.expert_benchmark.ExpertBoard`
objects, using `endplay`'s PBN parser (already a project dependency for
double-dummy solving — see `engine/dds_wrapper.py`).

Isolated from `benchmark/expert_benchmark.py` for the same reason
`engine/dds_wrapper.py` isolates its endplay-specific code: this module
can't be exercised or verified in an environment without `endplay`
installed (which includes the cloud sandbox this project was originally
built in), so every attribute access below is a best-effort reading of
endplay's documented API (https://endplay.readthedocs.io), not something
that's actually been run against a real file yet. Treat it exactly like
`engine/dds_wrapper.py`'s DDTable indexing was before it got fixed live
against a real install: **run `scripts/diag_pbn.py` against a real .pbn
file on your machine first** and tell me what it prints before trusting
this module's output — see docs/real_deals.md.

Where the source files come from — and why this is different from the
copyrighted-book restriction:

Real tournament PBN files (World Championship finals, etc.) record
factual information about what actually happened at the table: who held
what cards, what was bid, what was played, what the result was. That's
categorically different from the uploaded copyrighted book, which is
someone's original written analysis and commentary — this project has
never used and will never use that book as a source, but a plain factual
record of a real tournament board is not the same kind of thing at all,
any more than a chess game's move list or a box score is. See
docs/real_deals.md for exactly where these files come from.
"""
from __future__ import annotations

from typing import List, Optional

from engine.deal import Card, Suit, parse_pbn_deal
from benchmark.expert_benchmark import ExpertBoard

_DENOM_TO_STRAIN = {"spades": "S", "hearts": "H", "diamonds": "D", "clubs": "C", "nt": "NT"}
_PLAYER_TO_SEAT = {"north": "N", "east": "E", "south": "S", "west": "W"}


class PbnUnavailableError(RuntimeError):
    """endplay isn't installed, so real PBN files can't be parsed here."""


def pbn_available() -> bool:
    try:
        import endplay  # noqa: F401

        return True
    except ImportError:
        return False


def load_boards(path: str) -> List[ExpertBoard]:
    """Parse every board in a .pbn file into an `ExpertBoard`, silently
    skipping (not guessing about) any board this can't cleanly interpret:
    no recorded contract/declarer, an incomplete play record (a claim
    ended the hand early, or it just wasn't recorded), or an unrecognized
    strain/seat. Returns only boards worth benchmarking against."""
    if not pbn_available():
        raise PbnUnavailableError(
            "endplay is not installed, so real PBN tournament files can't be "
            "parsed. Run `pip install endplay` — see docs/real_deals.md."
        )
    import endplay.parsers.pbn as pbn

    with open(path, encoding="utf-8", errors="replace") as f:
        raw_boards = pbn.load(f)

    boards: List[ExpertBoard] = []
    for raw in raw_boards:
        board = _try_convert(raw)
        if board is not None:
            boards.append(board)
    return boards


def _try_convert(raw) -> Optional[ExpertBoard]:
    try:
        contract = raw.contract
        if contract is None or contract.declarer is None or contract.denom is None:
            return None  # passed out, or no contract recorded -- nothing to benchmark

        play = list(raw.play) if raw.play else []
        if len(play) != 52:
            return None  # claimed early / partially recorded -- skip rather than guess

        declarer_seat = _PLAYER_TO_SEAT[contract.declarer.name]
        strain = _DENOM_TO_STRAIN[contract.denom.name]
        trump = None if strain == "NT" else Suit(strain)

        deal = parse_pbn_deal(raw.deal.to_pbn(), dealer=declarer_seat)
        play_order = [_convert_card(c) for c in play]

        board_num = getattr(raw, "board_num", None)
        event = None
        try:
            event = raw.info.get("Event")
        except Exception:
            pass

        return ExpertBoard(
            deal=deal,
            declarer=declarer_seat,
            strain=strain,
            trump=trump,
            play_order=play_order,
            board_num=board_num,
            event=event,
        )
    except (AttributeError, KeyError, ValueError, TypeError):
        # Any single malformed/unexpected board is skipped, not fatal to
        # the whole file -- but see this module's docstring: until
        # verified against a real install, treat a *systematic* failure
        # (every board skipped) as "the attribute assumptions below are
        # probably wrong", not "this file has no usable boards".
        return None


def _convert_card(ep_card):
    suit = Suit(_DENOM_TO_STRAIN[ep_card.suit.name])
    rank = ep_card.rank.abbr
    return Card(suit, rank)
