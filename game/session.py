"""GameSession: ties dealing, bidding, card play, scoring, board
sequencing, seat rotation, and undo together into one long practice
session. This is the engine the Flask app (webapp/app.py) drives; it has
no web/UI dependencies itself, so it's directly unit-testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.deal import Card, Deal, Suit, deal_random
from engine.dds_wrapper import DDSUnavailableError, best_contract_for_seat
from bidding.calls import AuctionState, SEATS, partner_of
from bidding.auction_engine import AuctionEngine, AuctionDecision
from cardplay.trick_engine import PlayState, IllegalPlayError
from cardplay.bot_player import choose_card as heuristic_choose_card
from cardplay.dds_player import choose_card as dds_choose_card, dds_status, dds_available
from cardplay.endgame_solver import solve_endgame, SearchTooLargeError
from game.board import BoardInfo, board_info, user_seat_for_board
from game.scoring import ScoreResult, score_contract, imp_swing

ANALYZER_ENDGAME_CARD_LIMIT = 6  # see cardplay/endgame_solver.py's docstring for why (6, not 8, actually finishes)

PLAY_MODES = ("heuristic", "dds", "rl")
DEFAULT_RL_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl", "checkpoints", "latest.pt"
)


class IllegalActionError(ValueError):
    pass


@dataclass
class CallRecord:
    seat: str
    call: str
    explanation: str


@dataclass
class BoardResult:
    board_number: int
    contract_summary: str
    score_summary: str
    ns_score: int
    user_imps: Optional[float]  # signed from the human's partnership's perspective
    par_strain: Optional[str]
    par_tricks: Optional[int]


class GameSession:
    def __init__(
        self,
        starting_board: int = 1,
        auction_engine: Optional[AuctionEngine] = None,
        play_mode: str = "heuristic",
        rl_checkpoint_path: Optional[str] = None,
    ):
        if play_mode not in PLAY_MODES:
            raise ValueError(f"Unknown play_mode {play_mode!r}; must be one of {PLAY_MODES}")
        self.auction_engine = auction_engine or AuctionEngine()
        self.play_mode = play_mode
        self.rl_checkpoint_path = rl_checkpoint_path or DEFAULT_RL_CHECKPOINT
        self._rl_net = None  # lazily loaded the first time it's actually needed
        self.total_imps: float = 0.0
        self.total_raw_points: int = 0  # from the human partnership's perspective
        self.board_history: List[BoardResult] = []
        self.board_number: int = starting_board
        self.board: BoardInfo = board_info(starting_board)
        self.deal: Deal = deal_random(dealer=self.board.dealer, vulnerable=self.board.vulnerable)
        self.user_seat: str = user_seat_for_board(starting_board)
        self._reset_board_state()

    # -- board lifecycle ---------------------------------------------------

    def _reset_board_state(self) -> None:
        self.auction = AuctionState(dealer=self.board.dealer, calls=[])
        self.call_records: List[CallRecord] = []
        self.call_snapshots: List[List[str]] = []
        self.phase: str = "bidding"  # "bidding" | "play" | "board_complete"
        self.play_state: Optional[PlayState] = None
        self.play_snapshots: List[dict] = []
        self.contract = None  # (declarer, level, strain, dbl_state)
        self.last_board_result: Optional[BoardResult] = None
        self._last_finish_deltas: Optional[tuple] = None  # (raw_delta, imps_delta) for undo bookkeeping
        self._advance_bidding_bots()

    def start_board(self, number: int) -> None:
        self.board_number = number
        self.board = board_info(number)
        self.deal = deal_random(dealer=self.board.dealer, vulnerable=self.board.vulnerable)
        self.user_seat = user_seat_for_board(number)
        self._reset_board_state()

    def next_board(self) -> None:
        self.start_board(self.board_number + 1)

    def restart_current_board(self) -> None:
        """Undo everything on this board and re-bid/replay the *same*
        deal from scratch (deliberately doesn't re-deal — the point is to
        retry the hand you were just given)."""
        self._reset_board_state()

    # -- bidding -------------------------------------------------------

    def _advance_bidding_bots(self) -> None:
        if self.phase != "bidding":
            return
        while not self.auction.is_complete() and self.auction.whose_turn() != self.user_seat:
            seat = self.auction.whose_turn()
            hand = self.deal.hands[seat]
            decision = self.auction_engine.decide_call(seat, hand, self.auction)
            self.auction.add(decision.call)
            self.call_records.append(CallRecord(seat, decision.call, decision.explanation))
        if self.auction.is_complete():
            self._finish_auction()

    def current_bid_suggestion(self) -> Optional[AuctionDecision]:
        """What the engine recommends for the human right now — the "full
        guide through bidding" feature. None if it isn't the human's turn
        to call."""
        if self.phase != "bidding" or self.auction.is_complete():
            return None
        if self.auction.whose_turn() != self.user_seat:
            return None
        return self.auction_engine.decide_call(self.user_seat, self.deal.hands[self.user_seat], self.auction)

    def make_user_call(self, call: str) -> dict:
        if self.phase != "bidding":
            raise IllegalActionError("Not currently bidding.")
        if self.auction.whose_turn() != self.user_seat:
            raise IllegalActionError("It isn't your turn to call.")
        if call not in self.auction.legal_calls():
            raise IllegalActionError(f"{call} isn't a legal call right now.")

        suggestion = self.current_bid_suggestion()
        matched = suggestion.call == call if suggestion else None

        self.call_snapshots.append(list(self.auction.calls))
        self.auction.add(call)
        explanation = "Your call."
        if suggestion:
            explanation += (
                " Matches the suggested bid." if matched
                else f" Suggested instead: {suggestion.call} — {suggestion.explanation}"
            )
        self.call_records.append(CallRecord(self.user_seat, call, explanation))

        feedback = {
            "call": call,
            "matched_suggestion": matched,
            "suggested_call": suggestion.call if suggestion else None,
            "suggestion_explanation": suggestion.explanation if suggestion else None,
        }
        self._advance_bidding_bots()
        return feedback

    def _no_cards_played_yet(self) -> bool:
        return self.play_state is None or (
            len(self.play_state.completed_tricks) == 0 and len(self.play_state.current_trick.plays) == 0
        )

    def undo_last_call(self) -> bool:
        """Undo the human's most recent call. Allowed not just while still
        bidding, but also right after the auction just completed (either
        passed out, or opened the play phase but no card has been played
        yet) — otherwise a human whose call happened to end the auction
        (e.g. everyone passed after their opening pass) could never take
        it back, which defeats the point of "undo" as a learning aid.
        Once real card play has started, bidding can no longer be undone.
        """
        if not self.call_snapshots:
            return False
        if self.phase == "play" and not self._no_cards_played_yet():
            return False

        snapshot = self.call_snapshots.pop()
        self.auction.calls = snapshot
        self.call_records = self.call_records[: len(snapshot)]

        if self.phase != "bidding":
            # Undo whatever _finish_auction()/_finish_play() side effects
            # happened as a result of the call we just took back.
            if self.board_history and self.board_history[-1].board_number == self.board_number:
                self.board_history.pop()
            if self._last_finish_deltas is not None:
                raw_delta, imps_delta = self._last_finish_deltas
                self.total_raw_points -= raw_delta
                if imps_delta is not None:
                    self.total_imps -= imps_delta
                self._last_finish_deltas = None
            self.phase = "bidding"
            self.contract = None
            self.play_state = None
            self.play_snapshots = []
            self.last_board_result = None

        return True

    def _finish_auction(self) -> None:
        if self.auction.is_passed_out():
            result = BoardResult(
                board_number=self.board_number,
                contract_summary="Passed out",
                score_summary="Passed out — no score.",
                ns_score=0,
                user_imps=0.0,
                par_strain=None,
                par_tricks=None,
            )
            self.last_board_result = result
            self.board_history.append(result)
            self._last_finish_deltas = (0, 0.0)
            self.phase = "board_complete"
            return

        declarer, level, strain, dbl = self.auction.declarer_and_contract()
        self.contract = (declarer, level, strain, dbl)
        trump = None if strain == "NT" else Suit(strain)
        hands_copy = {s: list(self.deal.hands[s].cards) for s in SEATS}
        self.play_state = PlayState(hands=hands_copy, declarer=declarer, strain=strain, trump=trump, dealer=self.board.dealer)
        self.phase = "play"
        self._advance_play_bots()

    # -- card play -------------------------------------------------------

    def controlled_seats(self):
        if not self.contract:
            return set()
        declarer = self.contract[0]
        dummy = partner_of(declarer)
        if self.user_seat == declarer:
            return {declarer, dummy}
        if self.user_seat == dummy:
            return set()  # declarer (a bot) plays dummy's cards for them
        return {self.user_seat}

    def _bot_choose_card(self, seat: str) -> Card:
        """Dispatch to the configured play mode, with a transparent, logged
        fallback to the heuristic bot whenever the fancier mode can't
        actually run (DDS not installed, RL checkpoint missing/untrained).
        Never silently uses the book — every card-selection path here is
        either hand-written heuristics, the real DDS/PIMC solver, or a
        self-play-trained network; see cardplay/dds_player.py and
        rl/rl_player.py for exactly what each does."""
        if self.play_mode == "dds":
            return dds_choose_card(seat, self.play_state)
        if self.play_mode == "rl":
            net = self._get_rl_net()
            if net is not None:
                from rl.rl_player import choose_card as rl_choose_card

                return rl_choose_card(seat, self.play_state, net)
        return heuristic_choose_card(seat, self.play_state)

    def _get_rl_net(self):
        if self._rl_net is not None:
            return self._rl_net
        if not os.path.exists(self.rl_checkpoint_path):
            return None
        try:
            from rl.network import PolicyValueNet

            self._rl_net = PolicyValueNet.load(self.rl_checkpoint_path)
        except Exception:
            # Covers both "torch isn't installed" (ImportError) and any
            # checkpoint-loading problem — either way, fall back to the
            # heuristic bot rather than crashing the game.
            return None
        return self._rl_net

    def play_mode_status(self) -> str:
        if self.play_mode == "dds":
            return dds_status()
        if self.play_mode == "rl":
            try:
                from rl.rl_player import rl_checkpoint_status

                return rl_checkpoint_status(self.rl_checkpoint_path)
            except ImportError:
                return (
                    "PyTorch isn't installed, so 'Play against RL' falls back to "
                    "the heuristic bot for now. See docs/rl_training.md."
                )
        return "Playing against the built-in heuristic bot."

    def _advance_play_bots(self) -> None:
        if self.phase != "play":
            return
        while not self.play_state.is_complete() and self.play_state.next_to_play not in self.controlled_seats():
            seat = self.play_state.next_to_play
            card = self._bot_choose_card(seat)
            self.play_state.play_card(seat, card)
        if self.play_state.is_complete():
            self._finish_play()

    def legal_cards_for_user(self) -> List[Card]:
        if self.phase != "play" or self.play_state.next_to_play not in self.controlled_seats():
            return []
        return self.play_state.legal_cards_for(self.play_state.next_to_play)

    def play_user_card(self, seat: str, card: Card) -> None:
        if self.phase != "play":
            raise IllegalActionError("Not currently in the card-play phase.")
        if seat not in self.controlled_seats():
            raise IllegalActionError(f"You don't control seat {seat} right now.")
        if seat != self.play_state.next_to_play:
            raise IllegalActionError("It isn't that seat's turn to play.")
        if card not in self.play_state.legal_cards_for(seat):
            raise IllegalActionError(f"{card} isn't a legal play (must follow suit if able).")

        self.play_snapshots.append(self.play_state.clone_state())
        try:
            self.play_state.play_card(seat, card)
        except IllegalPlayError as exc:  # pragma: no cover - already validated above
            self.play_snapshots.pop()
            raise IllegalActionError(str(exc))
        self._advance_play_bots()

    def undo_last_card(self) -> bool:
        if self.phase != "play" or not self.play_snapshots:
            return False
        snapshot = self.play_snapshots.pop()
        self.play_state.restore_state(snapshot)
        return True

    def analyze_current_position(self) -> dict:
        """The "Analyze" feature: ask every card-play source this project
        has — the hand-written heuristic bot, the real DDS/PIMC solver, the
        self-play-trained RL network, and the from-scratch alpha-beta
        endgame solver — what they'd play right now, for whoever is on
        lead, and report them side by side. None of this touches the
        contents of the uploaded book; every suggestion here comes from
        either explicit hand-written rules, a real double-dummy solve, or
        a network trained purely by self-play (see cardplay/dds_player.py,
        rl/rl_player.py, cardplay/endgame_solver.py).

        Each source is independent and best-effort: a source that isn't
        available (no endplay, no trained checkpoint, position too big
        for the exact solver) reports why, rather than being omitted or
        crashing the others.
        """
        if self.phase != "play" or self.play_state is None:
            raise IllegalActionError("Nothing to analyze — not currently in the card-play phase.")
        ps = self.play_state
        if ps.is_complete():
            raise IllegalActionError("This board's play is already complete — nothing left to analyze.")

        seat = ps.next_to_play
        legal = ps.legal_cards_for(seat)
        cards_remaining = len(ps.hands[seat])

        result: dict = {
            "seat": seat,
            "cards_remaining": cards_remaining,
            "legal_cards": _cards_to_strs(legal),
        }

        # 1. Heuristic bot — always available, zero dependencies.
        try:
            heuristic_card = heuristic_choose_card(seat, ps, seed=0)
            result["heuristic"] = {
                "available": True,
                "suggested_card": str(heuristic_card),
                "note": "Hand-written club-level rules (no search).",
            }
        except Exception as exc:  # pragma: no cover - heuristic bot shouldn't fail
            result["heuristic"] = {"available": False, "suggested_card": None, "note": str(exc)}

        # 2. DDS/PIMC — real double-dummy solving at this trick boundary.
        if dds_available():
            try:
                dds_card = dds_choose_card(seat, ps, seed=0, num_samples=10)
                result["dds"] = {
                    "available": True,
                    "suggested_card": str(dds_card),
                    "note": "PIMC-sampled double-dummy solves (10 samples) — see cardplay/dds_player.py.",
                }
            except Exception as exc:
                result["dds"] = {"available": False, "suggested_card": None, "note": f"DDS analysis failed: {exc}"}
        else:
            result["dds"] = {"available": False, "suggested_card": None, "note": dds_status()}

        # 3. RL network — self-play-trained policy, PIMC-wrapped for real play.
        net = self._get_rl_net()
        if net is not None:
            try:
                from rl.rl_player import choose_card as rl_choose_card

                rl_card = rl_choose_card(seat, ps, net, seed=0, num_samples=20)
                result["rl"] = {
                    "available": True,
                    "suggested_card": str(rl_card),
                    "note": "Self-play-trained network, PIMC-wrapped (20 samples) — see rl/rl_player.py.",
                }
            except Exception as exc:
                result["rl"] = {"available": False, "suggested_card": None, "note": f"RL analysis failed: {exc}"}
        else:
            try:
                from rl.rl_player import rl_checkpoint_status

                note = rl_checkpoint_status(self.rl_checkpoint_path)
            except ImportError:
                note = "PyTorch isn't installed. See docs/rl_training.md."
            result["rl"] = {"available": False, "suggested_card": None, "note": note}

        # 4. From-scratch exact endgame solver — only for small enough
        # endgames (see cardplay/endgame_solver.py's honest scope
        # docstring for why it can't attempt full-size positions).
        if cards_remaining <= ANALYZER_ENDGAME_CARD_LIMIT:
            try:
                endgame_result = solve_endgame(ps, max_cards_per_hand=ANALYZER_ENDGAME_CARD_LIMIT)
                result["exact_endgame"] = {
                    "available": True,
                    "suggested_card": str(endgame_result.best_card),
                    "ns_tricks_from_here": endgame_result.ns_tricks_from_here,
                    "nodes_explored": endgame_result.nodes_explored,
                    "note": (
                        f"Exact minimax + alpha-beta search ({endgame_result.nodes_explored} nodes) — "
                        f"NS get {endgame_result.ns_tricks_from_here} of the remaining tricks under "
                        f"optimal play by both sides. See cardplay/endgame_solver.py."
                    ),
                }
            except SearchTooLargeError as exc:
                result["exact_endgame"] = {"available": False, "suggested_card": None, "note": str(exc)}
        else:
            result["exact_endgame"] = {
                "available": False,
                "suggested_card": None,
                "note": (
                    f"{cards_remaining} cards left per hand — the exact from-scratch solver is only "
                    f"attempted at {ANALYZER_ENDGAME_CARD_LIMIT} or fewer (no suit-symmetry reduction, "
                    f"unlike real DDS). Use the DDS suggestion above for full-size positions."
                ),
            }

        suggestions = {
            src: result[src]["suggested_card"]
            for src in ("heuristic", "dds", "rl", "exact_endgame")
            if result[src]["available"]
        }
        distinct = set(suggestions.values())
        result["consensus"] = (
            next(iter(distinct)) if len(distinct) == 1 and distinct else None
        )
        result["sources_agree"] = len(distinct) <= 1 and len(suggestions) > 1

        return result

    def _finish_play(self) -> None:
        declarer, level, strain, dbl = self.contract
        vul_ns = self.board.is_vulnerable("NS")
        vul_ew = self.board.is_vulnerable("EW")
        tricks_taken = self.play_state.declarer_tricks()
        result = score_contract(level, strain, dbl, tricks_taken, declarer, vul_ns, vul_ew)

        user_partnership = "NS" if self.user_seat in ("N", "S") else "EW"
        sign = 1 if user_partnership == "NS" else -1

        imps: Optional[float] = None
        par_strain: Optional[str] = None
        par_tricks: Optional[int] = None
        try:
            par_strain, par_tricks = best_contract_for_seat(self.deal, declarer)
            if par_tricks >= 7:
                comp_level = par_tricks - 6
                comp_result = score_contract(comp_level, par_strain, "", par_tricks, declarer, vul_ns, vul_ew)
                imps = imp_swing(result.ns_score, comp_result.ns_score) * sign
        except DDSUnavailableError:
            pass  # no comparison available without endplay installed

        raw_delta = result.ns_score * sign
        self.total_raw_points += raw_delta
        if imps is not None:
            self.total_imps += imps
        self._last_finish_deltas = (raw_delta, imps)

        board_result = BoardResult(
            board_number=self.board_number,
            contract_summary=f"{level}{strain}{dbl} by {declarer}",
            score_summary=result.summary,
            ns_score=result.ns_score,
            user_imps=imps,
            par_strain=par_strain,
            par_tricks=par_tricks,
        )
        self.last_board_result = board_result
        self.board_history.append(board_result)
        self.phase = "board_complete"

    # -- serialization for the web frontend -------------------------------

    def to_dict(self) -> dict:
        auction_view = [
            {"seat": r.seat, "call": r.call, "explanation": r.explanation} for r in self.call_records
        ]
        suggestion = self.current_bid_suggestion()

        hands_view = {}
        if self.phase in ("bidding",):
            hands_view[self.user_seat] = _hand_to_strs(self.deal.hands[self.user_seat])
        elif self.phase in ("play", "board_complete"):
            hands_view[self.user_seat] = _cards_to_strs(self.play_state.hands[self.user_seat]) if self.play_state else _hand_to_strs(self.deal.hands[self.user_seat])
            if self.play_state and self.play_state.opening_lead_made:
                dummy = self.play_state.dummy
                hands_view[dummy] = _cards_to_strs(self.play_state.hands[dummy])

        current_trick = None
        if self.play_state:
            current_trick = [{"seat": s, "card": str(c)} for s, c in self.play_state.current_trick.plays]

        return {
            "play_mode": self.play_mode,
            "play_mode_status": self.play_mode_status(),
            "board_number": self.board_number,
            "dealer": self.board.dealer,
            "vulnerable": self.board.vulnerable,
            "user_seat": self.user_seat,
            "phase": self.phase,
            "auction": auction_view,
            "legal_calls": self.auction.legal_calls() if self.phase == "bidding" else [],
            "whose_turn_to_call": self.auction.whose_turn() if self.phase == "bidding" and not self.auction.is_complete() else None,
            "suggestion": {"call": suggestion.call, "explanation": suggestion.explanation} if suggestion else None,
            "hands": hands_view,
            "contract": (
                {"declarer": self.contract[0], "level": self.contract[1], "strain": self.contract[2], "double": self.contract[3]}
                if self.contract else None
            ),
            "current_trick": current_trick,
            "tricks_by_side": self.play_state.tricks_by_side() if self.play_state else None,
            "next_to_play": self.play_state.next_to_play if self.play_state else None,
            "controlled_seats": sorted(self.controlled_seats()) if self.play_state else [],
            "legal_cards": _cards_to_strs(self.legal_cards_for_user()) if self.phase == "play" else [],
            "can_undo_call": bool(self.call_snapshots),
            "can_undo_card": bool(self.play_snapshots),
            "last_board_result": _board_result_to_dict(self.last_board_result) if self.last_board_result else None,
            "total_imps": round(self.total_imps, 1),
            "total_raw_points": self.total_raw_points,
            "boards_played": len(self.board_history),
        }


def _hand_to_strs(hand) -> List[str]:
    return _cards_to_strs(hand.cards)


def _cards_to_strs(cards) -> List[str]:
    return [str(c) for c in sorted(cards, key=lambda c: (c.suit.value, c.rank))]


def _board_result_to_dict(r: BoardResult) -> dict:
    return {
        "board_number": r.board_number,
        "contract_summary": r.contract_summary,
        "score_summary": r.score_summary,
        "ns_score": r.ns_score,
        "user_imps": r.user_imps,
        "par_strain": r.par_strain,
        "par_tricks": r.par_tricks,
    }
