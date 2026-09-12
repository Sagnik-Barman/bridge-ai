"""Save/load a GameSession to/from JSON, mid-board included.

Two things this enables:
  - Autosave: the web app writes the current session to `_autosave.json`
    after every action, and reloads it on startup instead of always
    starting a fresh session — so closing and reopening the app resumes
    exactly where you left off, mid-auction or mid-trick.
  - Named saves: an explicit checkpoint ("save this game as ...") you can
    return to later even after continuing to play past it, so you can
    revisit a specific hand for practice.

Everything needed to resume exactly — including the undo history — is
captured: the full deal, every call and its explanation, the play state
(hands, tricks so far, whose turn), the board history/score so far, and
the call/card undo stacks.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from engine.deal import Card, Deal, Hand, Suit, parse_card
from bidding.auction_engine import AuctionEngine
from cardplay.trick_engine import PlayState, Trick
from game.session import BoardResult, CallRecord, GameSession

SAVE_FORMAT_VERSION = 1
SAVES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saves")
AUTOSAVE_NAME = "_autosave"


def _ensure_saves_dir() -> None:
    os.makedirs(SAVES_DIR, exist_ok=True)


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()) or "save"
    return slug[:80]


def _path_for(name: str) -> str:
    return os.path.join(SAVES_DIR, f"{_safe_filename(name)}.json")


# ---------------------------------------------------------------------------
# Card/trick/hand JSON helpers
# ---------------------------------------------------------------------------

def _cards_to_json(cards: List[Card]) -> List[str]:
    return [str(c) for c in cards]


def _cards_from_json(strs: List[str]) -> List[Card]:
    return [parse_card(s) for s in strs]


def _trick_to_json(t: Trick) -> dict:
    return {"leader": t.leader, "plays": [[seat, str(card)] for seat, card in t.plays]}


def _trick_from_json(d: dict) -> Trick:
    return Trick(leader=d["leader"], plays=[(seat, parse_card(cs)) for seat, cs in d["plays"]])


def _play_state_to_json(ps: PlayState) -> dict:
    return {
        "hands": {seat: _cards_to_json(cards) for seat, cards in ps.hands.items()},
        "declarer": ps.declarer,
        "strain": ps.strain,
        "trump": ps.trump.value if ps.trump else None,
        "dealer": ps.dealer,
        "completed_tricks": [_trick_to_json(t) for t in ps.completed_tricks],
        "current_trick": _trick_to_json(ps.current_trick),
        "next_to_play": ps.next_to_play,
        "opening_lead_made": ps.opening_lead_made,
    }


def _play_state_from_json(d: dict) -> PlayState:
    hands = {seat: _cards_from_json(cards) for seat, cards in d["hands"].items()}
    trump = Suit(d["trump"]) if d["trump"] else None
    ps = PlayState(hands=hands, declarer=d["declarer"], strain=d["strain"], trump=trump, dealer=d["dealer"])
    ps.completed_tricks = [_trick_from_json(t) for t in d["completed_tricks"]]
    ps.current_trick = _trick_from_json(d["current_trick"])
    ps.next_to_play = d["next_to_play"]
    ps.opening_lead_made = d["opening_lead_made"]
    return ps


def _play_snapshot_to_json(snap: dict) -> dict:
    return {
        "hands": {seat: _cards_to_json(cards) for seat, cards in snap["hands"].items()},
        "completed_tricks": [_trick_to_json(t) for t in snap["completed_tricks"]],
        "current_trick": _trick_to_json(snap["current_trick"]),
        "next_to_play": snap["next_to_play"],
        "opening_lead_made": snap["opening_lead_made"],
    }


def _play_snapshot_from_json(d: dict) -> dict:
    return {
        "hands": {seat: _cards_from_json(cards) for seat, cards in d["hands"].items()},
        "completed_tricks": [_trick_from_json(t) for t in d["completed_tricks"]],
        "current_trick": _trick_from_json(d["current_trick"]),
        "next_to_play": d["next_to_play"],
        "opening_lead_made": d["opening_lead_made"],
    }


# ---------------------------------------------------------------------------
# GameSession <-> dict
# ---------------------------------------------------------------------------

def session_to_dict(session: GameSession) -> dict:
    return {
        "format_version": SAVE_FORMAT_VERSION,
        "saved_at": time.time(),
        "board_number": session.board_number,
        "user_seat": session.user_seat,
        "play_mode": session.play_mode,
        "rl_checkpoint_path": session.rl_checkpoint_path,
        "deal": {seat: _cards_to_json(hand.cards) for seat, hand in session.deal.hands.items()},
        "auction_calls": list(session.auction.calls),
        "call_records": [
            {"seat": r.seat, "call": r.call, "explanation": r.explanation} for r in session.call_records
        ],
        "call_snapshots": [list(s) for s in session.call_snapshots],
        "phase": session.phase,
        "contract": list(session.contract) if session.contract else None,
        "play_state": _play_state_to_json(session.play_state) if session.play_state else None,
        "play_snapshots": [_play_snapshot_to_json(s) for s in session.play_snapshots],
        "last_finish_deltas": list(session._last_finish_deltas) if session._last_finish_deltas else None,
        "last_board_result": _board_result_to_json(session.last_board_result) if session.last_board_result else None,
        "board_history": [_board_result_to_json(r) for r in session.board_history],
        "total_imps": session.total_imps,
        "total_raw_points": session.total_raw_points,
    }


def _board_result_to_json(r: BoardResult) -> dict:
    return {
        "board_number": r.board_number,
        "contract_summary": r.contract_summary,
        "score_summary": r.score_summary,
        "ns_score": r.ns_score,
        "user_imps": r.user_imps,
        "par_strain": r.par_strain,
        "par_tricks": r.par_tricks,
    }


def session_from_dict(data: dict, auction_engine: Optional[AuctionEngine] = None) -> GameSession:
    if data.get("format_version") != SAVE_FORMAT_VERSION:
        raise ValueError(f"Unsupported save format version: {data.get('format_version')!r}")

    session = object.__new__(GameSession)
    session.auction_engine = auction_engine or AuctionEngine()
    session.board_number = data["board_number"]
    session.play_mode = data.get("play_mode", "heuristic")
    session.rl_checkpoint_path = data.get("rl_checkpoint_path") or None
    from game.session import DEFAULT_RL_CHECKPOINT

    session.rl_checkpoint_path = session.rl_checkpoint_path or DEFAULT_RL_CHECKPOINT
    session._rl_net = None

    from game.board import board_info  # local import avoids a cycle at module load time
    session.board = board_info(session.board_number)
    session.deal = Deal(
        hands={seat: Hand(cards=_cards_from_json(cards)) for seat, cards in data["deal"].items()},
        dealer=session.board.dealer,
        vulnerable=session.board.vulnerable,
    )
    session.user_seat = data["user_seat"]

    from bidding.calls import AuctionState
    session.auction = AuctionState(dealer=session.board.dealer, calls=list(data["auction_calls"]))
    session.call_records = [CallRecord(**r) for r in data["call_records"]]
    session.call_snapshots = [list(s) for s in data["call_snapshots"]]
    session.phase = data["phase"]
    session.contract = tuple(data["contract"]) if data["contract"] else None
    session.play_state = _play_state_from_json(data["play_state"]) if data["play_state"] else None
    session.play_snapshots = [_play_snapshot_from_json(s) for s in data["play_snapshots"]]
    session._last_finish_deltas = tuple(data["last_finish_deltas"]) if data["last_finish_deltas"] else None
    session.last_board_result = BoardResult(**data["last_board_result"]) if data["last_board_result"] else None
    session.board_history = [BoardResult(**r) for r in data["board_history"]]
    session.total_imps = data["total_imps"]
    session.total_raw_points = data["total_raw_points"]
    return session


# ---------------------------------------------------------------------------
# File I/O + a save index for the "revisit a saved game" picker
# ---------------------------------------------------------------------------

def save_session(session: GameSession, name: str) -> str:
    _ensure_saves_dir()
    payload = session_to_dict(session)
    payload["name"] = name
    path = _path_for(name)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def load_session(name: str, auction_engine: Optional[AuctionEngine] = None) -> GameSession:
    path = _path_for(name)
    with open(path, "r") as f:
        data = json.load(f)
    return session_from_dict(data, auction_engine=auction_engine)


def autosave(session: GameSession) -> None:
    save_session(session, AUTOSAVE_NAME)


def load_autosave(auction_engine: Optional[AuctionEngine] = None) -> Optional[GameSession]:
    if not os.path.exists(_path_for(AUTOSAVE_NAME)):
        return None
    try:
        return load_session(AUTOSAVE_NAME, auction_engine=auction_engine)
    except Exception:
        # A corrupt or incompatible autosave should never prevent the app
        # from starting — fall back to a fresh session instead.
        return None


def list_saves(include_autosave: bool = False) -> List[dict]:
    """Metadata for every named save, newest first, for a "load game" picker."""
    _ensure_saves_dir()
    entries = []
    for fname in os.listdir(SAVES_DIR):
        if not fname.endswith(".json"):
            continue
        stem = fname[: -len(".json")]
        if stem == AUTOSAVE_NAME and not include_autosave:
            continue
        try:
            with open(os.path.join(SAVES_DIR, fname), "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "name": data.get("name", stem),
                "slug": stem,
                "board_number": data.get("board_number"),
                "phase": data.get("phase"),
                "boards_played": len(data.get("board_history", [])),
                "total_imps": data.get("total_imps"),
                "saved_at": data.get("saved_at"),
            }
        )
    entries.sort(key=lambda e: e["saved_at"] or 0, reverse=True)
    return entries


def delete_save(name: str) -> bool:
    path = _path_for(name)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
