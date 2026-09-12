import json
import os
import shutil

from game.session import GameSession
from game.persistence import (
    SAVES_DIR,
    session_to_dict,
    session_from_dict,
    save_session,
    load_session,
    list_saves,
    delete_save,
    autosave,
    load_autosave,
    AUTOSAVE_NAME,
)


def setup_function(_):
    if os.path.exists(SAVES_DIR):
        shutil.rmtree(SAVES_DIR)


def _bid_until(session, predicate, guard_limit=40):
    guard = 0
    while predicate(session):
        guard += 1
        assert guard < guard_limit
        sugg = session.current_bid_suggestion()
        if sugg is None:
            break
        session.make_user_call(sugg.call)


def test_round_trip_preserves_state_mid_bidding():
    session = None
    for _ in range(30):
        session = GameSession(starting_board=1)
        sugg = session.current_bid_suggestion()
        session.make_user_call(sugg.call)
        if session.phase == "bidding":
            break
    data = session_to_dict(session)
    json.dumps(data)  # must be JSON-serializable
    restored = session_from_dict(data)
    assert restored.to_dict() == session.to_dict()


def test_round_trip_preserves_state_mid_play_and_can_continue():
    session = None
    for _ in range(60):
        session = GameSession(starting_board=1)
        _bid_until(session, lambda s: s.phase == "bidding")
        if session.phase != "play":
            continue
        legal = session.legal_cards_for_user()
        if not legal:
            continue
        seat = session.play_state.next_to_play
        session.play_user_card(seat, legal[0])
        break
    assert session.phase == "play"
    data = session_to_dict(session)
    json.dumps(data)
    restored = session_from_dict(data)
    assert restored.to_dict() == session.to_dict()

    # Must be able to keep playing the restored session to completion.
    guard = 0
    while restored.phase == "play":
        guard += 1
        assert guard < 40
        legal = restored.legal_cards_for_user()
        if not legal:
            break
        seat = restored.play_state.next_to_play
        restored.play_user_card(seat, legal[0])
    assert restored.phase == "board_complete"


def test_undo_still_works_after_round_trip():
    session = None
    for _ in range(30):
        session = GameSession(starting_board=1)
        before = list(session.auction.calls)
        sugg = session.current_bid_suggestion()
        session.make_user_call(sugg.call)
        if session.phase == "bidding":
            break
    data = session_to_dict(session)
    restored = session_from_dict(data)
    assert restored.undo_last_call() is True
    assert restored.auction.calls == before


def test_save_and_load_by_name():
    session = GameSession(starting_board=1)
    save_session(session, "practice one")
    loaded = load_session("practice one")
    assert loaded.to_dict() == session.to_dict()


def test_list_saves_and_delete():
    session = GameSession(starting_board=1)
    save_session(session, "Game A")
    save_session(session, "Game B")
    names = {s["name"] for s in list_saves()}
    assert names == {"Game A", "Game B"}
    assert delete_save("Game A") is True
    assert {s["name"] for s in list_saves()} == {"Game B"}
    assert delete_save("does not exist") is False


def test_autosave_excluded_from_list_saves_by_default():
    session = GameSession(starting_board=1)
    autosave(session)
    save_session(session, "Visible Game")
    names = [s["name"] for s in list_saves()]
    assert "Visible Game" in names
    assert AUTOSAVE_NAME not in names


def test_load_autosave_round_trip():
    session = GameSession(starting_board=1)
    autosave(session)
    loaded = load_autosave()
    assert loaded is not None
    assert loaded.to_dict() == session.to_dict()


def test_load_autosave_returns_none_when_absent():
    assert load_autosave() is None


def test_round_trip_preserves_play_mode():
    session = GameSession(starting_board=1, play_mode="dds")
    data = session_to_dict(session)
    assert data["play_mode"] == "dds"
    restored = session_from_dict(data)
    assert restored.play_mode == "dds"
    assert restored.to_dict() == session.to_dict()
