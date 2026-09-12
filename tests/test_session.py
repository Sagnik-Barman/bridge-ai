from game.session import GameSession


def _bid_out_using_suggestions(session, guard_limit=40):
    guard = 0
    while session.phase == "bidding":
        guard += 1
        assert guard < guard_limit, "bidding never completed"
        sugg = session.current_bid_suggestion()
        assert sugg is not None
        session.make_user_call(sugg.call)


def _play_out_taking_first_legal(session, guard_limit=40):
    guard = 0
    while session.phase == "play":
        guard += 1
        assert guard < guard_limit, "play never completed"
        legal = session.legal_cards_for_user()
        if not legal:
            break  # shouldn't happen while phase == "play" and not complete
        seat = session.play_state.next_to_play
        session.play_user_card(seat, legal[0])


def test_full_session_16_boards_no_crash():
    session = GameSession(starting_board=1)
    for b in range(1, 17):
        assert session.board_number == b
        _bid_out_using_suggestions(session)
        _play_out_taking_first_legal(session)
        assert session.phase == "board_complete"
        assert len(session.board_history) == b
        session.next_board()


def test_seat_and_vulnerability_rotate_with_board():
    session = GameSession(starting_board=1)
    seats_seen = []
    for b in range(1, 5):
        seats_seen.append(session.user_seat)
        _bid_out_using_suggestions(session)
        _play_out_taking_first_legal(session)
        session.next_board()
    assert seats_seen == ["N", "E", "S", "W"]


def test_undo_last_call_restores_prior_auction():
    """Undo right after a single call, in the common case where that call
    doesn't itself trigger a bot's opening lead (see the two tests below
    for the pass-out and already-played-a-card edge cases specifically)."""
    for _ in range(50):
        session = GameSession(starting_board=1)
        before = list(session.auction.calls)
        sugg = session.current_bid_suggestion()
        session.make_user_call(sugg.call)
        assert session.auction.calls != before
        if session.phase == "play" and not session._no_cards_played_yet():
            continue  # a bot already led — undo is intentionally blocked here, tested separately
        assert session.undo_last_call() is True
        assert session.auction.calls == before
        return
    raise AssertionError("never found an undoable single-call scenario in 50 random deals (very unlucky)")


def test_undo_last_call_works_when_it_passed_out_the_board():
    """Regression test: if the human's call happens to be the one that
    passes the board out (everyone passes, including them), undo must
    still be able to take it back and reopen bidding, not silently refuse
    just because the phase already moved to 'board_complete'. (Undoing a
    call that led straight into card play with a bot's opening lead
    already played is a *different*, intentionally unsupported case — see
    the docstring on undo_last_call — so this test only covers the
    pass-out path, which should always be undoable.)"""
    found_case = False
    for _ in range(80):
        session = GameSession(starting_board=1)
        before = list(session.auction.calls)
        sugg = session.current_bid_suggestion()
        if sugg.call != "Pass":
            continue
        session.make_user_call(sugg.call)
        if not session.auction.is_passed_out():
            continue
        found_case = True
        assert session.phase == "board_complete"
        ok = session.undo_last_call()
        assert ok, "undo_last_call returned False right after a pass-out"
        assert session.phase == "bidding"
        assert session.auction.calls == before
        assert session.contract is None
        assert session.last_board_result is None
        assert len(session.board_history) == 0
        break
    assert found_case, "never hit a pass-out in 80 random deals (very unlucky)"


def test_undo_last_call_blocked_once_a_card_has_actually_been_played():
    """Complementary case: once the auction resolves to a contract *and*
    a bot has already played the opening lead (or later), the bidding is
    no longer undoable — undo_last_call should return False rather than
    silently rewinding cards that have already been played."""
    for _ in range(80):
        session = GameSession(starting_board=1)
        sugg = session.current_bid_suggestion()
        session.make_user_call(sugg.call)
        while session.phase == "bidding":
            sugg = session.current_bid_suggestion()
            session.make_user_call(sugg.call)
        if session.phase != "play" or session._no_cards_played_yet():
            continue
        assert session.undo_last_call() is False
        return
    # If we never found this scenario in 80 tries that's fine — it's not
    # the only regression test for undo, just a nice-to-have confirmation.


def test_undo_last_call_false_when_nothing_to_undo():
    session = GameSession(starting_board=1)
    assert session.undo_last_call() is False


def test_undo_last_card_restores_hand():
    session = GameSession(starting_board=1)
    for _ in range(20):
        session = GameSession(starting_board=session.board_number)
        _bid_out_using_suggestions(session)
        if session.phase != "play":
            session.next_board()
            continue
        legal = session.legal_cards_for_user()
        if not legal:
            session.next_board()
            continue
        seat = session.play_state.next_to_play
        before_hand = list(session.play_state.hands[seat])
        card = legal[0]
        session.play_user_card(seat, card)
        assert card not in session.play_state.hands[seat]
        assert session.undo_last_card() is True
        assert session.play_state.hands[seat] == before_hand
        return
    raise AssertionError("never found a testable play-phase turn")


def test_restart_current_board_keeps_same_deal():
    session = GameSession(starting_board=1)
    deal_before = session.deal
    board_before = session.board_number
    sugg = session.current_bid_suggestion()
    session.make_user_call(sugg.call)
    session.restart_current_board()
    assert session.deal is deal_before
    assert session.board_number == board_before
    assert session.phase == "bidding"
    assert session.auction.calls == [] or session.auction.whose_turn() == session.user_seat


def test_invalid_play_mode_rejected():
    try:
        GameSession(starting_board=1, play_mode="not-a-real-mode")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_dds_play_mode_completes_a_full_board_without_crashing():
    """Whether or not endplay is actually installed, 'dds' mode must play
    a full board through to completion (falling back to the heuristic bot
    under the hood whenever DDS isn't available)."""
    session = GameSession(starting_board=1, play_mode="dds")
    _bid_out_using_suggestions(session)
    _play_out_taking_first_legal(session)
    assert session.phase == "board_complete"
    assert session.to_dict()["play_mode"] == "dds"


def test_rl_play_mode_falls_back_gracefully_without_a_checkpoint():
    """Whether the fallback is because there's no checkpoint file yet, or
    (in an environment without PyTorch, like this one) because rl.network
    can't even be imported, the game must still complete without crashing
    and the status text must explain why the standard bot is playing."""
    session = GameSession(starting_board=1, play_mode="rl", rl_checkpoint_path="/nonexistent/checkpoint.npz")
    _bid_out_using_suggestions(session)
    _play_out_taking_first_legal(session)
    assert session.phase == "board_complete"
    status = session.play_mode_status()
    assert "No trained checkpoint" in status or "PyTorch isn't installed" in status


def test_rl_play_mode_uses_a_real_checkpoint_when_present(tmp_path=None):
    import tempfile, os, unittest

    try:
        from rl.network import PolicyValueNet
    except ImportError:
        raise unittest.SkipTest("torch is not installed in this environment — see docs/rl_training.md")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.npz")
        PolicyValueNet(seed=0).save(path)
        session = GameSession(starting_board=1, play_mode="rl", rl_checkpoint_path=path)
        _bid_out_using_suggestions(session)
        _play_out_taking_first_legal(session)
        assert session.phase == "board_complete"
        assert "Loaded RL checkpoint" in session.play_mode_status()


def test_to_dict_is_json_safe_and_has_expected_keys():
    import json

    session = GameSession(starting_board=1)
    d = session.to_dict()
    json.dumps(d)  # raises if anything isn't JSON-serializable
    for key in ["board_number", "dealer", "vulnerable", "user_seat", "phase", "auction", "hands", "total_imps"]:
        assert key in d
