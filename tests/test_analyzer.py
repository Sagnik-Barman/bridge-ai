from game.session import GameSession, IllegalActionError


def _bid_out_using_suggestions(session, guard_limit=40):
    guard = 0
    while session.phase == "bidding":
        guard += 1
        assert guard < guard_limit, "bidding never completed"
        sugg = session.current_bid_suggestion()
        assert sugg is not None
        session.make_user_call(sugg.call)


def test_analyze_raises_outside_play_phase():
    session = GameSession(starting_board=1)
    assert session.phase == "bidding"
    try:
        session.analyze_current_position()
        raise AssertionError("expected IllegalActionError while still bidding")
    except IllegalActionError:
        pass


def test_analyze_reports_all_four_sources_with_availability_flags():
    session = GameSession(starting_board=1)
    _bid_out_using_suggestions(session)
    if session.phase != "play":
        return  # passed out — nothing to analyze on this particular deal
    assert session.phase == "play"

    analysis = session.analyze_current_position()
    assert analysis["seat"] == session.play_state.next_to_play
    assert set(analysis["legal_cards"]) == {str(c) for c in session.play_state.legal_cards_for(analysis["seat"])}

    for source in ("heuristic", "dds", "rl", "exact_endgame"):
        assert source in analysis
        entry = analysis[source]
        assert "available" in entry and "suggested_card" in entry and "note" in entry
        if entry["available"]:
            assert entry["suggested_card"] in analysis["legal_cards"]

    # Heuristic never depends on optional installs, so it must always work.
    assert analysis["heuristic"]["available"] is True

    # "sources_agree"/"consensus" must be internally consistent with the
    # per-source suggestions actually reported.
    suggestions = {
        src: analysis[src]["suggested_card"]
        for src in ("heuristic", "dds", "rl", "exact_endgame")
        if analysis[src]["available"]
    }
    distinct = set(suggestions.values())
    if len(suggestions) > 1:
        assert analysis["sources_agree"] == (len(distinct) == 1)
    else:
        assert analysis["sources_agree"] is False


def test_analyze_endgame_solver_unavailable_note_mentions_card_count_early_in_the_deal():
    session = GameSession(starting_board=1)
    _bid_out_using_suggestions(session)
    if session.phase != "play":
        return
    analysis = session.analyze_current_position()
    # 13 cards left at the very start of play — far past the exact solver's
    # small-endgame limit, so it should explain why rather than attempt it.
    assert analysis["exact_endgame"]["available"] is False
    assert "cards left" in analysis["exact_endgame"]["note"]


def test_analyze_raises_once_board_is_fully_played():
    session = GameSession(starting_board=1)
    _bid_out_using_suggestions(session)
    if session.phase != "play":
        return
    guard = 0
    while session.phase == "play":
        guard += 1
        assert guard < 60
        legal = session.legal_cards_for_user()
        if not legal:
            break
        seat = session.play_state.next_to_play
        session.play_user_card(seat, legal[0])
    assert session.phase == "board_complete"
    try:
        session.analyze_current_position()
        raise AssertionError("expected IllegalActionError once play is complete")
    except IllegalActionError:
        pass
