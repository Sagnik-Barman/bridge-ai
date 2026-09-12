from engine.deal import deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState
from cardplay.bot_player import choose_card as heuristic_choose_card
from benchmark.expert_benchmark import ExpertBoard, benchmark_board, summarize


def _play_a_full_board_with_the_heuristic_bot(seed):
    """Builds a real, complete 52-card play record by having the
    heuristic bot play both sides -- a convenient way to get a fully
    legal, fully deterministic "expert" board to test the benchmark
    harness against, without needing a real PBN file."""
    deal = deal_random(dealer="N", seed=seed)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")
    play_order = []
    while not ps.is_complete():
        seat = ps.next_to_play
        card = heuristic_choose_card(seat, ps, seed=0)
        play_order.append(card)
        ps.play_card(seat, card)
    return ExpertBoard(deal=deal, declarer="N", strain="NT", trump=None, play_order=play_order, board_num=1, event="test fixture")


def test_benchmark_board_matches_the_engine_that_actually_generated_the_line():
    # The heuristic bot played every card here (seed=0, same seed the
    # benchmark uses internally) -- replaying it should recompute the
    # identical suggestion at every non-forced decision, since each
    # decision point is a real position the bot has genuinely seen before.
    board = _play_a_full_board_with_the_heuristic_bot(seed=11)
    result = benchmark_board(board)
    assert result.board_num == 1
    assert result.event == "test fixture"
    assert 0 <= result.actual_declarer_tricks <= 13
    rate = result.match_rate("heuristic")
    assert rate == 1.0, f"expected the heuristic bot to match its own recorded line, got {rate}"


def test_forced_decisions_are_excluded_from_scoring():
    board = _play_a_full_board_with_the_heuristic_bot(seed=12)
    result = benchmark_board(board)
    # The very last card of a hand, or any point where only one card is
    # legal, must be marked forced and carry no matches to score.
    forced = [d for d in result.decisions if d.forced]
    for d in forced:
        assert d.matches == {}
    non_forced = [d for d in result.decisions if not d.forced]
    for d in non_forced:
        assert d.matches  # at least "heuristic" was judged


def test_trick_numbers_are_1_through_13():
    board = _play_a_full_board_with_the_heuristic_bot(seed=13)
    result = benchmark_board(board)
    assert len(result.decisions) == 52
    assert [d.trick_number for d in result.decisions[:4]] == [1, 1, 1, 1]
    assert [d.trick_number for d in result.decisions[-4:]] == [13, 13, 13, 13]


def test_benchmark_board_rejects_an_incomplete_play_record():
    board = _play_a_full_board_with_the_heuristic_bot(seed=14)
    board.play_order = board.play_order[:30]
    try:
        benchmark_board(board)
        raise AssertionError("expected ValueError for an incomplete play record")
    except ValueError:
        pass


def test_summarize_aggregates_across_boards():
    boards = [_play_a_full_board_with_the_heuristic_bot(seed=s) for s in (21, 22, 23)]
    results = [benchmark_board(b) for b in boards]
    summary = summarize(results)
    assert summary["num_boards"] == 3
    assert summary["heuristic"]["avg_match_rate"] == 1.0
    assert summary["actual_declarer_tricks_avg"] is not None
