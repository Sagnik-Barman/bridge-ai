from engine.deal import Card, Hand, Suit, deal_random
from cardplay.trick_engine import PlayState, IllegalPlayError, legal_plays
from cardplay.bot_player import choose_card


def _play_whole_board(seed, declarer, strain):
    d = deal_random(seed=seed)
    hands = {seat: list(d.hands[seat].cards) for seat in ["N", "E", "S", "W"]}
    trump = None if strain == "NT" else Suit(strain)
    ps = PlayState(hands=hands, declarer=declarer, strain=strain, trump=trump, dealer=d.dealer)
    guard = 0
    while not ps.is_complete():
        guard += 1
        assert guard <= 60, "auction/play never terminated"
        seat = ps.next_to_play
        card = choose_card(seat, ps, seed=seed * 100 + guard)
        ps.play_card(seat, card)
    return ps


def test_opening_leader_is_lho_of_declarer():
    d = deal_random(seed=1)
    hands = {seat: list(d.hands[seat].cards) for seat in ["N", "E", "S", "W"]}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer=d.dealer)
    assert ps.next_to_play == "E"


def test_must_follow_suit_when_able():
    hand = [Card(Suit.SPADES, "5"), Card(Suit.HEARTS, "9")]
    legal = legal_plays(hand, Suit.HEARTS)
    assert legal == [Card(Suit.HEARTS, "9")]


def test_may_play_anything_when_void():
    hand = [Card(Suit.SPADES, "5"), Card(Suit.CLUBS, "9")]
    legal = legal_plays(hand, Suit.HEARTS)
    assert set(legal) == set(hand)


def test_trump_beats_led_suit():
    d = deal_random(seed=2)
    hands = {seat: list(d.hands[seat].cards) for seat in ["N", "E", "S", "W"]}
    ps = PlayState(hands=hands, declarer="N", strain="S", trump=Suit.SPADES, dealer=d.dealer)
    # Force a trick: N leads a heart, E follows low heart, S ruffs with a spade, W follows heart.
    leader = ps.next_to_play  # E, LHO of N
    # give control by directly constructing a trick rather than relying on hand contents
    from cardplay.trick_engine import Trick, _card_strength
    trick = Trick(leader="E", plays=[
        ("E", Card(Suit.HEARTS, "5")),
        ("S", Card(Suit.HEARTS, "6")),
        ("W", Card(Suit.SPADES, "2")),  # ruff
        ("N", Card(Suit.HEARTS, "K")),
    ])
    assert trick.winner(Suit.SPADES) == "W"  # the ruff wins even though N had the highest heart


def test_full_board_play_completes_with_13_tricks_multiple_deals():
    for seed in range(10):
        ps = _play_whole_board(seed, declarer=["N", "E", "S", "W"][seed % 4], strain=["NT", "S", "H", "D", "C"][seed % 5])
        totals = ps.tricks_by_side()
        assert sum(totals.values()) == 13
        assert all(len(h) == 0 for h in ps.hands.values())


def test_out_of_turn_play_rejected():
    d = deal_random(seed=3)
    hands = {seat: list(d.hands[seat].cards) for seat in ["N", "E", "S", "W"]}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer=d.dealer)
    wrong_seat = "N"  # declarer doesn't lead; E (LHO) does
    try:
        ps.play_card(wrong_seat, hands[wrong_seat][0])
        assert False, "should have raised IllegalPlayError"
    except IllegalPlayError:
        pass
