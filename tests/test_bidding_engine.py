from engine.deal import Card, Hand, Suit
from bidding.engine import BiddingEngine, load_convention, describe_conditions


def _hand_from_pbn_suits(spades, hearts, diamonds, clubs):
    cards = []
    for rank in spades:
        cards.append(Card(Suit.SPADES, rank))
    for rank in hearts:
        cards.append(Card(Suit.HEARTS, rank))
    for rank in diamonds:
        cards.append(Card(Suit.DIAMONDS, rank))
    for rank in clubs:
        cards.append(Card(Suit.CLUBS, rank))
    return Hand(cards=cards)


def test_load_convention():
    conv = load_convention()
    assert conv["system_name"] == "Standard American 2/1 Game Force"
    assert len(conv["opening_bids"]) > 0


def test_balanced_15_17_opens_1nt():
    # 4-3-3-3, HCP = A K Q of spades (4+3+2=9) + A K of hearts (4+3=7) = 16
    hand = _hand_from_pbn_suits(
        spades=["A", "K", "Q", "5"],
        hearts=["A", "K", "5"],
        diamonds=["5", "6", "7"],
        clubs=["5", "6", "7"],
    )
    assert hand.hcp() == 16
    assert hand.is_balanced()
    engine = BiddingEngine()
    assert engine.opening_bid(hand) == "1NT"


def test_weak_hand_passes():
    hand = _hand_from_pbn_suits(
        spades=["5", "6", "7"],
        hearts=["5", "6", "7"],
        diamonds=["5", "6", "7"],
        clubs=["5", "6", "7", "8"],
    )
    assert hand.hcp() == 0
    engine = BiddingEngine()
    assert engine.opening_bid(hand) == "Pass"


def test_five_card_major_opens_at_one_level():
    # 5-3-3-2 shape (not balanced by our 4333/4432/5422 definition), 17 HCP.
    hand = _hand_from_pbn_suits(
        spades=["A", "K", "Q", "J", "5"],
        hearts=["5", "6"],
        diamonds=["A", "K", "7"],
        clubs=["5", "6", "7"],
    )
    assert hand.hcp() == 17
    assert not hand.is_balanced()
    engine = BiddingEngine()
    assert engine.opening_bid(hand) == "1S"


def test_explanation_matches_plain_opening_bid():
    hand = _hand_from_pbn_suits(
        spades=["A", "K", "Q", "5"],
        hearts=["A", "K", "5"],
        diamonds=["5", "6", "7"],
        clubs=["5", "6", "7"],
    )
    engine = BiddingEngine()
    expl = engine.opening_bid_with_explanation(hand)
    assert expl.bid == engine.opening_bid(hand) == "1NT"
    assert "15-17" in expl.conditions_text
    assert "16 HCP" in expl.hand_summary


def test_rules_reference_lists_every_opening_bid():
    engine = BiddingEngine()
    ref = engine.rules_reference()
    for rule in engine.convention["opening_bids"]:
        assert rule["bid"] in ref
        assert rule["name"] in ref


def test_describe_conditions_handles_empty():
    assert describe_conditions({}) == "(no conditions)"


def test_preempt_opening_resolves_to_a_real_bid():
    # 6 HCP, 7-card diamond suit: should preempt 3D, not the literal
    # placeholder string "3-level preempt" from the YAML (regression test
    # for a bug where this fell through as an illegal call in real play).
    hand = _hand_from_pbn_suits(
        spades="2", hearts="53", diamonds="AQ98765", clubs="654"
    )
    engine = BiddingEngine()
    assert hand.hcp() == 6
    bid = engine.opening_bid(hand)
    assert bid == "3D"
    expl = engine.opening_bid_with_explanation(hand)
    assert expl.bid == "3D"
    from bidding.calls import is_suit_bid
    assert is_suit_bid(bid)
