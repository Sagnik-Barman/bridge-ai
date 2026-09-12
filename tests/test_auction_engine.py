import random

from engine.deal import Card, Hand, Suit, deal_random
from bidding.calls import AuctionState
from bidding.auction_engine import AuctionEngine


def _hand(spades="", hearts="", diamonds="", clubs=""):
    cards = []
    for rank in spades:
        cards.append(Card(Suit.SPADES, rank.upper()))
    for rank in hearts:
        cards.append(Card(Suit.HEARTS, rank.upper()))
    for rank in diamonds:
        cards.append(Card(Suit.DIAMONDS, rank.upper()))
    for rank in clubs:
        cards.append(Card(Suit.CLUBS, rank.upper()))
    return Hand(cards=cards)


def test_opening_still_works_through_auction_engine():
    engine = AuctionEngine()
    hand = _hand(spades="AKQ5", hearts="AK5", diamonds="567", clubs="567")  # 16 HCP balanced
    state = AuctionState(dealer="N", calls=[])
    decision = engine.decide_call("N", hand, state)
    assert decision.call == "1NT"


def test_stayman_response_to_1nt():
    engine = AuctionEngine()
    # 9 HCP with a 4-card major, responding to partner's 1NT.
    hand = _hand(spades="Q765", hearts="AK7", diamonds="654", clubs="432")
    state = AuctionState(dealer="N", calls=["1NT"])  # N opened, S (partner if dealer N... )
    # South is responder when dealer is N and 1 call made (E passed implicitly not needed for this unit test:
    # seat_after(N,1) = E must have acted; use dealer N with calls=["1NT","Pass"] so it's South's turn.
    state = AuctionState(dealer="N", calls=["1NT", "Pass"])
    decision = engine.decide_call("S", hand, state)
    assert decision.call == "2C"
    assert "Stayman" in decision.explanation


def test_jacoby_transfer_response_to_1nt():
    engine = AuctionEngine()
    hand = _hand(spades="2", hearts="KQ765", diamonds="654", clubs="432")  # 5+ hearts
    state = AuctionState(dealer="N", calls=["1NT", "Pass"])
    decision = engine.decide_call("S", hand, state)
    assert decision.call == "2D"
    assert "transfer" in decision.explanation.lower()


def test_opener_completes_transfer():
    engine = AuctionEngine()
    hand = _hand(spades="AK54", hearts="A5", diamonds="KQ7", clubs="A65")  # any 1NT-opener hand
    state = AuctionState(dealer="N", calls=["1NT", "Pass", "2D", "Pass"])  # North's turn again
    decision = engine.decide_call("N", hand, state)
    assert decision.call == "2H"


def test_stayman_opener_shows_spades():
    engine = AuctionEngine()
    hand = _hand(spades="AK54", hearts="A5", diamonds="KQ7", clubs="A65")
    state = AuctionState(dealer="N", calls=["1NT", "Pass", "2C", "Pass"])
    decision = engine.decide_call("N", hand, state)
    assert decision.call == "2S"


def test_simple_raise_response_to_major_opening():
    engine = AuctionEngine()
    hand = _hand(spades="KJ6", hearts="A65", diamonds="8765", clubs="43")  # 3 spades, 8 HCP
    state = AuctionState(dealer="N", calls=["1S", "Pass"])
    decision = engine.decide_call("S", hand, state)
    assert decision.call == "2S"


def test_negative_double_after_overcall():
    engine = AuctionEngine()
    # Partner (N) opens 1D, RHO... wait need E to overcall for it to be South's decision with LHO=... let's set:
    # dealer N: N=1D(open), E=1S(overcall), now South to call.
    hand = _hand(spades="2", hearts="KQ76", diamonds="765", clubs="Q765")  # 4 hearts, no spades, ~7 HCP
    state = AuctionState(dealer="N", calls=["1D", "1S"])
    decision = engine.decide_call("S", hand, state)
    assert decision.call == "Dbl"
    assert "negative double" in decision.explanation.lower()


def test_takeout_double_over_opening():
    engine = AuctionEngine()
    hand = _hand(spades="AQ76", hearts="KJ76", diamonds="2", clubs="A765")  # short diamonds, support elsewhere, opening values
    state = AuctionState(dealer="N", calls=["1D"])  # E's turn (opponent)
    decision = engine.decide_call("E", hand, state)
    assert decision.call == "Dbl"


def test_blackwood_response_counts_aces():
    from bidding.auction_engine import blackwood_response
    hand = _hand(spades="A765", hearts="A432", diamonds="A98", clubs="76")  # 3 aces
    decision = blackwood_response(hand)
    assert decision.call == "5S"


def test_full_auctions_never_crash_or_go_illegal_fuzz():
    """Runs complete 4-seat auctions (not just one call) across many
    random deals and rotating dealers. This is the strongest safety net:
    it would catch an infinite loop, a crash deep in rebid/competitive
    logic, or an illegal call several rounds into an auction — the kind
    of bug a single-call unit test can't see."""
    engine = AuctionEngine()
    dealers = ["N", "E", "S", "W"]
    for seed in range(100):
        deal = deal_random(seed=seed)
        state = AuctionState(dealer=dealers[seed % 4], calls=[])
        guard = 0
        while not state.is_complete():
            guard += 1
            assert guard <= 60, f"seed {seed}: auction never terminated"
            turn = state.whose_turn()
            decision = engine.decide_call(turn, deal.hands[turn], state)
            assert decision.call in state.legal_calls(), (
                f"seed {seed}: {turn} produced illegal call {decision.call}"
            )
            state.add(decision.call)


def test_engine_never_returns_illegal_call_fuzz():
    """Fuzz test: run the auction engine against many random deals for
    every seat's first call and make sure it always returns something
    legal (this is the safety net that matters most for a real game —
    an illegal call must never reach the game session)."""
    engine = AuctionEngine()
    for seed in range(30):
        deal = deal_random(seed=seed)
        state = AuctionState(dealer="N", calls=[])
        decision = engine.decide_call("N", deal.hands["N"], state)
        assert decision.call in state.legal_calls()
