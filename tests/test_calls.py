from bidding.calls import AuctionState, bid_rank, seat_after, partner_of


def test_seat_after_and_partner():
    assert seat_after("N") == "E"
    assert seat_after("N", 2) == "S"
    assert seat_after("W", 1) == "N"
    assert partner_of("N") == "S"
    assert partner_of("E") == "W"


def test_bid_rank_ordering():
    assert bid_rank("1C") < bid_rank("1D") < bid_rank("1H") < bid_rank("1S") < bid_rank("1NT")
    assert bid_rank("1NT") < bid_rank("2C")
    assert bid_rank("7NT") == 34


def test_whose_turn():
    state = AuctionState(dealer="N", calls=[])
    assert state.whose_turn() == "N"
    state.calls = ["Pass", "1S"]
    assert state.whose_turn() == "S"


def test_legal_calls_after_opening():
    state = AuctionState(dealer="N", calls=["1S"])
    legal = state.legal_calls()
    assert "Pass" in legal
    assert "1NT" in legal  # higher ranked
    assert "1H" not in legal  # lower ranked, illegal
    assert "Dbl" in legal  # East may double South's... wait N opened, E is opponent


def test_double_only_of_opponent():
    # N opens 1S; E (opponent) may double. But after E doubles, S (partner
    # of opener) may NOT re-double (redouble is only for the opener's side).
    state = AuctionState(dealer="N", calls=["1S", "Dbl"])
    legal = state.legal_calls()
    assert "Rdbl" in legal  # it's South's turn now, South is opener's partner


def test_auction_is_complete_after_three_passes():
    state = AuctionState(dealer="N", calls=["1S", "Pass", "Pass", "Pass"])
    assert state.is_complete()
    assert not state.is_passed_out()


def test_passed_out():
    state = AuctionState(dealer="N", calls=["Pass", "Pass", "Pass", "Pass"])
    assert state.is_complete()
    assert state.is_passed_out()


def test_declarer_is_first_of_partnership_to_bid_the_strain():
    # N opens 1S, S responds 2H (new suit), N raises 3H, S bids 4H, all pass.
    # South bid hearts first among the N/S partnership, so South is declarer.
    state = AuctionState(dealer="N", calls=["1S", "Pass", "2H", "Pass", "3H", "Pass", "4H", "Pass", "Pass", "Pass"])
    declarer, level, strain, dbl = state.declarer_and_contract()
    assert declarer == "S"
    assert level == 4
    assert strain == "H"
    assert dbl == ""


def test_declarer_with_double():
    state = AuctionState(dealer="N", calls=["1S", "Dbl", "Pass", "Pass", "Pass"])
    declarer, level, strain, dbl = state.declarer_and_contract()
    assert declarer == "N"
    assert level == 1
    assert strain == "S"
    assert dbl == "X"
