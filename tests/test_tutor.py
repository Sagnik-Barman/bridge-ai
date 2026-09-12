from bidding.tutor import normalize_bid


def test_normalize_bid_pass_variants():
    assert normalize_bid("pass") == "Pass"
    assert normalize_bid("Pass") == "Pass"
    assert normalize_bid("PASS") == "Pass"
    assert normalize_bid("p") == "Pass"


def test_normalize_bid_suit_bids_case_insensitive():
    assert normalize_bid("1s") == "1S"
    assert normalize_bid("1S") == "1S"
    assert normalize_bid("2c") == "2C"
    assert normalize_bid("1nt") == "1NT"
    assert normalize_bid("1NT") == "1NT"
    assert normalize_bid("7nt") == "7NT"


def test_normalize_bid_rejects_garbage():
    assert normalize_bid("") is None
    assert normalize_bid("banana") is None
    assert normalize_bid("8S") is None  # no 8-level bids
    assert normalize_bid("0C") is None
