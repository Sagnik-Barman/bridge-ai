from engine.deal import Suit, deal_random, parse_pbn_deal


def test_deal_random_deals_13_each():
    d = deal_random(seed=1)
    for seat in ["N", "E", "S", "W"]:
        assert len(d.hands[seat].cards) == 13


def test_deal_random_no_duplicate_cards():
    d = deal_random(seed=2)
    all_cards = [c for hand in d.hands.values() for c in hand.cards]
    assert len(all_cards) == len(set(all_cards)) == 52


def test_hcp_total_is_40():
    d = deal_random(seed=3)
    total = sum(d.hands[seat].hcp() for seat in ["N", "E", "S", "W"])
    assert total == 40


def test_pbn_roundtrip_format():
    d = deal_random(seed=4)
    pbn = d.pbn()
    assert pbn.startswith(f"{d.dealer}:")
    hands_part = pbn.split(":", 1)[1]
    assert len(hands_part.split(" ")) == 4


def test_parse_pbn_deal_round_trips_through_our_own_writer():
    d = deal_random(dealer="E", seed=9)
    parsed = parse_pbn_deal(d.pbn())
    assert parsed.dealer == "E"
    for seat in ["N", "E", "S", "W"]:
        assert sorted(parsed.hands[seat].cards, key=str) == sorted(d.hands[seat].cards, key=str)


def test_parse_pbn_deal_handles_a_starting_seat_other_than_the_real_dealer():
    # e.g. endplay's Deal.to_pbn() always starts the hand list from North,
    # regardless of the board's actual dealer -- callers pass the real
    # dealer separately.
    pbn = "N:AKQ.T98.765.4 J.AKQ.T98.765 T98.J.AKQ.T98 765.4.J.AKQJ"
    parsed = parse_pbn_deal(pbn, dealer="S")
    assert parsed.dealer == "S"
    assert [str(c) for c in parsed.hands["N"].cards] == ["AS", "KS", "QS", "TH", "9H", "8H", "7D", "6D", "5D", "4C"]
    assert [str(c) for c in parsed.hands["W"].cards] == ["7S", "6S", "5S", "4H", "JD", "AC", "KC", "QC", "JC"]


def test_parse_pbn_deal_rejects_malformed_strings():
    for bad in ["not a pbn string", "N:only.one.hand", "Q:AKQ.T98.765.4 J.AKQ.T98.765 T98.J.AKQ.T98 765.4.J.AKQJ"]:
        try:
            parse_pbn_deal(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass
