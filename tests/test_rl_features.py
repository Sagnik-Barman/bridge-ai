from engine.deal import SEATS, deal_random
from cardplay.trick_engine import PlayState
from rl import features


def _play_state(seed=1):
    deal = deal_random(dealer="N", seed=seed)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    return PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")


def test_encode_shape_and_range():
    ps = _play_state()
    x = features.encode("N", ps)
    assert x.shape == (features.FEATURE_SIZE,)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_own_hand_plane_matches_hand():
    ps = _play_state()
    x = features.encode("N", ps)
    own_plane = x[:52]
    expected_indices = {features.CARD_INDEX[c] for c in ps.hands["N"]}
    actual_indices = {i for i in range(52) if own_plane[i] > 0}
    assert actual_indices == expected_indices


def test_legal_mask_matches_legal_plays():
    ps = _play_state()
    mask = features.legal_mask("N", ps)
    # Opening lead: no led suit yet, so every card in hand should be legal.
    hand_indices = {features.CARD_INDEX[c] for c in ps.hands["N"]}
    legal_indices = {i for i in range(52) if mask[i] > 0}
    assert legal_indices == hand_indices


def test_legal_mask_after_a_card_is_led():
    ps = _play_state()
    seat = ps.next_to_play
    card = ps.hands[seat][0]
    ps.play_card(seat, card)
    next_seat = ps.next_to_play
    mask = features.legal_mask(next_seat, ps)
    from cardplay.trick_engine import legal_plays

    expected = {features.CARD_INDEX[c] for c in legal_plays(ps.hands[next_seat], ps.current_trick.led_suit())}
    actual = {i for i in range(52) if mask[i] > 0}
    assert actual == expected


def test_encode_updates_as_tricks_complete():
    ps = _play_state()
    x_before = features.encode(ps.next_to_play, ps)
    seat = ps.next_to_play
    card = ps.hands[seat][0]
    ps.play_card(seat, card)
    x_after = features.encode(seat, ps) if seat == ps.next_to_play else None
    # the trick tracker (last feature) should reflect one card played
    x_mid = features.encode(ps.next_to_play, ps)
    assert x_mid[-1] > 0  # cards_left_in_trick_before_mover / 3 > 0
