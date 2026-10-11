import random

from engine.deal import Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState
from cardplay.dds_player import choose_card, dds_available, dds_status


def _fresh_play_state(seed):
    deal = deal_random(dealer="N", seed=seed)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    return PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")


def test_dds_status_mentions_endplay():
    assert "endplay" in dds_status()


def test_choose_card_always_legal_across_many_deals():
    """Small deal count deliberately — when `endplay` is actually installed
    (unlike the sandbox this was written in), every trick-lead decision
    here does real double-dummy solves via PIMC sampling (see
    cardplay/dds_player.py), so a handful of full boards is already
    hundreds of real DDS calls. This is a correctness sanity check, not a
    strength benchmark — `rl/evaluate.py`-style benchmarking belongs in a
    separate, opt-in script, not the default fast test suite."""
    for seed in range(6):
        ps = _fresh_play_state(seed)
        guard = 0
        while not ps.is_complete():
            guard += 1
            assert guard < 60
            seat = ps.next_to_play
            card = choose_card(seat, ps, seed=seed * 100 + guard, num_samples=4)
            legal = ps.legal_cards_for(seat)
            assert card in legal, f"illegal card chosen: {card} not in {legal}"
            ps.play_card(seat, card)
        assert ps.is_complete()


def test_choose_card_full_board_trumped_contract():
    """Same as above but with a real trump suit, to exercise the trump
    comparison logic in the PIMC lead scoring, not just NT."""
    for seed in range(4):
        deal = deal_random(dealer="N", seed=seed + 1000)
        hands = {s: list(deal.hands[s].cards) for s in SEATS}
        ps = PlayState(hands=hands, declarer="N", strain="S", trump=Suit.SPADES, dealer="N")
        guard = 0
        while not ps.is_complete():
            guard += 1
            assert guard < 60
            seat = ps.next_to_play
            card = choose_card(seat, ps, seed=seed, num_samples=4)
            assert card in ps.legal_cards_for(seat)
            ps.play_card(seat, card)


def test_falls_back_cleanly_without_endplay():
    # In this environment endplay is not installed, so DDS leads should
    # gracefully fall back to the heuristic lead/follow logic rather than
    # raising — this is the behavior real users without endplay get too.
    if dds_available():
        return  # nothing to assert here in an environment where it IS installed
    ps = _fresh_play_state(seed=7)
    card = choose_card(ps.next_to_play, ps, seed=7)
    assert card in ps.legal_cards_for(ps.next_to_play)


def test_pimc_samples_never_contain_already_played_cards():
    """Regression: the sampler used to put cards from earlier tricks (and the
    current trick) back into the pool, so PIMC solved impossible deals."""
    from cardplay.bot_player import choose_card as heuristic
    from cardplay.dds_player import _known_hands_for, _sample_unseen_hands, played_cards

    for seed in range(5):
        ps = _fresh_play_state(seed)
        for _ in range(18):  # four full tricks plus two cards of the fifth
            ps.play_card(ps.next_to_play, heuristic(ps.next_to_play, ps, seed=seed))
        seat = ps.next_to_play
        known = _known_hands_for(seat, ps)
        unseen = [s for s in SEATS if s not in known]
        played = played_cards(ps)
        assert len(played) == 18
        sampled = _sample_unseen_hands(known, {s: len(ps.hands[s]) for s in unseen}, random.Random(seed), played)
        for s in unseen:
            assert len(sampled[s]) == len(ps.hands[s])
            assert not (set(sampled[s]) & played)
        # the sampled world uses exactly the cards still unplayed
        world = [c for s in SEATS for c in (sampled[s] if s in unseen else ps.hands[s])]
        assert len(world) == len(set(world)) == 52 - 18
