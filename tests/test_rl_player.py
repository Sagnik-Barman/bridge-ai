import unittest

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("torch is not installed in this environment — see docs/rl_training.md")

from engine.deal import SEATS, deal_random
from cardplay.trick_engine import PlayState
from rl.network import PolicyValueNet
from rl.rl_player import choose_card, rl_checkpoint_status


def _fresh_play_state(seed):
    deal = deal_random(dealer="N", seed=seed)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    return PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")


def test_choose_card_always_legal_with_an_untrained_network():
    net = PolicyValueNet(seed=0)
    for seed in range(15):
        ps = _fresh_play_state(seed)
        guard = 0
        while not ps.is_complete():
            guard += 1
            assert guard < 60
            seat = ps.next_to_play
            card = choose_card(seat, ps, net, seed=seed * 10 + guard, num_samples=3)
            assert card in ps.legal_cards_for(seat)
            ps.play_card(seat, card)


def test_choose_card_for_declarer_sampling_only_the_two_defenders():
    """Declarer sees its own hand and dummy's, so PIMC only needs to
    sample the two defenders' hands — exercise that path specifically by
    advancing to the point declarer (N) is on lead."""
    net = PolicyValueNet(seed=1)
    ps = _fresh_play_state(seed=42)
    guard = 0
    while ps.next_to_play != "N" and not ps.is_complete():
        guard += 1
        assert guard < 60
        seat = ps.next_to_play
        card = ps.legal_cards_for(seat)[0]
        ps.play_card(seat, card)
    assert not ps.is_complete()
    card = choose_card("N", ps, net, seed=1, num_samples=3)
    assert card in ps.legal_cards_for("N")


def test_rl_checkpoint_status_mentions_missing_when_absent():
    msg = rl_checkpoint_status("/nonexistent/path/checkpoint.npz")
    assert "No trained checkpoint" in msg


def test_v2_checkpoints_load_and_play_legal_cards():
    import os, tempfile
    from rl.network_v2 import CardPlayNet
    from rl.rl_player import load_any
    for obs in ("visible", "full"):
        path = os.path.join(tempfile.mkdtemp(), "v2.pt")
        CardPlayNet(obs=obs, width=32, blocks=1).save(path)
        net = load_any(path)
        assert getattr(net, "obs") == obs
        ps = _fresh_play_state(seed=5)
        guard = 0
        while not ps.is_complete():
            guard += 1
            assert guard < 60
            seat = ps.next_to_play
            card = choose_card(seat, ps, net, seed=guard, num_samples=2)
            assert card in ps.legal_cards_for(seat)
            ps.play_card(seat, card)
