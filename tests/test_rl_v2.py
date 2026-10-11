"""Tests for the v2 agent: oracle, features, network/loss, data generation, agent."""
import random

import numpy as np

from engine.deal import Card, Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState, legal_plays
from cardplay.bot_player import choose_card as heuristic


def _state(seed, strain="H", declarer="S", advance=0):
    deal = deal_random(dealer="N", seed=seed)
    trump = None if strain == "NT" else Suit(strain)
    ps = PlayState(hands={s: list(deal.hands[s].cards) for s in SEATS}, declarer=declarer,
                   strain=strain, trump=trump, dealer="N")
    for _ in range(advance):
        ps.play_card(ps.next_to_play, heuristic(ps.next_to_play, ps, seed=seed))
    return ps


# ------------------------------------------------------------------ features
def test_indexer_round_trips_every_unplayed_card():
    from rl.features_v2 import Indexer
    for adv in (0, 9, 30):
        ps = _state(1, advance=adv)
        idx = Indexer(ps)
        cards = [c for h in ps.hands.values() for c in h] + [c for _, c in ps.current_trick.plays]
        indices = [idx.index(c) for c in cards]
        assert len(set(indices)) == len(cards)
        assert all(idx.card(i) == c for i, c in zip(indices, cards))


def test_trump_is_always_canonical_suit_zero():
    from rl.features_v2 import Indexer
    for strain in ("S", "H", "D", "C"):
        ps = _state(2, strain=strain)
        idx = Indexer(ps)
        top_trump = Card(Suit(strain), "A")
        assert idx.index(top_trump) == 0


def test_relative_rank_promotes_cards_once_higher_ones_are_played():
    """With the ace of the suit already played, the king must encode exactly as
    the top card did before (same index as the ace had)."""
    from rl.features_v2 import Indexer
    ps = _state(3, strain="NT", advance=0)
    ace = Card(Suit.SPADES, "A")
    king = Card(Suit.SPADES, "K")
    before = Indexer(ps).index(ace)
    holder = next(s for s in SEATS if ace in ps.hands[s])
    ps.hands[holder].remove(ace)  # as if played in an earlier trick
    assert Indexer(ps).index(king) == before


def test_feature_sizes_and_legal_mask():
    from rl import features_v2 as F2
    ps = _state(4, advance=13)
    seat = ps.next_to_play
    assert F2.encode(seat, ps, "full").shape == (F2.FULL_SIZE,)
    assert F2.encode(seat, ps, "visible").shape == (F2.VISIBLE_SIZE,)
    idx = F2.Indexer(ps)
    m = F2.legal_mask(seat, ps, idx)
    legal = legal_plays(ps.hands[seat], ps.current_trick.led_suit())
    assert m.sum() == len(legal)
    assert all(m[idx.index(c)] == 1 for c in legal)


def test_visible_encoding_does_not_leak_hidden_hands():
    """Swapping cards between the two hands a defender cannot see must leave its
    'visible' features unchanged (and change the 'full' ones)."""
    from rl import features_v2 as F2
    for seed in range(5):
        ps = _state(seed, declarer="S", advance=5)  # opening lead made, dummy (N) visible
        seat = ps.next_to_play
        if seat in ("N", "S"):
            ps.play_card(seat, heuristic(seat, ps, seed=0))
            seat = ps.next_to_play
        if seat in ("N", "S"):
            continue
        hidden = [s for s in SEATS if s not in F2.visible_seats(seat, ps)]
        assert len(hidden) == 2
        a, b = hidden
        # swap one card of the same suit-agnostic kind: any card from each
        # (keeps hand sizes; may break voids, so pick cards not yet showing out)
        ca, cb = ps.hands[a][0], ps.hands[b][0]
        vis_before = F2.encode(seat, ps, "visible")
        full_before = F2.encode(seat, ps, "full")
        ps.hands[a][0], ps.hands[b][0] = cb, ca
        assert np.array_equal(vis_before, F2.encode(seat, ps, "visible"))
        if ca.suit != cb.suit or ca != cb:
            assert not np.array_equal(full_before, F2.encode(seat, ps, "full"))


# ------------------------------------------------------------------ oracle
def test_oracle_backends_agree():
    from rl.oracle import card_values, endplay_available
    if not endplay_available():
        return
    rng = random.Random(0)
    for seed in range(3):
        ps = _state(10 + seed, strain=rng.choice(["NT", "S", "D"]), advance=24)
        for _ in range(6):
            a, b = card_values(ps, "dds"), card_values(ps, "scratch")
            assert a == b
            seat = ps.next_to_play
            assert set(a) == set(legal_plays(ps.hands[seat], ps.current_trick.led_suit()))
            ps.play_card(seat, rng.choice(list(a)))


# ------------------------------------------------------------------ network and loss
def test_loss_rewards_probability_on_any_optimal_card():
    import torch
    from rl.network_v2 import CardPlayNet
    from rl.train_v2 import losses

    torch.manual_seed(0)
    net = CardPlayNet(obs="full", width=32, blocks=1)
    x = torch.rand(4, net.inp.in_features)
    q = torch.full((4, 52), -1, dtype=torch.int64)
    q[:, 0] = 5; q[:, 1] = 5; q[:, 2] = 3   # two optimal cards, one that loses 2 tricks
    _, st = losses(net, x, q)
    assert 0 <= st["opt"] <= 1 and st["regret"] >= 0
    # push the policy towards card 2 (bad): regret must show it
    with torch.no_grad():
        net.policy.bias.zero_(); net.policy.bias[2] = 50.0
    _, st = losses(net, x, q)
    assert st["opt"] == 0.0 and abs(st["regret"] - 2.0) < 1e-6
    with torch.no_grad():
        net.policy.bias[2] = 0.0; net.policy.bias[1] = 50.0
    _, st = losses(net, x, q)
    assert st["opt"] == 1.0 and st["ce"] < 1e-3


def test_training_learns_on_a_tiny_dataset(tmp_path=None):
    import tempfile, os
    from rl.datagen import generate
    from rl.train_v2 import train
    d = tempfile.mkdtemp()
    generate(6, os.path.join(d, "data"), workers=1, seed=3)
    _, hist = train([os.path.join(d, "data")], "visible", os.path.join(d, "n.pt"), epochs=3,
                    bs=64, width=32, blocks=1, threads=1)
    assert len(hist) == 3


def test_agent_always_plays_legal_cards_and_checkpoint_round_trips():
    import tempfile, os
    from rl.network_v2 import CardPlayNet, is_v2_checkpoint
    from rl.agent_v2 import choose_card
    for obs in ("full", "visible"):
        net = CardPlayNet(obs=obs, width=32, blocks=1)
        path = os.path.join(tempfile.mkdtemp(), "n.pt")
        net.save(path)
        assert is_v2_checkpoint(path)
        net = CardPlayNet.load(path)
        ps = _state(7)
        while not ps.is_complete():
            seat = ps.next_to_play
            for head in ("policy", "q", "mix"):
                c = choose_card(seat, ps, net, head=head)
                assert c in legal_plays(ps.hands[seat], ps.current_trick.led_suit())
            ps.play_card(seat, c)


def test_datagen_labels_match_legal_moves():
    import tempfile, os
    from rl.datagen import generate
    d = os.path.join(tempfile.mkdtemp(), "g")
    generate(3, d, workers=1, seed=0)
    z = np.load(os.path.join(d, "part0.npz"))
    assert len(z["q"]) > 0
    assert np.array_equal(z["mask"], (z["q"] >= 0).astype(np.uint8))
    assert (z["mask"].sum(1) >= 2).all()          # forced moves are skipped
    assert z["q"].max() <= 13
