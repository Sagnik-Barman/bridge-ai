import random
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("torch is not installed in this environment — see docs/rl_training.md")

from rl.env import play_game, play_self_play_game, decisions_to_training_batch
from rl.network import PolicyValueNet


def test_self_play_game_completes_with_13_tricks():
    net = PolicyValueNet(seed=0)
    py_rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    outcome = play_self_play_game(net, py_rng, np_rng)
    assert outcome.tricks["NS"] + outcome.tricks["EW"] == 13
    assert len(outcome.decisions) == 52  # 4 seats x 13 tricks


def test_decisions_to_training_batch_returns_are_per_trick_return_to_go():
    net = PolicyValueNet(seed=1)
    py_rng = random.Random(1)
    np_rng = np.random.default_rng(1)
    outcome = play_self_play_game(net, py_rng, np_rng)
    batch = decisions_to_training_batch(outcome)
    assert len(batch) == len(outcome.decisions)

    # A decision made during the very first trick (trick_index == 0) must
    # get the full final trick count for its side -- return-to-go from
    # the start of the game is, by definition, the whole game's total.
    for (x, mask, action, ret), decision in zip(batch, outcome.decisions):
        assert mask[action] == 1.0  # the action taken must have been legal
        if decision.trick_index == 0:
            assert ret == float(outcome.tricks[decision.side])

    # A decision made during the LAST trick can only ever be worth at
    # most 1 (there's only one trick left to win) -- strictly smaller
    # than the whole-game total whenever that total is more than 1,
    # which is the whole point of return-to-go over a flat final total.
    for d, (x, mask, action, ret) in zip(outcome.decisions, batch):
        if d.trick_index == 12:
            assert ret in (0.0, 1.0)

    # Both sides' return-to-go at any given trick index must sum to
    # exactly the number of tricks remaining from that point (every
    # trick is won by exactly one side).
    by_trick = {}
    for d, (_, _, _, ret) in zip(outcome.decisions, batch):
        by_trick.setdefault(d.trick_index, {})[d.side] = ret
    for trick_index, sides in by_trick.items():
        if len(sides) == 2:  # both sides had a recorded decision at this trick
            assert sides["NS"] + sides["EW"] == 13 - trick_index


def test_greedy_self_play_is_deterministic_given_same_seeds():
    net = PolicyValueNet(seed=2)
    outcome_a = play_self_play_game(net, random.Random(5), np.random.default_rng(5), greedy=True)
    outcome_b = play_self_play_game(net, random.Random(5), np.random.default_rng(5), greedy=True)
    assert outcome_a.tricks == outcome_b.tricks


def test_heuristic_opponent_side_records_only_the_networks_own_decisions():
    net = PolicyValueNet(seed=3)
    for heuristic_side in ("NS", "EW"):
        outcome = play_game(net, random.Random(6), np.random.default_rng(6), heuristic_side=heuristic_side)
        assert outcome.tricks["NS"] + outcome.tricks["EW"] == 13
        # Half the seats (26 of 52 decisions) were played by the heuristic
        # bot and must not show up as network decisions to train on.
        assert len(outcome.decisions) == 26
        assert all(d.side != heuristic_side for d in outcome.decisions)


def test_play_game_with_no_heuristic_side_matches_pure_self_play():
    net = PolicyValueNet(seed=4)
    outcome = play_game(net, random.Random(7), np.random.default_rng(7), heuristic_side=None)
    assert len(outcome.decisions) == 52
