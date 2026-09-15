"""Tests for rl/evaluate.py, including the fifth-attempt hybrid policy
(`_rl_policy_with_endgame_search`) added alongside the "add search on top"
idea, and its credit-attribution twin (`_heuristic_policy_with_endgame_
search`) that bolts the same exact search onto the plain heuristic bot
instead, so an improvement can be checked against "is this the network,
or just the search?" (see docs/rl_training.md's "A fifth attempt"
section). Torch-gated like tests/test_rl_network.py, since rl/evaluate.py
imports rl/network.py (the PyTorch version) -- see
tests/test_rl_network_numpy.py for the zero-dependency network tests."""
import random
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        "torch is not installed in this environment -- rl/evaluate.py "
        "imports rl/network.py, the PyTorch (CUDA-capable) version. See "
        "docs/rl_training.md for install instructions."
    )

from engine.deal import Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState
from cardplay.endgame_solver import solve_endgame
from rl.network import PolicyValueNet
from rl.evaluate import (
    _play_full_info,
    _rl_policy,
    _heuristic_policy,
    _rl_policy_with_endgame_search,
    _heuristic_policy_with_endgame_search,
    evaluate_vs_heuristic,
)


def _small_endgame_state(cards_per_hand=4, seed=0):
    """Build a real, legal PlayState with few cards left per hand by
    dealing full hands and then trimming each to the same small suit
    layout -- mirrors the setup style already used by
    tests/test_endgame_solver.py, kept self-contained here rather than
    importing that module's private helpers."""
    rng = random.Random(seed)
    deal = deal_random(dealer="N", seed=rng.randrange(1 << 30))
    hands = {s: list(deal.hands[s].cards)[:cards_per_hand] for s in SEATS}
    return PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")


def test_hybrid_matches_exact_solver_within_the_search_limit():
    """Below the endgame limit, the hybrid must return exactly what
    solve_endgame says -- the network's opinion shouldn't matter at all
    once the position is small enough to solve exactly. This is the one
    property that actually justifies calling this "search on top of the
    policy" rather than just another network variant."""
    net = PolicyValueNet(seed=0)
    hybrid = _rl_policy_with_endgame_search(net, endgame_limit=4)
    rng = random.Random(1)

    ps = _small_endgame_state(cards_per_hand=4, seed=2)
    expected = solve_endgame(ps, max_cards_per_hand=4).best_card
    got = hybrid(ps.next_to_play, ps, rng)
    assert got == expected


def test_hybrid_falls_back_to_the_network_above_the_search_limit():
    """Above the limit, the hybrid must behave exactly like the plain RL
    policy (same network, same seed draw from rng) -- no silent partial
    search, no crash, just the documented fallback."""
    net = PolicyValueNet(seed=0)
    hybrid = _rl_policy_with_endgame_search(net, endgame_limit=2)
    plain = _rl_policy(net)

    ps_a = _small_endgame_state(cards_per_hand=6, seed=3)
    ps_b = _small_endgame_state(cards_per_hand=6, seed=3)
    got = hybrid(ps_a.next_to_play, ps_a, random.Random(5))
    expected = plain(ps_b.next_to_play, ps_b, random.Random(5))
    assert got == expected


def test_hybrid_playthrough_completes_a_full_board():
    """End-to-end sanity check: a full board played out with the hybrid
    as declarer against the heuristic bot on defense must finish without
    error and return a legal trick count."""
    net = PolicyValueNet(seed=1)
    hybrid = _rl_policy_with_endgame_search(net, endgame_limit=6)
    heuristic = _heuristic_policy()

    deal = deal_random(dealer="N", seed=42)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")

    tricks = _play_full_info(ps, hybrid, heuristic, seed=42)
    assert 0 <= tricks <= 13


def test_evaluate_vs_heuristic_reports_hybrid_only_when_requested():
    net = PolicyValueNet(seed=2)

    plain_result = evaluate_vs_heuristic(net, num_deals=2, seed=0)
    assert "hybrid_declarer_avg_tricks" not in plain_result
    assert "heuristic_hybrid_declarer_avg_tricks" not in plain_result

    hybrid_result = evaluate_vs_heuristic(net, num_deals=2, seed=0, endgame_search_limit=6)
    assert "hybrid_declarer_avg_tricks" in hybrid_result
    assert "heuristic_hybrid_declarer_avg_tricks" in hybrid_result
    assert 0 <= hybrid_result["hybrid_declarer_avg_tricks"] <= 13
    assert 0 <= hybrid_result["heuristic_hybrid_declarer_avg_tricks"] <= 13
    # The plain RL and heuristic numbers must be unaffected by turning the
    # hybrid variants on -- they're extra playouts per deal, not changes to
    # the existing ones (same deal_seed drives every playout).
    assert hybrid_result["rl_declarer_avg_tricks"] == plain_result["rl_declarer_avg_tricks"]
    assert hybrid_result["heuristic_declarer_avg_tricks"] == plain_result["heuristic_declarer_avg_tricks"]


def test_heuristic_hybrid_matches_exact_solver_within_the_search_limit():
    """Same property as the RL hybrid, for the credit-attribution twin:
    below the limit it must defer entirely to the exact solver, regardless
    of what the heuristic bot itself would have chosen."""
    heuristic_hybrid = _heuristic_policy_with_endgame_search(endgame_limit=4)
    rng = random.Random(1)

    ps = _small_endgame_state(cards_per_hand=4, seed=2)
    expected = solve_endgame(ps, max_cards_per_hand=4).best_card
    got = heuristic_hybrid(ps.next_to_play, ps, rng)
    assert got == expected


def test_heuristic_hybrid_falls_back_to_the_heuristic_above_the_search_limit():
    heuristic_hybrid = _heuristic_policy_with_endgame_search(endgame_limit=2)
    plain = _heuristic_policy()

    ps_a = _small_endgame_state(cards_per_hand=6, seed=3)
    ps_b = _small_endgame_state(cards_per_hand=6, seed=3)
    got = heuristic_hybrid(ps_a.next_to_play, ps_a, random.Random(5))
    expected = plain(ps_b.next_to_play, ps_b, random.Random(5))
    assert got == expected
