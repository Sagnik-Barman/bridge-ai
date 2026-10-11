"""Tests for the from-scratch double-dummy solver (ddsolver/).

Correct answers come from three independent places: a brute-force minimax written
here in a few lines of Python (small positions), the project's own
cardplay/endgame_solver.py, and a full table computed by the DDS library.
The large-scale comparison against DDS is benchmark/dds_scratch_benchmark.py.
"""
import functools
import random
import unittest

import ddsolver as D

if not D.is_available():  # pragma: no cover
    raise unittest.SkipTest("ddsolver library could not be loaded or built (pip install ziglang)")


def _random_masks(rng, k):
    deck = [(s, r) for s in range(4) for r in range(13)]
    rng.shuffle(deck)
    masks = [[0] * 4 for _ in range(4)]
    for p in range(4):
        for s, r in deck[p * k:(p + 1) * k]:
            masks[p][s] |= 1 << r
    return masks


def _brute_force(masks, trump, leader, trick=()):
    """N-S tricks from here by exhaustive minimax (no pruning at all)."""
    @functools.lru_cache(maxsize=None)
    def value(hands, leader, trick):
        p = (leader + len(trick)) % 4
        if not any(any(h) for h in hands):
            return 0
        mine = hands[p]
        suits = [trick[0][0]] if trick and mine[trick[0][0]] else range(4)
        results = []
        for s in suits:
            for r in range(13):
                if not mine[s] >> r & 1:
                    continue
                nh = [list(h) for h in hands]
                nh[p][s] &= ~(1 << r)
                nh = tuple(tuple(h) for h in nh)
                t = trick + ((s, r),)
                if len(t) == 4:
                    w = 0
                    for i in range(1, 4):
                        if t[i][0] == t[w][0]:
                            if t[i][1] > t[w][1]:
                                w = i
                        elif t[i][0] == trump:
                            w = i
                    winner = (leader + w) % 4
                    results.append((winner % 2 == 0) + value(nh, winner, ()))
                else:
                    results.append(value(nh, leader, t))
        return max(results) if p % 2 == 0 else min(results)

    return value(tuple(tuple(h) for h in masks), leader, tuple(trick))


def test_single_last_trick():
    # N: ace of spades, the others small hearts; spades are trumps; N leads.
    masks = [[1 << 12, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]]
    assert D.solve(masks, trump=0, leader=0)[0] == 1
    # Same, but East holds the trump ace.
    masks = [[0, 1, 0, 0], [1 << 12, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]]
    assert D.solve(masks, trump=0, leader=0)[0] == 0


def test_matches_brute_force_on_small_positions():
    rng = random.Random(7)
    for k in (1, 2, 3):
        for _ in range(40):
            masks = _random_masks(rng, k)
            trump, leader = rng.randrange(5), rng.randrange(4)
            assert D.solve(masks, trump, leader)[0] == _brute_force(masks, trump, leader)


def test_matches_brute_force_in_the_middle_of_a_trick():
    rng = random.Random(8)
    for _ in range(60):
        k = 3
        masks = _random_masks(rng, k)
        trump, leader = rng.randrange(5), rng.randrange(4)
        # play 1-3 random legal cards into the first trick
        trick = []
        for i in range(rng.randrange(1, 4)):
            p = (leader + i) % 4
            led = trick[0][0] if trick else None
            options = [(s, r) for s in range(4) for r in range(13) if masks[p][s] >> r & 1
                       and (led is None or not masks[p][led] or s == led)]
            s, r = rng.choice(options)
            masks[p][s] &= ~(1 << r)
            trick.append((s, r))
        want = _brute_force(masks, trump, leader, tuple(trick))
        assert D.solve(masks, trump, leader, trick)[0] == want
        scored, _ = D.score_cards(masks, trump, leader, trick)
        best = max(v for _, _, v in scored) if (leader + len(trick)) % 2 == 0 else min(v for _, _, v in scored)
        assert best == want


def test_every_technique_gives_the_same_answer():
    rng = random.Random(9)
    configs = [D.MINIMAX, 0, D.ORDER, D.ORDER | D.EQUIV, D.ORDER | D.EQUIV | D.TT,
               D.ORDER | D.EQUIV | D.TT | D.RELRANK, D.ALL & ~D.QUICK, D.ALL]
    for _ in range(25):
        masks = _random_masks(rng, 5)
        trump, leader = rng.randrange(5), rng.randrange(4)
        answers = {D.solve(masks, trump, leader, flags=f)[0] for f in configs}
        assert len(answers) == 1, answers


def test_full_deal_table_matches_dds():
    # Reference table computed by the DDS library (dds-bridge/dds via endplay).
    pbn = "N:AKQ2.T98.5432.K4 J86.AK76.KJ.QJ98 T95.QJ32.AQ96.65 743.54.T87.AT732"
    dds = {"N": {"C": 6, "D": 9, "H": 8, "S": 8, "NT": 7},
           "E": {"C": 7, "D": 4, "H": 5, "S": 4, "NT": 5},
           "S": {"C": 6, "D": 9, "H": 8, "S": 8, "NT": 7},
           "W": {"C": 7, "D": 4, "H": 5, "S": 4, "NT": 5}}
    table, nodes = D.dd_table_pbn(pbn)
    assert table == dds
    assert nodes > 0


def test_node_limit_raises():
    masks = D.parse_pbn("N:AKQ2.T98.5432.K4 J86.AK76.KJ.QJ98 T95.QJ32.AQ96.65 743.54.T87.AT732")
    try:
        D.solve(masks, 4, 1, node_limit=10)
    except D.SolverError:
        return
    raise AssertionError("expected SolverError")


def test_agrees_with_the_project_endgame_solver():
    from engine.deal import Card, Suit, RANKS
    from cardplay.trick_engine import PlayState, Trick
    from cardplay.endgame_solver import solve_endgame

    def earlier_tricks(n):
        # the PlayState only ends after 13 tricks, so pretend 13 - 4 were played already
        return [Trick(leader="N", plays=[("N", Card(Suit.SPADES, "A")), ("E", Card(Suit.HEARTS, "2")),
                                         ("S", Card(Suit.HEARTS, "2")), ("W", Card(Suit.HEARTS, "2"))])
                for _ in range(n)]

    rng = random.Random(11)
    suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
    for _ in range(15):
        masks = _random_masks(rng, 4)
        hands = {seat: [Card(suits[s], RANKS[r]) for s in range(4) for r in range(13) if masks[p][s] >> r & 1]
                 for p, seat in enumerate("NESW")}
        strain = rng.choice(["S", "H", "D", "C", "NT"])
        trump = None if strain == "NT" else Suit(strain)
        ps = PlayState(hands=hands, declarer=rng.choice("NESW"), strain=strain, trump=trump, dealer="N")
        ps.completed_tricks = earlier_tricks(9)
        ps.opening_lead_made = True
        ours, _ = D.solve_play_state(ps)
        assert ours == solve_endgame(ps).ns_tricks_from_here
        card, after, _ = D.best_card(ps)
        assert after == ours
        assert card in ps.legal_cards_for(ps.next_to_play)
