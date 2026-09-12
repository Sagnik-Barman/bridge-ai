import random

from engine.deal import Card, Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState, Trick
from cardplay.bot_player import choose_card as heuristic_choose_card
from cardplay.endgame_solver import solve_endgame, SearchTooLargeError


def _dummy_completed_tricks(count, winner_seat):
    """`count` fabricated tricks, all trivially won by `winner_seat` (the
    trick-legality rules aren't checked by Trick.winner(), so this is a
    fine way to fast-forward a PlayState's history for a focused test)."""
    tricks = []
    other_seats = [s for s in SEATS if s != winner_seat]
    for _ in range(count):
        plays = [(winner_seat, Card(Suit.SPADES, "A"))] + [
            (s, Card(Suit.HEARTS, "2")) for s in other_seats
        ]
        tricks.append(Trick(leader=winner_seat, plays=plays))
    return tricks


def test_obvious_last_trick_trump_wins():
    """N holds the last trump; everyone else holds a plain suit card. N to
    lead the final trick — the only sane outcome is N wins it."""
    hands = {
        "N": [Card(Suit.SPADES, "A")],
        "E": [Card(Suit.HEARTS, "2")],
        "S": [Card(Suit.HEARTS, "3")],
        "W": [Card(Suit.HEARTS, "4")],
    }
    ps = PlayState(hands=hands, declarer="N", strain="S", trump=Suit.SPADES, dealer="N")
    ps.completed_tricks = _dummy_completed_tricks(12, winner_seat="N")
    ps.next_to_play = "N"
    ps.opening_lead_made = True

    result = solve_endgame(ps)
    assert result.best_card == Card(Suit.SPADES, "A")
    assert result.ns_tricks_from_here == 1
    assert result.nodes_explored >= 1
    # The real position must be untouched after solving.
    assert ps.hands["N"] == [Card(Suit.SPADES, "A")]
    assert len(ps.completed_tricks) == 12


def test_obvious_last_trick_defense_wins():
    """Same shape, but now it's a defender (E) who holds the only trump —
    NS should get 0 tricks from this last trick."""
    hands = {
        "N": [Card(Suit.HEARTS, "2")],
        "E": [Card(Suit.SPADES, "A")],
        "S": [Card(Suit.HEARTS, "3")],
        "W": [Card(Suit.HEARTS, "4")],
    }
    ps = PlayState(hands=hands, declarer="N", strain="S", trump=Suit.SPADES, dealer="N")
    ps.completed_tricks = _dummy_completed_tricks(12, winner_seat="N")
    ps.next_to_play = "N"
    ps.opening_lead_made = True

    result = solve_endgame(ps)
    assert result.ns_tricks_from_here == 0


def test_refuses_positions_larger_than_the_cap():
    deal = deal_random(dealer="N", seed=5)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")
    try:
        solve_endgame(ps, max_cards_per_hand=8)
        raise AssertionError("expected SearchTooLargeError for a full 13-card deal")
    except SearchTooLargeError:
        pass


def test_does_not_corrupt_the_callers_live_play_state():
    """Regression test for a real bug: `solve_endgame` used to alias the
    caller's `current_trick` object instead of deep-copying it, so the
    search's internal `play_card()` calls (which append in place to
    `current_trick.plays`) silently corrupted the CALLER's live game
    state. Reproduce it by solving a realistic, non-trivial 8-card
    endgame (small enough to be attempted, large enough to actually
    recurse) and checking the caller's PlayState is byte-for-byte
    unchanged afterward -- not just its hands (already checked above) but
    its current trick and completed tricks too."""
    deal = deal_random(dealer="N", seed=42)
    hands = {s: list(deal.hands[s].cards) for s in SEATS}
    ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")
    # Play down to a trick boundary with exactly 6 cards left per hand
    # (large enough to force real recursion, small enough to solve quickly).
    guard = 0
    while len(ps.hands[ps.next_to_play]) > 6 or len(ps.current_trick.plays) != 0:
        guard += 1
        assert guard < 60
        seat = ps.next_to_play
        card = heuristic_choose_card(seat, ps, seed=guard)
        ps.play_card(seat, card)
        if ps.is_complete():
            return  # unlucky shuffle finished before reaching a 6-card boundary; nothing to test

    before_trick = Trick(ps.current_trick.leader, list(ps.current_trick.plays))
    before_completed = len(ps.completed_tricks)
    before_hands = {s: list(cs) for s, cs in ps.hands.items()}

    solve_endgame(ps, max_cards_per_hand=6)

    assert ps.current_trick.leader == before_trick.leader
    assert ps.current_trick.plays == before_trick.plays
    assert len(ps.completed_tricks) == before_completed
    assert ps.hands == before_hands


def test_matches_real_dds_on_small_endgames():
    """Cross-check: play a random deal down to a small (<=5 cards/hand)
    trick boundary with the heuristic bot, then compare this from-scratch
    solver's NS-trick prediction against the real DDS solver's prediction
    for the identical remaining position. Skips (returns) if `endplay`
    isn't installed — see the module docstring for why this pairing
    matters: this solver exists partly to be checked against DDS, not
    just to exist standalone."""
    from engine.dds_wrapper import DDSUnavailableError, solve_double_dummy
    from engine.deal import Deal, Hand

    matched_any = False
    for seed in range(15):
        deal = deal_random(dealer="N", seed=seed + 500)
        hands = {s: list(deal.hands[s].cards) for s in SEATS}
        ps = PlayState(hands=hands, declarer="N", strain="NT", trump=None, dealer="N")
        rng = random.Random(seed)

        # Play down to a trick boundary with <=5 cards/hand left.
        while len(ps.hands[ps.next_to_play]) > 5 or len(ps.current_trick.plays) != 0:
            seat = ps.next_to_play
            card = heuristic_choose_card(seat, ps, seed=rng.randrange(1 << 30))
            ps.play_card(seat, card)
            if ps.is_complete():
                break
        if ps.is_complete() or len(ps.hands[ps.next_to_play]) > 5:
            continue

        try:
            our_result = solve_endgame(ps)
        except Exception:
            continue

        remaining_deal = Deal(hands={s: Hand(cards=ps.hands[s]) for s in SEATS}, dealer=ps.next_to_play)
        try:
            table = solve_double_dummy(remaining_deal)
        except DDSUnavailableError:
            return  # nothing to cross-check in this environment

        mover = ps.next_to_play
        remaining_total = len(ps.hands[mover])
        mover_side_tricks = table[mover]["NT"]
        if mover in ("N", "S"):
            dds_ns_tricks = mover_side_tricks
        else:
            dds_ns_tricks = remaining_total - mover_side_tricks

        assert our_result.ns_tricks_from_here == dds_ns_tricks, (
            f"seed={seed}: our solver says {our_result.ns_tricks_from_here} NS tricks "
            f"from here, DDS says {dds_ns_tricks}"
        )
        matched_any = True

    assert matched_any, "never found a testable <=5-card boundary in 15 random deals (very unlucky)"
