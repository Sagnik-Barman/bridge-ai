"""DDS/PIMC-based card play: a genuinely strong opponent, in contrast to
`cardplay/bot_player.py`'s dependency-free heuristics.

How a lead is chosen (the only decision this module handles differently
from the heuristic bot — see "Honest scope" below):

  1. Figure out what this seat can legitimately see: its own hand, plus
     dummy's hand once it's tabled (or from the start, if this seat is
     declarer or dummy, since declarer always sees both hands).
  2. Pick a short list of candidate lead cards — one representative per
     suit held (the sequence-top honor if there is one, else the lowest
     card), the same candidates a strong human would actually be choosing
     between.
  3. For each candidate, run several PIMC samples: deal the unseen cards
     to the unseen seats at random (this is *exact*, not approximate,
     because hand sizes are guaranteed equal at a trick boundary), then
     play out just that one trick using the existing well-tested
     heuristics for the other three seats, landing back on a fresh (equal
     -hands) boundary. From there, `engine.dds_wrapper.solve_double_dummy`
     gives the real double-dummy trick count for whoever's now on lead —
     exactly the primitive it's built for.
  4. Average the resulting trick count for our side across samples, per
     candidate, and lead whichever candidate scores best.

Honest scope: this is a legitimate, exact use of the double-dummy table
(`CalcDDTable`) applied only at trick boundaries, where its preconditions
(equal hand sizes, nobody partway through the current trick) genuinely
hold. It is *not* a full DDS engine — it doesn't use the more elaborate
`SolveBoard`-style "best card mid-trick" primitive some DDS bindings
expose, both because that API differs across bindings and versions and
because it can't be exercised or validated in this environment (no native
DDS binary is installed here, only the pure-Python bridge logic can be).
Mid-trick (2nd/3rd/4th to play), this module falls back to the same
club-level rules as the heuristic bot (win as cheaply as possible / duck /
third-hand-high), which are usually double-dummy-correct in those seats
anyway. If `endplay` isn't installed at all, everything falls back to the
heuristic bot, with `dds_status()` explaining why.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from engine.deal import Card, Deal, Hand, RANKS, SEATS, Suit
from engine.dds_wrapper import DDSUnavailableError, solve_double_dummy
from bidding.calls import seat_after
from cardplay.trick_engine import PlayState, Trick, legal_plays
from cardplay.bot_player import choose_follow, choose_opening_or_new_trick_lead, _has_sequence_top, _rank_value


def dds_available() -> bool:
    try:
        import endplay  # noqa: F401

        return True
    except ImportError:
        return False


def dds_status() -> str:
    if dds_available():
        return "endplay is installed — DDS/PIMC leads are active."
    return (
        "endplay is not installed, so 'Play against DDS' falls back to the "
        "same heuristics as the default bot for now. Run `pip install "
        "endplay` to unlock real double-dummy leads."
    )


def _known_hands_for(seat: str, play_state: PlayState) -> Dict[str, List[Card]]:
    """Cards this seat can legitimately see: its own hand, plus dummy's
    hand once tabled (or always, if this seat is declarer or dummy)."""
    known: Dict[str, List[Card]] = {seat: list(play_state.hands[seat])}
    dummy = play_state.dummy
    declaring_side = {play_state.declarer, dummy}
    if seat in declaring_side:
        known[play_state.declarer] = list(play_state.hands[play_state.declarer])
        known[dummy] = list(play_state.hands[dummy])
    elif play_state.opening_lead_made:
        known[dummy] = list(play_state.hands[dummy])
    return known


def _sample_unseen_hands(
    known: Dict[str, List[Card]], unseen_sizes: Dict[str, int], rng: random.Random
) -> Dict[str, List[Card]]:
    """Deal the cards nobody's revealed yet to the unseen seats, respecting
    each seat's known remaining hand size (equal at a trick boundary,
    which is the only time this is called — so this is an exact, not
    merely plausible, completion of the deal)."""
    seen = {c for cards in known.values() for c in cards}
    pool = [Card(s, r) for s in Suit for r in RANKS if Card(s, r) not in seen]
    rng.shuffle(pool)

    sampled: Dict[str, List[Card]] = {s: list(cards) for s, cards in known.items()}
    idx = 0
    for seat, size in unseen_sizes.items():
        sampled[seat] = pool[idx : idx + size]
        idx += size
    return sampled


def _lead_candidates(hand: List[Card]) -> List[Card]:
    """One representative lead per suit held: the top of a touching honor
    sequence if there is one, else the lowest card of that suit — the
    same short list a strong player would actually be weighing."""
    suits = sorted({c.suit for c in hand}, key=lambda s: s.value)
    candidates = []
    for suit in suits:
        seq_top = _has_sequence_top(hand, suit)
        if seq_top:
            candidates.append(seq_top)
            continue
        suit_cards = sorted((c for c in hand if c.suit == suit), key=_rank_value)
        candidates.append(suit_cards[0])
    return candidates


def _play_out_one_trick(
    leader: str, lead_card: Card, hands: Dict[str, List[Card]], play_state: PlayState, rng: random.Random
) -> PlayState:
    """Simulate exactly one trick — `leader` leads `lead_card`, the other
    three seats follow using the heuristic bot — landing on a fresh
    (equal-hands) trick boundary. Reuses the real, tested `PlayState`
    trick-completion logic rather than reimplementing it."""
    hands_copy = {s: list(cs) for s, cs in hands.items()}
    sim = PlayState(hands=hands_copy, declarer=play_state.declarer, strain=play_state.strain, trump=play_state.trump, dealer=play_state.dealer)
    sim.hands[leader].remove(lead_card)
    sim.current_trick = Trick(leader=leader, plays=[(leader, lead_card)])
    sim.completed_tricks = []
    sim.next_to_play = seat_after(leader, 1)
    sim.opening_lead_made = True
    for _ in range(3):
        cur = sim.next_to_play
        card = choose_follow(sim.hands[cur], sim, rng)
        sim.play_card(cur, card)
    return sim


def _score_lead_candidate(
    seat: str,
    card: Card,
    play_state: PlayState,
    known: Dict[str, List[Card]],
    unseen_sizes: Dict[str, int],
    rng: random.Random,
    num_samples: int,
) -> float:
    declaring_side = {play_state.declarer, play_state.dummy}
    our_side = declaring_side if seat in declaring_side else (set(SEATS) - declaring_side)

    total = 0.0
    valid_samples = 0
    for _ in range(num_samples):
        sampled = _sample_unseen_hands(known, unseen_sizes, rng)
        sim = _play_out_one_trick(seat, card, sampled, play_state, rng)
        winner = sim.completed_tricks[-1].winner(sim.trump)
        won_this_trick = 1 if winner in our_side else 0

        remaining_deal = Deal(hands={s: Hand(cards=sim.hands[s]) for s in SEATS}, dealer=winner)
        try:
            table = solve_double_dummy(remaining_deal)
        except DDSUnavailableError:
            return float("-inf")  # signal "can't score this way" to the caller

        remaining_tricks_total = len(sim.hands[winner])
        winner_future = table[winner][play_state.strain]
        if winner in our_side:
            our_future = winner_future
        else:
            our_future = remaining_tricks_total - winner_future

        total += won_this_trick + our_future
        valid_samples += 1

    return total / valid_samples if valid_samples else float("-inf")


def _best_lead_by_pimc(seat: str, play_state: PlayState, rng: random.Random, num_samples: int) -> Optional[Card]:
    candidates = _lead_candidates(play_state.hands[seat])
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    known = _known_hands_for(seat, play_state)
    unseen_seats = [s for s in SEATS if s not in known]
    if not unseen_seats:
        return None
    unseen_sizes = {s: len(play_state.hands[s]) for s in unseen_seats}

    scores = []
    for card in candidates:
        score = _score_lead_candidate(seat, card, play_state, known, unseen_sizes, rng, num_samples)
        scores.append((score, card))

    if all(s == float("-inf") for s, _ in scores):
        return None  # DDS unavailable partway through — let the caller fall back

    return max(scores, key=lambda sc: sc[0])[1]


def choose_card(seat: str, play_state: PlayState, seed: Optional[int] = None, num_samples: int = 5) -> Card:
    """`num_samples` trades strength for speed: each unit is a real
    double-dummy solve per lead candidate (see the module docstring). 5 is
    a reasonable default for interactive play (real `endplay` calls are
    fast individually, but this adds up across 13 leads/board and up to 4
    candidates/lead); raise it if you want stronger, slower leads, e.g.
    when calling this directly for analysis rather than live play."""
    rng = random.Random(seed)
    hand = play_state.hands[seat]

    if len(play_state.current_trick.plays) == 0 and dds_available():
        try:
            led = _best_lead_by_pimc(seat, play_state, rng, num_samples)
            if led is not None:
                return led
        except Exception:
            pass  # any DDS hiccup: fall through to the heuristic lead below

    if len(play_state.current_trick.plays) == 0:
        return choose_opening_or_new_trick_lead(hand, rng)

    return choose_follow(hand, play_state, rng)
