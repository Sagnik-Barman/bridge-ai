"""Perfect Information Monte Carlo (PIMC) card play.

Approach:
  1. Given the known cards (declarer's hand + dummy + cards played so far)
     and any constraints implied by the auction (suit lengths, HCP ranges
     inferred from bidding — not yet wired up, see TODO), sample many
     plausible full deals for the two unseen hands.
  2. Solve each sampled deal double-dummy via `engine.dds_wrapper`.
  3. For each legal card the side to play could lead/play, aggregate the
     double-dummy results across samples (e.g. average tricks, or simple
     voting on the best card) and pick accordingly.

This is a scaffold: sampling is currently unconstrained random dealing of
the unseen cards (not yet filtered by auction-implied constraints), and
aggregation is a simple average. Both are natural next steps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from engine.deal import Card, Deal, Hand, RANKS, Suit, SEATS
from engine.dds_wrapper import DDSUnavailableError, solve_double_dummy


@dataclass
class PIMCResult:
    strain: str
    seat: str
    samples: int
    mean_tricks: float
    tricks_per_sample: List[int]


def _unseen_cards(known: Dict[str, Hand]) -> List[Card]:
    all_cards = {Card(suit, rank) for suit in Suit for rank in RANKS}
    seen = {c for hand in known.values() for c in hand.cards}
    return list(all_cards - seen)


def sample_consistent_deal(
    known: Dict[str, Hand],
    unseen_seats: List[str],
    rng: random.Random,
) -> Deal:
    """Randomly deal the unseen cards to `unseen_seats`.

    TODO: constrain sampling using auction-implied information (HCP ranges
    and suit-length signals from the bidding, once `bidding/engine.py`
    exposes them) instead of dealing completely at random. That is the
    difference between "PIMC" done properly and a naive baseline.
    """
    pool = _unseen_cards(known)
    rng.shuffle(pool)

    hand_size = len(pool) // len(unseen_seats)
    sampled: Dict[str, Hand] = dict(known)
    for i, seat in enumerate(unseen_seats):
        sampled[seat] = Hand(cards=pool[i * hand_size : (i + 1) * hand_size])
    return Deal(hands=sampled)


def evaluate_contract_pimc(
    known: Dict[str, Hand],
    unseen_seats: List[str],
    strain: str,
    declarer_seat: str,
    num_samples: int = 50,
    seed: int | None = None,
) -> PIMCResult:
    """Estimate double-dummy tricks for `declarer_seat` in `strain` by
    averaging over `num_samples` random completions of the unseen hands.
    """
    rng = random.Random(seed)
    tricks: List[int] = []

    for _ in range(num_samples):
        deal = sample_consistent_deal(known, unseen_seats, rng)
        try:
            table = solve_double_dummy(deal)
        except DDSUnavailableError:
            # Without DDS installed we can't actually solve; surface a
            # clear signal rather than silently returning zero.
            raise
        tricks.append(table[declarer_seat][strain])

    mean_tricks = sum(tricks) / len(tricks) if tricks else 0.0
    return PIMCResult(
        strain=strain,
        seat=declarer_seat,
        samples=num_samples,
        mean_tricks=mean_tricks,
        tricks_per_sample=tricks,
    )


if __name__ == "__main__":
    from engine.deal import deal_random

    d = deal_random(seed=3)
    known = {"N": d.hands["N"], "S": d.hands["S"]}  # declarer's side is "known"
    unseen_seats = ["E", "W"]

    try:
        result = evaluate_contract_pimc(known, unseen_seats, strain="NT", declarer_seat="N", num_samples=20, seed=3)
        print(result)
    except DDSUnavailableError as e:
        print(f"[skipped: {e}]")
