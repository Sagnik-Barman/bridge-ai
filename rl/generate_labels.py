"""Generate an imitation-learning dataset for the RL card-play agent.

This exists to support the "imitation warm start" idea (see
docs/rl_training.md's "A fourth attempt" section): before self-play RL
begins, pretrain the policy network on a pile of (position, correct card)
examples with ordinary supervised learning, so self-play refines an
already-reasonable policy instead of discovering bridge from nothing.

**Honest scope -- read before trusting this data.** This project does not
have a validated way to get the true double-dummy-optimal card at an
arbitrary mid-hand decision point. The two real primitives it has are:

  - `cardplay.endgame_solver.solve_endgame`: a genuine, exact alpha-beta
    search. Correct at ANY decision point (lead or mid-trick), but only
    tractable up to `--endgame-limit` cards left per hand (this script
    defaults to a higher limit than the live Analyzer feature uses, since
    generating labels offline can afford to wait longer per position than
    a live "Analyze" click can).
  - `cardplay.dds_player.choose_card`: DDS/PIMC-informed, but -- by its
    own module docstring -- only meaningfully different from the
    heuristic bot at a fresh trick LEAD; mid-trick it already falls back
    to the same heuristic rules as `cardplay.bot_player`. Calling it
    mid-trick would just relabel the heuristic bot's own choice, which
    teaches the network nothing it doesn't already get from self-play.

So this script labels exactly two situations -- near the end of a hand
(exact, via the endgame solver), and trick leads earlier in the hand
(DDS/PIMC-informed, via dds_player, when `endplay` is installed) -- and
leaves every other decision (mid-trick, many cards still out) unlabeled.
That's a real, useful, but partial teacher, not a full double-dummy
oracle. See docs/rl_training.md for the full reasoning.

Usage -- time a small batch before committing to a big one, since the
per-deal cost depends heavily on your machine and on `--endgame-limit`
(each additional card can multiply the cost several-fold, not add a
little -- see docs/rl_training.md's "A fourth attempt" section):

    python -m rl.generate_labels --num-deals 100 --out /tmp/probe.npz
    # read the printed "~Ns remaining" estimate, then scale --num-deals
    # to whatever total time you're willing to spend, e.g.:
    python -m rl.generate_labels --num-deals 3000 --out rl/checkpoints/imitation_labels.npz
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import List, Tuple

import numpy as np

from engine.deal import Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState
from cardplay.bot_player import choose_card as heuristic_choose_card
from cardplay.endgame_solver import solve_endgame, SearchTooLargeError
from rl import features

STRAINS = ["C", "D", "H", "S", "NT"]

# Matches game/session.py's ANALYZER_ENDGAME_CARD_LIMIT and
# benchmark/expert_benchmark.py's default -- this project's own measured
# numbers (docs/endgame_solver.md) found 6 reliably finishes in 1-2s,
# while 7-8 frequently burn through the full 200,000-node search budget
# WITHOUT finishing (a slow failure, not a fast one) before raising
# SearchTooLargeError. An earlier version of this script defaulted to 8,
# reasoning that offline generation could afford to wait longer per
# position -- that was wrong: with roughly half of every hand's ~40
# decisions falling at 8-or-fewer cards left, and many of those exhausting
# the full budget before giving up, a few thousand deals at limit 8 can
# take hours, not "a couple of seconds each." 6 is the number this
# project can actually promise; raise it only if you're prepared for that.
DEFAULT_ENDGAME_LIMIT = 6


def _random_contract(rng: random.Random):
    declarer = rng.choice(SEATS)
    strain = rng.choice(STRAINS)
    return declarer, strain, declarer


def _dds_lead_or_none(seat: str, play_state: PlayState, py_rng: random.Random):
    """DDS/PIMC-informed lead, or None if endplay isn't installed. Import
    is local so this whole module still works (endgame-only labels) in an
    environment without endplay -- exactly like the rest of the codebase's
    DDS-optional pattern."""
    from cardplay.dds_player import choose_card as dds_choose_card, dds_available

    if not dds_available():
        return None
    return dds_choose_card(seat, play_state, seed=py_rng.randrange(1 << 30))


def generate_examples(
    num_deals: int,
    seed: int = 0,
    endgame_limit: int = DEFAULT_ENDGAME_LIMIT,
    dds_leads: bool = True,
    progress_every: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int, str]], int]:
    """Play out `num_deals` random hands (heuristic bot driving every
    seat -- realistic-ish trajectories, not optimal ones) and label the
    decision points described in the module docstring.

    `progress_every`: print a one-line progress update every this many
    deals (0 = silent). This script can legitimately take a long time at
    a large `--num-deals` or a high `--endgame-limit` -- each endgame
    search that exhausts its node budget without finishing is a genuinely
    slow failure, not a fast one -- so a silent multi-minute run is easy
    to mistake for a hang. `main()` below defaults this on.

    Returns (examples, num_endgame_searches_skipped_as_too_large), where
    each example is (x, mask, action_index, source) and source is
    "endgame" or "dds".
    """
    import time

    py_rng = random.Random(seed)
    examples: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
    skipped = 0
    t0 = time.time()

    for deal_idx in range(num_deals):
        if progress_every and deal_idx > 0 and deal_idx % progress_every == 0:
            elapsed = time.time() - t0
            rate = deal_idx / elapsed
            eta = (num_deals - deal_idx) / rate if rate > 0 else float("inf")
            print(
                f"  ...{deal_idx}/{num_deals} deals, {len(examples)} labels so far "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                flush=True,
            )
        declarer, strain, dealer = _random_contract(py_rng)
        deal = deal_random(dealer=dealer, seed=py_rng.randrange(1 << 30))
        trump = None if strain == "NT" else Suit(strain)
        hands = {s: list(deal.hands[s].cards) for s in SEATS}
        play_state = PlayState(hands=hands, declarer=declarer, strain=strain, trump=trump, dealer=dealer)

        while not play_state.is_complete():
            seat = play_state.next_to_play
            remaining = len(play_state.hands[seat])
            is_lead = len(play_state.current_trick.plays) == 0

            label_card = None
            source = None
            if remaining <= endgame_limit:
                try:
                    result = solve_endgame(play_state, max_cards_per_hand=endgame_limit)
                    label_card, source = result.best_card, "endgame"
                except SearchTooLargeError:
                    skipped += 1
            elif is_lead and dds_leads:
                label_card = _dds_lead_or_none(seat, play_state, py_rng)
                source = "dds" if label_card is not None else None

            if label_card is not None:
                x = features.encode(seat, play_state)
                mask = features.legal_mask(seat, play_state)
                action = features.CARD_INDEX[label_card]
                examples.append((x, mask, action, source))

            # Drive the actual game forward with the heuristic bot,
            # regardless of whether this decision got labeled -- we only
            # need realistic trajectories to walk through, not optimal
            # ones (that's what self-play is for, afterward).
            play_card = heuristic_choose_card(seat, play_state, seed=py_rng.randrange(1 << 30))
            play_state.play_card(seat, play_card)

    return examples, skipped


def save_examples(examples, path: str) -> None:
    xs = np.stack([e[0] for e in examples])
    masks = np.stack([e[1] for e in examples])
    actions = np.array([e[2] for e in examples], dtype=np.int64)
    sources = np.array([e[3] for e in examples])
    np.savez(path, x=xs, mask=masks, action=actions, source=sources)


def main() -> None:
    from cardplay.dds_player import dds_available

    parser = argparse.ArgumentParser(description="Generate an imitation-learning dataset for the RL agent.")
    parser.add_argument("--num-deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endgame-limit", type=int, default=DEFAULT_ENDGAME_LIMIT)
    parser.add_argument("--out", type=str, default="rl/checkpoints/imitation_labels.npz")
    parser.add_argument(
        "--no-dds-leads",
        action="store_true",
        help="Skip DDS-informed lead labels even if endplay is installed (endgame-solver labels only).",
    )
    args = parser.parse_args()

    if not dds_available() and not args.no_dds_leads:
        print(
            "endplay is not installed -- lead decisions won't get DDS-informed labels this run "
            "(endgame-solver labels near the end of each hand still work fine). "
            "Run `pip install endplay` to unlock those too."
        )

    print(f"Generating labels from {args.num_deals} deals (endgame-limit={args.endgame_limit})...")
    examples, skipped = generate_examples(
        args.num_deals,
        seed=args.seed,
        endgame_limit=args.endgame_limit,
        dds_leads=not args.no_dds_leads,
        progress_every=max(1, args.num_deals // 20),
    )
    print(f"Collected {len(examples)} labeled decisions from {args.num_deals} deals.")
    print(f"By source: {dict(Counter(e[3] for e in examples))}")
    print(f"({skipped} endgame searches skipped as too large for the {args.endgame_limit}-card limit.)")

    if not examples:
        raise SystemExit("No examples collected -- nothing to save. Check that endgame-limit is >= 2.")

    save_examples(examples, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
