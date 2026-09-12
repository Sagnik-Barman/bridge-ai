"""Interactive opening-bid tutor.

    python -m bidding.tutor

Deals you a random hand as dealer (so you're always the one deciding
whether/what to open — no prior calls to react to yet, since responses and
rebids aren't implemented in the convention file). You type your call; the
tutor immediately tells you whether it matches the Standard American 2/1
engine's recommendation and explains the rule behind that recommendation,
so you learn the underlying convention as you go — not just whether you
were "right".

Scope: opening bids only, matching the current state of
`bidding/conventions/standard_american_2over1.yaml`. Responses, rebids, and
competitive bidding aren't modeled yet, so this only ever asks "what would
you open?", never "what do you respond?".

Commands at any prompt:
  <a bid>   e.g. "1S", "1nt", "2c", "pass"  — make your call
  rules     print the full convention reference (all opening-bid rules)
  hand      re-print your current hand
  quit      exit and show your session score
"""
from __future__ import annotations

import re
import sys

from engine.deal import Hand, deal_random
from bidding.engine import BiddingEngine, BidExplanation

VALID_BID_RE = re.compile(r"^([1-7])(C|D|H|S|NT)$")


def normalize_bid(raw: str) -> str | None:
    """Turn loose user input ("1nt", "PASS", "2c") into the canonical form
    used by the convention file ("1NT", "Pass", "2C"), or None if it isn't
    recognizable as a bid at all.
    """
    text = raw.strip()
    if not text:
        return None
    if text.lower() in ("p", "pass"):
        return "Pass"
    m = VALID_BID_RE.match(text.upper().replace(" ", ""))
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None


def format_hand(hand: Hand) -> str:
    by_suit = hand.by_suit()
    lines = []
    from engine.deal import Suit

    for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
        cards = by_suit[suit]
        ranks = " ".join(c.rank for c in cards) if cards else "—"
        lines.append(f"  {suit.value}: {ranks}")
    return "\n".join(lines)


def explain(expl: BidExplanation, your_bid: str) -> None:
    match = "✓ Matches the engine's recommendation!" if your_bid == expl.bid else (
        f"✗ Engine recommends {expl.bid} instead."
    )
    print(f"\n{match}")
    print(f"  Recommended bid: {expl.bid}  ({expl.rule_name})")
    print(f"  Rule requires:   {expl.conditions_text}")
    print(f"  Your hand:       {expl.hand_summary}")


def run(seed: int | None = None) -> None:
    engine = BiddingEngine()
    correct = 0
    total = 0

    print("=" * 60)
    print("Opening Bid Tutor — Standard American 2/1")
    print("=" * 60)
    print("You are always the dealer, deciding whether/what to open.")
    print("Type a bid (e.g. 1S, 1NT, 2C, Pass), 'rules' to see the")
    print("convention reference, 'hand' to see your hand again, or")
    print("'quit' to stop.\n")

    rng_seed = seed
    while True:
        deal = deal_random(seed=rng_seed)
        rng_seed = None  # only the very first deal (if any) is seeded; rest are random
        hand = deal.hands[deal.dealer]

        print("-" * 60)
        print(f"New hand (you are dealer, seat {deal.dealer}):")
        print(format_hand(hand))

        while True:
            raw = input("\nYour call> ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                _print_summary(correct, total)
                return
            if raw.lower() == "rules":
                print()
                print(engine.rules_reference())
                continue
            if raw.lower() == "hand":
                print(format_hand(hand))
                continue

            bid = normalize_bid(raw)
            if bid is None:
                print("Didn't understand that. Try e.g. '1S', '1NT', '2C', 'Pass', "
                      "or 'rules' / 'hand' / 'quit'.")
                continue

            expl = engine.opening_bid_with_explanation(hand)
            explain(expl, bid)
            total += 1
            if bid == expl.bid:
                correct += 1
            break  # move on to the next dealt hand


def _print_summary(correct: int, total: int) -> None:
    print("\n" + "=" * 60)
    if total == 0:
        print("No hands bid this session.")
    else:
        pct = 100 * correct / total
        print(f"Session score: {correct}/{total} correct ({pct:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    seed_arg = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--seed="):
        seed_arg = int(sys.argv[1].split("=", 1)[1])
    try:
        run(seed=seed_arg)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
