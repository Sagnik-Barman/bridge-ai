"""Rules-based bidding state machine.

Reads a convention definition (e.g. `conventions/standard_american_2over1.yaml`)
as structured data and evaluates it against a Hand to determine bids. This
module contains no convention-specific knowledge itself — all bidding logic
lives in the YAML data files under `conventions/`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from engine.deal import Hand, Suit

CONVENTIONS_DIR = os.path.join(os.path.dirname(__file__), "conventions")


def load_convention(name: str = "standard_american_2over1") -> Dict[str, Any]:
    path = os.path.join(CONVENTIONS_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _suit_lengths_by_letter(hand: Hand) -> Dict[str, int]:
    return {s.value: n for s, n in hand.suit_lengths().items()}


def describe_conditions(conditions: Dict[str, Any]) -> str:
    """Render a rule's conditions as a short human-readable phrase, for
    both the tutor's per-bid explanations and a printable convention
    reference (`BiddingEngine.rules_reference()`).
    """
    parts: List[str] = []

    hcp_min = conditions.get("hcp_min")
    hcp_max = conditions.get("hcp_max")
    if hcp_min is not None and hcp_max is not None:
        parts.append(f"{hcp_min}-{hcp_max} HCP")
    elif hcp_min is not None:
        parts.append(f"{hcp_min}+ HCP")
    elif hcp_max is not None:
        parts.append(f"up to {hcp_max} HCP")

    if conditions.get("balanced"):
        parts.append("balanced shape")

    suit = conditions.get("suit")
    if suit is not None:
        min_len = conditions.get("suit_min_length", 1)
        phrase = f"{min_len}+ cards in {suit}"
        if conditions.get("longest_or_tied_longest"):
            phrase += " (longest or tied-longest suit)"
        parts.append(phrase)
    elif "suit_min_length" in conditions:
        parts.append(f"a suit of {conditions['suit_min_length']}+ cards")

    return ", ".join(parts) if parts else "(no conditions)"


def _matches(hand: Hand, conditions: Dict[str, Any]) -> bool:
    hcp = hand.hcp()
    lengths = _suit_lengths_by_letter(hand)

    if "hcp_min" in conditions and hcp < conditions["hcp_min"]:
        return False
    if "hcp_max" in conditions and hcp > conditions["hcp_max"]:
        return False
    if conditions.get("balanced") and not hand.is_balanced():
        return False

    suit = conditions.get("suit")
    if suit is not None:
        min_len = conditions.get("suit_min_length", 1)
        if lengths.get(suit, 0) < min_len:
            return False
        if conditions.get("longest_or_tied_longest"):
            if lengths[suit] < max(lengths.values()):
                return False

    if "suit_min_length" in conditions and suit is None:
        # e.g. preempt rule: any suit at least this long
        if max(lengths.values()) < conditions["suit_min_length"]:
            return False

    return True


@dataclass
class BidExplanation:
    """Why the engine recommends (or ruled out) a particular opening bid."""

    bid: str
    rule_name: str
    conditions_text: str
    hand_summary: str  # e.g. "16 HCP, shape S5 H3 D3 C2, balanced"


@dataclass
class Auction:
    dealer: str
    calls: List[str] = field(default_factory=list)  # e.g. ["1S", "Pass", "2H", ...]

    def next_bidder(self) -> str:
        order = ["N", "E", "S", "W"]
        start = order.index(self.dealer)
        return order[(start + len(self.calls)) % 4]

    def add(self, call: str) -> None:
        self.calls.append(call)

    def is_complete(self) -> bool:
        if len(self.calls) < 4:
            return False
        return all(c == "Pass" for c in self.calls[-3:])


class BiddingEngine:
    """Minimal opening-bid decision maker.

    Scaffold only: currently decides an opening bid from HCP/shape rules.
    Extending to full auctions requires the `responses` / `rebids` /
    `competitive` sections of the convention YAML to be filled in, and
    this class extended to track partnership state (opening bid made,
    responses so far) rather than just looking at one hand in isolation.
    """

    def __init__(self, convention: Optional[Dict[str, Any]] = None):
        self.convention = convention or load_convention()

    def _resolve_bid(self, rule: Dict[str, Any], hand: Hand) -> str:
        """Most rules' `bid` field is already the literal call. A rule can
        instead set `dynamic` to name a bid that depends on the hand
        (currently only the preempt rule, which bids 3-of-whatever-suit
        rather than a fixed strain) — resolved here rather than in the
        YAML, which can't express "3 of my longest suit".
        """
        dynamic = rule.get("dynamic")
        if dynamic == "preempt_longest_suit_at_3_level":
            lengths = _suit_lengths_by_letter(hand)
            longest = max(lengths, key=lengths.get)
            return f"3{longest}"
        return rule["bid"]

    def opening_bid(self, hand: Hand) -> str:
        for rule in self.convention["opening_bids"]:
            if _matches(hand, rule["conditions"]):
                return self._resolve_bid(rule, hand)
        return "Pass"  # fallback if no rule matches (shouldn't normally happen)

    def opening_bid_with_explanation(self, hand: Hand) -> BidExplanation:
        """Same decision as `opening_bid`, plus the rule name/conditions
        that justified it — what the tutor shows the learner.
        """
        lengths = _suit_lengths_by_letter(hand)
        shape = " ".join(f"{suit}{lengths[suit]}" for suit in ["S", "H", "D", "C"])
        hand_summary = f"{hand.hcp()} HCP, shape {shape}" + (
            ", balanced" if hand.is_balanced() else ", unbalanced"
        )

        for rule in self.convention["opening_bids"]:
            if _matches(hand, rule["conditions"]):
                return BidExplanation(
                    bid=self._resolve_bid(rule, hand),
                    rule_name=rule["name"],
                    conditions_text=describe_conditions(rule["conditions"]),
                    hand_summary=hand_summary,
                )
        return BidExplanation(
            bid="Pass",
            rule_name="No opening bid rule matched",
            conditions_text="(fell through every rule in the convention file)",
            hand_summary=hand_summary,
        )

    def rules_reference(self) -> str:
        """A printable, human-readable summary of every opening-bid rule
        in the loaded convention — the tutor's "learn the conventions" view.
        """
        lines = [f"{self.convention.get('system_name', 'Bidding system')} — opening bids:", ""]
        for rule in self.convention["opening_bids"]:
            lines.append(f"  {rule['bid']:<6} {rule['name']}")
            lines.append(f"         requires: {describe_conditions(rule['conditions'])}")
        return "\n".join(lines)

    def run_auction_stub(self, hands: Dict[str, Hand], dealer: str = "N") -> Auction:
        """Extremely simplified auction: everyone either opens (first bidder
        with a rule match) or passes. Not a real competitive auction —
        placeholder until `responses`/`rebids`/`competitive` are implemented.
        """
        auction = Auction(dealer=dealer)
        order = ["N", "E", "S", "W"]
        start = order.index(dealer)
        rotated = order[start:] + order[:start]

        opener = None
        for seat in rotated:
            bid = self.opening_bid(hands[seat])
            if bid != "Pass":
                auction.add(bid)
                opener = seat
                break
            auction.add("Pass")

        if opener is None:
            # Everyone passed out.
            auction.add("Pass")
            return auction

        # Placeholder: everyone else passes after the opening bid.
        while not auction.is_complete():
            auction.add("Pass")
        return auction


if __name__ == "__main__":
    from engine.deal import deal_random

    d = deal_random(seed=7)
    engine = BiddingEngine()
    auction = engine.run_auction_stub(d.hands, dealer=d.dealer)
    print(d)
    print("Auction:", auction.calls)
