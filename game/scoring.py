"""Standard duplicate bridge scoring and the IMP (International Match
Point) conversion scale.

Both tables here are the universal, public-domain scoring rules defined
by the WBF/ACBL and printed in every bridge rulebook and on the back of
most convention cards — not sourced from any book, and not specific to
any particular publisher.

Because this is solo practice against computer opponents (one table,
not a real teams match with a second table to compare against), there is
no second result to convert into a true IMP swing. The practical
approximation used here: after each board, compare your actual result to
the double-dummy "best your side could have done" result for that deal
(when a DDS binding is installed — see engine/dds_wrapper.py), and
express *that* difference in IMPs. This gives a genuinely IMP-flavored
sense of how costly a bidding or play mistake was, which is the spirit of
what team-of-four scoring rewards (missing games/slams, going down
unnecessarily) even without a live second table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TRICK_VALUE = {"C": 20, "D": 20, "H": 30, "S": 30, "NT": 30}  # NT first trick handled separately
NT_FIRST_TRICK_BONUS = 10  # NT first trick is 40 total (30 base + 10)

PART_SCORE_BONUS = 50
GAME_BONUS = {"vul": 500, "non_vul": 300}
SMALL_SLAM_BONUS = {"vul": 750, "non_vul": 500}
GRAND_SLAM_BONUS = {"vul": 1500, "non_vul": 1000}

DOUBLED_INSULT_BONUS = 50
REDOUBLED_INSULT_BONUS = 100

# (undertricks, vulnerable) -> penalty points, cumulative table.
UNDOUBLED_UNDERTRICK = {False: 50, True: 100}  # per undertrick


def _undoubled_undertrick_penalty(n: int, vul: bool) -> int:
    return n * UNDOUBLED_UNDERTRICK[vul]


def _doubled_undertrick_penalty(n: int, vul: bool) -> int:
    if not vul:
        # 100, 200, 200, 300, 300, ... (first 100, next two 200, rest 300)
        total = 0
        for i in range(1, n + 1):
            if i == 1:
                total += 100
            elif i <= 3:
                total += 200
            else:
                total += 300
        return total
    else:
        # 200, then 300 each subsequent
        total = 0
        for i in range(1, n + 1):
            total += 200 if i == 1 else 300
        return total


def _redoubled_undertrick_penalty(n: int, vul: bool) -> int:
    return 2 * _doubled_undertrick_penalty(n, vul)


@dataclass
class ScoreResult:
    declaring_side: str  # "NS" or "EW"
    ns_score: int  # signed score credited to NS (negative = EW gained)
    ew_score: int
    summary: str


def score_contract(
    level: int,
    strain: str,
    double_state: str,  # "", "X", "XX"
    tricks_taken: int,
    declarer_seat: str,
    vulnerable_ns: bool,
    vulnerable_ew: bool,
) -> ScoreResult:
    """Score one completed board's result from the declaring side's
    perspective, per standard duplicate scoring rules.
    """
    declaring_side = "NS" if declarer_seat in ("N", "S") else "EW"
    vul = vulnerable_ns if declaring_side == "NS" else vulnerable_ew

    needed = level + 6
    made = tricks_taken >= needed

    if not made:
        undertricks = needed - tricks_taken
        if double_state == "":
            penalty = _undoubled_undertrick_penalty(undertricks, vul)
        elif double_state == "X":
            penalty = _doubled_undertrick_penalty(undertricks, vul)
        else:
            penalty = _redoubled_undertrick_penalty(undertricks, vul)
        declaring_signed = -penalty
        summary = f"Down {undertricks}{' doubled' if double_state == 'X' else ' redoubled' if double_state == 'XX' else ''}: {declaring_side} {declaring_signed}"
    else:
        overtricks = tricks_taken - needed
        base_per_trick = TRICK_VALUE[strain]
        trick_score = level * base_per_trick + (NT_FIRST_TRICK_BONUS if strain == "NT" else 0)

        if double_state == "X":
            trick_score *= 2
        elif double_state == "XX":
            trick_score *= 4

        # Game/part-score bonus and slam bonus are additive — a slam is
        # automatically game too, and gets a bonus on top, not instead.
        bonus = 0
        if trick_score >= 100:
            bonus += GAME_BONUS["vul" if vul else "non_vul"]
        else:
            bonus += PART_SCORE_BONUS
        if level == 6:
            bonus += SMALL_SLAM_BONUS["vul" if vul else "non_vul"]
        elif level == 7:
            bonus += GRAND_SLAM_BONUS["vul" if vul else "non_vul"]

        if double_state == "X":
            bonus += DOUBLED_INSULT_BONUS
        elif double_state == "XX":
            bonus += REDOUBLED_INSULT_BONUS

        overtrick_value = 0
        if overtricks:
            if double_state == "":
                overtrick_value = overtricks * base_per_trick
            elif double_state == "X":
                overtrick_value = overtricks * (200 if vul else 100)
            else:
                overtrick_value = overtricks * (400 if vul else 200)

        declaring_signed = trick_score + bonus + overtrick_value
        summary = f"Made {level}{strain}{'' if not overtricks else f'+{overtricks}'}: {declaring_side} +{declaring_signed}"

    if declaring_side == "NS":
        return ScoreResult(declaring_side, declaring_signed, -declaring_signed, summary)
    return ScoreResult(declaring_side, -declaring_signed, declaring_signed, summary)


# ---------------------------------------------------------------------------
# IMP scale
# ---------------------------------------------------------------------------

_IMP_TABLE = [
    (10, 0), (40, 1), (80, 2), (120, 3), (160, 4), (210, 5), (260, 6),
    (310, 7), (360, 8), (420, 9), (490, 10), (590, 11), (740, 12),
    (890, 13), (1090, 14), (1290, 15), (1490, 16), (1740, 17), (1990, 18),
    (2240, 19), (2490, 20), (2990, 21), (3490, 22), (3990, 23),
]


def imps_for_diff(point_diff: int) -> int:
    """Standard WBF/ACBL IMP scale. `point_diff` is an absolute point
    difference between two results on the same board; returns the
    (unsigned) IMP value. Caller re-applies the sign.
    """
    diff = abs(point_diff)
    for threshold, imps in _IMP_TABLE:
        if diff <= threshold:
            return imps
    return 24


def imp_swing(your_ns_score: int, comparison_ns_score: int) -> int:
    """Signed IMPs gained (positive) or lost (negative) by NS, comparing
    your actual board result to a comparison result (e.g. the
    double-dummy par score) for the same board, both expressed as NS's
    signed point score."""
    diff = your_ns_score - comparison_ns_score
    imps = imps_for_diff(diff)
    return imps if diff >= 0 else -imps
