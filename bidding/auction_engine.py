"""Full-auction bidding engine: responses, opener rebids, competitive
bidding (overcalls, takeout doubles, negative doubles), and the standard
named conventions (Stayman, Jacoby transfers, Blackwood).

Everything here is drafted from first principles / general public
bidding-system knowledge (the level of detail found in an ACBL convention
card or a WBF system description) — no bridge book content was used.
See `docs/bidding_coverage.md` for an honest accounting of what is and
isn't modeled: this is a solid core, not an exhaustive tournament system.
Any auction shape this engine doesn't recognize falls through to a
clearly-labeled generic fallback rather than pretending to have an answer.

Design: `BiddingEngine` (bidding/engine.py) already decides opening bids
from the YAML convention file — that's reused unchanged here. Everything
*after* the opening bid (responses, rebids, competitive auctions,
conventions) is expressed as plain Python functions rather than more YAML,
because those decisions are inherently conditional on the auction history
in ways that are awkward to express as flat condition lists. Each function
is small and independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.deal import Hand
from bidding.engine import BiddingEngine, describe_conditions
from bidding.calls import (
    AuctionState,
    SEATS,
    is_suit_bid,
    partner_of,
    seat_after,
)

MAJORS = ["H", "S"]
MINORS = ["C", "D"]


def _lengths(hand: Hand) -> Dict[str, int]:
    return {s.value: n for s, n in hand.suit_lengths().items()}


def _shape_str(hand: Hand) -> str:
    L = _lengths(hand)
    return " ".join(f"{s}{L[s]}" for s in ["S", "H", "D", "C"])


def _summary(hand: Hand) -> str:
    return f"{hand.hcp()} HCP, shape {_shape_str(hand)}" + (
        ", balanced" if hand.is_balanced() else ", unbalanced"
    )


@dataclass
class AuctionDecision:
    call: str
    explanation: str
    tag: str = ""  # short machine-readable label, mainly for tests


# ---------------------------------------------------------------------------
# Responses to partner's opening bid (uncontested)
# ---------------------------------------------------------------------------

def respond_to_suit_opening(opener_suit: str, hand: Hand) -> AuctionDecision:
    """Responses to a 1-level suit opening (1C/1D/1H/1S), no interference.

    2/1 Game Force core structure:
      - 0-5 HCP: Pass.
      - New major at the 1-level (natural, non-forcing minimum but forcing
        one round) takes priority when available and unbid.
      - 6-9 HCP with 3+ card support: simple raise.
      - 10-12 HCP with 4+ card support: jump raise (invitational).
      - 13+ HCP with 4+ card support: raise straight to game (slam
        exploration beyond this is out of scope for v1).
      - 6-9 HCP, no fit, no biddable new major: 1NT.
      - 13+ HCP, no fit: new suit at the 2-level (game forcing) if one
        exists, else 2NT.
    """
    hcp = hand.hcp()
    L = _lengths(hand)
    is_major_open = opener_suit in MAJORS
    support = L[opener_suit]

    if hcp < 6:
        return AuctionDecision("Pass", f"{_summary(hand)} — too weak to respond (need 6+ HCP).", "resp_weak_pass")

    # New major at the 1-level, natural priority (only relevant when
    # opener bid a minor, or opened 1H and responder holds 4+ spades).
    if opener_suit != "S" and L["S"] >= 4 and not (opener_suit == "H" and support >= 4 and hcp >= 13):
        return AuctionDecision(
            "1S", f"{_summary(hand)} — 4+ spades, bid the new major up the line before supporting.", "resp_new_major_1S"
        )
    if opener_suit not in ("H", "S") and L["H"] >= 4:
        return AuctionDecision(
            "1H", f"{_summary(hand)} — 4+ hearts, bid the new major up the line before supporting.", "resp_new_major_1H"
        )

    if support >= 4 and hcp >= 13:
        game_level = 4 if is_major_open else 5
        return AuctionDecision(
            f"{game_level}{opener_suit}",
            f"{_summary(hand)} — {support}-card support and 13+ HCP: raise straight to game.",
            "resp_game_raise",
        )
    if support >= 4 and 10 <= hcp <= 12:
        return AuctionDecision(
            f"3{opener_suit}",
            f"{_summary(hand)} — {support}-card support and 10-12 HCP: jump (limit) raise, invitational to game.",
            "resp_limit_raise",
        )
    if support >= 3 and 6 <= hcp <= 9:
        return AuctionDecision(
            f"2{opener_suit}",
            f"{_summary(hand)} — {support}-card support and 6-9 HCP: simple raise.",
            "resp_simple_raise",
        )

    if hcp >= 13:
        # No fit: bid a new 4+ card suit at the 2-level (game forcing in
        # 2/1), preferring the longer/higher-ranking one; else 2NT.
        candidate_suits = [s for s in ["S", "H", "D", "C"] if s != opener_suit and L[s] >= 4]
        if candidate_suits:
            best = max(candidate_suits, key=lambda s: L[s])
            return AuctionDecision(
                f"2{best}",
                f"{_summary(hand)} — 13+ HCP, no fit: new suit at the 2-level, game forcing (2/1).",
                "resp_2over1",
            )
        return AuctionDecision(
            "2NT", f"{_summary(hand)} — 13+ HCP, no fit, no 4-card suit to show: natural 2NT.", "resp_2nt_gf"
        )

    return AuctionDecision(
        "1NT", f"{_summary(hand)} — 6-9 HCP, no fit, no 4-card major to show: 1NT.", "resp_1nt"
    )


def respond_to_1nt_opening(hand: Hand) -> AuctionDecision:
    """Responses to a 15-17 balanced 1NT opening: Stayman, Jacoby
    transfers, and natural raises — the standard modern structure.
    """
    hcp = hand.hcp()
    L = _lengths(hand)

    if L["H"] >= 5:
        return AuctionDecision(
            "2D", f"{_summary(hand)} — 5+ hearts: Jacoby transfer (asks opener to bid 2H).", "resp_1nt_transfer_H"
        )
    if L["S"] >= 5:
        return AuctionDecision(
            "2H", f"{_summary(hand)} — 5+ spades: Jacoby transfer (asks opener to bid 2S).", "resp_1nt_transfer_S"
        )
    if hcp >= 8 and (L["H"] >= 4 or L["S"] >= 4):
        return AuctionDecision(
            "2C", f"{_summary(hand)} — 8+ HCP with a 4-card major: Stayman, asking opener for a 4-card major.", "resp_1nt_stayman"
        )
    if hcp >= 10:
        return AuctionDecision("3NT", f"{_summary(hand)} — 10-15 HCP, balanced, no major: raise to game.", "resp_1nt_3nt")
    if hcp >= 8:
        return AuctionDecision("2NT", f"{_summary(hand)} — 8-9 HCP, balanced, no major: invitational raise.", "resp_1nt_2nt")
    return AuctionDecision("Pass", f"{_summary(hand)} — 0-7 HCP, no long major: pass.", "resp_1nt_pass")


def respond_to_2c_opening(hand: Hand) -> AuctionDecision:
    """Responses to the strong artificial 2C opening (22+ HCP or
    equivalent playing strength). 2D is the standard "waiting"/negative
    response; anything else here shows a genuine positive with a good
    suit and values — kept intentionally simple.
    """
    hcp = hand.hcp()
    L = _lengths(hand)
    if hcp >= 8:
        best_suit = max(["S", "H", "D", "C"], key=lambda s: L[s])
        if L[best_suit] >= 5:
            return AuctionDecision(
                f"2{best_suit}" if best_suit != "C" else "3C",
                f"{_summary(hand)} — 8+ HCP with a good 5+ card suit: positive response, showing that suit.",
                "resp_2c_positive",
            )
    return AuctionDecision(
        "2D", f"{_summary(hand)} — waiting/negative response (0-7 HCP or no clear suit to show).", "resp_2c_waiting"
    )


def respond_to_2nt_opening(hand: Hand) -> AuctionDecision:
    """Responses to a 20-21 balanced 2NT opening — same shape as 1NT
    responses, one level higher (3C Stayman, 3D/3H transfers)."""
    hcp = hand.hcp()
    L = _lengths(hand)
    if L["H"] >= 5:
        return AuctionDecision("3D", f"{_summary(hand)} — 5+ hearts: transfer to hearts.", "resp_2nt_transfer_H")
    if L["S"] >= 5:
        return AuctionDecision("3H", f"{_summary(hand)} — 5+ spades: transfer to spades.", "resp_2nt_transfer_S")
    if hcp >= 4 and (L["H"] >= 4 or L["S"] >= 4):
        return AuctionDecision("3C", f"{_summary(hand)} — a 4-card major: Stayman.", "resp_2nt_stayman")
    return AuctionDecision("3NT", f"{_summary(hand)} — no long/4-card major: raise to game.", "resp_2nt_3nt")


def respond_to_preempt(opener_bid: str, hand: Hand) -> AuctionDecision:
    """Very simplified response to a preemptive opening: raise with a fit
    and extra playing strength, otherwise pass."""
    strain = opener_bid[-1] if not opener_bid.endswith("NT") else "NT"
    hcp = hand.hcp()
    L = _lengths(hand)
    if strain in L and L[strain] >= 3 and hcp >= 10:
        level = int(opener_bid[0]) + 1
        return AuctionDecision(
            f"{level}{strain}",
            f"{_summary(hand)} — fit with partner's preempt and extra values: raise.",
            "resp_preempt_raise",
        )
    return AuctionDecision("Pass", f"{_summary(hand)} — no clear reason to disturb the preempt.", "resp_preempt_pass")


def respond_to_opening(opener_bid: str, hand: Hand) -> AuctionDecision:
    if opener_bid == "1NT":
        return respond_to_1nt_opening(hand)
    if opener_bid == "2NT":
        return respond_to_2nt_opening(hand)
    if opener_bid == "2C":
        return respond_to_2c_opening(hand)
    if is_suit_bid(opener_bid) and int(opener_bid[0]) == 1:
        return respond_to_suit_opening(opener_bid[1], hand)
    return respond_to_preempt(opener_bid, hand)


# ---------------------------------------------------------------------------
# Opener's rebid
# ---------------------------------------------------------------------------

def opener_rebid(opening_bid: str, response: str, hand: Hand) -> AuctionDecision:
    hcp = hand.hcp()
    L = _lengths(hand)

    if response == "Pass":
        return AuctionDecision("Pass", "Partner passed my opening — nothing more to say yet.", "rebid_after_pass")

    if opening_bid == "1NT" and response == "2C":  # Stayman
        if L["S"] >= 4:
            return AuctionDecision("2S", f"{_summary(hand)} — Stayman: showing 4+ spades.", "rebid_stayman_2S")
        if L["H"] >= 4:
            return AuctionDecision("2H", f"{_summary(hand)} — Stayman: showing 4+ hearts.", "rebid_stayman_2H")
        return AuctionDecision("2D", f"{_summary(hand)} — Stayman: no 4-card major to show.", "rebid_stayman_2D")

    if opening_bid == "1NT" and response == "2D":  # transfer to hearts
        return AuctionDecision("2H", "Completing partner's Jacoby transfer to hearts.", "rebid_transfer_complete_H")
    if opening_bid == "1NT" and response == "2H":  # transfer to spades
        return AuctionDecision("2S", "Completing partner's Jacoby transfer to spades.", "rebid_transfer_complete_S")

    if opening_bid == "2NT" and response == "3C":  # Stayman
        if L["S"] >= 4:
            return AuctionDecision("3S", f"{_summary(hand)} — Stayman: showing 4+ spades.", "rebid_stayman_3S")
        if L["H"] >= 4:
            return AuctionDecision("3H", f"{_summary(hand)} — Stayman: showing 4+ hearts.", "rebid_stayman_3H")
        return AuctionDecision("3D", f"{_summary(hand)} — Stayman: no 4-card major.", "rebid_stayman_3D")
    if opening_bid == "2NT" and response == "3D":
        return AuctionDecision("3H", "Completing partner's transfer to hearts.", "rebid_transfer_complete_2nt_H")
    if opening_bid == "2NT" and response == "3H":
        return AuctionDecision("3S", "Completing partner's transfer to spades.", "rebid_transfer_complete_2nt_S")

    if opening_bid == "2C" and response == "2D":  # waiting response to strong 2C
        best_suit = max(["S", "H", "D", "C"], key=lambda s: L[s])
        bid = f"2{best_suit}" if best_suit != "C" else "2NT"
        return AuctionDecision(
            bid, f"{_summary(hand)} — describing shape after partner's waiting 2D response to my strong 2C.", "rebid_2c_shape"
        )

    if is_suit_bid(response) and response[1:] == opening_bid[1:] and int(response[0]) > 1:
        # Partner raised our suit.
        support_shown = L[opening_bid[1]]
        if 19 <= hcp <= 21 or (16 <= hcp <= 21 and support_shown >= 5):
            level = int(response[0]) + 1
            return AuctionDecision(
                f"{level}{opening_bid[1]}",
                f"{_summary(hand)} — maximum for my opening: accept the invitation / bid on toward game.",
                "rebid_accept_raise",
            )
        return AuctionDecision(
            "Pass", f"{_summary(hand)} — minimum for my opening: nothing extra to show.", "rebid_pass_minimum"
        )

    if response == "1NT" and int(opening_bid[0]) == 1:
        if hcp >= 18:
            return AuctionDecision("2NT", f"{_summary(hand)} — 18-19 extra balanced, invite game.", "rebid_1nt_invite")
        if L[opening_bid[1]] >= 6:
            return AuctionDecision(
                f"2{opening_bid[1]}", f"{_summary(hand)} — 6+ card suit, minimum: rebid own suit.", "rebid_1nt_own_suit"
            )
        return AuctionDecision("Pass", f"{_summary(hand)} — minimum, balanced-ish: pass 1NT.", "rebid_1nt_pass")

    if is_suit_bid(response) and int(response[0]) == 2 and response[1:] != opening_bid[1:]:
        # 2/1 game-forcing new suit response: describe shape, forced to keep bidding.
        resp_suit = response[1:]
        if L[opening_bid[1]] >= 6:
            return AuctionDecision(
                f"2{opening_bid[1]}" if bid_is_above(opening_bid, response) else f"3{opening_bid[1]}",
                f"{_summary(hand)} — game force: rebid my 6+ card suit.",
                "rebid_2over1_own_suit",
            )
        if L[resp_suit] >= 3:
            level = int(response[0])
            return AuctionDecision(
                f"{level}{resp_suit}" if _rank_gt(f"{level}{resp_suit}", response) else f"{level+1}{resp_suit}",
                f"{_summary(hand)} — game force: support partner's suit.",
                "rebid_2over1_support",
            )
        return AuctionDecision("2NT", f"{_summary(hand)} — game force, no fit or long suit to show: 2NT.", "rebid_2over1_2nt")

    return AuctionDecision(
        "Pass", "Auction shape beyond this engine's rebid coverage — passing by default.", "rebid_fallback"
    )


def _rank_gt(a: str, b: str) -> bool:
    from bidding.calls import bid_rank
    return bid_rank(a) > bid_rank(b)


def bid_is_above(a: str, b: str) -> bool:
    from bidding.calls import bid_rank
    return bid_rank(a) > bid_rank(b)


# ---------------------------------------------------------------------------
# Competitive bidding: overcalls, takeout doubles, negative doubles
# ---------------------------------------------------------------------------

def overcall_or_double(opponent_bid: str, hand: Hand) -> AuctionDecision:
    """My first call, after (only) an opponent has opened the bidding."""
    hcp = hand.hcp()
    L = _lengths(hand)
    opp_suit = opponent_bid[1:] if opponent_bid != "1NT" else None

    if opp_suit:
        shortness = L[opp_suit] <= 2
        other_suits = [s for s in ["S", "H", "D", "C"] if s != opp_suit]
        support_all_others = all(L[s] >= 3 for s in other_suits)
        if 12 <= hcp <= 17 and shortness and support_all_others:
            return AuctionDecision(
                "Dbl",
                f"{_summary(hand)} — opening values, short in opponent's suit, support for the other suits: takeout double.",
                "comp_takeout_double",
            )

    candidate_suits = [s for s in ["S", "H", "D", "C"] if s != opp_suit and L[s] >= 5]
    if candidate_suits and 8 <= hcp <= 16:
        best = max(candidate_suits, key=lambda s: L[s])
        level = 1 if is_suit_bid(f"1{best}") and bid_is_above(f"1{best}", opponent_bid) else 2
        return AuctionDecision(
            f"{level}{best}",
            f"{_summary(hand)} — 5+ card suit and opening-range values: simple overcall.",
            "comp_overcall",
        )

    if hand.is_balanced() and 15 <= hcp <= 18 and opp_suit and L[opp_suit] >= 1:
        return AuctionDecision(
            "1NT", f"{_summary(hand)} — balanced 15-18 with a stopper in their suit: 1NT overcall.", "comp_1nt_overcall"
        )

    return AuctionDecision("Pass", f"{_summary(hand)} — not enough to act over their opening.", "comp_pass")


def response_to_overcall(partner_overcall: str, hand: Hand) -> AuctionDecision:
    hcp = hand.hcp()
    L = _lengths(hand)
    suit = partner_overcall[1:] if partner_overcall != "1NT" else None

    if suit and L[suit] >= 3 and hcp >= 6:
        level = int(partner_overcall[0]) + (1 if hcp >= 10 else 0)
        return AuctionDecision(
            f"{level}{suit}", f"{_summary(hand)} — support for partner's overcall: raise.", "comp_raise_overcall"
        )
    candidate = [s for s in ["S", "H", "D", "C"] if s != suit and L[s] >= 5 and hcp >= 8]
    if candidate:
        best = max(candidate, key=lambda s: L[s])
        return AuctionDecision(f"2{best}", f"{_summary(hand)} — own 5+ card suit: bid it.", "comp_new_suit_over_overcall")
    return AuctionDecision("Pass", f"{_summary(hand)} — nothing extra to add.", "comp_pass_overcall")


def response_to_takeout_double(opener_bid: str, hand: Hand) -> AuctionDecision:
    """Partner doubled the opponent's opening for takeout; I must
    normally find a bid even with a weak hand."""
    hcp = hand.hcp()
    L = _lengths(hand)
    opp_suit = opener_bid[1:] if opener_bid != "1NT" else None
    candidates = [s for s in ["S", "H", "D", "C"] if s != opp_suit]
    best = max(candidates, key=lambda s: L[s])
    base_level = 1 if is_suit_bid(f"1{best}") and bid_is_above(f"1{best}", opener_bid) else 2
    level = base_level + (1 if hcp >= 10 else 0)
    return AuctionDecision(
        f"{level}{best}",
        f"{_summary(hand)} — responding to partner's takeout double with my best suit.",
        "comp_double_response",
    )


def negative_double_or_response(opener_bid: str, overcall_bid: str, hand: Hand, still_legal_natural: Optional[str]) -> AuctionDecision:
    """Partner opened, RHO overcalled, now it's my turn. Either bid
    naturally if a comfortable natural call is still available, or
    consider a negative double showing the unbid major(s)."""
    hcp = hand.hcp()
    L = _lengths(hand)
    overcall_suit = overcall_bid[1:] if overcall_bid != "1NT" else None
    opener_suit = opener_bid[1:]
    unbid_majors = [s for s in MAJORS if s not in (opener_suit, overcall_suit)]

    if hcp >= 6 and unbid_majors and all(L[s] >= 4 for s in unbid_majors):
        return AuctionDecision(
            "Dbl",
            f"{_summary(hand)} — negative double: 4+ cards in the unbid major(s), values, no natural bid I prefer.",
            "comp_negative_double",
        )
    if still_legal_natural:
        return respond_to_opening(opener_bid, hand)  # approximate: bid as if uncontested
    return AuctionDecision("Pass", f"{_summary(hand)} — not enough for a negative double, nothing natural to bid.", "comp_pass_after_overcall")


# ---------------------------------------------------------------------------
# Blackwood
# ---------------------------------------------------------------------------

def blackwood_ask() -> AuctionDecision:
    return AuctionDecision("4NT", "Blackwood: asking partner how many aces they hold.", "blackwood_ask")


def blackwood_response(hand: Hand) -> AuctionDecision:
    aces = _count_aces(hand)
    table = {0: "5C", 4: "5C", 1: "5D", 2: "5H", 3: "5S"}
    call = table[aces]
    shown = "0 or 4" if aces in (0, 4) else str(aces)
    return AuctionDecision(call, f"Blackwood response: showing {shown} aces.", "blackwood_response")


def _count_aces(hand: Hand) -> int:
    return sum(1 for c in hand.cards if c.rank == "A")


def should_try_blackwood(hand: Hand, agreed_suit: str, partner_shown_extra: bool) -> bool:
    """Very conservative heuristic for the bot *initiating* a slam try:
    only with a big hand opposite a partner who has already shown extra
    values and an established fit. Deliberately cautious — see
    docs/bidding_coverage.md for why bot-initiated slam bidding is kept
    minimal in v1.
    """
    L = _lengths(hand)
    return hand.hcp() >= 18 and L.get(agreed_suit, 0) >= 4 and partner_shown_extra


# ---------------------------------------------------------------------------
# Top-level entry point: decide a call given the full auction so far
# ---------------------------------------------------------------------------

class AuctionEngine:
    def __init__(self, bidding_engine: Optional[BiddingEngine] = None):
        self.bidding_engine = bidding_engine or BiddingEngine()

    def decide_call(self, seat: str, hand: Hand, state: AuctionState) -> AuctionDecision:
        try:
            decision = self._decide(seat, hand, state)
        except Exception as exc:  # pragma: no cover - safety net
            decision = AuctionDecision("Pass", f"(internal error computing a call: {exc} — passing safely)", "error_fallback")

        legal = state.legal_calls()
        if decision.call not in legal:
            decision = AuctionDecision(
                "Pass",
                f"(computed {decision.call}, which isn't legal here — passing instead) {decision.explanation}",
                "illegal_fallback",
            )
        return decision

    def _decide(self, seat: str, hand: Hand, state: AuctionState) -> AuctionDecision:
        calls = state.calls
        dealer = state.dealer
        n = len(calls)

        def calls_of(who: str) -> List[str]:
            return [calls[i] for i in range(n) if seat_after(dealer, i) == who]

        partner = partner_of(seat)
        lho = seat_after(seat, 1)
        rho = seat_after(seat, -1)

        my_calls = calls_of(seat)
        partner_calls = calls_of(partner)
        lho_calls = calls_of(lho)
        rho_calls = calls_of(rho)
        opp_calls = lho_calls + rho_calls

        any_bid_made = any(is_suit_bid(c) or c in ("Dbl", "Rdbl") for c in calls)
        if not any_bid_made:
            expl = self.bidding_engine.opening_bid_with_explanation(hand)
            text = f"{expl.rule_name} — requires {expl.conditions_text}. Your hand: {expl.hand_summary}."
            return AuctionDecision(expl.bid, text, "opening")

        opener_index = next(i for i, c in enumerate(calls) if is_suit_bid(c) or c in ("Dbl", "Rdbl"))
        opener_seat = seat_after(dealer, opener_index)
        opener_bid = calls[opener_index] if is_suit_bid(calls[opener_index]) else None

        opponent_intervened = any(is_suit_bid(c) or c == "Dbl" for c in opp_calls)

        # Case: I opened, this is my rebid.
        if opener_seat == seat and my_calls:
            last_partner_call = partner_calls[-1] if partner_calls else "Pass"
            if len(my_calls) == 1:
                return opener_rebid(my_calls[0], last_partner_call, hand)
            return AuctionDecision("Pass", "Later rounds of my own auction aren't modeled yet — passing.", "rebid_fallback_deep")

        # Case: partner opened, this is my response.
        if opener_seat == partner and not my_calls:
            if not opponent_intervened:
                return respond_to_opening(opener_bid, hand)
            last_opp_suit_bid = next((c for c in reversed(opp_calls) if is_suit_bid(c)), None)
            if last_opp_suit_bid:
                still_legal = respond_to_opening(opener_bid, hand)
                still_legal_call = still_legal.call if still_legal.call in state.legal_calls() else None
                return negative_double_or_response(opener_bid, last_opp_suit_bid, hand, still_legal_call)
            # Opponent only doubled (takeout double of partner's opening):
            # simplified to bidding as if uncontested.
            return respond_to_opening(opener_bid, hand)

        # Case: partner opened, I've already responded — later rounds:
        # handle Stayman/transfer/Blackwood continuations specially.
        if opener_seat == partner and my_calls:
            last_partner_call = partner_calls[-1] if partner_calls else None
            if len(my_calls) == 1 and my_calls[0] == "2C" and opener_bid == "1NT" and last_partner_call in ("2D", "2H", "2S"):
                return _continue_after_stayman(hand, last_partner_call)
            if len(my_calls) == 1 and my_calls[0] in ("2D", "2H") and opener_bid == "1NT":
                transfer_suit = "H" if my_calls[0] == "2D" else "S"
                return _continue_after_transfer(hand, transfer_suit)
            if my_calls and my_calls[-1] == "4NT" and last_partner_call in ("5C", "5D", "5H", "5S"):
                return _continue_after_blackwood(hand, last_partner_call)
            return AuctionDecision("Pass", "Later rounds of responder's auction aren't modeled yet — passing.", "resp_fallback_deep")

        # Case: an opponent opened — overcall/double/balance decision.
        if opener_seat in (lho, rho):
            if not my_calls and not any(is_suit_bid(c) or c == "Dbl" for c in partner_calls):
                last_opp_suit_bid = next((c for c in reversed(opp_calls) if is_suit_bid(c)), opener_bid)
                return overcall_or_double(last_opp_suit_bid, hand)
            if partner_calls and partner_calls[-1] == "Dbl" and not my_calls:
                return response_to_takeout_double(opener_bid, hand)
            if partner_calls and is_suit_bid(partner_calls[-1]) and partner_calls[-1] != opener_bid and not my_calls:
                return response_to_overcall(partner_calls[-1], hand)
            return AuctionDecision("Pass", "Competitive auction beyond this engine's coverage — passing.", "comp_fallback_deep")

        return AuctionDecision("Pass", "Unrecognized auction shape — passing by default.", "unclassified_fallback")


def _continue_after_stayman(hand: Hand, opener_rebid_call: str) -> AuctionDecision:
    hcp = hand.hcp()
    L = _lengths(hand)
    found_major = opener_rebid_call in ("2H", "2S")
    if found_major:
        major = opener_rebid_call[1]
        if hcp >= 10:
            return AuctionDecision(f"4{major}", f"{_summary(hand)} — found the major, enough for game.", "stayman_cont_game")
        return AuctionDecision("Pass", f"{_summary(hand)} — found the major but only invitational values.", "stayman_cont_pass")
    if hcp >= 10:
        return AuctionDecision("3NT", f"{_summary(hand)} — no major fit found: settle in 3NT.", "stayman_cont_3nt")
    return AuctionDecision("Pass", f"{_summary(hand)} — no major fit, only invitational values: pass 2D.", "stayman_cont_pass_2d")


def _continue_after_transfer(hand: Hand, major: str) -> AuctionDecision:
    hcp = hand.hcp()
    L = _lengths(hand)
    if hcp >= 10:
        return AuctionDecision(f"4{major}", f"{_summary(hand)} — enough for game after completing the transfer.", "transfer_cont_game")
    if hcp >= 8 and L[major] >= 6:
        return AuctionDecision(f"3{major}", f"{_summary(hand)} — invitational with a 6-card suit.", "transfer_cont_invite")
    return AuctionDecision("Pass", f"{_summary(hand)} — minimum, happy to play the transfer suit at the 2-level.", "transfer_cont_pass")


def _continue_after_blackwood(hand: Hand, response: str) -> AuctionDecision:
    aces_shown = {"5C": "0 or 4", "5D": "1", "5H": "2", "5S": "3"}[response]
    # Extremely simplified: this engine doesn't track which suit the
    # partnership agreed on trump before Blackwood was used (that would
    # require richer auction-history modeling), so "bid slam" defaults to
    # 6NT rather than the correct 6-of-the-agreed-suit, and "sign off"
    # defaults to passing at partner's ace-showing response. Documented
    # as a known simplification in docs/bidding_coverage.md.
    if response in ("5H", "5S"):
        return AuctionDecision("6NT", f"Partner showed {aces_shown} aces — enough for a small slam.", "blackwood_slam")
    return AuctionDecision("Pass", f"Partner showed {aces_shown} aces — not enough for slam, sign off.", "blackwood_signoff")
