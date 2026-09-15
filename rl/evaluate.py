"""Evaluate a trained RL checkpoint honestly: compare it against the
existing heuristic bot, and (when `endplay` is installed) against the
double-dummy-optimal trick count, on the same random deals.

This is the "evidence" half of the resume story — a trained-but-modest
agent is only worth something if you can show, with real numbers, how it
stacks up against a plain rule-based baseline and against known-optimal
play. Nothing here is benchmarked against the uploaded book (there is
nothing in this codebase that ever reads it) — the comparison points are
the project's own heuristic bot and the DDS solver.

Run it directly:

    python -m rl.evaluate --checkpoint rl/checkpoints/latest.pt --num-deals 100
"""
from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np

from engine.deal import Suit, deal_random
from engine.dds_wrapper import DDSUnavailableError, best_contract_for_seat
from bidding.calls import SEATS, partner_of, seat_after
from cardplay.trick_engine import PlayState
from cardplay.bot_player import choose_card as heuristic_choose_card
from rl import features
from rl.network import PolicyValueNet

STRAINS = ["C", "D", "H", "S", "NT"]


def _play_full_info(play_state: PlayState, declarer_side_policy, defense_side_policy, seed: int) -> int:
    """Play out a board where the declaring side and the defending side
    each use their own card-chooser, with full information available to
    the RL network (this measures the network's raw skill in the same
    full-information setting it was trained in — see `rl/rl_player.py`
    for the imperfect-information wrapper used in actual gameplay)."""
    declaring_side = {play_state.declarer, play_state.dummy}
    rng = random.Random(seed)
    while not play_state.is_complete():
        seat = play_state.next_to_play
        chooser = declarer_side_policy if seat in declaring_side else defense_side_policy
        card = chooser(seat, play_state, rng)
        play_state.play_card(seat, card)
    return play_state.declarer_tricks()


def _rl_policy(net: PolicyValueNet):
    def choose(seat, play_state, rng):
        x = features.encode(seat, play_state)
        mask = features.legal_mask(seat, play_state)
        np_rng = np.random.default_rng(rng.randrange(1 << 30))
        action = net.act(x, mask, np_rng, greedy=True)
        return features.card_for_index(action)

    return choose


def _heuristic_policy():
    def choose(seat, play_state, rng):
        return heuristic_choose_card(seat, play_state, seed=rng.randrange(1 << 30))

    return choose


def _rl_policy_with_endgame_search(net: PolicyValueNet, endgame_limit: int = 6):
    """The fifth attempt (see docs/rl_training.md's "A fifth attempt"
    section): a hybrid that does NOT change anything the network learned.
    Instead, whenever few enough cards remain that the exact endgame
    solver can actually finish, use its answer instead of the network's;
    otherwise fall back to the network exactly as before. This is the
    same exact-search-near-the-end / learned-policy-otherwise shape
    already used elsewhere in this codebase (`cardplay/dds_player.py`,
    `game/session.py`'s Analyzer) — the first of the five attempts that
    changes *how much lookahead is used at decision time* rather than
    *how the network was trained*.

    This function is only meaningful here in `evaluate.py`, where play is
    full-information — `solve_endgame` requires knowing every hand, which
    is true here (matching how the network itself was trained) but NOT
    true of real gameplay, where a defender can't see declarer's and
    dummy's cards. Wiring this into real (imperfect-information) play
    would need the PIMC sampling `rl/rl_player.py` already does for the
    network, which is future work, not done here.
    """
    from cardplay.endgame_solver import solve_endgame, SearchTooLargeError

    rl_choose = _rl_policy(net)

    def choose(seat, play_state, rng):
        remaining = len(play_state.hands[seat])
        if remaining <= endgame_limit:
            try:
                result = solve_endgame(play_state, max_cards_per_hand=endgame_limit)
                return result.best_card
            except SearchTooLargeError:
                pass
        return rl_choose(seat, play_state, rng)

    return choose


def _heuristic_policy_with_endgame_search(endgame_limit: int = 6):
    """The same hybrid shape as `_rl_policy_with_endgame_search`, but with
    the *heuristic* bot as the fallback instead of the network. This
    exists purely for credit attribution: since the endgame solver is the
    same exact search either way, comparing this against the plain
    heuristic isolates how much of any RL+search improvement is really
    coming from the network's own play (mid-hand, before the search
    window opens) versus just from bolting exact search onto whatever
    policy is underneath. The honest question this answers: if the
    'RL + search' hybrid beats the plain heuristic, does the *heuristic*
    also improve by a similar amount once it gets the same search, or is
    the network actually adding something on top?"""
    from cardplay.endgame_solver import solve_endgame, SearchTooLargeError

    heuristic_choose = _heuristic_policy()

    def choose(seat, play_state, rng):
        remaining = len(play_state.hands[seat])
        if remaining <= endgame_limit:
            try:
                result = solve_endgame(play_state, max_cards_per_hand=endgame_limit)
                return result.best_card
            except SearchTooLargeError:
                pass
        return heuristic_choose(seat, play_state, rng)

    return choose


def evaluate_vs_heuristic(
    net: PolicyValueNet,
    num_deals: int = 50,
    seed: int = 0,
    endgame_search_limit: Optional[int] = None,
) -> dict:
    """`endgame_search_limit`: if set, also play a third variant per deal
    — the hybrid RL+exact-endgame-search policy (see
    `_rl_policy_with_endgame_search`) as declarer against the same
    heuristic defense — so its effect on the standard benchmark can be
    measured directly against the plain RL policy, not just asserted."""
    rl_policy = _rl_policy(net)
    heuristic_policy = _heuristic_policy()
    hybrid_policy = (
        _rl_policy_with_endgame_search(net, endgame_search_limit)
        if endgame_search_limit is not None
        else None
    )
    heuristic_hybrid_policy = (
        _heuristic_policy_with_endgame_search(endgame_search_limit)
        if endgame_search_limit is not None
        else None
    )
    py_rng = random.Random(seed)

    rl_declarer_tricks = []
    heuristic_declarer_tricks = []
    hybrid_declarer_tricks = []
    heuristic_hybrid_declarer_tricks = []
    par_tricks_list = []

    for i in range(num_deals):
        deal_seed = py_rng.randrange(1 << 30)
        declarer = py_rng.choice(SEATS)
        strain = py_rng.choice(STRAINS)
        deal = deal_random(dealer=declarer, seed=deal_seed)
        trump = None if strain == "NT" else Suit(strain)

        hands_a = {s: list(deal.hands[s].cards) for s in SEATS}
        ps_a = PlayState(hands=hands_a, declarer=declarer, strain=strain, trump=trump, dealer=declarer)
        rl_tricks = _play_full_info(ps_a, rl_policy, heuristic_policy, seed=deal_seed)
        rl_declarer_tricks.append(rl_tricks)

        hands_b = {s: list(deal.hands[s].cards) for s in SEATS}
        ps_b = PlayState(hands=hands_b, declarer=declarer, strain=strain, trump=trump, dealer=declarer)
        heuristic_tricks = _play_full_info(ps_b, heuristic_policy, heuristic_policy, seed=deal_seed)
        heuristic_declarer_tricks.append(heuristic_tricks)

        if hybrid_policy is not None:
            hands_c = {s: list(deal.hands[s].cards) for s in SEATS}
            ps_c = PlayState(hands=hands_c, declarer=declarer, strain=strain, trump=trump, dealer=declarer)
            hybrid_tricks = _play_full_info(ps_c, hybrid_policy, heuristic_policy, seed=deal_seed)
            hybrid_declarer_tricks.append(hybrid_tricks)

        if heuristic_hybrid_policy is not None:
            hands_d = {s: list(deal.hands[s].cards) for s in SEATS}
            ps_d = PlayState(hands=hands_d, declarer=declarer, strain=strain, trump=trump, dealer=declarer)
            heuristic_hybrid_tricks = _play_full_info(ps_d, heuristic_hybrid_policy, heuristic_policy, seed=deal_seed)
            heuristic_hybrid_declarer_tricks.append(heuristic_hybrid_tricks)

        try:
            par_strain, par = best_contract_for_seat(deal, declarer)
            par_tricks_list.append(par)
        except DDSUnavailableError:
            pass

    result = {
        "num_deals": num_deals,
        "rl_declarer_avg_tricks": float(np.mean(rl_declarer_tricks)),
        "heuristic_declarer_avg_tricks": float(np.mean(heuristic_declarer_tricks)),
    }
    if hybrid_declarer_tricks:
        result["hybrid_declarer_avg_tricks"] = float(np.mean(hybrid_declarer_tricks))
    if heuristic_hybrid_declarer_tricks:
        result["heuristic_hybrid_declarer_avg_tricks"] = float(np.mean(heuristic_hybrid_declarer_tricks))
    if par_tricks_list:
        result["dds_par_avg_tricks"] = float(np.mean(par_tricks_list))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-deals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--endgame-search-limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Also evaluate the fifth-attempt hybrid: use the exact endgame "
            "solver whenever <=N cards remain in the mover's hand, falling "
            "back to the RL policy otherwise (see docs/rl_training.md's "
            "'A fifth attempt' section). Also evaluates the same search "
            "bolted onto the plain heuristic bot instead, as a credit-"
            "attribution check -- if the heuristic gains just as much from "
            "the search as the RL policy does, the search is doing the "
            "work, not the network. Omit to skip both variants -- each "
            "adds a full extra playout per deal, so this roughly triples "
            "runtime. 6 matches the rest of this project's established "
            "safe limit."
        ),
    )
    args = parser.parse_args()

    net = PolicyValueNet.load(args.checkpoint)
    result = evaluate_vs_heuristic(
        net,
        num_deals=args.num_deals,
        seed=args.seed,
        endgame_search_limit=args.endgame_search_limit,
    )

    print(f"Evaluated on {result['num_deals']} random deals (each side declares & defends the same deals):")
    print(f"  RL agent as declarer:          {result['rl_declarer_avg_tricks']:.2f} tricks/board avg")
    print(f"  Heuristic bot as declarer:     {result['heuristic_declarer_avg_tricks']:.2f} tricks/board avg")
    if "hybrid_declarer_avg_tricks" in result:
        print(
            f"  RL + endgame search (<={args.endgame_search_limit} cards) as declarer: "
            f"{result['hybrid_declarer_avg_tricks']:.2f} tricks/board avg"
        )
    if "heuristic_hybrid_declarer_avg_tricks" in result:
        print(
            f"  Heuristic + endgame search (<={args.endgame_search_limit} cards) as declarer: "
            f"{result['heuristic_hybrid_declarer_avg_tricks']:.2f} tricks/board avg  "
            f"(credit check -- compare its gain over plain heuristic to the RL hybrid's gain over plain RL)"
        )
    if "dds_par_avg_tricks" in result:
        print(f"  Double-dummy optimal (par):    {result['dds_par_avg_tricks']:.2f} tricks/board avg")
    else:
        print("  (install `endplay` to also see the double-dummy-optimal par tricks for these deals)")


if __name__ == "__main__":
    main()
