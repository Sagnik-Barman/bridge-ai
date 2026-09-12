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


def evaluate_vs_heuristic(net: PolicyValueNet, num_deals: int = 50, seed: int = 0) -> dict:
    rl_policy = _rl_policy(net)
    heuristic_policy = _heuristic_policy()
    py_rng = random.Random(seed)

    rl_declarer_tricks = []
    heuristic_declarer_tricks = []
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
    if par_tricks_list:
        result["dds_par_avg_tricks"] = float(np.mean(par_tricks_list))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-deals", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    net = PolicyValueNet.load(args.checkpoint)
    result = evaluate_vs_heuristic(net, num_deals=args.num_deals, seed=args.seed)

    print(f"Evaluated on {result['num_deals']} random deals (each side declares & defends the same deals):")
    print(f"  RL agent as declarer:          {result['rl_declarer_avg_tricks']:.2f} tricks/board avg")
    print(f"  Heuristic bot as declarer:     {result['heuristic_declarer_avg_tricks']:.2f} tricks/board avg")
    if "dds_par_avg_tricks" in result:
        print(f"  Double-dummy optimal (par):    {result['dds_par_avg_tricks']:.2f} tricks/board avg")
    else:
        print("  (install `endplay` to also see the double-dummy-optimal par tricks for these deals)")


if __name__ == "__main__":
    main()
