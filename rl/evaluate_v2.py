"""Benchmark the v2 agent against the heuristic bot, the v1 RL agent and DDS.

Uses exactly the same 300 deals as rl/evaluate.py (`--seed 1` there and here
draws the same deal, declarer and strain sequence), so the numbers line up with
the five earlier attempts in docs/rl_training.md.

For each declarer policy, against the heuristic bot's defence:
  tricks      average tricks taken as declarer
  vs DD       average (tricks - double-dummy tricks for this exact contract):
              what declarer gained or lost relative to perfect play by both sides
  opt%        share of declarer-side decisions (more than one legal card) where
              the card played was double-dummy optimal
  thrown      average tricks thrown away per board by declarer-side decisions
              (sum over decisions of V* - Q*(card played), oracle-scored)

Policies (who can see what is stated, because it matters):
  heuristic        cardplay/bot_player.py            sees own hand + trick (honest)
  random           uniformly random legal card       (floor for opt%)
  v1 RL            rl/checkpoints/latest.pt (v1)     sees all four hands
  v2 full          v2 network, obs=full              sees all four hands
  v2 visible       v2 network, obs=visible           only what that player sees (honest)
  DDS (PIMC)       cardplay/dds_player.py, "Hard"    honest; slow, opt-in via --with-dds

Defence benchmark (`--defence`): the heuristic bot declares, the policy under
test defends; lower declarer tricks = better defence.

    python -m rl.evaluate_v2 --v2 rl/checkpoints/v2_visible.pt --v2-full rl/checkpoints/v2_full.pt \
        --v1 rl/checkpoints/v1_reinforce.pt --num-deals 300 --seed 1 --defence
"""
from __future__ import annotations

import argparse
import json
import random
import time
from typing import Callable, Dict, Optional

import numpy as np

from engine.deal import Suit, deal_random
from bidding.calls import SEATS
from cardplay.trick_engine import PlayState, legal_plays
from cardplay.bot_player import choose_card as heuristic_choose_card
from rl.oracle import card_values

STRAINS = ["C", "D", "H", "S", "NT"]


def benchmark_deals(num_deals: int, seed: int):
    """Same RNG sequence as rl/evaluate.py's evaluate_vs_heuristic."""
    py_rng = random.Random(seed)
    out = []
    for _ in range(num_deals):
        deal_seed = py_rng.randrange(1 << 30)
        declarer = py_rng.choice(SEATS)
        strain = py_rng.choice(STRAINS)
        out.append((deal_seed, declarer, strain))
    return out


def _fresh(deal_seed, declarer, strain) -> PlayState:
    deal = deal_random(dealer=declarer, seed=deal_seed)
    trump = None if strain == "NT" else Suit(strain)
    return PlayState(hands={s: list(deal.hands[s].cards) for s in SEATS}, declarer=declarer,
                     strain=strain, trump=trump, dealer=declarer)


def play_board(ps: PlayState, decl_policy: Callable, def_policy: Callable, seed: int,
               score_side: Optional[str] = "decl") -> dict:
    """Play out a board; oracle-score every decision of `score_side`."""
    rng = random.Random(seed)
    decl_side = {ps.declarer, ps.dummy}
    n_dec = n_opt = 0
    thrown = 0
    while not ps.is_complete():
        seat = ps.next_to_play
        ours = seat in decl_side
        pol = decl_policy if ours else def_policy
        card = pol(seat, ps, rng)
        scored = (score_side == "decl" and ours) or (score_side == "def" and not ours)
        if scored and len(legal_plays(ps.hands[seat], ps.current_trick.led_suit())) > 1:
            vals = card_values(ps)
            best = max(vals.values())
            n_dec += 1
            n_opt += int(vals[card] == best)
            thrown += best - vals[card]
        ps.play_card(seat, card)
    return {"tricks": ps.declarer_tricks(), "decisions": n_dec, "optimal": n_opt, "thrown": thrown}


def contract_dd_tricks(ps: PlayState) -> int:
    """Double-dummy tricks for declarer in this exact contract, with the real opening leader."""
    vals = card_values(ps)  # opening leader to play: values are for the defence
    return 13 - max(vals.values())


# ------------------------------------------------------------------ policies
def heuristic_policy():
    return lambda seat, ps, rng: heuristic_choose_card(seat, ps, seed=rng.randrange(1 << 30))


def random_policy():
    return lambda seat, ps, rng: rng.choice(legal_plays(ps.hands[seat], ps.current_trick.led_suit()))


def v1_policy(path: str):
    from rl.network import PolicyValueNet
    from rl import features
    net = PolicyValueNet.load(path, device="cpu")

    def choose(seat, ps, rng):
        x = features.encode(seat, ps)
        m = features.legal_mask(seat, ps)
        return features.card_for_index(net.act(x, m, None, greedy=True))
    return choose


def v2_policy(path: str, head: str = "policy"):
    from rl.network_v2 import CardPlayNet
    from rl.agent_v2 import choose_card
    net = CardPlayNet.load(path)
    return lambda seat, ps, rng: choose_card(seat, ps, net, head=head)


def dds_policy(samples: int = 5):
    from cardplay.dds_player import choose_card
    return lambda seat, ps, rng: choose_card(seat, ps, seed=rng.randrange(1 << 30), num_samples=samples)


# ------------------------------------------------------------------ runner
def run(policies: Dict[str, Callable], num_deals: int, seed: int, defence: bool = False,
        log_every: int = 50) -> dict:
    deals = benchmark_deals(num_deals, seed)
    heur = heuristic_policy()
    dd = []
    for d in deals:
        dd.append(contract_dd_tricks(_fresh(*d)))
    res = {"num_deals": num_deals, "seed": seed, "dd_contract_avg": float(np.mean(dd)), "declare": {}, "defend": {}}
    modes = [("declare", "decl")] + ([("defend", "def")] if defence else [])
    for mode, side in modes:
        for name, pol in policies.items():
            t0 = time.time()
            tr, diff, dec, opt, thrown = [], [], 0, 0, 0
            for i, d in enumerate(deals):
                ps = _fresh(*d)
                if mode == "declare":
                    r = play_board(ps, pol, heur, seed=d[0], score_side="decl")
                else:
                    r = play_board(ps, heur, pol, seed=d[0], score_side="def")
                tr.append(r["tricks"]); diff.append(r["tricks"] - dd[i])
                dec += r["decisions"]; opt += r["optimal"]; thrown += r["thrown"]
                if log_every and (i + 1) % log_every == 0:
                    print(f"    [{mode} {name}] {i + 1}/{num_deals}  avg tricks {np.mean(tr):.2f}", flush=True)
            se = float(np.std(tr, ddof=1) / np.sqrt(len(tr)))
            res[mode][name] = {"tricks": float(np.mean(tr)), "tricks_se": se, "vs_dd": float(np.mean(diff)),
                               "opt_rate": opt / max(dec, 1), "thrown_per_board": thrown / num_deals,
                               "seconds": time.time() - t0, "per_deal": tr}
            print(f"  {mode:7s} {name:22s} tricks {np.mean(tr):5.2f} (±{se:.2f})  vs DD {np.mean(diff):+5.2f}  "
                  f"opt {opt / max(dec, 1):6.1%}  thrown/board {thrown / num_deals:4.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return res


def paired_diff(res: dict, mode: str, a: str, b: str) -> tuple:
    """Mean per-deal difference a - b and its standard error (same deals, paired)."""
    x = np.array(res[mode][a]["per_deal"]) - np.array(res[mode][b]["per_deal"])
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v2", help="v2 checkpoint, obs=visible (honest)")
    ap.add_argument("--v2-full", help="v2 checkpoint, obs=full")
    ap.add_argument("--v1", help="v1 REINFORCE checkpoint")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                    help="additional v2 checkpoints to benchmark (repeatable), e.g. for ablations")
    ap.add_argument("--with-dds", action="store_true", help="also run the DDS/PIMC 'Hard' player (slow)")
    ap.add_argument("--only-declare", action="store_true")
    ap.add_argument("--with-random", action="store_true")
    ap.add_argument("--num-deals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--defence", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pols = {"heuristic": heuristic_policy()}
    if a.with_random:
        pols["random"] = random_policy()
    if a.v1:
        pols["v1 RL (sees all)"] = v1_policy(a.v1)
    if a.v2_full:
        pols["v2 full (sees all)"] = v2_policy(a.v2_full)
    for spec in a.extra:
        name, path = spec.split("=", 1)
        pols[name] = v2_policy(path)
    if a.v2:
        pols["v2 visible (honest)"] = v2_policy(a.v2)
    if a.with_dds:
        pols["DDS PIMC (honest)"] = dds_policy()
    res = run(pols, a.num_deals, a.seed, a.defence)
    print(f"\nDouble-dummy tricks for these contracts: {res['dd_contract_avg']:.2f}")
    for mode in res:
        if mode in ("declare", "defend") and "heuristic" in res[mode]:
            for name in res[mode]:
                if name != "heuristic":
                    m, se = paired_diff(res, mode, name, "heuristic")
                    print(f"  {mode}: {name} - heuristic = {m:+.2f} tricks/board (paired s.e. {se:.2f})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
