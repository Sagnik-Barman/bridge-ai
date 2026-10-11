"""Generate double-dummy-labelled training positions for the v2 agent.

Each game: a random deal and contract, played out by a *behaviour* policy for
each side, with every decision that has more than one legal card labelled by
the oracle (rl/oracle.py): the exact trick value of every legal card.

Behaviour policies (picked per side, per game):
  dd      - the oracle's best card, but a random legal card with probability eps
            (eps drawn per game from [0, 0.3]); covers near-optimal positions
  heur    - cardplay/bot_player.py, the hand-written heuristic bot
  random  - uniformly random legal card
  net     - the current v2 network (sampling from its policy), so later rounds
            are labelled on the positions the agent itself reaches (DAgger)

Contracts: half uniformly random (declarer and strain, like rl/evaluate.py's
benchmark), half "sensible" (the declaring side's longest combined suit if it
has 8+ cards, else no trump), so the agent also sees the contracts real
auctions reach.

    python -m rl.datagen --games 4000 --workers 2 --out data/v2/round0
    python -m rl.datagen --games 2000 --mix net=0.6,heur=0.25,dd=0.15 \
        --net rl/checkpoints/v2_round0.pt --out data/v2/round1

Output: one .npz per worker (x_full, x_vis as uint8 = 13 * feature value,
mask, q as int8 with -1 for illegal cards, and per-row game/trick/role ids).
Training data is self-generated end to end; nothing here reads any book.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import Dict, Optional

import numpy as np

from engine.deal import Suit, deal_random
from bidding.calls import SEATS, partner_of
from cardplay.trick_engine import PlayState, legal_plays
from cardplay.bot_player import choose_card as heuristic_choose_card
from rl import features_v2 as F2
from rl.oracle import card_values

STRAINS = ["C", "D", "H", "S", "NT"]
BEHAVIOURS = ["dd", "heur", "random", "net"]


def parse_mix(text: str) -> Dict[str, float]:
    mix = {}
    for part in text.split(","):
        k, v = part.split("=")
        assert k in BEHAVIOURS, k
        mix[k] = float(v)
    tot = sum(mix.values())
    return {k: v / tot for k, v in mix.items()}


def sensible_strain(deal, declarer: str) -> str:
    side = [declarer, partner_of(declarer)]
    lengths = {s: sum(1 for seat in side for c in deal.hands[seat].cards if c.suit == s) for s in Suit}
    best = max(lengths, key=lambda s: (lengths[s], s.value))
    return best.value if lengths[best] >= 8 else "NT"


def new_game(rng: random.Random) -> PlayState:
    declarer = rng.choice(SEATS)
    deal = deal_random(dealer=declarer, seed=rng.randrange(1 << 30))
    strain = rng.choice(STRAINS) if rng.random() < 0.5 else sensible_strain(deal, declarer)
    trump = None if strain == "NT" else Suit(strain)
    return PlayState(hands={s: list(deal.hands[s].cards) for s in SEATS},
                     declarer=declarer, strain=strain, trump=trump, dealer=declarer)


def _worker(args):
    wid, n_games, seed, mix, net_path, out_path, backend = args
    import torch
    torch.set_num_threads(1)
    net = None
    if net_path:
        from rl.network_v2 import CardPlayNet
        net = CardPlayNet.load(net_path)
    from rl.agent_v2 import choose_card as net_choose

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    names, probs = list(mix), list(mix.values())
    xf, xv, masks, qs, gid, tix, roles, beh = [], [], [], [], [], [], [], []
    t0 = time.time()
    for g in range(n_games):
        ps = new_game(rng)
        side_beh = {side: names[np_rng.choice(len(names), p=probs)] for side in ("NS", "EW")}
        eps = {side: rng.uniform(0.0, 0.3) for side in ("NS", "EW")}
        while not ps.is_complete():
            seat = ps.next_to_play
            side = "NS" if seat in ("N", "S") else "EW"
            legal = legal_plays(ps.hands[seat], ps.current_trick.led_suit())
            if len(legal) == 1:
                ps.play_card(seat, legal[0])
                continue
            values = card_values(ps, backend)
            idx = F2.Indexer(ps)
            q = np.full(52, -1, dtype=np.int8)
            for c, v in values.items():
                q[idx.index(c)] = v
            xf.append(np.rint(F2.encode(seat, ps, "full", idx) * 13).astype(np.uint8))
            xv.append(np.rint(F2.encode(seat, ps, "visible", idx) * 13).astype(np.uint8))
            masks.append((q >= 0).astype(np.uint8))
            qs.append(q)
            gid.append(wid * 10_000_000 + g)
            tix.append(len(ps.completed_tricks))
            roles.append(F2._role(seat, ps))
            b = side_beh[side]
            beh.append(BEHAVIOURS.index(b))

            if b == "dd":
                if rng.random() < eps[side]:
                    card = rng.choice(legal)
                else:
                    best = max(values.values())
                    card = rng.choice([c for c, v in values.items() if v == best])
            elif b == "heur":
                card = heuristic_choose_card(seat, ps, seed=rng.randrange(1 << 30))
            elif b == "random":
                card = rng.choice(legal)
            else:
                card = net_choose(seat, ps, net, rng=np_rng, temperature=1.0)
            ps.play_card(seat, card)
        if wid == 0 and (g + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  worker 0: {g + 1}/{n_games} games, {len(qs)} positions, "
                  f"{el:.0f}s elapsed, ~{el / (g + 1) * (n_games - g - 1):.0f}s left", flush=True)
    np.savez_compressed(out_path, x_full=np.array(xf), x_vis=np.array(xv), mask=np.array(masks),
                        q=np.array(qs), game=np.array(gid), trick=np.array(tix, dtype=np.int8),
                        role=np.array(roles, dtype=np.int8), behaviour=np.array(beh, dtype=np.int8))
    return out_path, len(qs)


def generate(games: int, out_dir: str, workers: int = 2, seed: int = 0,
             mix: Optional[Dict[str, float]] = None, net_path: Optional[str] = None,
             backend: str = "auto") -> list:
    mix = mix or {"dd": 0.45, "heur": 0.4, "random": 0.15}
    if "net" in mix and not net_path:
        raise ValueError("behaviour 'net' needs --net")
    os.makedirs(out_dir, exist_ok=True)
    from rl.oracle import backend_report
    print(backend_report(backend), flush=True)
    per = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]
    jobs = [(i, per[i], seed * 1000 + i, mix, net_path, os.path.join(out_dir, f"part{i}.npz"), backend)
            for i in range(workers)]
    if workers == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    print(f"wrote {sum(n for _, n in results)} positions to {out_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=str, default="dd=0.45,heur=0.4,random=0.15")
    ap.add_argument("--net", type=str, default=None)
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "dds", "scratch"])
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    generate(a.games, a.out, a.workers, a.seed, parse_mix(a.mix), a.net, a.backend)


if __name__ == "__main__":
    main()
