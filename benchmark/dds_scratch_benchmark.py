"""Check the from-scratch double-dummy solver (ddsolver/) against the DDS library and
measure what each search technique contributes.

    python -m benchmark.dds_scratch_benchmark verify --deals 200        # full 20-contract tables vs DDS
    python -m benchmark.dds_scratch_benchmark positions --n 300         # mid-play positions, every legal card
    python -m benchmark.dds_scratch_benchmark ablation --cards 7 --n 40 # techniques switched on one by one

`verify` and `positions` need endplay (pip install endplay), which wraps the DDS library
used as the reference answer. `ablation` needs nothing extra. Results go to results/.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ddsolver as D  # noqa: E402

SEATS = "NESW"
STRAINS = ["S", "H", "D", "C", "NT"]


def random_pbn(rng: random.Random, cards_per_hand: int = 13) -> str:
    deck = [(s, r) for s in range(4) for r in range(13)]
    rng.shuffle(deck)
    hands = []
    for p in range(4):
        cs = deck[p * cards_per_hand:(p + 1) * cards_per_hand]
        hands.append(".".join("".join(D.RANKS[r] for r in sorted((r for s2, r in cs if s2 == s), reverse=True))
                              for s in range(4)))
    return "N:" + " ".join(hands)


def summary(xs):
    xs = sorted(xs)
    if not xs:
        return "n/a"
    p90 = xs[min(len(xs) - 1, int(math.ceil(0.9 * len(xs))) - 1)]
    return f"median {statistics.median(xs):.3f} s, mean {statistics.mean(xs):.3f} s, 90th pct {p90:.3f} s, max {xs[-1]:.3f} s"


# ------------------------------------------------------------------ verify
def cmd_verify(args):
    from endplay.dds import calc_dd_table
    from endplay.types import Deal, Denom, Player
    den = {"S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds, "C": Denom.clubs, "NT": Denom.nt}
    pl = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dds_scratch_verify.csv")
    ours_t, ref_t, deals_ok, contracts_ok = [], [], 0, 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deal", "pbn", "match", "contracts_matching", "ours_s", "ours_nodes", "dds_s", "ours_table", "dds_table"])
        for i in range(args.deals):
            pbn = random_pbn(rng)
            t = time.perf_counter()
            table, nodes = D.dd_table_pbn(pbn)
            ours = time.perf_counter() - t
            t = time.perf_counter()
            ref = calc_dd_table(Deal(pbn))
            dds = time.perf_counter() - t
            same = sum(table[s][st] == ref[den[st], pl[s]] for s in SEATS for st in STRAINS)
            deals_ok += same == 20
            contracts_ok += same
            ours_t.append(ours)
            ref_t.append(dds)
            w.writerow([i, pbn, int(same == 20), same, f"{ours:.4f}", nodes, f"{dds:.4f}",
                        " ".join(str(table[s][st]) for s in SEATS for st in STRAINS),
                        " ".join(str(ref[den[st], pl[s]]) for s in SEATS for st in STRAINS)])
            f.flush()
            flag = "" if same == 20 else "   <-- MISMATCH"
            print(f"deal {i:4d}: {same}/20 contracts match, ours {ours:7.3f} s ({nodes:,} nodes), DDS {dds:.3f} s{flag}",
                  flush=True)
    n = args.deals
    print(f"\n{deals_ok}/{n} deals and {contracts_ok}/{20 * n} contracts identical to DDS")
    print(f"ours (20-contract table): {summary(ours_t)}")
    print(f"DDS  (20-contract table): {summary(ref_t)}")
    print(f"wrote {path}")
    return 0 if deals_ok == n else 1


# ------------------------------------------------------------------ positions
def _card_to_sr(card):
    return card.suit.value, int(card.rank.value).bit_length() - 3   # endplay rank values are 1 << (rank + 2)


def cmd_positions(args):
    from endplay.dds import solve_board
    from endplay.types import Deal, Denom, Player
    rng = random.Random(args.seed)
    ours_t, ref_t, ok, cards_checked = [], [], 0, 0
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dds_scratch_positions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "pbn_start", "trump", "cards_played", "match", "ours_s", "dds_s"])
        for i in range(args.n):
            pbn = random_pbn(rng)
            deal = Deal(pbn)
            trump = rng.randrange(5)
            deal.trump = Denom(trump)
            deal.first = Player(rng.randrange(4))
            played = rng.randrange(0, 40)   # 0..39 random legal cards: any trick, any position in it
            for _ in range(played):
                deal.play(rng.choice(list(deal.legal_moves())))
            masks = [[0] * 4 for _ in range(4)]
            for p, hand in enumerate([deal.north, deal.east, deal.south, deal.west]):
                for c in hand:
                    s, r = _card_to_sr(c)
                    masks[p][s] |= 1 << r
            trick = [_card_to_sr(c) for c in deal.curtrick]
            leader = deal.first.value
            t = time.perf_counter()
            scored, _ = D.score_cards(masks, trump, leader, trick)
            ours = time.perf_counter() - t
            t = time.perf_counter()
            ref = solve_board(deal)
            dds = time.perf_counter() - t
            mover = deal.curplayer.value
            left = sum(bin(m).count("1") for m in masks[mover])
            ref_ns = {}
            for card, tricks in ref:
                ref_ns[_card_to_sr(card)] = tricks if mover % 2 == 0 else left - tricks
            ours_ns = {(s, r): v for s, r, v in scored}
            match = ours_ns == ref_ns
            ok += match
            cards_checked += len(ref_ns)
            ours_t.append(ours)
            ref_t.append(dds)
            w.writerow([i, pbn, STRAINS[trump], played, int(match), f"{ours:.4f}", f"{dds:.4f}"])
            if not match:
                print(f"case {i}: MISMATCH ours={ours_ns} dds={ref_ns}", flush=True)
            elif (i + 1) % 25 == 0:
                print(f"{i + 1} positions checked, all matching so far", flush=True)
    print(f"\n{ok}/{args.n} positions identical to DDS ({cards_checked} legal cards scored)")
    print(f"ours (score every legal card): {summary(ours_t)}")
    print(f"DDS  (score every legal card): {summary(ref_t)}")
    print(f"wrote {path}")
    return 0 if ok == args.n else 1


# ------------------------------------------------------------------ ablation
CONFIGS = [
    ("plain minimax + alpha-beta", D.MINIMAX),
    ("+ yes/no (null-window) search", 0),
    ("+ move ordering", D.ORDER),
    ("+ equivalent cards", D.ORDER | D.EQUIV),
    ("+ transposition table, exact ranks", D.ORDER | D.EQUIV | D.TT),
    ("+ relative-rank keys", D.ORDER | D.EQUIV | D.TT | D.RELRANK),
    ("+ partition search", D.ORDER | D.EQUIV | D.TT | D.RELRANK | D.PARTITION),
    ("+ quick-trick / top-trump bounds (all)", D.ALL),
]


def cmd_ablation(args):
    rng = random.Random(args.seed)
    cases = []
    for _ in range(args.n):
        cases.append((D.parse_pbn(random_pbn(rng, args.cards)), rng.randrange(5), rng.randrange(4)))
    truth = [D.solve(m, tr, ld)[0] for m, tr, ld in cases]
    rows = []
    print(f"{args.n} random positions, {args.cards} cards per hand, random strain and leader; "
          f"node limit {args.node_limit:,} per position\n")
    print(f"| technique | solved | mean nodes | median nodes | mean time (ms) | x fewer nodes than baseline |")
    print("|---|---:|---:|---:|---:|---:|")
    base = None
    for name, flags in CONFIGS:
        nodes, times, solved = [], [], 0
        for (m, tr, ld), want in zip(cases, truth):
            t = time.perf_counter()
            try:
                v, n = D.solve(m, tr, ld, flags=flags, node_limit=args.node_limit)
            except D.SolverError:
                v, n = None, args.node_limit
            times.append(time.perf_counter() - t)
            nodes.append(n)
            if v is not None:
                solved += 1
                if v != want:
                    raise SystemExit(f"{name}: wrong answer {v} vs {want} -- bug")
        mean_n = statistics.mean(nodes)
        base = base or mean_n
        unfinished = "" if solved == len(cases) else " (lower bound: some hit the node limit)"
        rows.append([name, solved, len(cases), round(mean_n), round(statistics.median(nodes)),
                     round(1000 * statistics.mean(times), 2), round(base / mean_n, 1)])
        print(f"| {name} | {solved}/{len(cases)} | {mean_n:,.0f}{unfinished} | {statistics.median(nodes):,.0f} | "
              f"{1000 * statistics.mean(times):.2f} | {base / mean_n:,.1f} |", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"dds_scratch_ablation_{args.cards}cards.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "solved", "cases", "mean_nodes", "median_nodes", "mean_ms", "x_fewer_nodes"])
        w.writerows(rows)
    print(f"\nall configurations that finished agree with each other; wrote {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("verify")
    a.add_argument("--deals", type=int, default=100)
    a.add_argument("--seed", type=int, default=1)
    b = sub.add_parser("positions")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=2)
    c = sub.add_parser("ablation")
    c.add_argument("--cards", type=int, default=7, help="cards per hand")
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--seed", type=int, default=3)
    c.add_argument("--node-limit", type=int, default=20_000_000)
    for p in (a, b, c):
        p.add_argument("--out", default="results")
    args = ap.parse_args()
    return {"verify": cmd_verify, "positions": cmd_positions, "ablation": cmd_ablation}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
