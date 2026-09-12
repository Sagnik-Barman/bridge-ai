"""End-to-end demo pipeline: deal -> bid -> play -> report.

    python pipeline.py

Deals a random hand, runs the (currently minimal) opening-bid engine,
attempts a double-dummy / PIMC evaluation of the resulting contract if
`endplay` is installed, and prints a report. Falls back gracefully with a
clear message if the DDS binding isn't installed yet.
"""
from __future__ import annotations

import argparse

from engine.deal import deal_random
from engine.dds_wrapper import DDSUnavailableError, best_contract_for_seat
from bidding.engine import BiddingEngine
from cardplay.pimc import evaluate_contract_pimc


def main(seed: int | None = None) -> None:
    deal = deal_random(seed=seed)
    print("=" * 60)
    print("DEAL")
    print("=" * 60)
    print(deal)
    print()

    engine = BiddingEngine()
    auction = engine.run_auction_stub(deal.hands, dealer=deal.dealer)
    print("=" * 60)
    print("AUCTION (stub — opening bid only, no real competitive bidding yet)")
    print("=" * 60)
    order = ["N", "E", "S", "W"]
    start = order.index(deal.dealer)
    rotated = order[start:] + order[:start]
    for seat, call in zip(rotated * 2, auction.calls):
        print(f"  {seat}: {call}")
    print()

    print("=" * 60)
    print("CARD PLAY (double-dummy best contract per seat)")
    print("=" * 60)
    try:
        for seat in ["N", "E", "S", "W"]:
            strain, tricks = best_contract_for_seat(deal, seat)
            print(f"  {seat}: best strain {strain}, {tricks} tricks double-dummy")

        print()
        print("PIMC sanity check (N/S known, sampling E/W) for N in NT:")
        known = {"N": deal.hands["N"], "S": deal.hands["S"]}
        result = evaluate_contract_pimc(
            known, unseen_seats=["E", "W"], strain="NT", declarer_seat="N",
            num_samples=20, seed=seed,
        )
        print(f"  Mean tricks over {result.samples} samples: {result.mean_tricks:.2f}")

    except DDSUnavailableError as e:
        print(f"  [skipped — {e}]")

    print()
    print("Note: bidding and card-play logic here are early scaffolds, not a")
    print("competitive-strength bridge engine yet. See README.md and docs/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the bridge-ai demo pipeline.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible deals.")
    args = parser.parse_args()
    main(seed=args.seed)
