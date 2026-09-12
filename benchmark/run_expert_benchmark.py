"""CLI: benchmark this project's engines against real tournament play
recorded in one or more .pbn files.

    python -m benchmark.run_expert_benchmark data/real_deals/*.pbn
    python -m benchmark.run_expert_benchmark data/real_deals/bermuda_bowl_2019.pbn --rl-checkpoint rl/checkpoints/latest.pt

See docs/real_deals.md for where to get real .pbn files, and — before
trusting this on a real file — run `scripts/diag_pbn.py` first (see that
file and benchmark/pbn_loader.py's docstrings for why: the PBN-parsing
layer hasn't been verified against a real endplay install yet).
"""
from __future__ import annotations

import argparse
import glob
import sys

from benchmark.expert_benchmark import SOURCES, benchmark_board, summarize
from benchmark.pbn_loader import PbnUnavailableError, load_boards


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark engines against real tournament play.")
    parser.add_argument("pbn_files", nargs="+", help="One or more .pbn files (globs are expanded).")
    parser.add_argument("--rl-checkpoint", type=str, default=None, help="Path to a trained RL checkpoint to include.")
    parser.add_argument("--max-boards", type=int, default=None, help="Stop after this many boards (across all files).")
    args = parser.parse_args()

    paths = []
    for pattern in args.pbn_files:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    rl_net = None
    if args.rl_checkpoint:
        try:
            from rl.network import PolicyValueNet

            rl_net = PolicyValueNet.load(args.rl_checkpoint)
        except Exception as e:
            print(f"Couldn't load RL checkpoint ({e}) -- continuing without the RL comparison.")

    all_boards = []
    for path in paths:
        try:
            boards = load_boards(path)
        except PbnUnavailableError as e:
            print(str(e))
            sys.exit(1)
        print(f"{path}: {len(boards)} usable board(s) (complete play record + known contract).")
        all_boards.extend(boards)
        if args.max_boards and len(all_boards) >= args.max_boards:
            all_boards = all_boards[: args.max_boards]
            break

    if not all_boards:
        print("No usable boards found across the given file(s).")
        return

    print(f"\nBenchmarking {len(all_boards)} real board(s)...")
    results = []
    for i, board in enumerate(all_boards, 1):
        try:
            results.append(benchmark_board(board, rl_net=rl_net))
        except ValueError as e:
            print(f"  skipping board {board.board_num}: {e}")
        if i % 10 == 0:
            print(f"  ...{i}/{len(all_boards)}")

    summary = summarize(results)
    print(f"\n=== Results over {summary['num_boards']} real board(s) ===")
    print(f"Actual declarer tricks (avg, real tournament result): {summary['actual_declarer_tricks_avg']:.2f}")
    for source in SOURCES:
        s = summary[source]
        if s["boards_judged"] == 0:
            print(f"{source:>15}: not available for any board")
            continue
        decl = s["avg_match_rate_declaring_side"]
        overall = s["avg_match_rate"]
        decl_str = f"{decl * 100:.1f}%" if decl is not None else "n/a"
        overall_str = f"{overall * 100:.1f}%" if overall is not None else "n/a"
        print(f"{source:>15}: matched the real card {overall_str} of the time overall, {decl_str} as declarer/dummy "
              f"({s['boards_judged']} boards judged)")


if __name__ == "__main__":
    main()
