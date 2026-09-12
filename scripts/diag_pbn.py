"""Diagnostic script: run this on your own machine (where `endplay` is
actually installed) against a real .pbn file, BEFORE trusting
benchmark/pbn_loader.py's output. It prints exactly what endplay's PBN
parser actually hands back, so any wrong assumption in pbn_loader.py can
be fixed against real data — the same way engine/dds_wrapper.py's DDTable
indexing bug got found and fixed earlier in this project.

Usage:

    python scripts/diag_pbn.py path\\to\\some_tournament.pbn

Send me the full output and I'll fix anything that doesn't match what
pbn_loader.py currently assumes.
"""
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/diag_pbn.py <path-to-.pbn-file>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        import endplay.parsers.pbn as pbn
    except ImportError as e:
        print(f"endplay is not installed: {e}")
        sys.exit(1)

    with open(path, encoding="utf-8", errors="replace") as f:
        boards = pbn.load(f)

    print(f"Parsed {len(boards)} board(s) from {path!r}.\n")
    if not boards:
        print("No boards parsed -- nothing else to check.")
        return

    b = boards[0]
    print("=== First board ===")
    print("type(board):", type(b))
    print("dir(board):", [a for a in dir(b) if not a.startswith("_")])

    print("\n--- board.deal ---")
    print("type:", type(b.deal))
    print("to_pbn():", b.deal.to_pbn())

    print("\n--- board.contract ---")
    print("type:", type(b.contract))
    if b.contract is not None:
        print("dir:", [a for a in dir(b.contract) if not a.startswith("_")])
        print("declarer:", b.contract.declarer, type(b.contract.declarer), getattr(b.contract.declarer, "name", None))
        print("denom:", b.contract.denom, type(b.contract.denom), getattr(b.contract.denom, "name", None))
        print("level:", getattr(b.contract, "level", None))
        print("penalty:", getattr(b.contract, "penalty", None))
        print("result:", getattr(b.contract, "result", None))

    print("\n--- board.play ---")
    play = list(b.play) if b.play else []
    print("len(play):", len(play))
    if play:
        c = play[0]
        print("type(first card):", type(c))
        print("dir(first card):", [a for a in dir(c) if not a.startswith("_")])
        print("suit:", c.suit, type(c.suit), getattr(c.suit, "name", None))
        print("rank:", c.rank, type(c.rank), getattr(c.rank, "name", None), "abbr:", getattr(c.rank, "abbr", None))
        print("str(card):", str(c))
        print("first 8 cards:", [str(x) for x in play[:8]])

    print("\n--- board.dealer / board.board_num / board.info ---")
    print("dealer:", getattr(b, "dealer", None))
    print("board_num:", getattr(b, "board_num", None))
    info = getattr(b, "info", None)
    print("info type:", type(info))
    try:
        print("info['Event']:", info.get("Event"))
    except Exception as e:
        print("info.get('Event') failed:", e)

    print("\n=== Trying benchmark/pbn_loader.py against this file ===")
    sys.path.insert(0, ".")
    from benchmark.pbn_loader import load_boards

    converted = load_boards(path)
    print(f"pbn_loader.load_boards() converted {len(converted)} / {len(boards)} boards.")
    if converted:
        eb = converted[0]
        print("First converted ExpertBoard: declarer=%s strain=%s trump=%s len(play_order)=%d board_num=%s event=%s" % (
            eb.declarer, eb.strain, eb.trump, len(eb.play_order), eb.board_num, eb.event
        ))
        print("Deal check -- North's hand:", [str(c) for c in eb.deal.hands["N"].cards])


if __name__ == "__main__":
    main()
