"""Time each part of rl/datagen.py's per-decision work, to find what is slow.

    python scripts/datagen_speed_check.py

Prints, for a few games played with random cards: time per DDS solve (with
endplay's default resources and with the one-thread budget rl/oracle.py now
sets), time per feature encoding, and time per heuristic-bot move. Reference
numbers from a 2-core Linux machine: ~3-4 ms per solve, ~0.25 ms per encoding,
~0.02 ms per heuristic move.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    t0 = time.perf_counter()
    import rl.oracle as O
    from rl import features_v2 as F2
    from rl.datagen import new_game
    from cardplay.trick_engine import legal_plays
    from cardplay.bot_player import choose_card
    import endplay.dds  # noqa: F401
    print(f"imports: {time.perf_counter() - t0:.1f}s", flush=True)

    def replay(n_games, seed=5):
        rng = random.Random(seed)
        for _ in range(n_games):
            ps = new_game(rng)
            while not ps.is_complete():
                s = ps.next_to_play
                legal = legal_plays(ps.hands[s], ps.current_trick.led_suit())
                if len(legal) > 1:
                    yield ps, s
                ps.play_card(s, rng.choice(legal))

    # 1. DDS with endplay's default resources
    O._DDS_READY = True  # skip the one-thread setup for this pass
    t = time.perf_counter(); n = 0
    first = None
    for ps, s in replay(5):
        a = time.perf_counter()
        O.card_values(ps, "dds")
        if first is None:
            first = time.perf_counter() - a
        n += 1
    el = time.perf_counter() - t
    print(f"DDS, endplay defaults:  first solve {first * 1000:.0f} ms, then {el / n * 1000:.1f} ms/solve over {n}", flush=True)

    # 2. DDS with one thread and a fixed budget (what rl/oracle.py now does)
    O._DDS_READY = False
    t = time.perf_counter(); n = 0
    for ps, s in replay(5, seed=6):
        O.card_values(ps, "dds"); n += 1
    el = time.perf_counter() - t
    print(f"DDS, one thread/400MB:  {el / n * 1000:.1f} ms/solve over {n}", flush=True)

    # 3. feature encoding (both observation modes, as datagen does)
    t = time.perf_counter(); n = 0
    for ps, s in replay(5, seed=7):
        idx = F2.Indexer(ps)
        F2.encode(s, ps, "full", idx); F2.encode(s, ps, "visible", idx); n += 1
    el = time.perf_counter() - t
    print(f"feature encoding:       {el / n * 1000:.2f} ms/position over {n}", flush=True)

    # 4. heuristic bot
    t = time.perf_counter(); n = 0
    for ps, s in replay(5, seed=8):
        choose_card(s, ps, seed=1); n += 1
    el = time.perf_counter() - t
    print(f"heuristic bot:          {el / n * 1000:.2f} ms/move over {n}", flush=True)
    print("reference (2-core Linux): ~3-4 ms/solve, ~0.25 ms/encoding, ~0.02 ms/move")


if __name__ == "__main__":
    main()
