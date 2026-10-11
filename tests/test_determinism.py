"""The heuristic bot must play the same deal the same way in every process.

It used to break ties between equally short suits by iterating a Python set,
whose order depends on the per-process string hash seed, so benchmark numbers
drifted between runs of the same command (see cardplay/bot_player.py)."""
import os
import subprocess
import sys

_SCRIPT = """
import random
from engine.deal import Suit, deal_random
from cardplay.trick_engine import PlayState
from cardplay.bot_player import choose_card
out = []
for seed in range(12):
    deal = deal_random(dealer="N", seed=seed)
    ps = PlayState(hands={s: list(deal.hands[s].cards) for s in "NESW"}, declarer="S",
                   strain="NT", trump=None, dealer="N")
    rng = random.Random(seed)
    while not ps.is_complete():
        ps.play_card(ps.next_to_play, choose_card(ps.next_to_play, ps, seed=rng.randrange(1 << 30)))
    out.append("".join(str(c) for t in ps.completed_tricks for _, c in t.plays))
print(hash(tuple(out)) if False else "|".join(out))
"""


def test_heuristic_bot_is_independent_of_hash_seed():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = set()
    for hs in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=root)
        r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=root, env=env,
                           capture_output=True, text=True, check=True)
        results.add(r.stdout)
    assert len(results) == 1
