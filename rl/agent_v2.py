"""Choosing a card with a v2 network (rl/network_v2.py).

`choose_card(seat, play_state, net)` encodes the position the way the network
was trained (`net.obs`: "visible" = only what that player can see, "full" =
all four hands), picks the highest-scoring legal card and maps it back from
the canonical index to a real card.

head:
  "policy" - argmax of the policy head (default)
  "q"      - argmax of the predicted per-card trick values
  "mix"    - policy probability plus a small bonus from the q head; ties in the
             policy (equivalent cards) are broken by the q head
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from engine.deal import Card
from cardplay.trick_engine import PlayState, legal_plays
from rl import features_v2 as F2


def card_scores(seat: str, ps: PlayState, net, head: str = "policy"):
    idx = F2.Indexer(ps)
    x = F2.encode(seat, ps, net.obs, idx)
    mask = F2.legal_mask(seat, ps, idx)
    if head == "mix":
        p = net.scores(x, mask, "policy")
        q = net.scores(x, mask, "q")
        s = np.where(mask > 0, p + 0.05 * np.nan_to_num(q, neginf=0.0), -np.inf)
    else:
        s = net.scores(x, mask, head)
        s = np.where(mask > 0, s, -np.inf)
    return idx, s


def choose_card(seat: str, ps: PlayState, net, head: str = "policy",
                rng: Optional[np.random.Generator] = None, temperature: float = 0.0) -> Card:
    legal = legal_plays(ps.hands[seat], ps.current_trick.led_suit())
    if len(legal) == 1:
        return legal[0]
    idx, s = card_scores(seat, ps, net, head if temperature == 0 else "policy")
    if temperature > 0 and rng is not None:
        p = np.where(np.isfinite(s), s, 0.0) ** (1.0 / temperature)
        p = p / p.sum()
        return idx.card(int(rng.choice(len(p), p=p)))
    return idx.card(int(np.argmax(s)))
