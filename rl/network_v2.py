"""v2 card-play network: residual MLP with policy, per-card value and state value heads.

    trunk:  Linear(in, W) -> [LayerNorm -> Linear -> ReLU -> Linear, + skip] x B
    policy: W -> 52 logits (masked softmax over legal cards)
    q:      W -> 52   predicted tricks for the mover's side if that card is played
    value:  W -> 1    predicted tricks for the mover's side from here (best play)

Default size (W=384, B=3) is about 1.1M parameters, small enough to run on a
CPU in the deployed app (a single position takes well under a millisecond).

Training objective (rl/train_v2.py), per decision, with Q* the oracle's exact
values (rl/oracle.py) and regret(a) = V* - Q*(a) >= 0 the tricks a card throws away:

    L = -log sum_{a optimal} pi(a)        # put the probability on any optimal card
        + beta * sum_a pi(a) * regret(a)  # expected tricks thrown away under pi
        + c_q * mean_{a legal} (q(a) - Q*(a))^2
        + c_v * (v - V*)^2

The second term is the policy-gradient loss with the exact advantage
Q*(a) - V* summed over every legal action, rather than estimated from one
sampled action and a Monte Carlo return as v1's REINFORCE did.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.features_v2 import N_ACTIONS, feature_size


class _Block(nn.Module):
    def __init__(self, w: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, w)
        self.fc2 = nn.Linear(w, w)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return h + self.fc2(self.drop(F.relu(self.fc1(self.norm(h)))))


class CardPlayNet(nn.Module):
    def __init__(self, obs: str = "visible", width: int = 384, blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.obs, self.width, self.blocks, self.dropout = obs, width, blocks, dropout
        self.inp = nn.Linear(feature_size(obs), width)
        self.body = nn.Sequential(*[_Block(width, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, N_ACTIONS)
        self.q = nn.Linear(width, N_ACTIONS)
        self.value = nn.Linear(width, 1)

    def forward(self, x):
        h = F.relu(self.norm(self.body(F.relu(self.inp(x)))))
        return self.policy(h), self.q(h), self.value(h).squeeze(-1)

    # ---------------------------------------------------------------- helpers
    def config(self) -> dict:
        return {"obs": self.obs, "width": self.width, "blocks": self.blocks, "dropout": self.dropout}

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"version": 2, "config": self.config(), "state_dict": self.state_dict(),
                    "meta": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CardPlayNet":
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("version") != 2:
            raise ValueError(f"{path} is not a v2 checkpoint")
        net = cls(**ck["config"])
        net.load_state_dict(ck["state_dict"])
        net.meta = ck.get("meta", {})
        net.eval()
        return net

    @torch.no_grad()
    def scores(self, x: np.ndarray, mask: np.ndarray, head: str = "policy") -> np.ndarray:
        """Masked scores for a batch (or a single row): policy probabilities,
        or predicted tricks from the q head (illegal cards get -inf)."""
        single = x.ndim == 1
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        mt = torch.as_tensor(np.atleast_2d(mask), dtype=torch.float32)
        logits, q, _ = self(xt)
        if head == "q":
            out = q.masked_fill(mt == 0, float("-inf"))
        else:
            out = F.softmax(logits.masked_fill(mt == 0, -1e9), dim=-1)
        out = out.numpy()
        return out[0] if single else out


def is_v2_checkpoint(path: str) -> bool:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(ck, dict) and ck.get("version") == 2
    except Exception:
        return False
