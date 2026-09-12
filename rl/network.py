"""The card-play policy+value network — PyTorch, CUDA-capable.

This is the version actually used for training on your own machine
(`python -m rl.train`, see `docs/rl_training.md`): a small 2-hidden-layer
MLP with a policy head (52 logits, one per card) and a value head (a
scalar estimate of tricks-my-side-ends-up-with), trained with Adam and
run on your GPU via CUDA whenever one's available — falls back to CPU
automatically otherwise, so the same code runs on a laptop with no GPU
too, just slower.

Why this file exists alongside `rl/network_numpy.py`: this whole `rl/`
pipeline was originally designed and proven out in a cloud sandbox that
had no PyTorch (or any way to install it) available at all, so the first
working version hand-rolled the forward/backward pass in NumPy. That
NumPy version is kept as `rl/network_numpy.py` — a correctness reference
and a zero-dependency fallback — but this PyTorch version is what you
actually want for real training: it's simpler code, it's the standard
tool for the job, and it uses your CUDA GPU instead of your CPU.

External interface deliberately matches `rl/network_numpy.py` exactly
(`forward`, `policy`, `act`, `train_step`, `save`/`load`) so nothing else
in `rl/` (env.py, train.py, evaluate.py, rl_player.py) needed to change
when this file was swapped in.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.features import FEATURE_SIZE, N_CARDS

HIDDEN1 = 128
HIDDEN2 = 64


def get_device(prefer_cuda: bool = True) -> torch.device:
    """CUDA if available and requested, else CPU. Called with no arguments
    this is exactly "use my GPU if I have one" — print it once at the
    start of a training run (`rl/train.py` does) so you can see which one
    you actually got."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_SIZE, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.policy_head = nn.Linear(HIDDEN2, N_CARDS)
        self.value_head = nn.Linear(HIDDEN2, 1)
        # Small init on the heads (matches the NumPy version's 0.1x scaling)
        # so early-training logits/values start close to zero/uniform
        # rather than the network being confidently wrong from step one.
        nn.init.uniform_(self.policy_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.policy_head(h2)
        value = self.value_head(h2).squeeze(-1)
        return logits, value


class PolicyValueNet:
    def __init__(self, seed: int = 0, device: Optional[str] = None):
        torch.manual_seed(seed)
        self.device = torch.device(device) if device else get_device()
        self.net = _Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

    # -- forward -------------------------------------------------------

    def forward(self, x: np.ndarray):
        """x: (FEATURE_SIZE,) numpy array. Returns (logits(52,) numpy,
        value: float, cache=None) — the cache slot is kept only so callers
        written against the NumPy version's signature don't need to
        change; PyTorch's autograd makes an explicit cache unnecessary."""
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(xt)
        return logits.squeeze(0).cpu().numpy(), float(value.item()), None

    def policy(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masked softmax over the 52 cards, as a NumPy array (kept as
        NumPy at this boundary since every caller in `rl/` — env.py,
        evaluate.py, rl_player.py — works with plain NumPy arrays and
        Python randomness, not torch tensors)."""
        logits, _, _ = self.forward(x)
        return _masked_softmax(logits, mask)

    def act(self, x: np.ndarray, mask: np.ndarray, rng, greedy: bool = False) -> int:
        probs = self.policy(x, mask)
        if greedy:
            return int(np.argmax(probs))
        return int(rng.choice(N_CARDS, p=probs))

    # -- training --------------------------------------------------------

    def train_step(self, batch, lr: float = 3e-4) -> dict:
        """batch: list of (x, mask, action_index, return_) tuples. Runs one
        Adam step of REINFORCE-with-baseline (actor-critic): the policy
        loss pushes up log-probability of the action taken, weighted by
        the advantage (return minus the value head's prediction); the
        value loss regresses the value head towards the observed return.
        Both losses are summed and backpropagated together in one step,
        which is standard practice and keeps the two heads' shared trunk
        consistent."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        xs = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b[2] for b in batch], dtype=torch.long, device=self.device)
        returns = torch.as_tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)

        logits, values = self.net(xs)
        masked_logits = logits.masked_fill(masks == 0, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        advantage = (returns - values).detach()
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(chosen_log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item())}

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PolicyValueNet":
        net = cls.__new__(cls)
        net.device = torch.device(device) if device else get_device()
        net.net = _Net().to(net.device)
        checkpoint = torch.load(path, map_location=net.device)
        net.net.load_state_dict(checkpoint["state_dict"])
        net.optimizer = torch.optim.Adam(net.net.parameters(), lr=3e-4)
        return net


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    m = np.max(masked)
    exp = np.exp(masked - m) * mask
    total = exp.sum()
    if total <= 0:
        return mask / max(mask.sum(), 1.0)
    return exp / total
