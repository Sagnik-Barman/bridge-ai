"""A small policy+value network, implemented from scratch in NumPy —
CPU-only reference/fallback implementation.

`rl/network.py` (the module the rest of `rl/` actually imports) is now a
PyTorch version that uses your CUDA GPU when available, since you're
training on your own machine. This NumPy version is kept around for two
reasons: it's what the whole `rl/` pipeline was originally built and
proven against (in a cloud sandbox with no PyTorch available at all), and
it still works with zero dependencies beyond NumPy if you ever need to run
somewhere without PyTorch. It is not imported by anything by default — see
`docs/rl_training.md` for how to switch back to it if you ever want to.

This hand-rolls the forward pass, backward pass, and an Adam-style update
for a 2-hidden-layer MLP with a shared trunk and two heads:

  - a policy head: 52 logits, one per card (masked to legal cards before
    turning into a probability distribution)
  - a value head: a scalar estimate of "tricks my side ends up with from
    this position onward" (used as the REINFORCE baseline in
    `rl/train.py`, i.e. this is a small actor-critic)

This is deliberately simple — a few hundred input features, two hidden
layers of 128 units, ReLU activations — appropriately scoped for training
from scratch on a laptop/cloud CPU in a reasonable amount of time, not a
large model. See `docs/rl_training.md` for the full training story.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from rl.features import FEATURE_SIZE, N_CARDS

HIDDEN1 = 128
HIDDEN2 = 64


def _init_layer(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    # He initialization, appropriate for ReLU trunk layers.
    scale = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)


class PolicyValueNet:
    """Params are plain NumPy arrays in a dict so the whole thing is
    trivially saveable with `np.savez`."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": _init_layer(rng, FEATURE_SIZE, HIDDEN1),
            "b1": np.zeros(HIDDEN1, dtype=np.float32),
            "W2": _init_layer(rng, HIDDEN1, HIDDEN2),
            "b2": np.zeros(HIDDEN2, dtype=np.float32),
            "Wp": _init_layer(rng, HIDDEN2, N_CARDS) * 0.1,
            "bp": np.zeros(N_CARDS, dtype=np.float32),
            "Wv": _init_layer(rng, HIDDEN2, 1) * 0.1,
            "bv": np.zeros(1, dtype=np.float32),
        }
        # Adam moment estimates, same shapes as params.
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # -- forward -----------------------------------------------------------

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float, dict]:
        """x: (FEATURE_SIZE,). Returns (policy_logits(52,), value(scalar), cache)."""
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ p["Wp"] + p["bp"]
        value = float((h2 @ p["Wv"] + p["bv"])[0])
        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "logits": logits}
        return logits, value, cache

    def policy(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masked softmax over the 52 cards."""
        logits, _, _ = self.forward(x)
        return _masked_softmax(logits, mask)

    def act(self, x: np.ndarray, mask: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> int:
        probs = self.policy(x, mask)
        if greedy:
            return int(np.argmax(probs))
        return int(rng.choice(N_CARDS, p=probs))

    # -- training: one step of REINFORCE-with-baseline (actor-critic) ------

    def train_step(self, batch, lr: float = 3e-4) -> dict:
        """batch: list of (x, mask, action_index, return_) tuples, all from
        one mini-batch of self-play decisions. Returns loss diagnostics."""
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        policy_loss_total = 0.0
        value_loss_total = 0.0

        for x, mask, action, ret in batch:
            logits, value, cache = self.forward(x)
            probs = _masked_softmax(logits, mask)
            advantage = ret - value

            # Policy gradient for a masked softmax + REINFORCE:
            # d(loss)/d(logits) = (probs - one_hot(action)) * advantage
            dlogits = probs.copy()
            dlogits[action] -= 1.0
            dlogits *= advantage
            dlogits *= mask  # never push probability mass onto illegal cards

            # Value head: simple squared-error to the observed return.
            dvalue = -2.0 * advantage  # d(advantage^2)/d(value) = -2*advantage

            self._backward(cache, dlogits, dvalue, grads)

            policy_loss_total += -float(np.log(max(probs[action], 1e-8))) * advantage
            value_loss_total += advantage ** 2

        n = len(batch)
        for k in grads:
            grads[k] /= max(n, 1)
        self._adam_update(grads, lr)

        return {"policy_loss": policy_loss_total / max(n, 1), "value_loss": value_loss_total / max(n, 1)}

    def _backward(self, cache, dlogits, dvalue, grads) -> None:
        p = self.params
        x, z1, h1, z2, h2 = cache["x"], cache["z1"], cache["h1"], cache["z2"], cache["h2"]

        grads["Wp"] += np.outer(h2, dlogits)
        grads["bp"] += dlogits
        grads["Wv"] += np.outer(h2, np.array([dvalue], dtype=np.float32))
        grads["bv"] += np.array([dvalue], dtype=np.float32)

        dh2 = dlogits @ p["Wp"].T + dvalue * p["Wv"][:, 0]
        dz2 = dh2 * (z2 > 0)

        grads["W2"] += np.outer(h1, dz2)
        grads["b2"] += dz2

        dh1 = dz2 @ p["W2"].T
        dz1 = dh1 * (z1 > 0)

        grads["W1"] += np.outer(x, dz1)
        grads["b1"] += dz1

    def _adam_update(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g ** 2)
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path: str) -> "PolicyValueNet":
        data = np.load(path)
        net = cls.__new__(cls)
        net.params = {k: data[k] for k in data.files}
        net._m = {k: np.zeros_like(v) for k, v in net.params.items()}
        net._v = {k: np.zeros_like(v) for k, v in net.params.items()}
        net._t = 0
        return net


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    m = np.max(masked)
    exp = np.exp(masked - m) * mask
    total = exp.sum()
    if total <= 0:
        # Degenerate (shouldn't happen with a valid mask): uniform over legal.
        return mask / max(mask.sum(), 1.0)
    return exp / total
