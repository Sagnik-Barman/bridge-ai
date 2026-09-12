import os
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest(
        "torch is not installed in this environment — rl/network.py is the "
        "PyTorch (CUDA-capable) version; see rl/network_numpy.py for the "
        "zero-dependency reference implementation these tests used to cover, "
        "and docs/rl_training.md for install instructions."
    )

from rl.network import PolicyValueNet
from rl.features import FEATURE_SIZE, N_CARDS


def test_forward_shapes():
    net = PolicyValueNet(seed=0)
    x = np.random.default_rng(0).normal(size=FEATURE_SIZE).astype(np.float32)
    logits, value, cache = net.forward(x)
    assert logits.shape == (N_CARDS,)
    assert isinstance(value, float)


def test_policy_respects_mask():
    net = PolicyValueNet(seed=0)
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[[3, 7, 40]] = 1.0
    probs = net.policy(x, mask)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    illegal = [i for i in range(N_CARDS) if i not in (3, 7, 40)]
    assert np.allclose(probs[illegal], 0.0)


def test_act_only_returns_legal_indices():
    net = PolicyValueNet(seed=1)
    rng = np.random.default_rng(1)
    x = np.random.default_rng(2).normal(size=FEATURE_SIZE).astype(np.float32)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[[1, 2, 5]] = 1.0
    for _ in range(20):
        a = net.act(x, mask, rng)
        assert a in (1, 2, 5)
    a_greedy = net.act(x, mask, rng, greedy=True)
    assert a_greedy in (1, 2, 5)


def test_train_step_reduces_loss_on_a_fixed_batch():
    """Sanity check that the hand-rolled backprop actually moves parameters
    in a loss-reducing direction: repeatedly training on the same
    (state, action, return) should push the network towards predicting
    that action/return, i.e. policy_loss and value_loss should trend down."""
    net = PolicyValueNet(seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=FEATURE_SIZE).astype(np.float32)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[[0, 1, 2]] = 1.0
    batch = [(x, mask, 1, 5.0)]

    losses = []
    for _ in range(50):
        stats = net.train_step(batch, lr=0.01)
        losses.append(stats["value_loss"])

    assert losses[-1] < losses[0]


def test_save_and_load_roundtrip():
    net = PolicyValueNet(seed=3)
    x = np.random.default_rng(3).normal(size=FEATURE_SIZE).astype(np.float32)
    mask = np.ones(N_CARDS, dtype=np.float32)
    probs_before = net.policy(x, mask)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.npz")
        net.save(path)
        loaded = PolicyValueNet.load(path)

    probs_after = loaded.policy(x, mask)
    assert np.allclose(probs_before, probs_after)
