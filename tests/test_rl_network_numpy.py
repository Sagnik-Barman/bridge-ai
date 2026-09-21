"""Tests for the dependency-free rl/network_numpy.py, including
`pretrain_step` (the supervised imitation-warm-start step added alongside
rl/generate_labels.py and rl/pretrain.py). Kept separate from
tests/test_rl_network.py, which covers the PyTorch version and is
torch-gated -- everything here runs with only NumPy."""
import numpy as np

from rl.network_numpy import PolicyValueNet, _masked_softmax
from rl.features import FEATURE_SIZE, N_CARDS


def _greedy_action(net, x, mask):
    logits, _, _ = net.forward(x)
    probs = _masked_softmax(logits, mask)
    return int(np.argmax(probs))


def test_pretrain_step_reduces_loss_on_a_fixed_batch():
    net = PolicyValueNet(seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=FEATURE_SIZE).astype(np.float32)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[[4, 10, 22]] = 1.0
    batch = [(x, mask, 10)]

    losses = []
    for _ in range(30):
        stats = net.pretrain_step(batch, lr=5e-3)
        losses.append(stats["loss"])

    assert losses[-1] < losses[0]


def test_pretrain_step_learns_a_small_labeled_dataset():
    """Same shape of check as scripts/rl_sanity_check.py, but exercising
    the actual `pretrain_step` API used by rl/pretrain.py: a handful of
    fixed (context, correct action) pairs should become solvable with
    ordinary supervised training."""
    rng_np = np.random.default_rng(3)
    n_contexts = 6
    legal_per_context = 5
    dataset = []
    for _ in range(n_contexts):
        x = rng_np.normal(size=FEATURE_SIZE).astype(np.float32)
        legal = rng_np.choice(N_CARDS, size=legal_per_context, replace=False)
        mask = np.zeros(N_CARDS, dtype=np.float32)
        mask[legal] = 1.0
        good_action = int(rng_np.choice(legal))
        dataset.append((x, mask, good_action))

    net = PolicyValueNet(seed=5)
    for _ in range(150):
        net.pretrain_step(dataset, lr=3e-3)

    correct = sum(1 for x, mask, action in dataset if _greedy_action(net, x, mask) == action)
    assert correct == n_contexts


def test_pretrain_step_never_pushes_probability_onto_illegal_cards():
    net = PolicyValueNet(seed=1)
    x = np.random.default_rng(2).normal(size=FEATURE_SIZE).astype(np.float32)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[[0, 15, 30]] = 1.0
    for _ in range(10):
        net.pretrain_step([(x, mask, 15)], lr=2e-3)
    probs = net.policy(x, mask)
    illegal = [i for i in range(N_CARDS) if i not in (0, 15, 30)]
    assert np.allclose(probs[illegal], 0.0)
