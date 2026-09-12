"""
Synthetic sanity check for rl/network_numpy.py's PolicyValueNet + train_step.

Goal: isolate "does this actor-critic training loop work AT ALL" from
"is 52-card bridge card-play just a hard problem for it" by training on a
much simpler synthetic task that uses the EXACT SAME interface
(x: FEATURE_SIZE vector, mask: N_CARDS legal-actions mask, action index,
scalar return) that rl/env.py + rl/train.py feed into train_step.

Task: a set of fixed random "contexts" (x vectors), each with a small
legal-action subset (mask) and one designated "good" action. Reward is a
sparse binary signal, matching bridge's characteristics: 1.0 if the agent
picks the good action, 0.0 otherwise (analogous to "won this trick or
not"). This mirrors the *shape* of the bridge REINFORCE problem
(masked softmax over many actions, sparse 0/1-ish reward, actor-critic
baseline) without any bridge-domain complexity or partial observability.

If train_step/PolicyValueNet is fundamentally sound, average reward
should climb from "random guessing among legal actions" towards ~1.0
(always picking the good action) within a few hundred iterations.

Run from the project root: `python scripts/rl_sanity_check.py`.

This is the diagnostic run when three consecutive training configurations
(pure self-play, +opponent diversity, +per-trick reward shaping) all
landed at essentially the same evaluation result — see the "A third real
result" section of docs/rl_training.md for the actual output and what it
ruled in/out.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import random

from rl.network_numpy import PolicyValueNet
from rl.features import FEATURE_SIZE, N_CARDS

rng_np = np.random.default_rng(42)
py_rng = random.Random(42)

N_CONTEXTS = 8
LEGAL_PER_CONTEXT = 6  # mirrors bridge: usually a handful of legal cards, not all 52

contexts = []
for i in range(N_CONTEXTS):
    x = rng_np.normal(0, 1, size=FEATURE_SIZE).astype(np.float32)
    legal = rng_np.choice(N_CARDS, size=LEGAL_PER_CONTEXT, replace=False)
    mask = np.zeros(N_CARDS, dtype=np.float32)
    mask[legal] = 1.0
    good_action = int(rng_np.choice(legal))
    contexts.append({"x": x, "mask": mask, "legal": legal, "good": good_action})

net = PolicyValueNet(seed=7)

GAMES_PER_ITER = 40
ITERATIONS = 400

print(f"Random-guessing baseline avg reward: {1.0/LEGAL_PER_CONTEXT:.3f}")
print(f"Optimal avg reward: 1.000")
print()

history = []
for it in range(1, ITERATIONS + 1):
    batch = []
    rewards = []
    for _ in range(GAMES_PER_ITER):
        ctx = contexts[py_rng.randrange(N_CONTEXTS)]
        action = net.act(ctx["x"], ctx["mask"], rng_np, greedy=False)
        reward = 1.0 if action == ctx["good"] else 0.0
        batch.append((ctx["x"], ctx["mask"], action, reward))
        rewards.append(reward)
    stats = net.train_step(batch, lr=3e-4)
    history.append(np.mean(rewards))
    if it % 40 == 0 or it == 1:
        recent = np.mean(history[-40:])
        print(f"iter {it:4d}/{ITERATIONS}  avg_reward(last40)={recent:.3f}  "
              f"policy_loss={stats['policy_loss']:.3f}  value_loss={stats['value_loss']:.3f}")

# Final check: greedy policy on each context
print()
print("Final greedy policy per context (does it pick the designated good action?):")
n_correct = 0
for i, ctx in enumerate(contexts):
    action = net.act(ctx["x"], ctx["mask"], rng_np, greedy=True)
    correct = action == ctx["good"]
    n_correct += correct
    print(f"  context {i}: good={ctx['good']:2d}  greedy_pick={action:2d}  {'OK' if correct else 'WRONG'}")
print(f"\n{n_correct}/{N_CONTEXTS} contexts solved greedily.")
