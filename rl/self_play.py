"""Superseded — kept only as a pointer so nothing breaks if something still
imports this module by its old name.

The self-play RL pipeline is now real, working code, split across:
  - rl/env.py        (self-play game generation)
  - rl/network.py    (the policy+value network)
  - rl/train.py      (the training loop — `python -m rl.train`)
  - rl/evaluate.py   (benchmarking a checkpoint — `python -m rl.evaluate`)
  - rl/rl_player.py  (wiring a trained checkpoint into actual gameplay)

See docs/rl_training.md for the full explanation. You can delete this
file — nothing in the project imports `rl.self_play` anymore.
"""
from __future__ import annotations

raise ImportError(
    "rl.self_play no longer exists as working code — see rl/train.py, "
    "rl/env.py, rl/network.py, rl/evaluate.py, rl/rl_player.py, and "
    "docs/rl_training.md."
)
