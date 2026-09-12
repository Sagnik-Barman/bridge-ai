# Self-play RL card-play agent

This used to be a stub with only design notes. It's now real, working
code — see **`docs/rl_training.md`** at the project root for the full
explanation (state representation, network architecture, the self-play
training algorithm, how to run training yourself, how to evaluate a
checkpoint, and an honest account of what's simplified).

Quick pointers into this folder:

- `features.py` — turns a card-play position into a fixed-size vector.
- `network.py` — a small policy+value MLP, in PyTorch (CUDA-capable —
  used for actual training on your machine, see `docs/rl_training.md`).
- `network_numpy.py` — the same architecture, hand-written in NumPy
  (forward pass, backprop, Adam) from when this project's environment had
  no PyTorch available at all; kept as a zero-dependency reference.
- `env.py` — the full-information self-play environment used for training.
- `train.py` — the training loop (`python -m rl.train`).
- `evaluate.py` — benchmarks a checkpoint against the heuristic bot and,
  if `endplay` is installed, double-dummy-optimal par (`python -m rl.evaluate`).
- `rl_player.py` — wraps a trained checkpoint with PIMC sampling so it can
  play under real (imperfect-information) conditions; this is what
  `game/session.py` calls for the "Play against RL" opponent in the web app.
- `checkpoints/` — trained weights land here by default (gitignored) —
  nothing is checked in, since a checkpoint is only as good as the
  training run that produced it.

As with the rest of this project, training data is 100% self-play
(the program dealing itself random hands and playing them out) — nothing
here reads the uploaded bridge book or any other copyrighted material.
