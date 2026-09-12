# References — free/open resources only

Links only. No content from any book is reproduced or summarized here or
anywhere else in this repository.

## Double-dummy solving

- [dds-bridge/dds](https://github.com/dds-bridge/dds) — the open-source
  Double Dummy Solver C library this project's `engine/dds_wrapper.py`
  ultimately depends on.
- [endplay](https://github.com/dominicprice/endplay) — maintained Python
  package wrapping DDS, with PBN parsing/writing and other bridge utilities.
  Its own docs and source are the primary reference for `engine/dds_wrapper.py`.
- [dds-bridge/python-dds](https://github.com/dds-bridge/python-dds) —
  alternative/earlier Python binding to DDS, kept here for reference in case
  `endplay` proves insufficient for some use case.

## Bidding systems (public documentation, not books)

- [World Bridge Federation (WBF) system policy and system descriptions](https://www.worldbridge.org/) —
  official body; publishes system regulations and reference material for
  major systems including Standard American variants.
- [ACBL (American Contract Bridge League) convention cards](https://www.acbl.org/) —
  standard convention card documents describing Standard American 2/1 Game
  Force, SAYC, and other systems at a structured, rules level (exactly the
  kind of source appropriate for drafting `bidding/conventions/*.yaml`).
- [Bridge Base Online (BBO) system documentation](https://www.bridgebase.com/) —
  hosts various public system write-ups and is a common venue for
  online play data.

## Hand-record / deal datasets

- [PBN (Portable Bridge Notation) format specification](https://www.tistis.nl/pbn/) —
  the file format this project's `Deal.pbn()` method targets, used widely
  for exchanging hand records between bridge software.
- Various tournament hand-record archives published by national bridge
  organizations (e.g. ACBL, national federations) often distribute PBN
  files freely for post-event analysis — check individual organizations'
  terms before bulk use.

## Academic / research background

- Ginsberg, M. L. — "GIB: Imperfect Information in a Computationally
  Challenging Game" (J. Artificial Intelligence Research, 2001) — one of
  the foundational papers on applying Monte Carlo sampling + double-dummy
  solving (i.e. the PIMC approach) to computer bridge.
- Research on AlphaZero-style self-play applied to imperfect-information
  card games (search terms: "self-play reinforcement learning bridge
  bidding", "determinized Monte Carlo tree search imperfect information")
  — background for the general approach behind `rl/`.
- Williams, R. J. — "Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning" (Machine Learning, 1992) — the
  original REINFORCE paper; `rl/network.py`'s training step is a small
  actor-critic variant of this (a learned value baseline reduces the
  variance of the REINFORCE gradient estimate).
- Sutton, R. S. & Barto, A. G. — "Reinforcement Learning: An Introduction"
  (2nd ed., freely available from the authors) — standard reference for
  the actor-critic / policy-gradient-with-baseline formulation used in
  `rl/train.py`.
- See `docs/rl_training.md` for the full write-up of how `rl/` actually
  works, tailored to this project's specific state representation,
  network, and training loop.

## Explicitly excluded

Any bridge instructional book (including but not limited to books in the
Master Bridge Series) is excluded as a source for this project's code, rule
data, or training data, per the project's own setup instructions.
