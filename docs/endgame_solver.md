# The from-scratch endgame solver, and the Analyzer feature

This document explains `cardplay/endgame_solver.py` and how it fits into
the "Analyze" button in the web app (`game/session.py`'s
`analyze_current_position()`).

## Why this exists

The project already has a real, fast double-dummy solver
(`engine/dds_wrapper.py`, wrapping the `dds-bridge/dds` C library via
`endplay`). That solver is not something this project built — it's an
industrial-strength piece of software with years of optimization behind
it. `cardplay/endgame_solver.py` is different: it's a small, from-scratch
implementation of the classical algorithm real solvers are built on
(minimax search with alpha-beta pruning and a transposition table),
written to understand and demonstrate the technique itself, and to have
an independent way to cross-check the real solver on positions small
enough for both to handle.

This is a legitimate algorithms/DSA project in its own right — game
trees, adversarial search, pruning, memoized recursion, and the
complexity tradeoffs involved are all real computer science, not busywork.

## What it deliberately does NOT do

Real DDS engines get their speed from optimizations this module skips
entirely:

- **Suit-symmetry reduction.** If neither side has a card between the 4
  and 2 of a suit, those two cards are strategically interchangeable —
  a real solver collapses them into one equivalence class, which shrinks
  the branching factor enormously. This module treats every card as
  distinct.
- **Move ordering heuristics** that make alpha-beta cutoffs happen much
  earlier in practice.
- **Quick-tricks / partial-search bounds** that let a real solver stop
  early once a contract's fate is already decided.

Because of this, the solver here is only usable on *small* endgames —
by default, up to 6 cards left in each hand (`max_cards_per_hand=6`,
also enforced as a hard `node_budget` in case a position is
pathologically slow even under that limit). Beyond that it raises
`SearchTooLargeError` rather than pretending to have an answer or
hanging. A full 13-card deal is completely out of reach for this
approach — that's exactly why real DDS exists.

The 6-card default isn't a round number picked for looks — it's a
measured one. On ordinary hardware: 5 cards left per hand solves in well
under a second; 6 usually finishes in a second or two; 7-8 frequently
exhaust the 200,000-node budget *without finishing at all*. A limit that
usually can't finish isn't a meaningfully bigger limit, so 6 is the
number this module actually delivers on.

## Correctness details worth understanding

- **Only exact values are cached.** When alpha-beta prunes a branch, the
  value it returns is a *bound* (e.g. "at least this good"), not the
  exact minimax value. Caching a bound as if it were exact is a classic
  transposition-table bug — it can make later lookups return a wrong
  answer for a *different* alpha/beta window. This module only writes to
  its memo table when the search over a node's full window completed
  without a cutoff.
- **The solver never mutates the caller's real position.** `solve_endgame`
  clones the incoming `PlayState` up front; search-time backtracking
  reuses the existing, well-tested `PlayState.clone_state()` /
  `restore_state()` machinery rather than a second, independent
  copy-and-mutate scheme. This promise was actually broken for a while:
  an earlier version aliased the caller's `current_trick` object instead
  of deep-copying it, so the very first simulated play inside the search
  silently corrupted the caller's real, live trick state (`Trick` is a
  mutable dataclass, and `play_card()` appends to it in place). It went
  undetected by the original test suite because those tests only ever
  called `solve_endgame` once per position and checked `ps.hands`
  afterward, never a *sequence* of calls across a full hand, and never
  `ps.current_trick`. `benchmark/expert_benchmark.py`'s replay-a-whole-
  real-board style of testing — which calls `solve_endgame` dozens of
  times in a row against one long-lived `PlayState` — surfaced it
  immediately (an `OverflowError` deep in the search, from a hand that
  had gone empty when it shouldn't have). Fixed by deep-copying
  `current_trick` and `completed_tricks`; `tests/test_endgame_solver.py`'s
  `test_does_not_corrupt_the_callers_live_play_state` is the regression
  test.
- **Verification against real DDS.** `tests/test_endgame_solver.py`
  includes `test_matches_real_dds_on_small_endgames`, which plays a
  random deal down (using the ordinary heuristic bot) to a trick boundary
  with 5 or fewer cards left per hand, then checks that this solver's
  predicted NS trick count agrees exactly with `engine/dds_wrapper.py`'s
  real double-dummy answer for the identical position. When `endplay`
  isn't installed, this check is skipped rather than failing — there's
  nothing to cross-check against.

## How it's used: the Analyzer feature

`GameSession.analyze_current_position()` gathers a suggestion from every
card-play source in the project for whoever is on lead right now:

1. **Heuristic bot** (`cardplay/bot_player.py`) — always available.
2. **DDS/PIMC** (`cardplay/dds_player.py`) — available when `endplay` is
   installed; handles positions of any size.
3. **RL network** (`rl/rl_player.py`) — available when a trained
   checkpoint exists and PyTorch is installed.
4. **This exact endgame solver** — available only when 6 or fewer cards
   remain in the hand on lead; otherwise it reports why it's sitting out
   and defers to the DDS suggestion.

Each source is independent and best-effort: if one isn't available (no
`endplay`, no checkpoint, position too big), it says so rather than
crashing the others or being silently omitted. The web app's "Analyze"
button (visible any time it's your turn in the card-play phase) shows all
four side by side, along with whether they agree.

None of this — the endgame solver, the DDS wrapper, the RL network, or
the Analyzer feature — reads from, is trained on, or otherwise derives
from the uploaded copyrighted bridge book. Every suggestion here comes
from hand-written rules, a real double-dummy solve, an from-scratch
search over the rules of bridge, or a network trained purely by self-play.

## A note on game theory

Double-dummy bridge (everyone's hand known) is a finite, perfect
information, zero-sum extensive-form game — exactly the setting von
Neumann's minimax theorem and backward induction apply to, and exactly
what both this solver and real DDS engines compute: the game's value and
an optimal line of play, by exhaustive (or pruned) search over the game
tree.

Real bridge — where each player only sees their own hand (and dummy's,
once tabled) — is not that game. It's a game of *imperfect information*,
formally closer to a Bayesian extensive-form game with information sets:
a truly game-theoretic treatment would reason about mixed strategies and
beliefs over the opponents' hidden hands, not just search a known tree.
PIMC (used by `cardplay/dds_player.py` and `rl/rl_player.py`) is a
practical, widely-used *heuristic* approximation to that — sample
plausible worlds, solve each one perfectly, average — not a true
equilibrium solver. That gap (PIMC's known weaknesses around information-
gathering plays, like a safety play that's only right because it protects
against a distribution the sampling didn't weight correctly) is real and
worth naming honestly rather than glossing over.
