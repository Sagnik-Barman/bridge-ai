# A double-dummy solver written from scratch (`ddsolver/`)

`ddsolver/` is a complete double-dummy solver for full 13-card deals, written for
this project in C++ (about 780 lines, `ddsolver/src/solver.cpp`) with a small
Python wrapper. It gives exactly the same answers as the DDS library
(dds-bridge/dds, which the rest of the project uses through `endplay`), checked
on all 2,000 contracts of 100 random deals and on 300 mid-play positions. It is
slower than DDS, which has many more years of tuning behind it: a median of 2.6 s for
a full 20-contract table against DDS's 0.13 s. The numbers are below.

It is separate from `cardplay/endgame_solver.py`, the small Python endgame solver
behind the Analyze button (exact up to about 6 cards per hand). That one shows the
basic algorithm; this one adds the techniques that make a full deal solvable.

## Using it

```python
from ddsolver import dd_table_pbn, solve_double_dummy, best_card, solve_play_state

table, nodes = dd_table_pbn("N:AKQ2.T98.5432.K4 J86.AK76.KJ.QJ98 T95.QJ32.AQ96.65 743.54.T87.AT732")
table["N"]["NT"]         # 7: tricks for North-South with North declaring no trump

solve_double_dummy(deal)  # same output format as engine/dds_wrapper.solve_double_dummy
best_card(play_state)     # (best card, N-S tricks after it, nodes) for any position, even mid-trick
```

The compiled library is in `ddsolver/lib/`: `dds_scratch.dll` (Windows, x86-64)
and `libdds_scratch.so` (Linux, x86-64). To rebuild it after changing the C++:

```bash
pip install ziglang          # a C/C++ compiler that installs with pip (any OS)
python -m ddsolver.build     # writes ddsolver/lib/<library for this machine>
```

g++, clang++ or MSVC also work if one is on the PATH.

## The problem

Double dummy means every hand is visible and both sides play perfectly. From any
position the question is: how many of the remaining tricks does North-South take?
It is a two-player, zero-sum game with perfect information, so it can be solved
exactly by searching the game tree: N-S pick the card that maximises their
tricks, E-W the one that minimises them (minimax).

The tree for a full deal is far too large to search directly. Each technique below
removes a large part of it, and the benchmark switches them on one at a time to
measure how much each one removes.

## The techniques

**1. Bitmask state.** Each hand is stored as four 13-bit numbers, one per suit
(bit 0 = the 2, bit 12 = the ace). Playing a card, checking whether a player can
follow suit, and finding the highest card of a suit are each one or two machine
instructions. Undoing a move restores the same few numbers, so the search never
copies a position.

**2. Yes/no search instead of "how many tricks?"** Rather than computing the exact
number directly, the search answers a yes/no question: can N-S take at least t
tricks? At an N-S turn the answer is yes if any card works; at an E-W turn it is yes
only if every card works. A yes/no question stops at the first card that settles it,
which prunes far more than searching for a number (this is a null-window
alpha-beta search). The exact count is found by asking the question a few times:
starting from a guess (the result for the previous declarer in the same strain) and
stepping up or down, or by binary search between 0 and the tricks left.

**3. Equivalent cards.** If a hand holds the Q and the J of a suit and the K is gone
(or is in the same hand), playing the Q or the J leads to the same result. The rule:
two cards of one hand are equivalent when no card held by anyone else, and no card
already on the table in this trick, lies between them. Only one card of each such
group is searched.

**4. Transposition table with relative ranks.** The same position is reached
through many different orders of play. Results are cached at the start of each
trick, as a lower bound and an upper bound on the N-S tricks still to come (one
yes/no search proves only one of the two). The cache key uses relative ranks: once
the A, K and Q of spades are gone, the J of spades is the top spade and plays exactly
like an ace. So the key records, for each suit, only who holds the remaining cards
from the top down, not their printed ranks.

**5. Partition search: forgetting the small cards that didn't matter.** This is the
largest single gain on full deals. When a search finishes, it also reports which
cards its answer depended on: a card counts if it won a trick by beating another
card of the same suit, or if a bound or cached result relied on it. Every card below
the lowest such card in a suit is treated as an interchangeable "small card". The
cache entry then fixes only the suit lengths of each hand and the owners of the
cards from that cut-off upwards, so one entry answers every position that differs
only in its small cards. Entries are grouped by the suit lengths of all four hands;
within a group they are indexed by how many top cards of each suit they fix.

One subtlety, found by testing: equivalent-card pruning (technique 3) and partition
search interact. If the cut-off between "exact" and "small" cards falls inside a
group of equivalent cards that was searched through one member only, then a
different position with the same exact cards can split that group, and offer a card
that was never searched. The search therefore also reports the rank range of every
group it searched through one member, and the cut-off is moved below any group it
would otherwise split. Without that rule the solver gave wrong answers (the first
showed up within five test deals); with it, every result matches DDS.

**6. Cheap bounds.** Two quick checks at the start of each trick often settle the
question without any search:
- *Quick tricks*: the top cards the player on lead can cash straight away, or
  partner's top cards if the leader has a card in that suit to lead to them. In a
  side suit, cashing stops before any hand that holds trumps runs out of the suit,
  so the count is never too high.
- *Top trumps*: a trump higher than every trump the opponents hold wins the trick
  whenever it is played, and a hand plays each card on a different trick. So a side
  takes at least as many tricks as either of its hands has such trumps. This gives a
  lower bound for the side on lead and an upper bound for the other side.

**7. Move ordering.** Alpha-beta prunes most when the best card is tried first.
Leads: cash a winner; lead low to partner's winner; give partner a ruff; otherwise
lead low from length; try first the suit that won last time this position was
searched. Following: play low when partner is already winning; win as cheaply as
possible; ruff as cheaply as possible; discard low from a long suit. In one measured
full-deal solve, the first card tried settled about 94% of the positions where a single
card was enough to decide.

## Results

All numbers below were measured in a cloud Linux container (one core per run, with
other jobs sharing the machine), so absolute times are rough; the comparisons with DDS
were run side by side on the same machine. Re-run them with
`python -m benchmark.dds_scratch_benchmark ...` (raw results from these runs are in `results/ddsolver/`).

### Correctness against DDS

| Check | Result |
|---|---|
| Full deals: 100 random deals, all 20 contracts each (`verify --deals 100`) | **100/100 deals, 2000/2000 contracts identical to DDS** |
| Mid-play positions: 300 random positions at any point of the play, including mid-trick; every legal card scored (`positions --n 300`) | **300/300 positions identical (1,369 cards scored)** |
| Small positions against a brute-force search, and every technique against every other (`tests/test_ddsolver.py`) | all identical |

### Speed against DDS (the same 100 deals, full 20-contract table)

| | median | mean | 90th percentile | slowest |
|---|---:|---:|---:|---:|
| this solver | 2.6 s | 9.0 s | 12.4 s | 313 s |
| DDS | 0.13 s | 0.23 s | 0.55 s | 2.6 s |

The median deal takes about 19 times as long as with DDS. A quarter of the deals
finish in under a second, but a few hard deals (long, split trump suits; 7 of 100 took
over 20 s) take far longer. DDS has many more refinements: much more detailed move
ordering and quick-trick rules, further bounds, and a tree-structured table.

### What each technique contributes

The same random positions solved with the techniques switched on one at a time
(`ablation`). "Solved" means finished within the node limit; where some positions hit
the limit, the node count shown is a lower bound.

Endgames with 8 cards per hand (30 positions, node limit 20 million):

| Technique (cumulative) | Solved | Mean nodes | Mean time |
|---|---:|---:|---:|
| plain minimax + alpha-beta | 4/30 | over 18.7 million | 416 ms |
| + yes/no (null-window) search | 12/30 | over 14.8 million | 373 ms |
| + move ordering | 27/30 | over 5.5 million | 195 ms |
| + equivalent cards | 30/30 | 347,558 | 14.4 ms |
| + transposition table, exact ranks | 30/30 | 40,378 | 6.9 ms |
| + relative-rank keys | 30/30 | 17,409 | 2.0 ms |
| + partition search | 30/30 | 6,459 | 1.6 ms |
| + quick-trick and top-trump bounds (everything) | 30/30 | 4,941 | 0.7 ms |

All techniques together search at least 3,800 times fewer nodes than plain minimax
with alpha-beta.

Full deals, 13 cards per hand (12 positions: one random strain and opening leader
each, node limit 30 million):

| Technique (cumulative) | Solved | Mean nodes | Mean time |
|---|---:|---:|---:|
| up to and including equivalent cards | 0/12 | over 30 million | 0.6 to 1.2 s before giving up |
| + transposition table, exact ranks | 1/12 | over 28 million | 3.8 s |
| + relative-rank keys | 10/12 | over 18.6 million | 2.6 s |
| + partition search | 12/12 | 4.1 million | 0.89 s |
| + quick-trick and top-trump bounds (everything) | 12/12 | 3.8 million | 0.95 s |

On full deals, partition search is what makes every position solvable. The cheap bounds
remove a further 6% of nodes there (the timing difference is within noise), while on
8-card endgames they more than halve the time.

## How correctness was checked

- **Against DDS on full deals** (`benchmark/dds_scratch_benchmark.py verify`): every
  one of the 20 contracts (5 strains x 4 declarers) of each random deal.
- **Against DDS in the middle of play** (`... positions`): random deals played forward
  with random legal cards to any point, including the middle of a trick; every legal
  card of the player to move is scored and compared with DDS's `solve_board`.
- **Against a brute-force search** (`tests/test_ddsolver.py`): a few lines of plain
  minimax with no pruning at all, on small positions, including the middle of a trick.
- **Every technique against every other** (`tests/test_ddsolver.py` and the ablation):
  all configurations must return the same number of tricks for every position.

## Limitations

- Slower than DDS (see above), and it computes one deal at a time on one CPU core. DDS
  also runs several deals in parallel.
- Memory: about 140 MB of tables, allocated on first use (plus 67 MB for the simpler
  table, only when a comparison configuration without partition search is run).
- The Python wrapper is not thread-safe: the tables are shared by all calls.

## What changed elsewhere because of this

Cross-checking against this solver found two bugs in `cardplay/endgame_solver.py`
(the small solver behind the Analyze button): its memo stored alpha-beta bounds as
if they were exact values, and it keyed absolute trick counts on a position that
doesn't record tricks already won. Both are fixed; see `docs/endgame_solver.md`.
The RL results that used that solver for endgame search ("fifth attempt" in
`docs/rl_training.md`) were measured before the fix and should be re-run.
