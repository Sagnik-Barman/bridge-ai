# bridge-ai

An AI system that plays contract bridge. Two ways to use it:

1. **A full web app to play a long bridge session against the computer** (`webapp/`) — standard board/dealer/vulnerability rotation, full guided bidding (a suggested call + the reasoning behind it, before every one of your calls), IMP-flavored scoring, and undo (step back one call/card, or restart the whole board).
2. **The underlying engine components** (double-dummy solving, PIMC card play, a from-scratch bidding system) as a standalone library — a CLI opening-bid tutor and a demo pipeline script are also included.

No bridge book content (text, hand analyses, or prose explanations from copyrighted instructional books) is used anywhere in this codebase or as training data. Bidding rules are drafted from first principles / public system documentation (WBF system descriptions, ACBL convention cards) as structured data, not transcribed from books. See `docs/references.md` for the only source materials this project draws on, and `docs/bidding_coverage.md` for an honest account of exactly what the bidding engine does and doesn't cover.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`endplay` (the DDS binding) ships prebuilt binaries for common platforms; if installation fails on your platform, see [endplay's install notes](https://github.com/dominicprice/endplay#installation). Everything else — the whole game, bidding, and scoring — works without it; the only thing `endplay` unlocks is the IMPs-vs-double-dummy-par comparison shown after each board, and a real solver for `pipeline.py`'s card-play demo.

## Playing the full game

```bash
python webapp/app.py
```

Then open `http://127.0.0.1:5000`. What you get:

- **A long session of boards**, numbered from 1, with the standard duplicate dealer/vulnerability cycle (see `game/board.py`) — dealer rotates N/E/S/W and vulnerability follows the usual 16-board pattern.
- **Your seat rotates each board** (N, E, S, W, N, ...), so a long session works you through all four positions, as requested. When your rotated seat happens to be the dummy on a given board, you have nothing to do during card play on that board (per the real rules of bridge, declarer plays dummy's cards) — you'll still bid on it.
- **A suggested call and its explanation before every one of your calls** — the "full guide through bidding" — plus feedback right after you call showing whether you matched the suggestion.
- **Bot-controlled seats** for everyone you're not currently playing: opponents always, and your partner too except when you're declarer (then you play both hands, as normal).
- **Two selectable opponent strengths, picked from "New game"**: the default club-level heuristic bot, **Play against DDS** (real double-dummy/PIMC-informed leads once `endplay` is installed — see `cardplay/dds_player.py`), and **Play against RL** (a neural network trained on double-dummy-labelled positions that plays only on what it can see; it beats the heuristic bot by about 1.3-1.4 tricks per board, see `docs/rl_training.md`). Both fall back transparently to the heuristic bot when their dependency/checkpoint isn't available, and the top bar always shows which mode is active and why.
- **Analyze button**, any time it's your turn during card play: asks the heuristic bot, the DDS/PIMC solver, the RL network, and a from-scratch alpha-beta endgame solver (small endgames only — see `docs/endgame_solver.md`) what they'd play, and shows all four side by side with whether they agree — see `game/session.py`'s `analyze_current_position()`.
- **IMP-flavored scoring**: after each board, your result is compared to the double-dummy best-possible result for your side (needs `endplay` installed — see Setup) and expressed as an IMP swing, alongside the plain contract score.
- **Undo**: step back one call or one card at a time, or restart the whole board from the same deal.
- **Save/resume, including mid-board**: the current game autosaves after every action (`saves/_autosave.json`) and reloads automatically on startup — closing the app and reopening it later resumes exactly where you left off, mid-auction or mid-trick. The "Saved games" button also lets you save a checkpoint under a name and come back to it later even after continuing to play past it, or load/delete any of your named saves — see `game/persistence.py`.

This is a single-user, single-process local tool. See `docs/bidding_coverage.md` before trusting the suggestions on anything beyond a fairly standard auction — deep competitive sequences and slam bidding are intentionally simplified in v1.

## Architecture

```
bridge-ai/
├── ddsolver/           # A full double-dummy solver written from scratch in C++ (see docs/dds_from_scratch.md)
│   ├── src/solver.cpp  # The search: bitmasks, yes/no search, equivalent cards, partition-search table, bounds, ordering
│   ├── __init__.py     # Python wrapper (ctypes): full tables, single positions, best card for a PlayState
│   ├── build.py        # Rebuilds the library (pip install ziglang, or any C++ compiler)
│   └── lib/            # Prebuilt dds_scratch.dll (Windows x64) and libdds_scratch.so (Linux x64)
├── engine/             # Core hand/deal representation + DDS wrapper
│   ├── deal.py         # Card, Hand, Deal classes; random dealing; card string parsing
│   └── dds_wrapper.py  # Thin wrapper around a DDS Python binding (endplay)
├── bidding/
│   ├── engine.py       # Opening-bid decisions from the YAML convention file
│   ├── calls.py        # Call legality, whose-turn, declarer/contract determination
│   ├── auction_engine.py # Full auction: responses, rebids, competitive bidding, conventions
│   ├── tutor.py         # CLI opening-bid trainer
│   └── conventions/
│       └── standard_american_2over1.yaml
├── cardplay/
│   ├── trick_engine.py # Follow-suit legality, trick winner, one board's play state
│   ├── bot_player.py   # Heuristic card-play bot (no DDS dependency) — the default opponent
│   ├── dds_player.py   # DDS/PIMC-based leads for the "Play against DDS" opponent
│   ├── endgame_solver.py # From-scratch minimax + alpha-beta + memoization, small endgames only — see docs/endgame_solver.md
│   └── pimc.py          # PIMC (Perfect Information Monte Carlo) sampling helpers
├── rl/                  # Self-play RL card-play agent (see docs/rl_training.md)
│   ├── oracle.py           # v2: exact double-dummy value of every legal card (DDS, or ddsolver/)
│   ├── features_v2.py      # v2: trump-canonical, relative-rank encoding; 'visible' = only what that player sees
│   ├── network_v2.py       # v2: residual MLP with policy, per-card value and state value heads
│   ├── datagen.py          # v2: oracle-labelled positions from mixed / on-policy (DAgger) games
│   ├── train_v2.py         # v2: training (set cross-entropy + exact-advantage regret + value losses)
│   ├── agent_v2.py         # v2: pick a card with a v2 network
│   ├── evaluate_v2.py      # v2: benchmark vs heuristic / v1 / DDS, opt% and tricks thrown
│   ├── features.py       # Fixed-size state encoding fed to the network
│   ├── network.py        # Policy+value MLP (PyTorch, CUDA-capable) — see docs/rl_training.md
│   ├── network_numpy.py  # Same architecture, hand-rolled in NumPy — zero-dependency reference/fallback
│   ├── env.py             # Full-information self-play environment
│   ├── generate_labels.py # Builds an imitation-warm-start dataset (endgame solver + DDS leads)
│   ├── pretrain.py         # Supervised warm start on that dataset, before self-play
│   ├── train.py           # Self-play training loop (CLI)
│   ├── evaluate.py         # Evaluate a checkpoint vs. the heuristic bot / DDS par
│   ├── rl_player.py        # PIMC-wrapped inference for the "Play against RL" opponent
│   └── checkpoints/        # Trained weights land here by default (gitignored)
├── benchmark/             # Benchmark engines against real tournament play (see docs/real_deals.md)
│   ├── expert_benchmark.py # Replay real boards, compare engine suggestions to the actual card played (endplay-independent)
│   ├── pbn_loader.py       # endplay-specific: turns real .pbn files into ExpertBoard objects
│   ├── dds_scratch_benchmark.py # ddsolver vs DDS: full tables, mid-play positions, technique-by-technique ablation
│   └── run_expert_benchmark.py # CLI entry point
├── scripts/
│   ├── diag_pbn.py          # Run against a real .pbn file before trusting pbn_loader.py — see docs/real_deals.md
│   └── rl_sanity_check.py   # Synthetic sanity check for the RL training loop — see docs/rl_training.md
├── game/
│   ├── board.py         # Standard dealer/vulnerability rotation
│   ├── scoring.py        # Duplicate scoring table + IMP scale
│   ├── session.py        # Ties it all together: board sequencing, undo, scoreboard
│   └── persistence.py    # Save/load a GameSession to/from JSON, mid-board included
├── webapp/
│   ├── app.py            # Flask API + page route
│   ├── templates/index.html
│   └── static/{app.js,style.css}
├── saves/                # Autosave + named saved games, JSON (gitignored)
├── data/                 # PBN hand records, generated datasets (gitignored)
├── tests/                # Unit + fuzz tests (100+ tests; `python run_tests.py` or pytest)
├── docs/
│   ├── architecture.md
│   ├── bidding_coverage.md  # Honest scope of the bidding engine
│   ├── rl_training.md       # How the RL agent works and how to train it yourself
│   ├── endgame_solver.md    # The from-scratch alpha-beta solver + the Analyze feature; honest scope + game theory notes
│   ├── dds_from_scratch.md  # The from-scratch full-deal double-dummy solver: techniques, verification, results
│   ├── real_deals.md        # Benchmarking against real tournament play — sourcing, honest scope, how to run it
│   └── references.md        # Open/free resources only — links, no reproduced content
├── pipeline.py            # End-to-end demo: deal -> bid -> play -> report
└── requirements.txt
```

## Stack

- **Python 3.10+** for everything.
- **Flask** for the local web app.
- **DDS (Double Dummy Solver)** — the C/C++ library at [dds-bridge/dds](https://github.com/dds-bridge/dds), accessed via [endplay](https://github.com/dominicprice/endplay). Optional (see above); powers the "Play against DDS" opponent.
- **PyTorch** — the RL card-play agent's network and training loop (`rl/network.py`), CUDA-capable; see `docs/rl_training.md` for GPU/VS Code setup. A dependency-free NumPy version (`rl/network_numpy.py`, hand-rolled forward/backward pass) is kept as a reference/fallback from when this was built in an environment with no PyTorch available.
- **PyYAML** for opening-bid convention rules as structured data.
- **pytest** (or the dependency-free `python run_tests.py`) for tests.

## Running the tests

```bash
python run_tests.py     # no dependencies beyond PyYAML/Flask needed
# or, if you have pytest installed:
pytest
```

## Other entry points

```bash
python pipeline.py          # deal -> bid -> play -> report, one hand
python -m bidding.tutor     # CLI opening-bid-only trainer with rule explanations
```

## Current state

- [x] Full web app: long session, board/vulnerability rotation, seat rotation, guided bidding, card play, IMP scoring, undo (`webapp/`, `game/`)
- [x] Full 2/1 auction engine: responses, opener rebids, Stayman, Jacoby transfers, Blackwood, overcalls, takeout/negative doubles — see `docs/bidding_coverage.md` for exact scope and known gaps
- [x] Trick-play engine + heuristic bot (no DDS dependency required to play)
- [x] Standard duplicate scoring + IMP scale, verified against known reference values
- [x] `engine/dds_wrapper.py` — real double-dummy solving once `endplay` is installed
- [x] `cardplay/dds_player.py` — DDS/PIMC-informed leads wired in as the "Play against DDS" opponent (honest scope: exact at trick-lead boundaries, heuristic mid-trick — see the module docstring)
- [x] `rl/` — a genuine, from-scratch (NumPy + PyTorch/CUDA) self-play RL card-play agent: state encoding, a policy+value network (implemented twice, hand-rolled NumPy and PyTorch), an actor-critic training loop with opponent diversity and per-trick reward shaping, and an evaluation script benchmarking it against the heuristic bot and DDS par. Honest result: three training configurations were tried and diagnosed in turn, a synthetic sanity check confirmed the training loop itself is correct, and the agent still plateaus below the heuristic bot (~5.85-5.91 vs ~6.29-6.34 tricks/board) — see `docs/rl_training.md` for the full story, the diagnostic process, and how to train one yourself
- [x] `rl/generate_labels.py` + `rl/pretrain.py` — a fourth attempt at the plateau above: a supervised imitation warm start (behavior cloning on exact endgame-solver labels + DDS-informed trick-lead labels) run before self-play, with its own honest scope caveat (it can't label the messy middle of the hand). Result: a fourth null result, 5.86 vs 6.40 tricks/board — essentially identical to the third attempt, strengthening the case for a structural ceiling rather than a fixable training-loop bug — see `docs/rl_training.md`'s "A fourth attempt" section for the full run
- [x] `rl/evaluate.py --endgame-search-limit` — a fifth attempt, and the first that changes decision-time lookahead instead of the training loop: layers the exact `cardplay/endgame_solver.py` search on top of a policy whenever few enough cards remain. Two separate, honest findings, not one: (1) the search itself is a real win, +0.79-0.86 tricks/board, reproduced on two 300-deal runs; (2) a credit-attribution control (the identical search on the plain heuristic bot instead of the network) showed the heuristic gains at least as much from it, and heuristic+search (7.14) beats RL+search (6.59-6.63) outright — so the trained policy still doesn't outperform the heuristic, with or without search. Measured only in the full-information evaluation harness (not yet wired into real imperfect-information gameplay) — see `docs/rl_training.md`'s "A fifth attempt" section for the full numbers and both runs
- [x] **v2 RL agent (October 2026)**: trained on 3.4M positions labelled by an exact double-dummy oracle (DDS, cross-checked against this project's own `ddsolver/`), with a trump-canonical relative-rank encoding and an observation limited to what the player can see. 300-deal benchmark: **+1.34 tricks/board over the heuristic bot as declarer, -1.40 as defender** (paired s.e. 0.10); +1.42 / -1.25 on a second, unseen set of 300 deals. Pipeline reproduced end to end on a second machine. The earlier 2.2M-position model scored +1.17 / -1.28. Against the DDS/PIMC 'Hard' opponent (2.2M model): +0.87 as declarer, -0.70 as defender. The v1 REINFORCE agent was -0.57 on the same deals. Found and fixed along the way: PIMC sampling impossible deals (played cards back in the pool), the heuristic bot depending on Python's hash seed, and an endgame-solver overflow. See `docs/rl_training.md`, "v2"
- [x] Selectable opponents in the web app ("New game" → Standard bot / DDS / RL), each falling back transparently when its dependency/checkpoint isn't available
- [x] `ddsolver/` — a full double-dummy solver written from scratch in C++ (yes/no alpha-beta search, equivalent-card pruning, a partition-search transposition table, quick-trick and top-trump bounds, move ordering), identical to DDS on 2,000/2,000 contracts of 100 random deals and on 300 mid-play positions; slower than DDS (median 2.6 s vs 0.13 s for a full 20-contract table) — see `docs/dds_from_scratch.md` for the numbers and a technique-by-technique ablation
- [x] `cardplay/endgame_solver.py` — from-scratch minimax + alpha-beta + transposition-table search, exact on small endgames, cross-checked against real DDS and against `ddsolver/` in tests — see `docs/endgame_solver.md` for honest scope. Two memo bugs (bounds cached as exact values, absolute trick counts under a key that ignores tricks already won) were found by that cross-check and fixed; results that used it before the fix (the RL "fifth attempt" numbers above) need re-running
- [x] **Analyze** button in the web app: combines the heuristic bot, DDS/PIMC, RL, and the endgame solver into one side-by-side view (`game/session.py`'s `analyze_current_position()`, `/api/analyze`)
- [x] `benchmark/` — benchmarks every engine against real recorded play from actual championship boards (move-matching against the real card played, not just random deals) — see `docs/real_deals.md`. The endplay-specific PBN-parsing layer (`benchmark/pbn_loader.py`) hasn't been verified against a real install yet (same situation `engine/dds_wrapper.py` was in before its DDTable bug was found) — run `scripts/diag_pbn.py` against a real file first
- [x] `bidding/tutor.py` — standalone opening-bid CLI trainer
- [x] A trained RL checkpoint ships with the repo: `rl/checkpoints/latest.pt` is the v2 agent (retrain with the pipeline in `docs/rl_training.md`)
- [ ] Deeper competitive-auction and slam-bidding coverage (see `docs/bidding_coverage.md`)

## Next steps

1. Install `endplay` to unlock real double-dummy solving, the IMP-vs-par comparison, and the "Play against DDS" opponent's PIMC leads.
2. The trained v2 RL checkpoint ships as `rl/checkpoints/latest.pt`. Benchmark it with `python -m rl.evaluate_v2 --v2 rl/checkpoints/latest.pt --defence`, or retrain it with the pipeline in `docs/rl_training.md` ("v2").
3. Extend `bidding/auction_engine.py`'s deeper-round fallback logic per `docs/bidding_coverage.md`'s gap list, as you run into them in real play.
