# Benchmarking against real tournament play

`benchmark/` compares this project's engines (heuristic bot, DDS/PIMC, RL
network, the from-scratch endgame solver) against what real players
actually did on real boards from major bridge championships — not just
random deals.

## Why this is different from the copyrighted-book restriction

This project has never used, and will never use, the uploaded copyrighted
book ("Adventures in Card Play", Ottlik & Kelsey) as source material,
code basis, or training data — that restriction stands, unconditionally.

Real tournament PBN files are a different kind of thing entirely: they're
a factual record of what happened at a real table — who held which
cards, what was bid, what was played, what the result was. That's data
about an event, like a chess game's move list or a sports box score, not
anyone's original written analysis or commentary. Nothing here is trained
on these files either (they're used only for after-the-fact benchmarking,
never fed into `rl/train.py`), but even if it were, the copyright analysis
would be completely different from the book.

## Where to get real PBN files

- **[World Championship & NABC Finals in PBN format](https://www.computerbridge.se/finals-world-championship-in-pbn/)**
  — zipped PBN files for recent Bermuda Bowl, Venice Cup, Vanderbilt,
  Spingold, Soloway KO, and Swedish Cup events. Good for a modest,
  recent, high-quality set.
- **[The Vugraph Project](https://www.sarantakos.com/bridge/vugraph.html)**
  — a much larger historical archive (1955–2013).
- **[BBO Vugraph Archive](https://www.thebridgechannel.se/bbo-vugraph-archive/)**
  — ongoing online tournament vugraph records.

Download a `.pbn` file (or unzip an archive of them) into
`data/real_deals/` — that directory is gitignored (like the rest of
`data/`), so downloaded files never get committed to the repo; only the
code that reads them does.

## Before trusting this on a real file: run the diagnostic first

`benchmark/pbn_loader.py` (the endplay-specific PBN parsing layer) was
written against endplay's *documented* API, but — exactly like
`engine/dds_wrapper.py`'s DDTable indexing bug earlier in this project —
it has never actually been run against a real install, because `endplay`
can't be installed in the sandbox this was built in. Before relying on
it:

```
python scripts/diag_pbn.py data/real_deals/whatever.pbn
```

This prints exactly what endplay's parser hands back for a real file —
the `Contract`, `Card`, `Deal` attribute values and types — and then
tries `benchmark/pbn_loader.py` against the same file so you can see
whether it converted boards successfully. Send me the output; if any
assumption in `pbn_loader.py` is wrong (e.g. an attribute is named
differently than expected), it gets fixed against real data, the same
way the DDS wrapper bug did.

## Running the benchmark

Once `scripts/diag_pbn.py` looks right:

```
python -m benchmark.run_expert_benchmark data/real_deals/*.pbn
python -m benchmark.run_expert_benchmark data/real_deals/bermuda_bowl_2019.pbn --rl-checkpoint rl/checkpoints/latest.pt
```

## How the comparison works — and its honest limits

For each real board with a complete 52-card play record and a known
contract, the benchmark replays the *exact* sequence of cards as they
were really played. At every non-forced decision point (more than one
legal card available), before playing the real card, it asks each
available engine what *it* would have played in that exact, real
position — then plays the real card regardless of what any engine
suggested, and moves on. This "move-matching" style keeps the comparison
meaningful trick after trick, since every engine is always being asked
about a position that genuinely occurred, never one built from an earlier
wrong guess (letting engines branch into their own hypothetical lines
would compound differences and stop meaning anything after a few tricks).

What this measures, honestly: how often each engine's move *matches* the
real expert's move — not whether the expert was right. A real expert can
make a genuine mistake, take an unusual, table-feel-based safety line
that isn't double-dummy-optimal, or make a lead based on partnership
agreements/bidding inferences no engine here has access to (bidding is
entirely out of scope for `benchmark/`, deliberately — only the play
phase is compared). A low match rate against a specific hand doesn't
automatically mean an engine played worse; a genuinely strong engine that
disagrees with a human on a close, table-feel decision is not "wrong" the
way disagreeing with a real double-dummy answer would be. Treat the
match-rate numbers as a rough proxy for "plays like a strong human" for
resume purposes, not as ground truth the way `engine/dds_wrapper.py`'s
double-dummy analysis is.
