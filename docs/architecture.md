# Architecture notes

## Pipeline overview

```
deal_random()              engine/deal.py
      │
      ▼
BiddingEngine.opening_bid  bidding/engine.py  ── reads ──▶ bidding/conventions/*.yaml
      │  (auction, eventually a contract + declarer)
      ▼
evaluate_contract_pimc()   cardplay/pimc.py
      │  samples unseen hands, for each sample:
      ▼
solve_double_dummy()       engine/dds_wrapper.py  ── calls ──▶ endplay ── wraps ──▶ DDS (C library)
```

## Design decisions and why

- **DDS via endplay, not raw ctypes**: endplay is actively maintained,
  ships prebuilt binaries for common platforms, and already understands
  PBN. Wrapping DDS ourselves would duplicate that work. `engine/dds_wrapper.py`
  isolates the endplay-specific calls so we could swap bindings later
  without touching `cardplay/` or `bidding/`.

- **Bidding rules as YAML, not code**: keeps convention logic declarative
  and inspectable/editable without touching `bidding/engine.py`. The engine
  itself is a generic rule-matcher; all Standard American 2/1-specific
  knowledge lives in `conventions/standard_american_2over1.yaml`. This also
  makes it straightforward to add alternate systems (e.g. a Precision Club
  variant) later as a second YAML file.

- **PIMC as sampling + DDS + aggregation**: this is the standard approach
  used by strong bridge-playing programs for the card-play phase, since
  full-information solving (DDS) is fast and reliable but the real game is
  imperfect-information. Sampling many "determinizations" (full deals
  consistent with what's known) and double-dummy solving each is a tractable
  approximation. The current scaffold samples uniformly at random from the
  unseen cards — the next real improvement is constraining samples using
  auction-implied information (HCP ranges, suit lengths) once the bidding
  engine exposes that as structured output rather than just a final bid string.

- **RL is deliberately unscoped for now**: self-play RL for an
  imperfect-information game like bridge is a substantial research project
  in its own right (see `rl/README.md`). Scaffolding it prematurely would
  either produce misleading placeholder code or lock in premature
  architectural decisions before the base rules-based pipeline is validated.

## Known gaps / TODOs

- `bidding/engine.py` only decides an opening bid; it does not yet track a
  full auction (responses, rebids, competitive bidding). `Auction.run_auction_stub`
  is a placeholder that assumes the first hand with a non-pass opening
  becomes declarer and everyone else passes — not remotely a real auction.
- `cardplay/pimc.py` samples completely unconstrained random deals for the
  unseen hands; it does not yet condition on the auction.
- No connection yet between the bidding engine's output (a contract) and
  the PIMC evaluator's input (currently called with a hardcoded strain/seat
  in the `__main__` block of each module).
- No PBN dataset ingestion yet; `data/` is empty aside from `.gitkeep`.
