# Bidding coverage — what's actually modeled

This is an honest accounting of `bidding/auction_engine.py` (built on top of
`bidding/engine.py`'s opening bids), so "the engine recommends X" comes with
a clear sense of how far to trust it. Everything below is drafted from
first-principles / general public bidding-system knowledge (roughly ACBL
convention-card depth) — no bridge book content was used anywhere.

## Solidly covered

- **Opening bids** (1C/1D/1H/1S/1NT/2C/2NT/preempts/pass) — `bidding/conventions/standard_american_2over1.yaml`, unchanged from the original scaffold.
- **Responses to a 1-level suit opening**, uncontested: pass/simple raise/limit raise/game raise by support+HCP, natural new major up the line, 1NT, 2/1 game-forcing new suit.
- **Responses to 1NT and 2NT**: Stayman, Jacoby transfers, natural raises.
- **Responses to the strong 2C opening**: 2D waiting/negative, simple positive responses.
- **Opener's rebids** after: partner's Pass, Stayman, a transfer, a simple raise, a 1NT response, a 2/1 game-forcing response.
- **Simple overcalls, 1NT overcalls, and takeout doubles** over an opponent's opening.
- **Responses to partner's overcall or takeout double.**
- **Basic negative doubles** (partner opens, RHO overcalls, I have 4+ cards in the unbid major(s) and values).
- **Blackwood mechanics**: 4NT ace-ask and the standard 5C/5D/5H/5S ace-count responses are handled correctly if *you* initiate Blackwood. The bot only initiates it itself under a narrow, conservative heuristic (`should_try_blackwood` in `auction_engine.py`) — it is not a general slam-bidding brain.
- **Every call the engine ever returns is checked against `AuctionState.legal_calls()` before being used** (`AuctionEngine.decide_call`'s safety net) — an internal bug here always degrades to a safe `Pass` with a note, never an illegal call or a crash. This is fuzz-tested (`tests/test_auction_engine.py`) across hundreds of random deals through complete 4-seat auctions.

## Known simplifications (by design, for v1)

- **Second and later rounds of a partnership's auction beyond the specific patterns above** (a second round of competitive bidding, a responder's second bid outside the Stayman/transfer/Blackwood continuations, opener's second rebid) fall through to a generic "pass by default" with an explicit explanation that the auction shape is beyond current coverage. This is the single biggest gap — real auctions often go deeper than this engine tracks.
- **No trump-suit memory across the auction.** Decisions are made from (my hand, the raw call sequence) rather than a richer model of "what trump suit has this partnership agreed on." This mostly shows up in the Blackwood continuation (`_continue_after_blackwood`), which can't sign off in the *actual* agreed suit and defaults to 6NT/Pass instead.
- **No Bergen raises, Jacoby 2NT, reverse-bid rules, fourth-suit-forcing, support doubles, or 1430 RKCB** — the modern refinements on top of the core system aren't modeled.
- **Competitive bidding beyond the first round of intervention** (e.g. a second opponent's bid, balancing decisions after two passes) uses a simplified "look at the most recent opponent bid" heuristic rather than reconstructing the whole competitive picture.
- **Preempt responses and rebids** are a single simple raise-or-pass rule, not the fuller structure (e.g. 2NT ask, new-suit feature-asking) some partnerships use.
- **Suit-quality requirements are not checked** — e.g. the preemptive-opening rule only checks length (7+), not honor content, so it will occasionally recommend a preempt on a suit most players wouldn't preempt with.

## Where this shows up in the app

The game (`game/session.py`) shows the recommended call and its explanation before every one of your calls (`current_bid_suggestion`), and every bot call is logged with its own explanation too, visible by hovering/inspecting the auction history — so when the engine falls back to the generic "beyond coverage, passing" behavior, that's visible in the explanation text rather than silently pretending to have a real answer.
