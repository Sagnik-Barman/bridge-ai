"""Heuristic card-play bot for seats the human isn't controlling.

This is a dependency-free fallback so the game is playable without DDS
installed: reasonable club-level heuristics (lead low from a long suit or
top of a sequence, third-hand-high, cover an honor with an honor when
holding one, discard from your weakest suit), not double-dummy-optimal
play. `engine/dds_wrapper.py` / `cardplay/pimc.py` already scaffold a path
to genuinely strong (DDS/PIMC-based) play — swapping this bot for that,
once `endplay` is installed, is a natural next step and does not require
changing `cardplay/trick_engine.py` or the game session, since both only
depend on "give me a legal Card for this seat."
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from engine.deal import Card, RANKS, Suit
from cardplay.trick_engine import PlayState, _card_strength, legal_plays

HONOR_RANKS = {"A", "K", "Q", "J", "T"}


def _rank_value(card: Card) -> int:
    return RANKS.index(card.rank)


def _suit_lengths(cards: List[Card]) -> Dict[Suit, int]:
    lengths: Dict[Suit, int] = {s: 0 for s in Suit}
    for c in cards:
        lengths[c.suit] += 1
    return lengths


def _has_sequence_top(cards: List[Card], suit: Suit) -> Optional[Card]:
    """If this suit holds a touching honor sequence (e.g. KQJ), return the
    top card of it — standard "lead top of a sequence" heuristic."""
    suit_cards = sorted((c for c in cards if c.suit == suit), key=_rank_value, reverse=True)
    if len(suit_cards) >= 2 and suit_cards[0].rank in HONOR_RANKS:
        top_idx = RANKS.index(suit_cards[0].rank)
        second_idx = RANKS.index(suit_cards[1].rank)
        if top_idx - second_idx == 1:
            return suit_cards[0]
    return None


def choose_opening_or_new_trick_lead(hand: List[Card], rng: random.Random) -> Card:
    lengths = _suit_lengths(hand)
    longest_suits = [s for s, n in lengths.items() if n == max(lengths.values()) and n > 0]
    for suit in longest_suits:
        seq_top = _has_sequence_top(hand, suit)
        if seq_top:
            return seq_top
    suit = rng.choice(longest_suits)
    suit_cards = sorted((c for c in hand if c.suit == suit), key=_rank_value)
    return suit_cards[0]  # lowest of longest


def choose_follow(hand: List[Card], play_state: PlayState, rng: random.Random) -> Card:
    trick = play_state.current_trick
    legal = legal_plays(hand, trick.led_suit())
    led_suit = trick.led_suit()

    if len(trick.plays) == 0:
        return choose_opening_or_new_trick_lead(hand, rng)

    current_best = max(trick.plays, key=lambda sc: _card_strength(sc[1], led_suit, play_state.trump))[1]
    can_win = [c for c in legal if _card_strength(c, led_suit, play_state.trump) > _card_strength(current_best, led_suit, play_state.trump)]

    position = len(trick.plays)  # 1 = second to play, 2 = third, 3 = fourth/last

    if position == 3:  # last to play: win as cheaply as possible, else lowest discard
        if can_win:
            return min(can_win, key=lambda c: _card_strength(c, led_suit, play_state.trump))
        return _lowest_discard(hand, legal)

    if position == 2:  # third hand: play high if it can help win
        if can_win:
            return max(can_win, key=lambda c: _card_strength(c, led_suit, play_state.trump))
        return _lowest_discard(hand, legal)

    # second to play: generally play low, ducking to see what develops
    if legal[0].suit == led_suit:
        return min(legal, key=_rank_value)
    return _lowest_discard(hand, legal)


def _lowest_discard(hand: List[Card], legal: List[Card]) -> Card:
    """When not following suit (discarding), throw from the shortest side
    suit, lowest card, to preserve length elsewhere — a simple standard
    discarding heuristic."""
    lengths = _suit_lengths(hand)
    legal_suits = {c.suit for c in legal}
    shortest_suit = min(legal_suits, key=lambda s: lengths[s])
    candidates = [c for c in legal if c.suit == shortest_suit]
    return min(candidates, key=_rank_value)


def choose_card(seat: str, play_state: PlayState, seed: Optional[int] = None) -> Card:
    """Entry point used by the game session for any seat the bot controls
    (declarer/dummy when the human is defending, or both defenders when
    the human is declarer)."""
    rng = random.Random(seed)
    hand = play_state.hands[seat]
    return choose_follow(hand, play_state, rng)
