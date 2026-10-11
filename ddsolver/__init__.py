"""A from-scratch double-dummy solver (C++ core in src/solver.cpp, loaded with ctypes).

    from ddsolver import dd_table_pbn
    table, nodes = dd_table_pbn("N:AKQ2.T98.5432.K4 J86.AK76.KJ.QJ98 T95.QJ32.AQ96.65 743.54.T87.AT732")
    table["N"]["NT"]   # tricks North-South take in no trump with North declaring

The compiled library lives in ddsolver/lib/. If it is missing, it is built
automatically the first time it is needed (see ddsolver/build.py; the only
requirement is `pip install ziglang`, or g++ / clang++ on the PATH).

Flags switch individual techniques on or off, so their effect can be measured
(benchmark/dds_scratch_benchmark.py). ALL is the normal setting.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

TT = 1          # transposition table
RELRANK = 2     # relative-rank table keys
EQUIV = 4       # equivalent-card pruning
QUICK = 8       # quick-trick and top-trump bounds
ORDER = 16      # move ordering
MINIMAX = 32    # baseline: plain minimax alpha-beta
PARTITION = 64  # table entries ignore low cards that never mattered
ALL = TT | RELRANK | PARTITION | EQUIV | QUICK | ORDER

SEATS = ["N", "E", "S", "W"]
SUITS = ["S", "H", "D", "C"]             # internal suit order
STRAINS = ["S", "H", "D", "C", "NT"]     # internal strain order (trump index 4 = NT)
RANKS = "23456789TJQKA"

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


class SolverError(RuntimeError):
    pass


def lib_filename() -> str:
    if sys.platform.startswith("win"):
        return "dds_scratch.dll"
    if sys.platform == "darwin":
        return "libdds_scratch.dylib"
    return "libdds_scratch.so"


def _load():
    global _lib
    if _lib is not None:
        return _lib
    path = os.path.join(_HERE, "lib", lib_filename())
    if not os.path.exists(path):
        from ddsolver.build import build
        build()
    lib = ctypes.CDLL(path)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    ip = ctypes.POINTER(ctypes.c_int)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.dds_solve.argtypes = [u16p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int,
                              ctypes.c_uint64, u64p]
    lib.dds_solve.restype = ctypes.c_int
    lib.dds_score_cards.argtypes = [u16p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int,
                                    ctypes.c_uint64, ip, ip, u64p]
    lib.dds_score_cards.restype = ctypes.c_int
    lib.dds_table.argtypes = [u16p, ctypes.c_int, ctypes.c_uint64, ip, u64p]
    lib.dds_table.restype = ctypes.c_int
    lib.dds_clear_tt.argtypes = []
    lib.dds_clear_tt.restype = None
    _lib = lib
    return lib


def is_available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ hands
def parse_pbn(pbn: str) -> List[List[int]]:
    """'N:AKQ.T98.76.5432 ...' -> masks[player][suit] (player N,E,S,W; suit S,H,D,C)."""
    first, _, rest = pbn.strip().partition(":")
    hands = rest.split()
    if first.upper() not in SEATS or len(hands) != 4:
        raise ValueError(f"not a PBN deal: {pbn!r}")
    start = SEATS.index(first.upper())
    masks = [[0] * 4 for _ in range(4)]
    for i, h in enumerate(hands):
        suits = h.split(".")
        if len(suits) != 4:
            raise ValueError(f"hand {h!r} needs 4 suits")
        for s, ranks in enumerate(suits):
            for ch in ranks.upper():
                masks[(start + i) % 4][s] |= 1 << RANKS.index(ch)
    return masks


def _hands_array(masks: Sequence[Sequence[int]]):
    arr = (ctypes.c_uint16 * 16)()
    for p in range(4):
        for s in range(4):
            arr[p * 4 + s] = masks[p][s]
    return arr


def _trick_array(trick: Sequence[Tuple[int, int]]):
    arr = (ctypes.c_int * 4)()
    for i, (s, r) in enumerate(trick):
        arr[i] = s * 16 + r
    return arr


# ------------------------------------------------------------------ core calls
def solve(masks, trump: int, leader: int, trick: Sequence[Tuple[int, int]] = (),
          flags: int = ALL, node_limit: int = 0) -> Tuple[int, int]:
    """N-S tricks from this position on (the current trick included), and nodes searched.

    trump: 0..3 = S,H,D,C, 4 = no trump. leader: the player (0..3 = N,E,S,W) who led the
    current trick; trick: the (suit, rank) cards already played to it, in order.
    Raises SolverError if node_limit (0 = none) was reached first.
    """
    lib = _load()
    nodes = ctypes.c_uint64(0)
    v = lib.dds_solve(_hands_array(masks), trump, leader, len(trick), _trick_array(trick), flags,
                      node_limit, ctypes.byref(nodes))
    if v < 0:
        raise SolverError("bad position or node limit reached")
    return v, nodes.value


def score_cards(masks, trump: int, leader: int, trick: Sequence[Tuple[int, int]] = (),
                flags: int = ALL, node_limit: int = 0) -> Tuple[List[Tuple[int, int, int]], int]:
    """For every legal card of the player to act: (suit, rank, N-S tricks if it is played)."""
    lib = _load()
    cards = (ctypes.c_int * 13)()
    ns = (ctypes.c_int * 13)()
    nodes = ctypes.c_uint64(0)
    n = lib.dds_score_cards(_hands_array(masks), trump, leader, len(trick), _trick_array(trick), flags,
                            node_limit, cards, ns, ctypes.byref(nodes))
    if n < 0:
        raise SolverError("bad position or node limit reached")
    return [(cards[i] >> 4, cards[i] & 15, ns[i]) for i in range(n)], nodes.value


def dd_table(masks, flags: int = ALL, node_limit: int = 0) -> Tuple[Dict[str, Dict[str, int]], int]:
    """Full double-dummy table {declarer: {strain: tricks}} (strains C, D, H, S, NT) and nodes."""
    lib = _load()
    out = (ctypes.c_int * 20)()
    nodes = ctypes.c_uint64(0)
    if lib.dds_table(_hands_array(masks), flags, node_limit, out, ctypes.byref(nodes)) < 0:
        raise SolverError("bad deal or node limit reached")
    table = {seat: {} for seat in SEATS}
    for st in range(5):
        for d in range(4):
            table[SEATS[d]][STRAINS[st]] = out[st * 4 + d]
    return table, nodes.value


def dd_table_pbn(pbn: str, flags: int = ALL, node_limit: int = 0):
    return dd_table(parse_pbn(pbn), flags, node_limit)


def clear_tt() -> None:
    _load().dds_clear_tt()


# ------------------------------------------------------------------ project types
def masks_from_cards(hands: Dict[str, list]) -> List[List[int]]:
    """{seat: [engine.deal.Card, ...]} -> masks[player][suit]."""
    masks = [[0] * 4 for _ in range(4)]
    for seat, cards in hands.items():
        p = SEATS.index(seat)
        for c in cards:
            masks[p][SUITS.index(c.suit.value)] |= 1 << RANKS.index(c.rank)
    return masks


def solve_double_dummy(deal) -> Dict[str, Dict[str, int]]:
    """Drop-in for engine.dds_wrapper.solve_double_dummy: {seat: {strain: tricks}}."""
    masks = masks_from_cards({s: h.cards for s, h in deal.hands.items()})
    table, _ = dd_table(masks)
    return table


def _play_state_args(ps):
    masks = masks_from_cards(ps.hands)
    trump = 4 if ps.trump is None else SUITS.index(ps.trump.value)
    leader = SEATS.index(ps.current_trick.leader)
    trick = [(SUITS.index(c.suit.value), RANKS.index(c.rank)) for _, c in ps.current_trick.plays]
    return masks, trump, leader, trick


def solve_play_state(ps, flags: int = ALL, node_limit: int = 0) -> Tuple[int, int]:
    """N-S tricks from a cardplay.trick_engine.PlayState onward, and nodes searched."""
    masks, trump, leader, trick = _play_state_args(ps)
    return solve(masks, trump, leader, trick, flags, node_limit)


def score_play_state(ps, flags: int = ALL, node_limit: int = 0):
    """[(Card, N-S tricks from here if that card is played now)] for the player to act."""
    from engine.deal import Card, Suit
    masks, trump, leader, trick = _play_state_args(ps)
    scored, nodes = score_cards(masks, trump, leader, trick, flags, node_limit)
    return [(Card(Suit(SUITS[s]), RANKS[r]), v) for s, r, v in scored], nodes


def best_card(ps, flags: int = ALL):
    """(best Card for the player to act, N-S tricks from here after it, nodes)."""
    scored, nodes = score_play_state(ps, flags)
    ns_to_act = ps.next_to_play in ("N", "S")
    card, v = (max if ns_to_act else min)(scored, key=lambda cv: cv[1])
    return card, v, nodes
