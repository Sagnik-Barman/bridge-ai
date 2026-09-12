"""Thin wrapper around a DDS (Double Dummy Solver) Python binding.

All endplay-specific (or, later, raw-ctypes-specific) calls are isolated
here so the rest of the codebase depends only on this module's small
interface, not on a particular binding.

DDS project: https://github.com/dds-bridge/dds
Binding used: https://github.com/dominicprice/endplay (pip install endplay)
"""
from __future__ import annotations

from typing import Dict

from engine.deal import Deal, Suit

STRAINS = ["C", "D", "H", "S", "NT"]


class DDSUnavailableError(RuntimeError):
    """Raised when the endplay/DDS binding is not installed or fails to load."""


def _require_endplay():
    try:
        import endplay  # noqa: F401
        from endplay.dds import calc_dd_table
        from endplay.types import Deal as EndplayDeal, Denom, Player

        return calc_dd_table, EndplayDeal, Denom, Player
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise DDSUnavailableError(
            "endplay is not installed. Run `pip install endplay` to enable "
            "double-dummy solving."
        ) from exc


# `endplay`'s DDTable is indexed as table[Denom, Player] -> int (a single
# tuple key, not chained table[strain][seat]) and has no get_score method
# in the installed version (0.5.12) — confirmed by hand against a live
# install, since this project's own dev environment can't install endplay
# at all. See engine/dds_wrapper.py's git history / docs/rl_training.md
# for context on why this had to be debugged live rather than up front.
_STRAIN_TO_DENOM_NAME = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs", "NT": "nt"}
_SEAT_TO_PLAYER_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}


def solve_double_dummy(deal: Deal) -> Dict[str, Dict[str, int]]:
    """Return the full double-dummy table: {seat: {strain: tricks}}.

    This is the "perfect information" ground truth used both directly
    (double-dummy analysis) and as the inner solve step of PIMC, where
    many sampled deals are each solved this way and the results averaged.
    """
    calc_dd_table, EndplayDeal, Denom, Player = _require_endplay()

    ep_deal = EndplayDeal(deal.pbn())
    table = calc_dd_table(ep_deal)

    result: Dict[str, Dict[str, int]] = {}
    for seat in ["N", "E", "S", "W"]:
        result[seat] = {}
        player = getattr(Player, _SEAT_TO_PLAYER_NAME[seat])
        for strain in STRAINS:
            denom = getattr(Denom, _STRAIN_TO_DENOM_NAME[strain])
            result[seat][strain] = table[denom, player]
    return result


def best_contract_for_seat(deal: Deal, seat: str) -> tuple[str, int]:
    """Return (strain, tricks) of the best double-dummy contract for `seat`."""
    table = solve_double_dummy(deal)
    tricks_by_strain = table[seat]
    best_strain = max(tricks_by_strain, key=tricks_by_strain.get)
    return best_strain, tricks_by_strain[best_strain]


if __name__ == "__main__":
    from engine.deal import deal_random

    d = deal_random(seed=1)
    try:
        print(solve_double_dummy(d))
    except DDSUnavailableError as e:
        print(f"[skipped: {e}]")
