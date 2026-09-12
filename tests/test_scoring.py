"""Reference-value tests for game/scoring.py against the well-known
standard duplicate scoring table (the same numbers printed on any bridge
scorecard/reference sheet — not from any book)."""
from game.scoring import score_contract, imps_for_diff, imp_swing


def _ns_score(level, strain, dbl, tricks, declarer, vul_ns=False, vul_ew=False):
    return score_contract(level, strain, dbl, tricks, declarer, vul_ns, vul_ew).ns_score


def test_partscore_1nt_making_exactly_nonvul():
    assert _ns_score(1, "NT", "", 7, "N") == 90


def test_game_3nt_making_exactly_nonvul():
    assert _ns_score(3, "NT", "", 9, "N") == 400


def test_game_3nt_making_exactly_vul():
    assert _ns_score(3, "NT", "", 9, "N", vul_ns=True) == 600


def test_game_4_major_nonvul():
    assert _ns_score(4, "S", "", 10, "N") == 420


def test_small_slam_major_nonvul():
    assert _ns_score(6, "S", "", 12, "N") == 980


def test_small_slam_minor_nonvul():
    assert _ns_score(6, "C", "", 12, "N") == 920


def test_grand_slam_notrump_vul():
    assert _ns_score(7, "NT", "", 13, "N", vul_ns=True) == 2220


def test_grand_slam_notrump_nonvul():
    assert _ns_score(7, "NT", "", 13, "N") == 1520


def test_down_one_undoubled_nonvul_is_negative_50_for_declarer():
    # South (NS) bids 4S, only takes 9 tricks -> down 1, undoubled, non-vul.
    assert _ns_score(4, "S", "", 9, "S") == -50


def test_down_one_undoubled_vul_is_negative_100():
    assert _ns_score(4, "S", "", 9, "S", vul_ns=True) == -100


def test_doubled_down_four_nonvul_is_negative_800():
    assert _ns_score(4, "S", "X", 6, "S") == -800  # down 4


def test_doubled_down_two_vul_is_negative_500():
    assert _ns_score(4, "S", "X", 8, "S", vul_ns=True) == -500  # down 2, vul doubled: 200+300


def test_declaring_side_is_ew_flips_sign():
    result = score_contract(4, "S", "", 10, "E", vulnerable_ns=False, vulnerable_ew=False)
    assert result.declaring_side == "EW"
    assert result.ew_score == 420
    assert result.ns_score == -420


def test_doubled_making_with_overtrick_gets_insult_bonus():
    # 3NT doubled, making exactly 10 tricks (1 overtrick), non-vul.
    # Trick score: (3*30+10)*2 = 200 -> game bonus 300 -> insult 50 -> overtrick 100 (doubled, non-vul) = 650
    assert _ns_score(3, "NT", "X", 10, "N") == 650


def test_imp_scale_known_points():
    assert imps_for_diff(0) == 0
    assert imps_for_diff(20) == 1
    assert imps_for_diff(500) == 11
    assert imps_for_diff(3000) == 22
    assert imps_for_diff(10000) == 24


def test_imp_swing_sign():
    assert imp_swing(620, 100) > 0
    assert imp_swing(100, 620) < 0
    assert imp_swing(400, 400) == 0
