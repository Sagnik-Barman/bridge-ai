from game.board import board_info, user_seat_for_board


def test_board_1():
    b = board_info(1)
    assert b.dealer == "N" and b.vulnerable == "None"


def test_board_5_repeats_dealer_with_different_vul():
    b = board_info(5)
    assert b.dealer == "N" and b.vulnerable == "NS"


def test_board_16():
    b = board_info(16)
    assert b.dealer == "W" and b.vulnerable == "EW"


def test_cycle_repeats_after_16():
    assert board_info(17).dealer == board_info(1).dealer
    assert board_info(17).vulnerable == board_info(1).vulnerable
    assert board_info(33).vulnerable == board_info(1).vulnerable


def test_is_vulnerable_helper():
    b = board_info(4)  # All vulnerable
    assert b.is_vulnerable("NS")
    assert b.is_vulnerable("EW")
    b2 = board_info(2)  # NS only
    assert b2.is_vulnerable("NS")
    assert not b2.is_vulnerable("EW")


def test_user_seat_rotates_each_board():
    seats = [user_seat_for_board(n) for n in range(1, 9)]
    assert seats == ["N", "E", "S", "W", "N", "E", "S", "W"]
