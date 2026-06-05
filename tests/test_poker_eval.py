from __future__ import annotations

from app.poker_eval import best_made_hand, describe_showdown, parse_card_text


def test_parse_card_text_normalizes_ten() -> None:
    assert parse_card_text("10s Ah Kd") == ["Ts", "Ah", "Kd"]


def test_best_made_hand_detects_full_house_from_any_seven_cards() -> None:
    made = best_made_hand(parse_card_text("8c 8d 8h Ac Ks 2d 2s"))

    assert made.label == "葫芦"
    assert "三张 8" in made.detail
    assert "一对 2" in made.detail


def test_best_made_hand_detects_flush_without_hardcoded_cards() -> None:
    made = best_made_hand(parse_card_text("As 9s 7s 4s 2s Kd Qh"))

    assert made.label == "同花"
    assert "最大牌 A" in made.detail


def test_describe_showdown_compares_both_players() -> None:
    explanation = describe_showdown(
        hero_hand="Qh Qd",
        villain_hand="As Kh",
        board="Qs 2d 2c 7h 9c",
    )

    assert explanation is not None
    assert explanation.hero.label == "葫芦"
    assert explanation.villain.label == "一对"
    assert explanation.winner == "hero"
    assert "Hero 的最佳五张牌" in explanation.content
