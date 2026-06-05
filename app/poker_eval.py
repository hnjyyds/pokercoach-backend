from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations


RANK_VALUE = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
VALUE_LABEL = {
    14: "A",
    13: "K",
    12: "Q",
    11: "J",
    10: "10",
    9: "9",
    8: "8",
    7: "7",
    6: "6",
    5: "5",
    4: "4",
    3: "3",
    2: "2",
}


@dataclass(frozen=True)
class MadeHandExplanation:
    label: str
    detail: str
    score: tuple[int, ...]
    best_cards: tuple[str, ...]


@dataclass(frozen=True)
class ShowdownExplanation:
    hero: MadeHandExplanation
    villain: MadeHandExplanation | None
    winner: str
    content: str


def describe_showdown(
    hero_hand: str,
    villain_hand: str,
    board: str,
) -> ShowdownExplanation | None:
    hero_cards = parse_card_text(hero_hand)
    villain_cards = parse_card_text(villain_hand)
    board_cards = parse_card_text(board)
    if len(hero_cards) < 2 or len(board_cards) < 5:
        return None

    hero_made = best_made_hand(hero_cards + board_cards)
    villain_made = best_made_hand(villain_cards + board_cards) if len(villain_cards) >= 2 else None
    if villain_made is None:
        return ShowdownExplanation(
            hero=hero_made,
            villain=None,
            winner="hero",
            content=f"先看 Hero 的最佳五张牌：{hero_made.detail}，所以牌型是{hero_made.label}。",
        )

    if hero_made.score > villain_made.score:
        winner = "hero"
        winner_sentence = "Hero 的牌型更高，所以 Hero 赢。"
    elif hero_made.score < villain_made.score:
        winner = "villain"
        winner_sentence = "对手的牌型更高，所以对手赢。"
    else:
        winner = "tie"
        winner_sentence = "双方最佳五张牌相同，这手是平局。"

    content = (
        f"先看 Hero 的最佳五张牌：{hero_made.detail}，牌型是{hero_made.label}。"
        f"对手最佳五张牌是：{villain_made.detail}，牌型是{villain_made.label}。"
        f"{winner_sentence}"
    )
    return ShowdownExplanation(hero=hero_made, villain=villain_made, winner=winner, content=content)


def parse_card_text(text: str) -> list[str]:
    cards: list[str] = []
    for raw in text.replace(",", " ").split():
        code = normalize_card_code(raw)
        if code:
            cards.append(code)
    return cards


def normalize_card_code(raw: str) -> str | None:
    value = raw.strip()
    if len(value) < 2:
        return None
    rank = value[0].upper()
    suit = value[1].lower()
    if rank == "1" and len(value) >= 3 and value[1] == "0":
        rank = "T"
        suit = value[2].lower()
    if rank not in RANK_VALUE or suit not in {"c", "d", "h", "s"}:
        return None
    return f"{rank}{suit}"


def best_made_hand(cards: list[str]) -> MadeHandExplanation:
    if len(cards) < 5:
        raise ValueError("at least five cards are required")
    return max((classify_five_cards(tuple(combo)) for combo in combinations(cards, 5)), key=lambda hand: hand.score)


def classify_five_cards(cards: tuple[str, ...]) -> MadeHandExplanation:
    values = [RANK_VALUE[card[0]] for card in cards]
    suits = [card[1] for card in cards]
    counts = Counter(values)
    straight_high = straight_high_card(values)
    is_flush = len(set(suits)) == 1

    if straight_high and is_flush:
        return made("同花顺", f"五张同花连续牌，最高到 {rank_label(straight_high)}", (8, straight_high), cards)

    quads = ranks_with_count(counts, 4)
    if quads:
        quad = quads[0]
        kicker = max(value for value in values if value != quad)
        return made("四条", f"四张 {rank_label(quad)} 加 {rank_label(kicker)} 踢脚", (7, quad, kicker), cards)

    trips = ranks_with_count(counts, 3)
    pairs = ranks_with_count(counts, 2)
    if trips and pairs:
        trip = trips[0]
        pair = pairs[0]
        return made("葫芦", f"三张 {rank_label(trip)} 加一对 {rank_label(pair)}", (6, trip, pair), cards)

    if is_flush:
        kickers = sorted(values, reverse=True)
        return made("同花", f"五张同花，最大牌 {rank_label(kickers[0])}", (5, *kickers), cards)

    if straight_high:
        return made("顺子", f"五张连续牌，最高到 {rank_label(straight_high)}", (4, straight_high), cards)

    if trips:
        trip = trips[0]
        kickers = sorted((value for value in values if value != trip), reverse=True)
        return made("三条", f"三张 {rank_label(trip)}，再用 {rank_label(kickers[0])} 和 {rank_label(kickers[1])} 作补牌", (3, trip, *kickers), cards)

    if len(pairs) >= 2:
        high_pair, low_pair = pairs[:2]
        kicker = max(value for value in values if value not in {high_pair, low_pair})
        return made("两对", f"一对 {rank_label(high_pair)} 加一对 {rank_label(low_pair)}，再用 {rank_label(kicker)} 作补牌", (2, high_pair, low_pair, kicker), cards)

    if pairs:
        pair = pairs[0]
        kickers = sorted((value for value in values if value != pair), reverse=True)
        return made("一对", f"一对 {rank_label(pair)}，再比较剩余三张补牌", (1, pair, *kickers), cards)

    kickers = sorted(values, reverse=True)
    return made("高牌", f"没有对子或更强牌型，最大牌是 {rank_label(kickers[0])}", (0, *kickers), cards)


def made(label: str, detail: str, score: tuple[int, ...], cards: tuple[str, ...]) -> MadeHandExplanation:
    return MadeHandExplanation(label=label, detail=detail, score=score, best_cards=cards)


def ranks_with_count(counts: Counter[int], count: int) -> list[int]:
    return sorted((rank for rank, amount in counts.items() if amount == count), reverse=True)


def straight_high_card(values: list[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 5 - 1, -1):
        if all(value in unique for value in range(high - 4, high + 1)):
            return high
    return None


def rank_label(value: int) -> str:
    return VALUE_LABEL[value]
