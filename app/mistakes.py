from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException, status
from pydantic import BaseModel, Field


CoachRole = Literal["agent", "user"]


class BattleMistakeCandidate(BaseModel):
    action: str
    label: str
    target_total_bb: float
    ev_bb: float
    weight: float
    is_recommended: bool
    reason: str


class BattleMistakeTableSeat(BaseModel):
    seat_index: int
    position: str
    name: str
    stack_bb: float
    committed_bb: float
    status: str
    is_hero: bool = False


class BattleMistakeScenario(BaseModel):
    session_id: str
    hand_number: int
    table_size: int
    street: str
    position: str
    hero_name: str
    hero_cards: list[str]
    board: list[str]
    pot_bb: float
    current_bet_bb: float
    stack_bb: float
    committed_bb: float
    spr: float
    table_seats: list[BattleMistakeTableSeat]
    tags: list[str]


class CoachMessageSnapshot(BaseModel):
    id: str
    role: CoachRole
    content: str
    created_at: str


class BattleMistakeSummary(BaseModel):
    id: str
    title: str
    subtitle: str
    street: str
    position: str
    hero_cards: list[str]
    board: list[str]
    user_action_label: str
    recommended_action_label: str
    ev_delta_bb: float
    icon: str
    accent: str
    created_at: str


class BattleMistakeDetail(BattleMistakeSummary):
    owner_id: str
    session_id: str
    hand_number: int
    action_id: str | None = None
    user_action: str
    recommended_action: str
    user_total_bb: float
    recommended_total_bb: float
    scenario: BattleMistakeScenario
    candidates: list[BattleMistakeCandidate]
    why_wrong: str
    correct_play: str
    coach_messages: list[CoachMessageSnapshot] = Field(default_factory=list)


class BattleMistakeCreate(BaseModel):
    owner_id: str
    session_id: str
    hand_number: int
    action_id: str | None = None
    title: str
    subtitle: str
    street: str
    position: str
    hero_cards: list[str]
    board: list[str]
    user_action: str
    user_action_label: str
    user_total_bb: float
    recommended_action: str
    recommended_action_label: str
    recommended_total_bb: float
    ev_delta_bb: float
    scenario: BattleMistakeScenario
    candidates: list[BattleMistakeCandidate]
    why_wrong: str
    correct_play: str
    icon: str = "exclamationmark.bubble.fill"
    accent: str = "#EF4444"


class CoachMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=600)


MISTAKES: dict[str, BattleMistakeDetail] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_identifier(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in {"_", "-"})


def mistake_store_dir() -> Path:
    configured = os.environ.get("POKERCOACH_MISTAKE_STORE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / ".data" / "mistakes"


def mistake_store_path(mistake_id: str) -> Path:
    return mistake_store_dir() / f"{safe_identifier(mistake_id)}.json"


def record_battle_mistake(payload: BattleMistakeCreate) -> BattleMistakeDetail:
    mistake = BattleMistakeDetail(
        id=f"mis_{uuid4().hex[:12]}",
        owner_id=payload.owner_id,
        session_id=payload.session_id,
        hand_number=payload.hand_number,
        action_id=payload.action_id,
        title=payload.title,
        subtitle=payload.subtitle,
        street=payload.street,
        position=payload.position,
        hero_cards=payload.hero_cards,
        board=payload.board,
        user_action=payload.user_action,
        user_action_label=payload.user_action_label,
        user_total_bb=payload.user_total_bb,
        recommended_action=payload.recommended_action,
        recommended_action_label=payload.recommended_action_label,
        recommended_total_bb=payload.recommended_total_bb,
        ev_delta_bb=round(payload.ev_delta_bb, 2),
        scenario=payload.scenario,
        candidates=payload.candidates,
        why_wrong=payload.why_wrong,
        correct_play=payload.correct_play,
        icon=payload.icon,
        accent=payload.accent,
        created_at=now_iso(),
        coach_messages=[
            CoachMessageSnapshot(
                id=f"msg_{uuid4().hex[:10]}",
                role="agent",
                content=initial_coach_message(payload),
                created_at=now_iso(),
            )
        ],
    )
    MISTAKES[mistake.id] = mistake
    persist_mistake(mistake)
    return mistake


def list_mistakes(owner_id: str) -> list[BattleMistakeSummary]:
    details = load_owner_mistakes(owner_id)
    if not details:
        details = seed_mistakes(owner_id)
    return [
        BattleMistakeSummary.model_validate(detail.model_dump())
        for detail in sorted(details, key=lambda item: item.created_at, reverse=True)
    ][:20]


def get_mistake(mistake_id: str, owner_id: str) -> BattleMistakeDetail:
    detail = load_mistake(mistake_id)
    if detail is None:
        detail = next((item for item in seed_mistakes(owner_id) if item.id == mistake_id), None)
    if detail is None or detail.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mistake not found")
    return detail


def add_coach_message(mistake_id: str, owner_id: str, request: CoachMessageRequest) -> BattleMistakeDetail:
    detail = get_mistake(mistake_id, owner_id)
    detail = detail.model_copy(deep=True)
    detail.coach_messages.append(
        CoachMessageSnapshot(
            id=f"msg_{uuid4().hex[:10]}",
            role="user",
            content=request.message.strip(),
            created_at=now_iso(),
        )
    )
    detail.coach_messages.append(
        CoachMessageSnapshot(
            id=f"msg_{uuid4().hex[:10]}",
            role="agent",
            content=coach_reply(detail, request.message),
            created_at=now_iso(),
        )
    )
    MISTAKES[detail.id] = detail
    persist_mistake(detail)
    return detail


def dashboard_mistake_texts(owner_id: str) -> list[str]:
    return [
        f"{mistake.position} {' '.join(mistake.hero_cards)}：{mistake.user_action_label}偏离推荐，建议{mistake.recommended_action_label}。"
        for mistake in list_mistakes(owner_id)[:2]
    ]


def persist_mistake(mistake: BattleMistakeDetail) -> None:
    mistake_store_dir().mkdir(parents=True, exist_ok=True)
    path = mistake_store_path(mistake.id)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(mistake.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_owner_mistakes(owner_id: str) -> list[BattleMistakeDetail]:
    mistakes_by_id = {
        mistake_id: mistake
        for mistake_id, mistake in MISTAKES.items()
        if mistake.owner_id == owner_id
    }
    for path in mistake_store_dir().glob("*.json"):
        detail = load_mistake_from_path(path)
        if detail is None or detail.owner_id != owner_id:
            continue
        mistakes_by_id[detail.id] = detail
        MISTAKES[detail.id] = detail
    return list(mistakes_by_id.values())


def load_mistake(mistake_id: str) -> BattleMistakeDetail | None:
    if mistake_id in MISTAKES:
        return MISTAKES[mistake_id]
    detail = load_mistake_from_path(mistake_store_path(mistake_id))
    if detail is not None:
        MISTAKES[detail.id] = detail
    return detail


def load_mistake_from_path(path: Path) -> BattleMistakeDetail | None:
    if not path.exists():
        return None
    try:
        return BattleMistakeDetail.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def initial_coach_message(payload: BattleMistakeCreate) -> str:
    return (
        f"这手 {payload.position} {' '.join(payload.hero_cards)} 的关键偏差是："
        f"你选择了{payload.user_action_label}，系统推荐{payload.recommended_action_label}。"
        f"EV 差约 {payload.ev_delta_bb:.1f}BB。{payload.correct_play}"
    )


def coach_reply(detail: BattleMistakeDetail, user_message: str) -> str:
    lowered = user_message.lower()
    scenario = detail.scenario
    core = (
        f"复盘这个 spot 时先看三件事：位置 {scenario.position}、底池 {scenario.pot_bb:.1f}BB、"
        f"有效后手约 {scenario.stack_bb:.1f}BB，SPR {scenario.spr:.1f}。"
    )
    if "为什么" in user_message or "why" in lowered:
        return f"{core}{detail.why_wrong}"
    if "怎么" in user_message or "应该" in user_message or "how" in lowered:
        return f"{core}{detail.correct_play}"
    if "范围" in user_message or "range" in lowered:
        tags = "、".join(scenario.tags[:4]) or "位置、手牌、SPR"
        return f"{core}这手要按 {tags} 来缩放范围；不要只看自己的两张牌，先确认对手范围和可承受的底池大小。"
    return (
        f"{core}你的动作是{detail.user_action_label}，推荐是{detail.recommended_action_label}。"
        f"主要原因：{detail.why_wrong}"
    )


def seed_mistakes(owner_id: str) -> list[BattleMistakeDetail]:
    safe_owner = safe_identifier(owner_id or "demo")
    return [
        BattleMistakeDetail(
            id=f"seed_{safe_owner}_utg_kqo",
            owner_id=owner_id,
            session_id="seed_session",
            hand_number=1,
            action_id=None,
            title="KQo 面对 UTG 加注",
            subtitle="CO · Preflop · 反向隐含赔率",
            street="preflop",
            position="CO",
            hero_cards=["Kc", "Qd"],
            board=[],
            user_action="call",
            user_action_label="跟注",
            user_total_bb=2.5,
            recommended_action="fold",
            recommended_action_label="弃牌",
            recommended_total_bb=0,
            ev_delta_bb=1.1,
            scenario=BattleMistakeScenario(
                session_id="seed_session",
                hand_number=1,
                table_size=6,
                street="preflop",
                position="CO",
                hero_name="Alex",
                hero_cards=["Kc", "Qd"],
                board=[],
                pot_bb=4.0,
                current_bet_bb=2.5,
                stack_bb=100,
                committed_bb=0,
                spr=25,
                table_seats=[
                    BattleMistakeTableSeat(seat_index=3, position="UTG", name="River", stack_bb=98, committed_bb=2.5, status="active"),
                    BattleMistakeTableSeat(seat_index=5, position="CO", name="Alex", stack_bb=100, committed_bb=0, status="active", is_hero=True),
                ],
                tags=["CO", "KQo", "被主导", "高SPR"],
            ),
            candidates=[
                BattleMistakeCandidate(action="fold", label="弃牌", target_total_bb=0, ev_bb=0.2, weight=0.55, is_recommended=True, reason="UTG 范围强，KQo 容易被 AQ/AK/KQs 主导。"),
                BattleMistakeCandidate(action="call", label="跟注", target_total_bb=2.5, ev_bb=-0.9, weight=0.30, is_recommended=False, reason="高 SPR 下反向隐含赔率变大，翻后容易支付强牌。"),
                BattleMistakeCandidate(action="raise", label="3-bet", target_total_bb=8, ev_bb=-0.3, weight=0.15, is_recommended=False, reason="阻断不足，面对 4-bet 难继续。"),
            ],
            why_wrong="KQo 看起来像高张强牌，但面对 UTG 紧范围，经常被 AK、AQ、QQ+ 主导；深筹码时跟注会把自己带进难打的大底池。",
            correct_play="新手阶段直接弃牌，保留筹码进入位置更好、范围更清晰的 spot。",
            icon="suit.club.fill",
            accent="#F59E0B",
            created_at="2026-06-05T00:00:00+00:00",
            coach_messages=[
                CoachMessageSnapshot(
                    id="seed_msg_utg_1",
                    role="agent",
                    content="这手重点不是 KQ 漂亮，而是 UTG 范围太强。先把被主导风险排除，会少输很多大底池。",
                    created_at="2026-06-05T00:00:00+00:00",
                )
            ],
        ),
        BattleMistakeDetail(
            id=f"seed_{safe_owner}_sb_defense",
            owner_id=owner_id,
            session_id="seed_session",
            hand_number=2,
            action_id=None,
            title="小盲位宽跟注",
            subtitle="SB · Preflop · 位置劣势",
            street="preflop",
            position="SB",
            hero_cards=["Qh", "8h"],
            board=[],
            user_action="call",
            user_action_label="跟注",
            user_total_bb=2.5,
            recommended_action="fold",
            recommended_action_label="弃牌",
            recommended_total_bb=0.5,
            ev_delta_bb=0.8,
            scenario=BattleMistakeScenario(
                session_id="seed_session",
                hand_number=2,
                table_size=6,
                street="preflop",
                position="SB",
                hero_name="Alex",
                hero_cards=["Qh", "8h"],
                board=[],
                pot_bb=4.0,
                current_bet_bb=2.5,
                stack_bb=99.5,
                committed_bb=0.5,
                spr=24.9,
                table_seats=[
                    BattleMistakeTableSeat(seat_index=0, position="BTN", name="Nova", stack_bb=97.5, committed_bb=2.5, status="active"),
                    BattleMistakeTableSeat(seat_index=1, position="SB", name="Alex", stack_bb=99.5, committed_bb=0.5, status="active", is_hero=True),
                    BattleMistakeTableSeat(seat_index=2, position="BB", name="Ivy", stack_bb=99, committed_bb=1, status="active"),
                ],
                tags=["SB", "Q8s", "位置劣势", "高SPR"],
            ),
            candidates=[
                BattleMistakeCandidate(action="fold", label="弃牌", target_total_bb=0.5, ev_bb=0.1, weight=0.50, is_recommended=True, reason="小盲位翻后全程失位，边缘同花牌实现权益困难。"),
                BattleMistakeCandidate(action="call", label="跟注", target_total_bb=2.5, ev_bb=-0.7, weight=0.34, is_recommended=False, reason="容易形成被动多人池，翻后难以控池。"),
                BattleMistakeCandidate(action="raise", label="3-bet", target_total_bb=9, ev_bb=-0.2, weight=0.16, is_recommended=False, reason="阻断牌不足，面对继续范围权益不够。"),
            ],
            why_wrong="小盲位没有位置，Q8s 的同花潜力不足以弥补翻后实现权益差；跟注还会给大盲好价格进入底池。",
            correct_play="直接弃牌或只在明确 exploit 对手过度开池时低频 3-bet，不要默认平跟。",
            icon="location.fill",
            accent="#EF4444",
            created_at="2026-06-04T00:00:00+00:00",
            coach_messages=[
                CoachMessageSnapshot(
                    id="seed_msg_sb_1",
                    role="agent",
                    content="小盲位先默认更紧。你要用更高质量的牌进入底池，因为翻后没有位置，错误会被放大。",
                    created_at="2026-06-04T00:00:00+00:00",
                )
            ],
        ),
    ]
