from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from treys import Card, Evaluator

from app.mistakes import (
    BattleMistakeCandidate,
    BattleMistakeCreate,
    BattleMistakeScenario,
    BattleMistakeTableSeat,
    record_battle_mistake,
)


BattleStreet = Literal["preflop", "flop", "turn", "river", "showdown", "complete"]
BattleActionType = Literal["blind", "fold", "check", "call", "bet", "raise", "all_in"]
BattleTableEventType = Literal[
    "hand_start",
    "blind_posted",
    "burn",
    "deal_flop",
    "deal_turn",
    "deal_river",
    "showdown",
    "uncontested",
    "hand_complete",
]
BattleReplayEventType = Literal["table", "action"]
SeatStatus = Literal["active", "folded", "all_in", "showdown"]
BattleSessionMode = Literal["spectate", "play"]
BattlePlayerAction = Literal["fold", "check", "call", "bet", "raise"]


class BattleSessionCreate(BaseModel):
    table_size: Literal[2, 6, 9] = 6
    observer_seat: int = Field(default=2, ge=0)
    starting_stack_bb: float = Field(default=100, ge=20, le=300)
    mode: BattleSessionMode = "spectate"
    player_seat: int | None = Field(default=None, ge=0)
    seed: str | None = Field(default=None, min_length=4, max_length=64)


class BattleAdvanceRequest(BaseModel):
    observer_seat: int = Field(default=0, ge=0)
    steps: int = Field(default=1, ge=1, le=24)


class BattleNextHandRequest(BaseModel):
    observer_seat: int = Field(default=0, ge=0)


class BattlePlayerActionRequest(BaseModel):
    observer_seat: int = Field(default=0, ge=0)
    action: BattlePlayerAction
    target_total_bb: float | None = Field(default=None, ge=0)


class BattleAgentProfile(BaseModel):
    id: str
    name: str
    style: str
    avatar_seed: str
    accent: str
    bio: str
    archetype: str
    mastery_label: str
    gto_score: int
    exploit_score: int
    postflop_score: int
    risk_profile: str
    strategy_tags: list[str]


class BattleDecisionCandidateSnapshot(BaseModel):
    action: BattleActionType
    label: str
    target_total_bb: float
    ev_bb: float
    weight: float
    is_chosen: bool
    reason: str


class BattleDecisionSnapshot(BaseModel):
    source: str
    engine: str
    equity_samples: int
    policy_profile: str
    chosen_action: BattleActionType | None = None
    chosen_label: str | None = None
    chosen_ev_bb: float | None = None
    best_alternative_action: BattleActionType | None = None
    best_alternative_label: str | None = None
    best_alternative_ev_bb: float | None = None
    ev_delta_bb: float | None = None
    hand_class: str
    range_bucket: str
    range_role: str
    range_frequency: float
    board_texture: str
    equity: float | None = None
    pot_odds: float
    spr: float
    pressure: float
    confidence: float
    recommended_total_bb: float
    tags: list[str]
    summary: str
    candidates: list[BattleDecisionCandidateSnapshot] = Field(default_factory=list)


class BattleActionSnapshot(BaseModel):
    id: str
    seat_index: int
    position: str
    agent_id: str
    agent_name: str
    street: BattleStreet
    action: BattleActionType
    label: str
    amount_bb: float
    total_bet_bb: float
    pot_bb: float
    equity: float | None = None
    pot_odds: float | None = None
    note: str
    decision: BattleDecisionSnapshot | None = None
    created_at: str


class BattleActionStreetSnapshot(BaseModel):
    street: BattleStreet
    label: str
    actions: list[BattleActionSnapshot]


class BattleSeatSnapshot(BaseModel):
    index: int
    agent: BattleAgentProfile
    position: str
    stack_bb: float
    street_bet_bb: float
    total_committed_bb: float
    status: SeatStatus
    is_dealer: bool
    is_small_blind: bool
    is_big_blind: bool
    is_observer: bool
    is_active: bool
    is_human: bool
    hole_cards: list[str] | None
    last_action: BattleActionSnapshot | None


class BattleTaskSnapshot(BaseModel):
    id: str
    title: str
    subtitle: str
    icon: str
    accent: str
    state: Literal["done", "running", "queued"]


class BattleSidePotSnapshot(BaseModel):
    amount_bb: float
    eligible_seats: list[int]
    winners: list[int]


class BattleShowdownHandSnapshot(BaseModel):
    seat_index: int
    position: str
    agent_id: str
    agent_name: str
    hole_cards: list[str]
    made_hand: str
    hand_rank: int
    is_winner: bool
    won_bb: float


class BattleResultSnapshot(BaseModel):
    winners: list[int]
    summary: str
    showdown: list[str]
    showdown_details: list[BattleShowdownHandSnapshot] = Field(default_factory=list)
    side_pots: list[BattleSidePotSnapshot] = Field(default_factory=list)


class BattleTableEventSnapshot(BaseModel):
    id: str
    event: BattleTableEventType
    street: BattleStreet
    label: str
    seat_index: int | None = None
    cards: list[str] = Field(default_factory=list)
    burn_card: str | None = None
    pot_bb: float
    created_at: str


class BattleReplayEventSnapshot(BaseModel):
    id: str
    sequence: int
    kind: BattleReplayEventType
    street: BattleStreet
    label: str
    created_at: str
    seat_index: int | None = None
    position: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    action_id: str | None = None
    table_event_id: str | None = None
    action: BattleActionType | None = None
    table_event: BattleTableEventType | None = None
    cards: list[str] = Field(default_factory=list)
    burn_card: str | None = None
    amount_bb: float = 0
    total_bet_bb: float = 0
    pot_bb: float = 0
    equity: float | None = None
    pot_odds: float | None = None
    note: str | None = None
    decision: BattleDecisionSnapshot | None = None


class BattleReviewInsightSnapshot(BaseModel):
    id: str
    title: str
    detail: str
    icon: str
    accent: str
    seat_index: int | None = None
    action_id: str | None = None


class BattleHandSummarySnapshot(BaseModel):
    id: str
    session_id: str
    hand_number: int
    board: list[str]
    winners: list[int]
    summary: str
    action_count: int
    decision_count: int
    replay_count: int
    completed_at: str


class BattleHandHistorySnapshot(BaseModel):
    id: str
    session_id: str
    hand_number: int
    table_size: int
    seed: str
    street: BattleStreet
    stage_label: str
    board: list[str]
    burned_cards: list[str]
    observer_seat: int
    seats: list[BattleSeatSnapshot]
    action_timeline: list[BattleActionStreetSnapshot]
    table_events: list[BattleTableEventSnapshot]
    replay_events: list[BattleReplayEventSnapshot]
    result: BattleResultSnapshot | None = None
    review_insights: list[BattleReviewInsightSnapshot]
    action_count: int
    decision_count: int
    is_complete: bool


class BattleSessionSnapshot(BaseModel):
    id: str
    mode: BattleSessionMode
    player_seat: int | None
    table_size: int
    hand_number: int
    street: BattleStreet
    stage_label: str
    pot_bb: float
    current_bet_bb: float
    min_raise_bb: float
    board: list[str]
    burned_cards: list[str]
    observer_seat: int
    active_seat: int | None
    seats: list[BattleSeatSnapshot]
    recent_actions: list[BattleActionSnapshot]
    action_timeline: list[BattleActionStreetSnapshot]
    table_events: list[BattleTableEventSnapshot]
    replay_events: list[BattleReplayEventSnapshot]
    tasks: list[BattleTaskSnapshot]
    result: BattleResultSnapshot | None = None
    is_complete: bool
    is_session_complete: bool


class BattleSessionSummarySnapshot(BaseModel):
    id: str
    table_size: int
    hand_number: int
    street: BattleStreet
    stage_label: str
    pot_bb: float
    active_seat: int | None
    completed_hand_count: int
    is_complete: bool
    is_session_complete: bool
    last_event_label: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AgentPersona:
    id: str
    name: str
    style: str
    avatar_seed: str
    accent: str
    bio: str
    looseness: float
    aggression: float
    bluff: float
    discipline: float

    @property
    def gto_score(self) -> int:
        balance = 1 - abs(self.looseness - 0.50)
        return min(99, round(86 + self.discipline * 8 + balance * 4))

    @property
    def exploit_score(self) -> int:
        pressure = self.aggression * 0.58 + self.bluff * 0.42
        return min(99, round(84 + pressure * 14))

    @property
    def postflop_score(self) -> int:
        texture_skill = self.discipline * 0.45 + self.aggression * 0.35 + (1 - abs(self.bluff - 0.18)) * 0.20
        return min(99, round(85 + texture_skill * 13))

    @property
    def strategy_tags(self) -> list[str]:
        tags = ["GTO", "范围", "SPR"]
        if self.aggression >= 0.70:
            tags.append("极化")
        if self.discipline >= 0.88:
            tags.append("纪律")
        if self.bluff >= 0.22:
            tags.append("红线")
        if self.looseness <= 0.44:
            tags.append("紧范围")
        if "短码" in self.style:
            tags.append("短码")
        if "池控" in self.style or "跟注" in self.style:
            tags.append("控池")
        return tags[:5]

    @property
    def archetype(self) -> str:
        if "均衡" in self.style:
            return "Solver 均衡"
        if "压迫" in self.style or "进攻" in self.style or self.aggression >= 0.74:
            return "极化进攻"
        if "池控" in self.style or "跟注" in self.style:
            return "控池抓诈"
        if "短码" in self.style:
            return "短码专家"
        return "范围读牌"

    @property
    def risk_profile(self) -> str:
        if self.aggression >= 0.74 or self.bluff >= 0.24:
            return "进攻"
        if self.discipline >= 0.90:
            return "纪律"
        if self.looseness <= 0.44:
            return "紧凶"
        return "均衡"

    def snapshot(self) -> BattleAgentProfile:
        return BattleAgentProfile(
            id=self.id,
            name=self.name,
            style=self.style,
            avatar_seed=self.avatar_seed,
            accent=self.accent,
            bio=self.bio,
            archetype=self.archetype,
            mastery_label="大师级",
            gto_score=self.gto_score,
            exploit_score=self.exploit_score,
            postflop_score=self.postflop_score,
            risk_profile=self.risk_profile,
            strategy_tags=self.strategy_tags,
        )


@dataclass
class PlayerState:
    index: int
    agent: AgentPersona
    position: str
    stack_bb: float
    is_human: bool = False
    hole_cards: list[str] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    street_bet_bb: float = 0
    total_committed_bb: float = 0
    last_action: "ActionEvent | None" = None

    @property
    def status(self) -> SeatStatus:
        if self.folded:
            return "folded"
        if self.all_in:
            return "all_in"
        return "active"


@dataclass
class ActionEvent:
    id: str
    seat_index: int
    position: str
    agent_id: str
    agent_name: str
    street: BattleStreet
    action: BattleActionType
    label: str
    amount_bb: float
    total_bet_bb: float
    pot_bb: float
    equity: float | None
    pot_odds: float | None
    note: str
    decision: "DecisionTrace | None"
    created_at: str

    def snapshot(self) -> BattleActionSnapshot:
        return BattleActionSnapshot(
            id=self.id,
            seat_index=self.seat_index,
            position=self.position,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            street=self.street,
            action=self.action,
            label=self.label,
            amount_bb=round_bb(self.amount_bb),
            total_bet_bb=round_bb(self.total_bet_bb),
            pot_bb=round_bb(self.pot_bb),
            equity=None if self.equity is None else round(self.equity, 3),
            pot_odds=None if self.pot_odds is None else round(self.pot_odds, 3),
            note=self.note,
            decision=self.decision.snapshot() if self.decision else None,
            created_at=self.created_at,
        )


@dataclass
class TableEvent:
    id: str
    event: BattleTableEventType
    street: BattleStreet
    label: str
    seat_index: int | None
    cards: list[str]
    burn_card: str | None
    pot_bb: float
    created_at: str

    def snapshot(self) -> BattleTableEventSnapshot:
        return BattleTableEventSnapshot(
            id=self.id,
            event=self.event,
            street=self.street,
            label=self.label,
            seat_index=self.seat_index,
            cards=self.cards,
            burn_card=self.burn_card,
            pot_bb=round_bb(self.pot_bb),
            created_at=self.created_at,
        )


@dataclass
class DecisionCandidate:
    action: BattleActionType
    target_total_bb: float
    ev_bb: float
    weight: float
    reason: str
    is_chosen: bool = False

    def snapshot(self) -> BattleDecisionCandidateSnapshot:
        return BattleDecisionCandidateSnapshot(
            action=self.action,
            label=action_label(self.action),
            target_total_bb=round_bb(self.target_total_bb),
            ev_bb=round_bb(self.ev_bb),
            weight=round(self.weight, 3),
            is_chosen=self.is_chosen,
            reason=self.reason,
        )


@dataclass(frozen=True)
class RangeProfile:
    tier: str
    role: str
    frequency: float
    note: str
    bet_fraction: float | None = None
    raise_fraction: float | None = None


@dataclass(frozen=True)
class BoardProfile:
    texture: str
    wetness: float
    paired: bool
    monotone: bool
    two_tone: bool
    connected: bool
    high_card: int


@dataclass
class DecisionTrace:
    source: str
    engine: str
    equity_samples: int
    policy_profile: str
    hand_class: str
    range_bucket: str
    range_role: str
    range_frequency: float
    board_texture: str
    equity: float | None
    pot_odds: float
    spr: float
    pressure: float
    confidence: float
    recommended_total_bb: float
    tags: list[str]
    summary: str
    candidates: list[DecisionCandidate] = field(default_factory=list)

    def snapshot(self) -> BattleDecisionSnapshot:
        chosen = chosen_candidate(self.candidates)
        alternative = best_alternative_candidate(self.candidates, chosen)
        ev_delta = None
        if chosen and alternative:
            ev_delta = round_bb(chosen.ev_bb - alternative.ev_bb)

        return BattleDecisionSnapshot(
            source=self.source,
            engine=self.engine,
            equity_samples=self.equity_samples,
            policy_profile=self.policy_profile,
            chosen_action=chosen.action if chosen else None,
            chosen_label=action_label(chosen.action) if chosen else None,
            chosen_ev_bb=round_bb(chosen.ev_bb) if chosen else None,
            best_alternative_action=alternative.action if alternative else None,
            best_alternative_label=action_label(alternative.action) if alternative else None,
            best_alternative_ev_bb=round_bb(alternative.ev_bb) if alternative else None,
            ev_delta_bb=ev_delta,
            hand_class=self.hand_class,
            range_bucket=self.range_bucket,
            range_role=self.range_role,
            range_frequency=round(self.range_frequency, 3),
            board_texture=self.board_texture,
            equity=None if self.equity is None else round(self.equity, 3),
            pot_odds=round(self.pot_odds, 3),
            spr=round_bb(self.spr),
            pressure=round(self.pressure, 3),
            confidence=round(self.confidence, 3),
            recommended_total_bb=round_bb(self.recommended_total_bb),
            tags=self.tags,
            summary=self.summary,
            candidates=[candidate.snapshot() for candidate in self.candidates],
        )


@dataclass
class BrainDecision:
    action: BattleActionType
    target_total_bb: float
    equity: float | None
    pot_odds: float | None
    note: str
    trace: DecisionTrace | None


@dataclass
class SidePot:
    amount_bb: float
    eligible_players: list[PlayerState]


@dataclass
class BattleSession:
    id: str
    mode: BattleSessionMode
    player_seat: int | None
    table_size: int
    hand_number: int
    starting_stack_bb: float
    seed: str
    rng: Random
    players: list[PlayerState]
    deck: list[str]
    owner_id: str | None = None
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())
    board: list[str] = field(default_factory=list)
    burned_cards: list[str] = field(default_factory=list)
    pot_bb: float = 0
    street: BattleStreet = "preflop"
    dealer_index: int = 0
    current_bet_bb: float = 1
    min_raise_bb: float = 1
    current_actor: int | None = None
    acted_this_street: set[int] = field(default_factory=set)
    action_log: list[ActionEvent] = field(default_factory=list)
    table_events: list[TableEvent] = field(default_factory=list)
    completed_hands: list[BattleHandHistorySnapshot] = field(default_factory=list)
    result: BattleResultSnapshot | None = None
    is_session_complete: bool = False

    def snapshot(self, observer_seat: int) -> BattleSessionSnapshot:
        observer_seat = normalize_seat(observer_seat, self.table_size)
        recent_actions = [event.snapshot() for event in self.action_log[-12:]]
        action_timeline = build_action_timeline(self.action_log)
        table_events = [event.snapshot() for event in self.table_events]
        replay_events = build_replay_events(self.action_log, self.table_events)
        seats = [
            BattleSeatSnapshot(
                index=player.index,
                agent=player.agent.snapshot(),
                position=player.position,
                stack_bb=round_bb(player.stack_bb),
                street_bet_bb=round_bb(player.street_bet_bb),
                total_committed_bb=round_bb(player.total_committed_bb),
                status=player.status,
                is_dealer=player.index == self.dealer_index,
                is_small_blind=player.position == "SB",
                is_big_blind=player.position == "BB",
                is_observer=player.index == observer_seat,
                is_active=player.index == self.current_actor,
                is_human=player.is_human,
                hole_cards=player.hole_cards if player.index == observer_seat or self.street in ("showdown", "complete") else None,
                last_action=player.last_action.snapshot() if player.last_action else None,
            )
            for player in self.players
        ]

        return BattleSessionSnapshot(
            id=self.id,
            mode=self.mode,
            player_seat=self.player_seat,
            table_size=self.table_size,
            hand_number=self.hand_number,
            street=self.street,
            stage_label=street_label(self.street),
            pot_bb=round_bb(self.pot_bb),
            current_bet_bb=round_bb(self.current_bet_bb),
            min_raise_bb=round_bb(self.min_raise_bb),
            board=self.board,
            burned_cards=self.burned_cards,
            observer_seat=observer_seat,
            active_seat=self.current_actor,
            seats=seats,
            recent_actions=recent_actions,
            action_timeline=action_timeline,
            table_events=table_events,
            replay_events=replay_events,
            tasks=build_tasks(self, observer_seat),
            result=self.result,
            is_complete=self.street == "complete",
            is_session_complete=self.is_session_complete,
        )


AGENTS: list[AgentPersona] = [
    AgentPersona("agent_river", "River", "GTO 压迫", "river-gto-pro", "#8B5CF6", "以范围优势和极化下注闻名。", 0.52, 0.72, 0.22, 0.86),
    AgentPersona("agent_nash", "Nash", "均衡派", "nash-solver", "#13C8A6", "接近 solver 的低波动决策。", 0.47, 0.58, 0.13, 0.94),
    AgentPersona("agent_ivy", "Ivy", "池控专家", "ivy-control", "#F59E0B", "擅长跟注、控池和抓诈唬。", 0.44, 0.42, 0.09, 0.90),
    AgentPersona("agent_mira", "Mira", "红线进攻", "mira-aggro", "#EF4444", "喜欢主动争夺弃牌权益。", 0.58, 0.80, 0.27, 0.80),
    AgentPersona("agent_leo", "Leo", "松凶 exploit", "leo-lag", "#3B82F6", "高频施压，但会根据赔率收手。", 0.64, 0.74, 0.24, 0.76),
    AgentPersona("agent_nova", "Nova", "读牌派", "nova-reader", "#EC4899", "偏重阻断牌、牌面结构和对手范围。", 0.50, 0.62, 0.18, 0.88),
    AgentPersona("agent_kane", "Kane", "短码 ICM", "kane-short", "#64748B", "短码和再加注全下策略更成熟。", 0.40, 0.67, 0.10, 0.91),
    AgentPersona("agent_echo", "Echo", "跟注站克星", "echo-exploit", "#14B8A6", "擅长薄价值下注和过牌-加注。", 0.49, 0.61, 0.15, 0.84),
    AgentPersona("agent_ace", "Ace", "高压牌手", "ace-pressure", "#0F172A", "在按钮位和盲注战极具攻击性。", 0.60, 0.78, 0.23, 0.82),
]

SESSIONS: dict[str, BattleSession] = {}
AGENT_BY_ID = {agent.id: agent for agent in AGENTS}


def human_persona(owner_id: str | None = None, name: str = "Alex") -> AgentPersona:
    safe_owner = "".join(character for character in (owner_id or "local") if character.isalnum() or character in {"_", "-"})
    return AgentPersona(
        id=f"human_{safe_owner}",
        name=name,
        style="玩家",
        avatar_seed=f"human-{safe_owner}",
        accent="#071226",
        bio="用户加入牌局后的真人座位。",
        looseness=0.50,
        aggression=0.55,
        bluff=0.14,
        discipline=0.86,
    )


EVALUATOR = Evaluator()
EQUITY_SAMPLE_COUNT = 96
RANK_VALUE = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
POSITION_OPEN_FLOORS = {
    "UTG": 0.58,
    "UTG+1": 0.57,
    "MP": 0.54,
    "LJ": 0.52,
    "HJ": 0.50,
    "CO": 0.46,
    "BTN": 0.41,
    "SB": 0.49,
    "BB": 0.50,
}
PREMIUM_HANDS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
VALUE_3BET_HANDS = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo"}
MIXED_3BET_HANDS = {"A5s", "A4s", "A3s", "KQs", "KJs", "QJs", "JTs", "T9s"}
PLAYABLE_DEFENDS = {
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
    "AKo", "AQo", "AJo", "KQo",
}
EARLY_OPEN_HANDS = {
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
    "AKs", "AQs", "AJs", "ATs", "A5s", "A4s", "KQs", "KJs", "QJs", "JTs",
    "AKo", "AQo",
}
MIDDLE_OPEN_HANDS = EARLY_OPEN_HANDS | {
    "55", "44", "A9s", "A8s", "A7s", "KTs", "QTs", "T9s", "98s", "AJo", "KQo",
}
LATE_OPEN_HANDS = MIDDLE_OPEN_HANDS | {
    "33", "22", "A6s", "A3s", "A2s", "K9s", "Q9s", "J9s", "T8s", "87s", "76s", "65s", "54s",
    "ATo", "KJo", "QJo",
}
BUTTON_OPEN_HANDS = LATE_OPEN_HANDS | {
    "K8s", "K7s", "K6s", "Q8s", "J8s", "97s", "86s", "75s", "64s",
    "A9o", "A8o", "KTo", "QTo", "JTo", "T9o", "98o",
}
SMALL_BLIND_OPEN_HANDS = LATE_OPEN_HANDS | {"K8s", "Q8s", "J8s", "A9o", "KTo", "QTo", "JTo"}
POSITION_OPEN_RANGES = {
    "UTG": EARLY_OPEN_HANDS,
    "UTG+1": EARLY_OPEN_HANDS,
    "MP": MIDDLE_OPEN_HANDS,
    "LJ": MIDDLE_OPEN_HANDS,
    "HJ": MIDDLE_OPEN_HANDS,
    "CO": LATE_OPEN_HANDS,
    "BTN": BUTTON_OPEN_HANDS,
    "SB": SMALL_BLIND_OPEN_HANDS,
}
POSTFLOP_AGGRESSIVE_ROLES = {"极化价值下注", "保护下注", "小频率范围下注", "半诈唬下注", "低频延迟诈唬"}


def battle_session_store_dir() -> Path:
    configured = os.environ.get("POKERCOACH_BATTLE_STORE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / ".data" / "battle_sessions"


def battle_session_store_path(session_id: str) -> Path:
    safe_id = "".join(character for character in session_id if character.isalnum() or character in {"_", "-"})
    return battle_session_store_dir() / f"{safe_id}.json"


def persist_battle_session(session: BattleSession) -> None:
    store_dir = battle_session_store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    path = battle_session_store_path(session.id)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(serialize_session(session), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_persisted_battle_session(session_id: str) -> BattleSession | None:
    path = battle_session_store_path(session_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_session(payload)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def serialize_session(session: BattleSession) -> dict[str, Any]:
    return {
        "version": 1,
        "id": session.id,
        "owner_id": session.owner_id,
        "mode": session.mode,
        "player_seat": session.player_seat,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "table_size": session.table_size,
        "hand_number": session.hand_number,
        "starting_stack_bb": session.starting_stack_bb,
        "seed": session.seed,
        "rng_state": json_safe_random_state(session.rng.getstate()),
        "players": [serialize_player(player) for player in session.players],
        "deck": list(session.deck),
        "board": list(session.board),
        "burned_cards": list(session.burned_cards),
        "pot_bb": session.pot_bb,
        "street": session.street,
        "dealer_index": session.dealer_index,
        "current_bet_bb": session.current_bet_bb,
        "min_raise_bb": session.min_raise_bb,
        "current_actor": session.current_actor,
        "acted_this_street": sorted(session.acted_this_street),
        "action_log": [serialize_action_event(event) for event in session.action_log],
        "table_events": [serialize_table_event(event) for event in session.table_events],
        "completed_hands": [history.model_dump(mode="json") for history in session.completed_hands],
        "result": session.result.model_dump(mode="json") if session.result else None,
        "is_session_complete": session.is_session_complete,
    }


def deserialize_session(payload: dict[str, Any]) -> BattleSession:
    rng = Random(payload["seed"])
    rng.setstate(tuple_random_state(payload["rng_state"]))
    action_log = [deserialize_action_event(event) for event in payload.get("action_log", [])]
    action_by_id = {event.id: event for event in action_log}
    players = [
        deserialize_player(player, action_by_id)
        for player in payload.get("players", [])
    ]
    session = BattleSession(
        id=payload["id"],
        mode=payload.get("mode", "spectate"),
        player_seat=payload.get("player_seat"),
        table_size=int(payload["table_size"]),
        hand_number=int(payload["hand_number"]),
        starting_stack_bb=float(payload["starting_stack_bb"]),
        seed=payload["seed"],
        rng=rng,
        players=players,
        deck=list(payload.get("deck", [])),
        owner_id=payload.get("owner_id"),
        created_at=payload.get("created_at") or inferred_session_time(payload),
        updated_at=payload.get("updated_at") or inferred_session_time(payload),
        board=list(payload.get("board", [])),
        burned_cards=list(payload.get("burned_cards", [])),
        pot_bb=float(payload.get("pot_bb", 0)),
        street=payload.get("street", "preflop"),
        dealer_index=int(payload.get("dealer_index", 0)),
        current_bet_bb=float(payload.get("current_bet_bb", 1)),
        min_raise_bb=float(payload.get("min_raise_bb", 1)),
        current_actor=payload.get("current_actor"),
        acted_this_street=set(payload.get("acted_this_street", [])),
        action_log=action_log,
        table_events=[deserialize_table_event(event) for event in payload.get("table_events", [])],
        completed_hands=[
            BattleHandHistorySnapshot.model_validate(history)
            for history in payload.get("completed_hands", [])
        ],
        result=BattleResultSnapshot.model_validate(payload["result"]) if payload.get("result") else None,
        is_session_complete=bool(payload.get("is_session_complete", False)),
    )
    return session


def serialize_player(player: PlayerState) -> dict[str, Any]:
    return {
        "index": player.index,
        "agent_id": player.agent.id,
        "agent_name": player.agent.name,
        "is_human": player.is_human,
        "position": player.position,
        "stack_bb": player.stack_bb,
        "hole_cards": list(player.hole_cards),
        "folded": player.folded,
        "all_in": player.all_in,
        "street_bet_bb": player.street_bet_bb,
        "total_committed_bb": player.total_committed_bb,
        "last_action_id": player.last_action.id if player.last_action else None,
    }


def deserialize_player(payload: dict[str, Any], action_by_id: dict[str, ActionEvent]) -> PlayerState:
    is_human = bool(payload.get("is_human", False))
    agent = human_persona(payload.get("agent_id", "local").removeprefix("human_"), payload.get("agent_name", "Alex")) if is_human else AGENT_BY_ID.get(payload["agent_id"], AGENTS[int(payload["index"])])
    player = PlayerState(
        index=int(payload["index"]),
        agent=agent,
        position=payload.get("position", ""),
        stack_bb=float(payload.get("stack_bb", 0)),
        is_human=is_human,
        hole_cards=list(payload.get("hole_cards", [])),
        folded=bool(payload.get("folded", False)),
        all_in=bool(payload.get("all_in", False)),
        street_bet_bb=float(payload.get("street_bet_bb", 0)),
        total_committed_bb=float(payload.get("total_committed_bb", 0)),
    )
    last_action_id = payload.get("last_action_id")
    player.last_action = action_by_id.get(last_action_id) if last_action_id else None
    return player


def serialize_action_event(event: ActionEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "seat_index": event.seat_index,
        "position": event.position,
        "agent_id": event.agent_id,
        "agent_name": event.agent_name,
        "street": event.street,
        "action": event.action,
        "label": event.label,
        "amount_bb": event.amount_bb,
        "total_bet_bb": event.total_bet_bb,
        "pot_bb": event.pot_bb,
        "equity": event.equity,
        "pot_odds": event.pot_odds,
        "note": event.note,
        "decision": serialize_decision_trace(event.decision) if event.decision else None,
        "created_at": event.created_at,
    }


def deserialize_action_event(payload: dict[str, Any]) -> ActionEvent:
    return ActionEvent(
        id=payload["id"],
        seat_index=int(payload["seat_index"]),
        position=payload["position"],
        agent_id=payload["agent_id"],
        agent_name=payload["agent_name"],
        street=payload["street"],
        action=payload["action"],
        label=payload["label"],
        amount_bb=float(payload.get("amount_bb", 0)),
        total_bet_bb=float(payload.get("total_bet_bb", 0)),
        pot_bb=float(payload.get("pot_bb", 0)),
        equity=payload.get("equity"),
        pot_odds=payload.get("pot_odds"),
        note=payload.get("note", ""),
        decision=deserialize_decision_trace(payload["decision"]) if payload.get("decision") else None,
        created_at=payload["created_at"],
    )


def serialize_table_event(event: TableEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "event": event.event,
        "street": event.street,
        "label": event.label,
        "seat_index": event.seat_index,
        "cards": list(event.cards),
        "burn_card": event.burn_card,
        "pot_bb": event.pot_bb,
        "created_at": event.created_at,
    }


def deserialize_table_event(payload: dict[str, Any]) -> TableEvent:
    return TableEvent(
        id=payload["id"],
        event=payload["event"],
        street=payload["street"],
        label=payload["label"],
        seat_index=payload.get("seat_index"),
        cards=list(payload.get("cards", [])),
        burn_card=payload.get("burn_card"),
        pot_bb=float(payload.get("pot_bb", 0)),
        created_at=payload["created_at"],
    )


def serialize_decision_trace(trace: DecisionTrace) -> dict[str, Any]:
    return {
        "source": trace.source,
        "engine": trace.engine,
        "equity_samples": trace.equity_samples,
        "policy_profile": trace.policy_profile,
        "hand_class": trace.hand_class,
        "range_bucket": trace.range_bucket,
        "range_role": trace.range_role,
        "range_frequency": trace.range_frequency,
        "board_texture": trace.board_texture,
        "equity": trace.equity,
        "pot_odds": trace.pot_odds,
        "spr": trace.spr,
        "pressure": trace.pressure,
        "confidence": trace.confidence,
        "recommended_total_bb": trace.recommended_total_bb,
        "tags": list(trace.tags),
        "summary": trace.summary,
        "candidates": [serialize_decision_candidate(candidate) for candidate in trace.candidates],
    }


def deserialize_decision_trace(payload: dict[str, Any]) -> DecisionTrace:
    return DecisionTrace(
        source=payload["source"],
        engine=payload["engine"],
        equity_samples=int(payload.get("equity_samples", 0)),
        policy_profile=payload["policy_profile"],
        hand_class=payload["hand_class"],
        range_bucket=payload["range_bucket"],
        range_role=payload["range_role"],
        range_frequency=float(payload.get("range_frequency", 0)),
        board_texture=payload["board_texture"],
        equity=payload.get("equity"),
        pot_odds=float(payload.get("pot_odds", 0)),
        spr=float(payload.get("spr", 0)),
        pressure=float(payload.get("pressure", 0)),
        confidence=float(payload.get("confidence", 0)),
        recommended_total_bb=float(payload.get("recommended_total_bb", 0)),
        tags=list(payload.get("tags", [])),
        summary=payload["summary"],
        candidates=[
            deserialize_decision_candidate(candidate)
            for candidate in payload.get("candidates", [])
        ],
    )


def serialize_decision_candidate(candidate: DecisionCandidate) -> dict[str, Any]:
    return {
        "action": candidate.action,
        "target_total_bb": candidate.target_total_bb,
        "ev_bb": candidate.ev_bb,
        "weight": candidate.weight,
        "reason": candidate.reason,
        "is_chosen": candidate.is_chosen,
    }


def deserialize_decision_candidate(payload: dict[str, Any]) -> DecisionCandidate:
    return DecisionCandidate(
        action=payload["action"],
        target_total_bb=float(payload.get("target_total_bb", 0)),
        ev_bb=float(payload.get("ev_bb", 0)),
        weight=float(payload.get("weight", 0)),
        reason=payload.get("reason", ""),
        is_chosen=bool(payload.get("is_chosen", False)),
    )


def json_safe_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [json_safe_random_state(item) for item in value]
    return value


def tuple_random_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tuple_random_state(item) for item in value)
    return value


def list_agents() -> list[BattleAgentProfile]:
    return [agent.snapshot() for agent in AGENTS]


def list_battle_sessions(owner_id: str) -> list[BattleSessionSummarySnapshot]:
    sessions_by_id: dict[str, BattleSession] = {}
    for session_id, session in SESSIONS.items():
        if session.owner_id == owner_id:
            sessions_by_id[session_id] = session

    for path in battle_session_store_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("owner_id") != owner_id:
            continue
        session_id = payload.get("id")
        if not isinstance(session_id, str) or session_id in sessions_by_id:
            continue
        session = deserialize_session(payload)
        sessions_by_id[session.id] = session
        SESSIONS[session.id] = session

    return sorted(
        (battle_session_summary(session) for session in sessions_by_id.values()),
        key=lambda summary: summary.updated_at,
        reverse=True,
    )


def battle_session_summary(session: BattleSession) -> BattleSessionSummarySnapshot:
    last_table_event = session.table_events[-1] if session.table_events else None
    last_action = session.action_log[-1] if session.action_log else None
    if last_table_event and last_action:
        if last_table_event.created_at >= last_action.created_at:
            last_event_label = last_table_event.label
        else:
            last_event_label = f"{last_action.position} {last_action.agent_name} {last_action.label}"
    elif last_table_event:
        last_event_label = last_table_event.label
    elif last_action:
        last_event_label = f"{last_action.position} {last_action.agent_name} {last_action.label}"
    else:
        last_event_label = "牌局准备中"

    return BattleSessionSummarySnapshot(
        id=session.id,
        table_size=session.table_size,
        hand_number=session.hand_number,
        street=session.street,
        stage_label=street_label(session.street),
        pot_bb=round_bb(session.pot_bb),
        active_seat=session.current_actor,
        completed_hand_count=len(session.completed_hands),
        is_complete=session.street == "complete",
        is_session_complete=session.is_session_complete,
        last_event_label=last_event_label,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def create_battle_session(payload: BattleSessionCreate, owner_id: str | None = None, owner_name: str = "Alex") -> BattleSession:
    if payload.observer_seat >= payload.table_size:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="observer_seat outside table")
    if payload.mode == "play":
        player_seat = payload.player_seat if payload.player_seat is not None else payload.observer_seat
        if player_seat >= payload.table_size:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="player_seat outside table")
        observer_seat = player_seat
    else:
        player_seat = None
        observer_seat = payload.observer_seat

    seed = payload.seed or uuid4().hex
    rng = Random(seed)
    deck = fresh_deck(rng)
    created_at = now_iso()
    players = [
        PlayerState(
            index=index,
            agent=human_persona(owner_id, owner_name) if player_seat == index else AGENTS[index],
            position="",
            stack_bb=payload.starting_stack_bb,
            is_human=player_seat == index,
        )
        for index in range(payload.table_size)
    ]
    session = BattleSession(
        id=f"bat_{uuid4().hex[:12]}",
        mode=payload.mode,
        player_seat=player_seat,
        table_size=payload.table_size,
        hand_number=1,
        starting_stack_bb=payload.starting_stack_bb,
        seed=seed,
        rng=rng,
        players=players,
        deck=deck,
        owner_id=owner_id,
        created_at=created_at,
        updated_at=created_at,
    )
    assign_positions(session)
    deal_hole_cards(session)
    record_table_event(session, "hand_start", f"第 {session.hand_number} 手开始")
    post_blinds(session)
    session.current_actor = preflop_first_actor(session)
    SESSIONS[session.id] = session
    persist_battle_session(session)
    return session


def get_battle_session(session_id: str, owner_id: str | None = None) -> BattleSession:
    session = SESSIONS.get(session_id)
    if session is None:
        session = load_persisted_battle_session(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Battle session not found")
        SESSIONS[session.id] = session
    ensure_battle_session_owner(session, owner_id)
    return session


def advance_battle_session(session_id: str, steps: int, owner_id: str | None = None) -> BattleSession:
    session = get_battle_session(session_id, owner_id=owner_id)
    for _ in range(steps):
        if session.is_session_complete or session.street == "complete" or session_waiting_for_player(session):
            break
        advance_one_action(session)
    touch_battle_session(session)
    persist_battle_session(session)
    return session


def apply_player_battle_action(
    session_id: str,
    payload: BattlePlayerActionRequest,
    owner_id: str | None = None,
) -> BattleSession:
    session = get_battle_session(session_id, owner_id=owner_id)
    if session.mode != "play":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session is spectator-only")
    if session.street == "complete" or session.is_session_complete:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Current hand is complete")
    if session.current_actor is None:
        session.current_actor = next_actor_after(session, session.dealer_index)
    if session.current_actor is None:
        advance_street(session)
        touch_battle_session(session)
        persist_battle_session(session)
        return session

    player = session.players[session.current_actor]
    if not player.is_human:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Waiting for an agent action")

    recommended = recommended_player_decision(session, player)
    decision = build_player_decision(session, player, payload.action, payload.target_total_bb)
    record_player_action_mistake(session, player, decision, recommended, owner_id or session.owner_id)
    apply_decision(session, player, decision)

    if count_contenders(session) <= 1:
        award_uncontested_pot(session)
    elif betting_closed_for_hand(session):
        settle_showdown(session)
    elif betting_round_complete(session):
        advance_street(session)
    else:
        session.current_actor = next_actor_after(session, player.index)

    touch_battle_session(session)
    persist_battle_session(session)
    return session


def build_player_decision(
    session: BattleSession,
    player: PlayerState,
    action: BattlePlayerAction,
    target_total_bb: float | None,
) -> BrainDecision:
    call_needed = max(session.current_bet_bb - player.street_bet_bb, 0)
    normalized_action: BattleActionType = action
    if action == "check" and call_needed > 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cannot check facing a bet")
    if action == "call" and call_needed <= 0:
        normalized_action = "check"
    if action == "bet" and session.current_bet_bb > 0:
        normalized_action = "raise"
    if action == "raise" and call_needed <= 0 and session.current_bet_bb <= 0:
        normalized_action = "bet"

    if normalized_action in {"fold", "check", "call"}:
        target = session.current_bet_bb if normalized_action == "call" else player.street_bet_bb
    else:
        minimum_target = session.current_bet_bb + session.min_raise_bb if session.current_bet_bb > 0 else max(1.0, session.min_raise_bb)
        target = max(target_total_bb or minimum_target, minimum_target)
        target = min(target, player.stack_bb + player.street_bet_bb)
        if target <= player.street_bet_bb + call_needed + 0.001:
            normalized_action = "call" if call_needed > 0 else "check"
            target = session.current_bet_bb if call_needed > 0 else player.street_bet_bb

    equity = estimate_equity(session, player)
    pot_odds = call_needed / (session.pot_bb + call_needed) if call_needed > 0 else 0
    note = f"玩家选择 {action_label(normalized_action)}。"
    return BrainDecision(normalized_action, target, equity, pot_odds, note, None)


def recommended_player_decision(session: BattleSession, player: PlayerState) -> BrainDecision:
    rng_state = session.rng.getstate()
    try:
        return decide_action(session, player)
    finally:
        session.rng.setstate(rng_state)


def record_player_action_mistake(
    session: BattleSession,
    player: PlayerState,
    player_decision: BrainDecision,
    recommended: BrainDecision,
    owner_id: str | None,
) -> None:
    if not owner_id or not recommended.trace:
        return
    if not should_record_player_mistake(player_decision, recommended):
        return

    trace = recommended.trace
    recommended_candidate = chosen_candidate(trace.candidates) or max(trace.candidates, key=lambda item: item.ev_bb, default=None)
    user_candidate = closest_candidate_for_action(trace.candidates, player_decision.action, player_decision.target_total_bb)
    ev_delta = mistake_ev_delta(player_decision, recommended, user_candidate, recommended_candidate)
    user_reason = f" {user_candidate.reason}" if user_candidate else ""
    recommended_sizing = (
        f"到 {round_bb(recommended.target_total_bb)}BB"
        if recommended.action in {"bet", "raise", "all_in"}
        else ""
    )

    record_battle_mistake(
        BattleMistakeCreate(
            owner_id=owner_id,
            session_id=session.id,
            hand_number=session.hand_number,
            title=f"{player.position} 决策偏差",
            subtitle=f"{street_label(session.street)} · 底池 {round_bb(session.pot_bb)}BB · 推荐{action_label(recommended.action)}",
            street=session.street,
            position=player.position,
            hero_cards=list(player.hole_cards),
            board=list(session.board),
            user_action=player_decision.action,
            user_action_label=action_label(player_decision.action),
            user_total_bb=round_bb(player_decision.target_total_bb),
            recommended_action=recommended.action,
            recommended_action_label=action_label(recommended.action),
            recommended_total_bb=round_bb(recommended.target_total_bb),
            ev_delta_bb=ev_delta,
            scenario=BattleMistakeScenario(
                session_id=session.id,
                hand_number=session.hand_number,
                table_size=session.table_size,
                street=session.street,
                position=player.position,
                hero_name=player.agent.name,
                hero_cards=list(player.hole_cards),
                board=list(session.board),
                pot_bb=round_bb(session.pot_bb),
                current_bet_bb=round_bb(session.current_bet_bb),
                stack_bb=round_bb(player.stack_bb),
                committed_bb=round_bb(player.total_committed_bb),
                spr=round_bb(trace.spr),
                table_seats=[
                    BattleMistakeTableSeat(
                        seat_index=seat.index,
                        position=seat.position,
                        name=seat.agent.name,
                        stack_bb=round_bb(seat.stack_bb),
                        committed_bb=round_bb(seat.total_committed_bb),
                        status=seat.status,
                        is_hero=seat.index == player.index,
                    )
                    for seat in session.players
                ],
                tags=trace.tags,
            ),
            candidates=[
                BattleMistakeCandidate(
                    action=candidate.action,
                    label=action_label(candidate.action),
                    target_total_bb=round_bb(candidate.target_total_bb),
                    ev_bb=round_bb(candidate.ev_bb),
                    weight=round(candidate.weight, 3),
                    is_recommended=candidate.is_chosen,
                    reason=candidate.reason,
                )
                for candidate in trace.candidates
            ],
            why_wrong=(
                f"推荐引擎给出的主线是「{trace.summary}」。"
                f"你的{action_label(player_decision.action)}相对推荐少约 {ev_delta:.1f}BB EV。{user_reason}"
            ),
            correct_play=(
                f"这类 {trace.range_bucket} spot 优先{action_label(recommended.action)}{recommended_sizing}。"
                f"{recommended_candidate.reason if recommended_candidate else recommended.note}"
            ),
        )
    )


def should_record_player_mistake(player_decision: BrainDecision, recommended: BrainDecision) -> bool:
    if not recommended.trace:
        return False
    player_action = canonical_mistake_action(player_decision.action)
    recommended_action = canonical_mistake_action(recommended.action)
    if player_action != recommended_action:
        return True
    if player_action in {"bet_raise"}:
        sizing_gap = abs(player_decision.target_total_bb - recommended.target_total_bb)
        return sizing_gap >= max(2.0, recommended.target_total_bb * 0.35)
    user_candidate = closest_candidate_for_action(
        recommended.trace.candidates,
        player_decision.action,
        player_decision.target_total_bb,
    )
    recommended_candidate = chosen_candidate(recommended.trace.candidates)
    if user_candidate and recommended_candidate:
        return recommended_candidate.ev_bb - user_candidate.ev_bb >= 0.35
    return False


def canonical_mistake_action(action: BattleActionType) -> str:
    if action in {"bet", "raise", "all_in"}:
        return "bet_raise"
    if action in {"check", "call"}:
        return "continue"
    return action


def closest_candidate_for_action(
    candidates: list[DecisionCandidate],
    action: BattleActionType,
    target_total_bb: float,
) -> DecisionCandidate | None:
    canonical = canonical_mistake_action(action)
    matching = [
        candidate
        for candidate in candidates
        if canonical_mistake_action(candidate.action) == canonical
    ]
    if not matching:
        return None
    return min(matching, key=lambda candidate: abs(candidate.target_total_bb - target_total_bb))


def mistake_ev_delta(
    player_decision: BrainDecision,
    recommended: BrainDecision,
    user_candidate: DecisionCandidate | None,
    recommended_candidate: DecisionCandidate | None,
) -> float:
    if user_candidate and recommended_candidate:
        return round_bb(max(recommended_candidate.ev_bb - user_candidate.ev_bb, 0.1))
    if canonical_mistake_action(player_decision.action) != canonical_mistake_action(recommended.action):
        return 0.6
    return 0.1


def start_next_hand(session_id: str, owner_id: str | None = None) -> BattleSession:
    session = get_battle_session(session_id, owner_id=owner_id)
    if session.is_session_complete:
        return session

    if session.street != "complete":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Current hand is still active")

    archive_completed_hand(session)
    if not prepare_next_hand_roster(session):
        session.is_session_complete = True
        touch_battle_session(session)
        persist_battle_session(session)
        return session

    session.hand_number += 1
    session.deck = fresh_deck(session.rng)
    session.board = []
    session.burned_cards = []
    session.pot_bb = 0
    session.street = "preflop"
    session.current_bet_bb = 1
    session.min_raise_bb = 1
    session.current_actor = None
    session.acted_this_street = set()
    session.action_log = []
    session.table_events = []
    session.result = None
    session.is_session_complete = False

    for player in session.players:
        player.hole_cards = []
        player.folded = False
        player.all_in = False
        player.street_bet_bb = 0
        player.total_committed_bb = 0
        player.last_action = None

    assign_positions(session)
    deal_hole_cards(session)
    record_table_event(session, "hand_start", f"第 {session.hand_number} 手开始")
    post_blinds(session)
    session.current_actor = preflop_first_actor(session)
    touch_battle_session(session)
    persist_battle_session(session)
    return session


def mark_session_complete_if_terminal(session: BattleSession) -> None:
    if len([player for player in session.players if player.stack_bb > 0.001]) < 2:
        session.is_session_complete = True


def prepare_next_hand_roster(session: BattleSession) -> bool:
    survivors = [player for player in session.players if player.stack_bb > 0.001]
    if len(survivors) < 2:
        return False
    if session.mode == "play" and session.player_seat is not None and all(not player.is_human for player in survivors):
        return False

    previous_dealer = session.dealer_index
    next_dealer_old_index = next(
        (player.index for player in survivors if player.index > previous_dealer),
        survivors[0].index,
    )
    old_to_new = {player.index: new_index for new_index, player in enumerate(survivors)}

    for new_index, player in enumerate(survivors):
        player.index = new_index

    session.players = survivors
    session.table_size = len(survivors)
    session.dealer_index = old_to_new[next_dealer_old_index]
    session.player_seat = next((player.index for player in survivors if player.is_human), None)
    return True


def battle_hand_history(
    session_id: str,
    observer_seat: int,
    hand_number: int | None = None,
    owner_id: str | None = None,
) -> BattleHandHistorySnapshot:
    session = get_battle_session(session_id, owner_id=owner_id)
    observer_seat = normalize_seat(observer_seat, session.table_size)
    if hand_number is not None and hand_number != session.hand_number:
        archived = next((hand for hand in session.completed_hands if hand.hand_number == hand_number), None)
        if archived is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Battle hand history not found")
        return history_for_observer(archived, observer_seat)

    if hand_number is None and session.street != "complete" and session.completed_hands:
        return history_for_observer(session.completed_hands[-1], observer_seat)

    return build_hand_history(session, observer_seat)


def list_battle_hand_summaries(session_id: str, owner_id: str | None = None) -> list[BattleHandSummarySnapshot]:
    session = get_battle_session(session_id, owner_id=owner_id)
    return [hand_summary(hand) for hand in session.completed_hands]


def ensure_battle_session_owner(session: BattleSession, owner_id: str | None) -> None:
    if owner_id is None or session.owner_id is None:
        return
    if session.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Battle session not found")


def touch_battle_session(session: BattleSession) -> None:
    session.updated_at = now_iso()


def inferred_session_time(payload: dict[str, Any]) -> str:
    for collection_name in ("table_events", "action_log"):
        collection = payload.get(collection_name)
        if isinstance(collection, list) and collection:
            created_at = collection[-1].get("created_at")
            if isinstance(created_at, str):
                return created_at
    return now_iso()


def archive_completed_hand(session: BattleSession) -> None:
    if session.street != "complete":
        return
    if any(hand.hand_number == session.hand_number for hand in session.completed_hands):
        return
    session.completed_hands.append(build_hand_history(session, observer_seat=0))


def build_hand_history(session: BattleSession, observer_seat: int) -> BattleHandHistorySnapshot:
    snapshot = session.snapshot(observer_seat)
    decisions = [event for event in session.action_log if event.decision is not None]
    return BattleHandHistorySnapshot(
        id=f"{session.id}_hand_{session.hand_number}",
        session_id=session.id,
        hand_number=session.hand_number,
        table_size=session.table_size,
        seed=session.seed,
        street=session.street,
        stage_label=street_label(session.street),
        board=session.board,
        burned_cards=session.burned_cards,
        observer_seat=snapshot.observer_seat,
        seats=snapshot.seats,
        action_timeline=build_action_timeline(session.action_log),
        table_events=[event.snapshot() for event in session.table_events],
        replay_events=build_replay_events(session.action_log, session.table_events),
        result=session.result,
        review_insights=build_review_insights(session, decisions),
        action_count=len(session.action_log),
        decision_count=len(decisions),
        is_complete=session.street == "complete",
    )


def history_for_observer(history: BattleHandHistorySnapshot, observer_seat: int) -> BattleHandHistorySnapshot:
    seats = [
        seat.model_copy(update={"is_observer": seat.index == observer_seat, "is_active": False})
        for seat in history.seats
    ]
    return history.model_copy(update={"observer_seat": observer_seat, "seats": seats})


def hand_summary(history: BattleHandHistorySnapshot) -> BattleHandSummarySnapshot:
    completed_at = history.table_events[-1].created_at if history.table_events else now_iso()
    return BattleHandSummarySnapshot(
        id=history.id,
        session_id=history.session_id,
        hand_number=history.hand_number,
        board=history.board,
        winners=history.result.winners if history.result else [],
        summary=history.result.summary if history.result else "牌局进行中",
        action_count=history.action_count,
        decision_count=history.decision_count,
        replay_count=len(history.replay_events),
        completed_at=completed_at,
    )


def advance_one_action(session: BattleSession) -> None:
    if count_contenders(session) <= 1:
        award_uncontested_pot(session)
        return

    if betting_closed_for_hand(session):
        settle_showdown(session)
        return

    if session.current_actor is None:
        session.current_actor = next_actor_after(session, session.dealer_index)
        if session.current_actor is None:
            advance_street(session)
            return

    player = session.players[session.current_actor]
    if player.is_human:
        return
    if player.folded or player.all_in:
        session.current_actor = next_actor_after(session, session.current_actor)
        return

    decision = decide_action(session, player)
    apply_decision(session, player, decision)

    if count_contenders(session) <= 1:
        award_uncontested_pot(session)
        return

    if betting_closed_for_hand(session):
        settle_showdown(session)
        return

    if betting_round_complete(session):
        advance_street(session)
    else:
        session.current_actor = next_actor_after(session, player.index)


def session_waiting_for_player(session: BattleSession) -> bool:
    if session.mode != "play" or session.current_actor is None:
        return False
    if session.current_actor >= len(session.players):
        return False
    player = session.players[session.current_actor]
    return player.is_human and not player.folded and not player.all_in and session.street != "complete"


def decide_action(session: BattleSession, player: PlayerState) -> BrainDecision:
    call_needed = max(session.current_bet_bb - player.street_bet_bb, 0)
    pot_odds = call_needed / (session.pot_bb + call_needed) if call_needed > 0 else 0
    equity = estimate_equity(session, player)
    hand_class = hand_notation(player.hole_cards)
    preflop_strength = estimate_preflop_strength(player.hole_cards, player.position)
    open_spot = is_preflop_open_spot(session)
    if session.street == "preflop":
        range_profile = build_range_profile(session, player, hand_class, call_needed, open_spot)
        strength = max(0.05, min(preflop_strength * 0.72 + range_profile.frequency * 0.28, 0.98))
    else:
        strength = equity
        range_profile = build_postflop_range_profile(session, player, strength, pot_odds, call_needed)
    profile = player.agent
    pressure = profile.aggression * 0.08 + profile.bluff * 0.05 + (profile.exploit_score - 86) / 100 * 0.035
    discipline = profile.discipline * 0.06 + (profile.gto_score - 86) / 100 * 0.035
    spr = stack_to_pot_ratio(session, player)
    short_stack_pressure = max(0.0, min((45 - player.stack_bb) / 45, 1.0)) * 0.055
    low_spr_pressure = max(0.0, min((5 - spr) / 5, 1.0)) * 0.035
    deep_stack_discipline = max(0.0, min((player.stack_bb - 120) / 180, 1.0)) * 0.035
    pressure += short_stack_pressure + low_spr_pressure
    discipline += deep_stack_discipline
    mix = session.rng.random()
    can_raise = call_needed <= 0 or player.index not in session.acted_this_street
    candidates = build_decision_candidates(
        session=session,
        player=player,
        strength=strength,
        pot_odds=pot_odds,
        pressure=pressure,
        can_raise=can_raise,
        call_needed=call_needed,
        range_profile=range_profile,
        open_spot=open_spot,
    )

    def decision(action: BattleActionType, target_total_bb: float, message: str) -> BrainDecision:
        note = thinking_note(session, strength, pot_odds, message)
        trace = build_decision_trace(
            session=session,
            player=player,
            action=action,
            target_total_bb=target_total_bb,
            hand_class=hand_class,
            strength=strength,
            equity=equity,
            pot_odds=pot_odds,
            pressure=pressure,
            message=message,
            candidates=candidates,
            range_profile=range_profile,
        )
        return BrainDecision(action, target_total_bb, equity, pot_odds, note, trace)

    if call_needed <= 0 or open_spot:
        if session.street == "preflop":
            bet_threshold = POSITION_OPEN_FLOORS.get(player.position, 0.52) - (profile.looseness - 0.50) * 0.07
            range_open = range_profile.role in {"标准开池", "混合开池", "价值开池"}
        else:
            bet_threshold = postflop_bet_threshold(range_profile)
            range_open = range_profile.role in POSTFLOP_AGGRESSIVE_ROLES
        bluff_threshold = profile.bluff * (0.28 if session.street != "preflop" else 0.16)
        if strength >= bet_threshold or (range_open and mix <= range_profile.frequency) or mix < bluff_threshold:
            target = choose_bet_size(session, player, strength, range_profile)
            action: BattleActionType = "bet" if session.current_bet_bb == 0 else "raise"
            return decision(action, target, f"{range_profile.note} 主动下注，利用范围/权益优势给对手压力。")
        if open_spot:
            if range_profile.role == "混合开池" and mix <= range_profile.frequency * 0.35:
                return decision("call", session.current_bet_bb, f"{range_profile.note} 低频跟入，保留底池参与权。")
            return decision("fold", player.street_bet_bb, f"当前组合不在 {player.position} 开池范围内，弃牌。")
        return decision("check", player.street_bet_bb, "无须投入更多筹码，保留范围。")

    continue_threshold = max(pot_odds - 0.02, 0.08)
    raise_threshold = min(max(pot_odds + 0.32 - pressure, 0.56), 0.86)
    if session.street == "preflop":
        if range_profile.role == "价值3bet":
            raise_threshold = min(raise_threshold, 0.66)
        elif range_profile.role == "混合3bet":
            raise_threshold = min(raise_threshold, 0.72)
        elif range_profile.role == "弃牌":
            continue_threshold = max(continue_threshold, 0.62)
            raise_threshold = 1.0
    else:
        if range_profile.role == "价值加注":
            raise_threshold = min(raise_threshold, 0.68)
        elif range_profile.role == "半诈唬加注":
            raise_threshold = min(raise_threshold, 0.72)
            continue_threshold = max(min(continue_threshold, pot_odds + 0.08), 0.12)
        elif range_profile.role in {"赔率防守", "边缘抓诈"}:
            continue_threshold = max(min(continue_threshold, pot_odds + 0.06), 0.10)
        elif range_profile.role == "纪律弃牌":
            continue_threshold = max(continue_threshold, 0.56)
            raise_threshold = 1.0
    if session.street == "preflop" and session.current_bet_bb >= 8:
        raise_threshold = max(raise_threshold, 0.88)
        continue_threshold = max(continue_threshold, 0.48)
    if session.current_bet_bb >= 24 and (strength < 0.94 or hand_class not in PREMIUM_HANDS):
        raise_threshold = 1.0
    bluff_raise = mix < profile.bluff * 0.06 and call_needed <= session.pot_bb * 0.32

    range_raise = (
        session.street == "preflop" and range_profile.role in {"价值3bet", "混合3bet"} and mix <= range_profile.frequency
    ) or (
        session.street != "preflop" and range_profile.role in {"价值加注", "半诈唬加注"} and mix <= range_profile.frequency
    )
    if can_raise and (strength >= raise_threshold or range_raise or bluff_raise):
        target = choose_raise_size(session, player, strength, range_profile)
        if target > session.current_bet_bb and target > player.street_bet_bb + call_needed:
            return decision("raise", target, f"{range_profile.note} 权益超过继续阈值，采用加注扩大价值或制造弃牌权益。")

    range_continue = (
        session.street == "preflop" and range_profile.role == "防守跟注" and mix <= range_profile.frequency
    ) or (
        session.street != "preflop" and range_profile.role in {"赔率防守", "边缘抓诈"} and mix <= range_profile.frequency
    )
    if strength + pressure + discipline >= continue_threshold or range_continue:
        return decision("call", session.current_bet_bb, f"{range_profile.note} 赔率允许继续，跟注保留对手较宽范围。")

    return decision("fold", player.street_bet_bb, "当前组合不在继续范围内，弃牌。")


def apply_decision(session: BattleSession, player: PlayerState, decision: BrainDecision) -> None:
    call_needed = max(session.current_bet_bb - player.street_bet_bb, 0)
    amount_added = 0.0
    action = decision.action
    previous_current_bet = session.current_bet_bb
    previous_min_raise = session.min_raise_bb

    if action == "fold":
        player.folded = True
    elif action == "check":
        pass
    elif action == "call":
        amount_added = commit_to_pot(session, player, call_needed)
        if player.all_in:
            action = "all_in"
    elif action in ("bet", "raise"):
        target_total = min(max(decision.target_total_bb, session.current_bet_bb + session.min_raise_bb), player.stack_bb + player.street_bet_bb)
        amount_added = commit_to_pot(session, player, max(target_total - player.street_bet_bb, 0))
        if player.street_bet_bb > session.current_bet_bb:
            raise_delta = player.street_bet_bb - previous_current_bet
            session.current_bet_bb = player.street_bet_bb
            if raise_delta + 0.001 >= previous_min_raise:
                session.min_raise_bb = max(raise_delta, 1)
                session.acted_this_street = set()
        if player.all_in:
            action = "all_in"

    session.acted_this_street.add(player.index)
    event = ActionEvent(
        id=f"act_{len(session.action_log) + 1:04d}",
        seat_index=player.index,
        position=player.position,
        agent_id=player.agent.id,
        agent_name=player.agent.name,
        street=session.street,
        action=action,
        label=action_label(action),
        amount_bb=amount_added,
        total_bet_bb=player.street_bet_bb,
        pot_bb=session.pot_bb,
        equity=decision.equity,
        pot_odds=decision.pot_odds,
        note=decision.note,
        decision=decision.trace,
        created_at=now_iso(),
    )
    player.last_action = event
    session.action_log.append(event)


def betting_round_complete(session: BattleSession) -> bool:
    eligible = [player for player in session.players if not player.folded and not player.all_in]
    if len(eligible) <= 1:
        return True
    return all(
        player.index in session.acted_this_street and abs(player.street_bet_bb - session.current_bet_bb) < 0.001
        for player in eligible
    )


def is_preflop_open_spot(session: BattleSession) -> bool:
    return (
        session.street == "preflop"
        and session.current_bet_bb <= 1.0
        and all(event.action == "blind" for event in session.action_log)
    )


def advance_street(session: BattleSession) -> None:
    if session.street == "preflop":
        deal_board_cards(session, "flop", 3)
        session.street = "flop"
    elif session.street == "flop":
        deal_board_cards(session, "turn", 1)
        session.street = "turn"
    elif session.street == "turn":
        deal_board_cards(session, "river", 1)
        session.street = "river"
    elif session.street == "river":
        settle_showdown(session)
        return

    reset_street_bets(session)
    if betting_closed_for_hand(session):
        settle_showdown(session)
        return
    session.current_actor = postflop_first_actor(session)


def settle_showdown(session: BattleSession) -> None:
    complete_board_with_burns(session)
    contenders = [player for player in session.players if not player.folded]
    if not contenders:
        session.street = "complete"
        record_table_event(session, "hand_complete", "牌局结束", street="complete")
        mark_session_complete_if_terminal(session)
        archive_completed_hand(session)
        return

    board_cards = [Card.new(card) for card in session.board]
    scores_by_seat: dict[int, int] = {}
    for player in contenders:
        score = EVALUATOR.evaluate(board_cards, [Card.new(card) for card in player.hole_cards])
        scores_by_seat[player.index] = score

    side_pots = build_side_pots(session)
    result_side_pots: list[BattleSidePotSnapshot] = []
    summary_parts: list[str] = []
    all_winner_ids: list[int] = []
    awards_by_seat: dict[int, float] = {player.index: 0.0 for player in contenders}

    for pot_index, side_pot in enumerate(side_pots):
        eligible = [player for player in side_pot.eligible_players if player.index in scores_by_seat]
        if not eligible:
            continue

        best_score = min(scores_by_seat[player.index] for player in eligible)
        winners = [player for player in eligible if scores_by_seat[player.index] == best_score]
        share = side_pot.amount_bb / len(winners)
        for player in winners:
            player.stack_bb = round(player.stack_bb + share, 4)
            awards_by_seat[player.index] = awards_by_seat.get(player.index, 0) + share
            if player.index not in all_winner_ids:
                all_winner_ids.append(player.index)

        pot_label = "主池" if pot_index == 0 else f"边池 {pot_index}"
        winner_names = "、".join(f"{player.position} {player.agent.name}" for player in winners)
        class_name = made_hand_label(best_score)
        summary_parts.append(f"{winner_names} 以 {class_name} 赢下 {pot_label} {round_bb(side_pot.amount_bb)}BB")
        result_side_pots.append(
            BattleSidePotSnapshot(
                amount_bb=round_bb(side_pot.amount_bb),
                eligible_seats=[player.index for player in eligible],
                winners=[player.index for player in winners],
            )
        )

    session.result = BattleResultSnapshot(
        winners=all_winner_ids,
        summary="；".join(summary_parts),
        showdown=[f"{player.position} {player.agent.name}: {' '.join(player.hole_cards)}" for player in contenders],
        showdown_details=[
            BattleShowdownHandSnapshot(
                seat_index=player.index,
                position=player.position,
                agent_id=player.agent.id,
                agent_name=player.agent.name,
                hole_cards=player.hole_cards,
                made_hand=made_hand_label(scores_by_seat[player.index]),
                hand_rank=scores_by_seat[player.index],
                is_winner=player.index in all_winner_ids,
                won_bb=round_money(awards_by_seat.get(player.index, 0)),
            )
            for player in contenders
        ],
        side_pots=result_side_pots,
    )
    session.current_actor = None
    record_table_event(session, "showdown", "摊牌结算", street="showdown", cards=session.board)
    session.street = "complete"
    record_table_event(session, "hand_complete", "牌局结束", street="complete")
    mark_session_complete_if_terminal(session)
    archive_completed_hand(session)


def award_uncontested_pot(session: BattleSession) -> None:
    winner = next((player for player in session.players if not player.folded), None)
    if winner is None:
        session.street = "complete"
        record_table_event(session, "hand_complete", "牌局结束", street="complete")
        mark_session_complete_if_terminal(session)
        archive_completed_hand(session)
        return
    winner.stack_bb += session.pot_bb
    session.result = BattleResultSnapshot(
        winners=[winner.index],
        summary=f"{winner.position} {winner.agent.name} 无摊牌赢下 {round_bb(session.pot_bb)}BB",
        showdown=[],
        showdown_details=[],
        side_pots=[
            BattleSidePotSnapshot(
                amount_bb=round_bb(session.pot_bb),
                eligible_seats=[winner.index],
                winners=[winner.index],
            )
        ],
    )
    session.current_actor = None
    record_table_event(
        session,
        "uncontested",
        f"{winner.position} 无摊牌赢下底池",
        street=session.street,
        seat_index=winner.index,
    )
    session.street = "complete"
    record_table_event(session, "hand_complete", "牌局结束", street="complete")
    mark_session_complete_if_terminal(session)
    archive_completed_hand(session)


def build_side_pots(session: BattleSession) -> list[SidePot]:
    committed_levels = sorted(
        {
            round_bb(player.total_committed_bb)
            for player in session.players
            if player.total_committed_bb > 0
        }
    )
    side_pots: list[SidePot] = []
    previous_level = 0.0

    for level in committed_levels:
        contributors = [
            player
            for player in session.players
            if player.total_committed_bb + 0.001 >= level
        ]
        amount = round_bb((level - previous_level) * len(contributors))
        eligible_players = [player for player in contributors if not player.folded]

        if amount > 0 and eligible_players:
            if side_pots and [player.index for player in side_pots[-1].eligible_players] == [player.index for player in eligible_players]:
                side_pots[-1].amount_bb = round_bb(side_pots[-1].amount_bb + amount)
            else:
                side_pots.append(SidePot(amount_bb=amount, eligible_players=eligible_players))
        previous_level = level

    if not side_pots and session.pot_bb > 0:
        side_pots.append(
            SidePot(
                amount_bb=round_bb(session.pot_bb),
                eligible_players=[player for player in session.players if not player.folded],
            )
        )

    return side_pots


def estimate_equity(session: BattleSession, player: PlayerState, samples: int = EQUITY_SAMPLE_COUNT) -> float:
    if session.street == "preflop":
        return estimate_preflop_strength(player.hole_cards, player.position)

    known = set(player.hole_cards + session.board + session.burned_cards)
    remaining = [card for card in full_deck_codes() if card not in known]
    wins = 0.0
    board_needed = 5 - len(session.board)
    opponents = max(count_contenders(session) - 1, 1)

    for _ in range(samples):
        sample = session.rng.sample(remaining, min(len(remaining), board_needed + opponents * 2))
        runout = session.board + sample[:board_needed]
        cursor = board_needed
        hero_score = EVALUATOR.evaluate([Card.new(card) for card in runout], [Card.new(card) for card in player.hole_cards])
        best_opp = None
        for _ in range(opponents):
            opp_hole = sample[cursor : cursor + 2]
            cursor += 2
            if len(opp_hole) < 2:
                continue
            opp_score = EVALUATOR.evaluate([Card.new(card) for card in runout], [Card.new(card) for card in opp_hole])
            best_opp = opp_score if best_opp is None else min(best_opp, opp_score)
        if best_opp is None or hero_score < best_opp:
            wins += 1
        elif hero_score == best_opp:
            wins += 0.5
    return max(0.02, min(wins / samples, 0.98))


def estimate_preflop_strength(cards: list[str], position: str) -> float:
    if len(cards) != 2:
        return 0.5
    ranks = sorted((RANK_VALUE[card[0]] for card in cards), reverse=True)
    high, low = ranks
    suited = cards[0][1] == cards[1][1]
    pair = high == low
    gap = high - low

    if pair:
        base = 0.52 + (high - 2) / 12 * 0.36
    else:
        base = (high + low) / 28
        if suited:
            base += 0.08
        if high == 14:
            base += 0.05
        if gap <= 2:
            base += 0.04
        if gap >= 5:
            base -= 0.08
    if position in {"BTN", "CO"}:
        base += 0.05
    elif position in {"SB", "BB"}:
        base -= 0.01
    elif position.startswith("UTG"):
        base -= 0.05
    return max(0.05, min(base, 0.98))


def hand_notation(cards: list[str]) -> str:
    if len(cards) != 2:
        return "--"
    first, second = cards
    ranks = sorted([first[0].upper(), second[0].upper()], key=lambda rank: RANK_VALUE[rank], reverse=True)
    if ranks[0] == ranks[1]:
        return f"{ranks[0]}{ranks[1]}"
    suited = first[1] == second[1]
    return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"


def build_range_profile(
    session: BattleSession,
    player: PlayerState,
    hand_class: str,
    call_needed: float,
    open_spot: bool,
) -> RangeProfile:
    if session.street != "preflop":
        return RangeProfile(
            tier="翻后权益",
            role="胜率驱动",
            frequency=1.0,
            note="翻后改由胜率、赔率和牌面结构主导。",
        )

    if call_needed <= 0 or open_spot:
        open_range = POSITION_OPEN_RANGES.get(player.position, MIDDLE_OPEN_HANDS)
        if hand_class in PREMIUM_HANDS:
            return RangeProfile("核心价值", "价值开池", 1.0, f"{player.position} 核心价值范围。")
        if hand_class in open_range:
            return RangeProfile("标准范围", "标准开池", 0.92, f"{player.position} 标准开池范围。")
        if hand_class in PLAYABLE_DEFENDS and player.position in {"CO", "BTN", "SB"}:
            return RangeProfile("边缘混合", "混合开池", 0.38 + player.agent.looseness * 0.22, f"{player.position} 边缘混合开池。")
        return RangeProfile("弃牌范围", "弃牌", 0.08 + player.agent.bluff * 0.08, f"{player.position} 默认不进入开池范围。")

    if hand_class in PREMIUM_HANDS:
        return RangeProfile("核心价值", "价值3bet", 1.0, "核心价值手牌优先再加注。")
    if hand_class in VALUE_3BET_HANDS:
        return RangeProfile("价值3bet", "价值3bet", 0.86, "强牌进入高频 3bet 范围。")
    if hand_class in MIXED_3BET_HANDS:
        return RangeProfile("混合3bet", "混合3bet", 0.32 + player.agent.aggression * 0.24, "阻断牌和可玩性支持混合 3bet。")
    if hand_class in PLAYABLE_DEFENDS:
        return RangeProfile("防守范围", "防守跟注", 0.72, "手牌具备跟注防守价值。")
    return RangeProfile("弃牌范围", "弃牌", 0.04 + player.agent.bluff * 0.05, "不在当前防守范围内。")


def build_postflop_range_profile(
    session: BattleSession,
    player: PlayerState,
    strength: float,
    pot_odds: float,
    call_needed: float,
) -> RangeProfile:
    board = analyze_board(session.board)
    spr = stack_to_pot_ratio(session, player)
    multiway = count_contenders(session) > 2
    in_position = is_last_to_act(session, player)
    draw = has_draw_potential(player.hole_cards, session.board)
    aggression = player.agent.aggression
    bluff = player.agent.bluff
    discipline = player.agent.discipline
    postflop_lift = (player.agent.postflop_score - 86) / 100

    if call_needed <= 0:
        if strength >= 0.82 or (spr <= 2.5 and strength >= 0.66):
            fraction = 0.74 if board.wetness >= 0.55 or spr <= 3 else 0.58
            return RangeProfile(
                "坚果/强价值",
                "极化价值下注",
                min(0.96, 0.78 + aggression * 0.18 + postflop_lift * 0.10),
                f"{board.texture}，SPR {spr:.1f}，强价值范围优先扩大底池。",
                bet_fraction=fraction,
                raise_fraction=0.86,
            )
        if strength >= 0.58:
            if board.wetness >= 0.55 or multiway:
                return RangeProfile(
                    "中强价值",
                    "保护下注",
                    min(0.88, 0.58 + aggression * 0.22 + postflop_lift * 0.10),
                    f"{board.texture} 听牌密度高，中强牌需要收费保护。",
                    bet_fraction=0.62,
                    raise_fraction=0.72,
                )
            return RangeProfile(
                "范围优势",
                "小频率范围下注",
                min(0.78, 0.44 + aggression * 0.24 + (0.08 if in_position else 0) + postflop_lift * 0.08),
                f"{board.texture} 对跟注方命中率较低，适合小尺度持续下注。",
                bet_fraction=0.34,
                raise_fraction=0.62,
            )
        if draw:
            return RangeProfile(
                "听牌权益",
                "半诈唬下注",
                min(0.72, 0.30 + bluff * 0.34 + aggression * 0.10 + postflop_lift * 0.10),
                f"{board.texture} 上有补牌权益，用半诈唬争取弃牌权益。",
                bet_fraction=0.48 if board.wetness >= 0.55 else 0.36,
                raise_fraction=0.74,
            )
        if in_position and board.wetness <= 0.38 and not multiway:
            return RangeProfile(
                "空气/阻断",
                "低频延迟诈唬",
                min(0.32, 0.10 + bluff * 0.24),
                f"{board.texture} 且有位置，保留少量低频施压组合。",
                bet_fraction=0.33,
                raise_fraction=0.60,
            )
        return RangeProfile(
            "摊牌价值",
            "摊牌控池",
            max(0.08, 0.24 - discipline * 0.10),
            f"{board.texture}，权益不足以大规模下注，优先控池实现摊牌价值。",
            bet_fraction=0.33,
            raise_fraction=0.56,
        )

    if strength >= 0.82 or (spr <= 2.5 and strength >= 0.66):
        return RangeProfile(
            "强价值防守",
            "价值加注",
            min(0.92, 0.66 + aggression * 0.22 + postflop_lift * 0.08),
            f"{board.texture}，对手下注后强价值可以加注争取筹码入池。",
            bet_fraction=0.66,
            raise_fraction=0.88,
        )
    if draw and strength + bluff * 0.20 >= pot_odds + 0.12:
        return RangeProfile(
            "听牌防守",
            "半诈唬加注",
            min(0.66, 0.24 + bluff * 0.32 + aggression * 0.12 + postflop_lift * 0.10),
            f"{board.texture} 上有可观补牌权益，混合加注制造弃牌权益。",
            bet_fraction=0.50,
            raise_fraction=0.76,
        )
    if strength >= max(pot_odds + 0.10, 0.32):
        return RangeProfile(
            "跟注防守",
            "赔率防守",
            min(0.86, 0.56 + discipline * 0.22 + postflop_lift * 0.08),
            f"权益覆盖 {pot_odds:.0%} 赔率门槛，保留对手诈唬范围。",
            bet_fraction=0.45,
            raise_fraction=0.60,
        )
    if strength >= pot_odds and player.agent.discipline >= 0.82 and board.wetness <= 0.42:
        return RangeProfile(
            "薄防守",
            "边缘抓诈",
            0.32 + player.agent.discipline * 0.18,
            f"{board.texture} 下对手诈唬不足时低频抓诈，纪律性控制损失。",
            bet_fraction=0.40,
            raise_fraction=0.56,
        )
    return RangeProfile(
        "弃牌范围",
        "纪律弃牌",
        max(0.04, 0.16 - discipline * 0.08),
        f"权益低于赔率门槛，弃掉底部范围保留筹码。",
        bet_fraction=0.33,
        raise_fraction=0.54,
    )


def choose_bet_size(
    session: BattleSession,
    player: PlayerState,
    strength: float,
    range_profile: RangeProfile | None = None,
) -> float:
    if session.street == "preflop":
        return min(player.stack_bb + player.street_bet_bb, 2.5 if player.position in {"BTN", "CO"} else 3.0)
    fraction = range_profile.bet_fraction if range_profile and range_profile.bet_fraction is not None else None
    if fraction is None:
        fraction = 0.42 if strength < 0.62 else 0.66 if strength < 0.80 else 0.88
    return round_bb(min(player.stack_bb + player.street_bet_bb, max(1.0, session.pot_bb * fraction)))


def choose_raise_size(
    session: BattleSession,
    player: PlayerState,
    strength: float,
    range_profile: RangeProfile | None = None,
) -> float:
    available_total = player.stack_bb + player.street_bet_bb
    if session.street == "preflop":
        multiplier = 3.2 if player.position not in {"SB", "BB"} else 3.8
        cap = available_total if strength >= 0.94 else min(available_total, 24.0)
        return round_bb(min(cap, max(2.5, session.current_bet_bb * multiplier)))
    fraction = range_profile.raise_fraction if range_profile and range_profile.raise_fraction is not None else None
    if fraction is None:
        fraction = 0.45 if strength < 0.72 else 0.72
    raise_by = max(session.min_raise_bb, session.pot_bb * fraction)
    cap = available_total if strength >= 0.93 else min(available_total, session.current_bet_bb + max(1.0, session.pot_bb * 0.92))
    return round_bb(min(cap, session.current_bet_bb + raise_by))


def build_decision_candidates(
    session: BattleSession,
    player: PlayerState,
    strength: float,
    pot_odds: float,
    pressure: float,
    can_raise: bool,
    call_needed: float,
    range_profile: RangeProfile,
    open_spot: bool,
) -> list[DecisionCandidate]:
    fold_equity = min(max(player.agent.aggression * 0.16 + player.agent.bluff * 0.18 + pressure, 0.04), 0.38)
    candidates: list[DecisionCandidate] = []

    def add_candidate(action: BattleActionType, target_total_bb: float, ev_bb: float, reason: str) -> None:
        candidates.append(
            DecisionCandidate(
                action=action,
                target_total_bb=round_bb(target_total_bb),
                ev_bb=round_bb(ev_bb),
                weight=0,
                reason=reason,
            )
        )

    if open_spot:
        add_candidate("fold", player.street_bet_bb, 0, "不进入开池范围时直接放弃。")

        call_ev = strength * (session.pot_bb + call_needed) - call_needed
        add_candidate("call", session.current_bet_bb, call_ev * 0.45, "低频跟入保留底池权益，但主动性较差。")

        target = choose_bet_size(session, player, strength, range_profile)
        added = max(target - player.street_bet_bb, 0)
        raise_ev = strength * (session.pot_bb + added) + fold_equity * session.pot_bb - (1 - strength) * added
        add_candidate("raise", target, raise_ev, f"{range_profile.role}：标准开池建立主动权。")
    elif call_needed <= 0:
        check_ev = strength * session.pot_bb
        add_candidate("check", player.street_bet_bb, check_ev, "免费实现权益，保留中弱牌范围。")

        target = choose_bet_size(session, player, strength, range_profile)
        added = max(target - player.street_bet_bb, 0)
        action: BattleActionType = "bet" if session.current_bet_bb == 0 else "raise"
        bet_ev = strength * (session.pot_bb + added) + fold_equity * session.pot_bb - (1 - strength) * added
        add_candidate(action, target, bet_ev, f"{range_profile.role}：用范围优势和弃牌权益施压。")
    else:
        add_candidate("fold", player.street_bet_bb, 0, "放弃底池，保留剩余筹码。")

        call_ev = strength * (session.pot_bb + call_needed) - call_needed
        add_candidate("call", session.current_bet_bb, call_ev, f"{range_profile.role}：权益覆盖底池赔率时继续。")

        if can_raise:
            target = choose_raise_size(session, player, strength, range_profile)
            if target > session.current_bet_bb and target > player.street_bet_bb + call_needed:
                added = max(target - player.street_bet_bb, 0)
                raise_fold_equity = min(fold_equity + 0.10, 0.48)
                raise_ev = (
                    strength * (session.pot_bb + added)
                    + raise_fold_equity * (session.pot_bb + call_needed)
                    - (1 - strength) * added
                )
                add_candidate("raise", target, raise_ev, f"{range_profile.role}：极化价值或半诈唬，迫使弱范围弃牌。")

    return weight_candidates(candidates)


def weight_candidates(candidates: list[DecisionCandidate]) -> list[DecisionCandidate]:
    if not candidates:
        return []
    lowest_ev = min(candidate.ev_bb for candidate in candidates)
    scores = [max(0.06, candidate.ev_bb - lowest_ev + 0.35) for candidate in candidates]
    total = sum(scores) or 1
    for candidate, score in zip(candidates, scores):
        candidate.weight = score / total
    return sorted(candidates, key=lambda candidate: candidate.weight, reverse=True)


def mark_chosen_candidate(
    candidates: list[DecisionCandidate],
    chosen_action: BattleActionType,
    target_total_bb: float,
) -> list[DecisionCandidate]:
    if not candidates:
        return []
    best_index = 0
    best_distance = float("inf")
    for index, candidate in enumerate(candidates):
        action_distance = 0 if candidate.action == chosen_action else 1
        amount_distance = abs(candidate.target_total_bb - target_total_bb) / max(target_total_bb, 1)
        distance = action_distance + amount_distance
        if distance < best_distance:
            best_index = index
            best_distance = distance
    for index, candidate in enumerate(candidates):
        candidate.is_chosen = index == best_index
    return candidates


def chosen_candidate(candidates: list[DecisionCandidate]) -> DecisionCandidate | None:
    return next((candidate for candidate in candidates if candidate.is_chosen), None)


def best_alternative_candidate(
    candidates: list[DecisionCandidate],
    chosen: DecisionCandidate | None,
) -> DecisionCandidate | None:
    alternatives = [candidate for candidate in candidates if candidate is not chosen]
    if not alternatives:
        return None
    return max(alternatives, key=lambda candidate: candidate.ev_bb)


def thinking_note(session: BattleSession, strength: float, pot_odds: float, message: str) -> str:
    source = "翻前范围表" if session.street == "preflop" else "Treys Monte Carlo 胜率 + 牌面纹理"
    return f"{source}: 权益 {strength:.0%}，赔率门槛 {pot_odds:.0%}。{message}"


def build_decision_trace(
    session: BattleSession,
    player: PlayerState,
    action: BattleActionType,
    target_total_bb: float,
    hand_class: str,
    strength: float,
    equity: float | None,
    pot_odds: float,
    pressure: float,
    message: str,
    candidates: list[DecisionCandidate],
    range_profile: RangeProfile,
) -> DecisionTrace:
    source = "GTO 翻前范围 + 位置开池表" if session.street == "preflop" else "Treys Monte Carlo 胜率 + 牌面纹理/SPR 策略"
    engine = "range_chart" if session.street == "preflop" else "treys_monte_carlo"
    equity_samples = 0 if session.street == "preflop" else EQUITY_SAMPLE_COUNT
    policy_profile = f"{player.agent.archetype} · 大师级"
    spr = stack_to_pot_ratio(session, player)
    bucket = range_profile.tier if session.street == "preflop" else range_bucket_for(action, strength)
    texture = board_texture(session.board)
    profile_pressure = min(
        player.agent.aggression * 0.62
        + player.agent.bluff * 0.24
        + (player.agent.exploit_score - 86) / 100 * 0.14,
        1.0,
    )
    confidence = decision_confidence(action, strength, pot_odds, player.agent.discipline, pressure)
    tags = decision_tags(
        session=session,
        player=player,
        action=action,
        hand_class=hand_class,
        range_bucket=bucket,
        board_texture=texture,
        spr=spr,
    )
    return DecisionTrace(
        source=source,
        engine=engine,
        equity_samples=equity_samples,
        policy_profile=policy_profile,
        hand_class=hand_class,
        range_bucket=bucket,
        range_role=range_profile.role,
        range_frequency=range_profile.frequency,
        board_texture=texture,
        equity=equity,
        pot_odds=pot_odds,
        spr=spr,
        pressure=profile_pressure,
        confidence=confidence,
        recommended_total_bb=target_total_bb,
        tags=tags,
        summary=f"{bucket}：{message}",
        candidates=mark_chosen_candidate(candidates, action, target_total_bb),
    )


def range_bucket_for(action: BattleActionType, strength: float) -> str:
    if action in {"bet", "raise", "all_in"}:
        if strength >= 0.78:
            return "价值进攻"
        if strength >= 0.58:
            return "范围施压"
        return "低频诈唬"
    if action in {"call", "check"}:
        if strength >= 0.62:
            return "摊牌价值"
        return "边缘继续"
    return "弃牌范围"


def board_texture(board: list[str]) -> str:
    return analyze_board(board).texture


def analyze_board(board: list[str]) -> BoardProfile:
    if len(board) < 3:
        return BoardProfile(
            texture="翻前",
            wetness=0.0,
            paired=False,
            monotone=False,
            two_tone=False,
            connected=False,
            high_card=0,
        )

    ranks = sorted([RANK_VALUE[card[0]] for card in board])
    suits = [card[1] for card in board]
    paired = len(set(ranks)) < len(ranks)
    monotone = len(set(suits)) == 1
    two_tone = len(set(suits)) == 2
    connected = max(ranks) - min(ranks) <= 4
    high_card = max(ranks)

    if paired and monotone:
        texture = "配对同花面"
    elif monotone:
        texture = "单花湿润面"
    elif paired:
        texture = "配对干燥面"
    elif connected or two_tone:
        texture = "湿润听牌面"
    else:
        texture = "干燥高张面"

    wetness = 0.16
    if monotone:
        wetness += 0.45
    elif two_tone:
        wetness += 0.22
    if connected:
        wetness += 0.25
    if paired:
        wetness -= 0.10
    if high_card >= 12 and not connected and not two_tone and not monotone:
        wetness -= 0.06

    return BoardProfile(
        texture=texture,
        wetness=max(0.05, min(wetness, 0.95)),
        paired=paired,
        monotone=monotone,
        two_tone=two_tone,
        connected=connected,
        high_card=high_card,
    )


def postflop_bet_threshold(range_profile: RangeProfile) -> float:
    thresholds = {
        "极化价值下注": 0.72,
        "保护下注": 0.54,
        "小频率范围下注": 0.50,
        "半诈唬下注": 0.64,
        "低频延迟诈唬": 0.78,
        "摊牌控池": 0.86,
    }
    return thresholds.get(range_profile.role, 0.62)


def is_last_to_act(session: BattleSession, player: PlayerState) -> bool:
    order: list[int] = []
    for offset in range(1, session.table_size + 1):
        index = (session.dealer_index + offset) % session.table_size
        candidate = session.players[index]
        if candidate.folded or candidate.all_in:
            continue
        order.append(index)
    return bool(order) and order[-1] == player.index


def has_draw_potential(hole_cards: list[str], board: list[str]) -> bool:
    cards = hole_cards + board
    if len(board) < 3 or len(cards) < 5:
        return False

    suit_counts: dict[str, int] = {}
    for card in cards:
        suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
    if max(suit_counts.values(), default=0) >= 4:
        return True

    ranks = {RANK_VALUE[card[0]] for card in cards}
    if 14 in ranks:
        ranks.add(1)
    for low in range(1, 11):
        if len({rank for rank in ranks if low <= rank <= low + 4}) >= 4:
            return True
    return False


def stack_to_pot_ratio(session: BattleSession, player: PlayerState) -> float:
    active_opponents = [opponent.stack_bb for opponent in session.players if opponent.index != player.index and not opponent.folded]
    villain_stack = max(active_opponents, default=player.stack_bb)
    effective_stack = min(player.stack_bb, villain_stack)
    return effective_stack / max(session.pot_bb, 1.0)


def decision_confidence(
    action: BattleActionType,
    strength: float,
    pot_odds: float,
    discipline: float,
    pressure: float,
) -> float:
    if action == "fold":
        threshold = max(pot_odds, 0.48)
    elif action in {"call", "check"}:
        threshold = max(pot_odds, 0.34)
    else:
        threshold = 0.58 - pressure
    edge = abs(strength - threshold)
    return max(0.45, min(0.98, 0.56 + edge * 0.72 + discipline * 0.08))


def decision_tags(
    session: BattleSession,
    player: PlayerState,
    action: BattleActionType,
    hand_class: str,
    range_bucket: str,
    board_texture: str,
    spr: float,
) -> list[str]:
    tags = [player.position, range_bucket]
    tags.append("GTO范围" if session.street == "preflop" else board_texture)
    if action in {"bet", "raise", "all_in"}:
        tags.append("弃牌权益")
    elif action == "call":
        tags.append("底池赔率")
    if spr <= 3:
        tags.append("低SPR")
    elif spr >= 8:
        tags.append("高SPR")
    return tags[:5]


def post_blinds(session: BattleSession) -> None:
    small_blind = next(player for player in session.players if player.position == "SB")
    big_blind = next(player for player in session.players if player.position == "BB")
    post_blind(session, small_blind, 0.5)
    post_blind(session, big_blind, 1.0)
    session.current_bet_bb = 1.0
    session.min_raise_bb = 1.0


def post_blind(session: BattleSession, player: PlayerState, amount: float) -> None:
    paid = commit_to_pot(session, player, amount)
    event = ActionEvent(
        id=f"act_{len(session.action_log) + 1:04d}",
        seat_index=player.index,
        position=player.position,
        agent_id=player.agent.id,
        agent_name=player.agent.name,
        street=session.street,
        action="blind",
        label="盲注",
        amount_bb=paid,
        total_bet_bb=player.street_bet_bb,
        pot_bb=session.pot_bb,
        equity=None,
        pot_odds=None,
        note="系统自动发布盲注。",
        decision=None,
        created_at=now_iso(),
    )
    player.last_action = event
    session.action_log.append(event)
    record_table_event(
        session,
        "blind_posted",
        f"{player.position} 发布 {round_bb(paid)}BB 盲注",
        seat_index=player.index,
    )


def reset_street_bets(session: BattleSession) -> None:
    session.current_bet_bb = 0
    session.min_raise_bb = 1
    session.acted_this_street = set()
    for player in session.players:
        player.street_bet_bb = 0
        player.last_action = None


def commit_to_pot(session: BattleSession, player: PlayerState, amount: float) -> float:
    paid = round_bb(min(max(amount, 0), player.stack_bb))
    player.stack_bb = round_bb(player.stack_bb - paid)
    player.street_bet_bb = round_bb(player.street_bet_bb + paid)
    player.total_committed_bb = round_bb(player.total_committed_bb + paid)
    session.pot_bb = round_bb(session.pot_bb + paid)
    if player.stack_bb <= 0.001:
        player.stack_bb = 0
        player.all_in = True
    return paid


def deal_hole_cards(session: BattleSession) -> None:
    for _ in range(2):
        for player in session.players:
            player.hole_cards.extend(draw(session, 1))


def complete_board_with_burns(session: BattleSession) -> None:
    while len(session.board) < 5:
        if len(session.board) < 3:
            deal_board_cards(session, "flop", 3 - len(session.board))
            continue
        if len(session.board) == 3:
            deal_board_cards(session, "turn", 1)
            continue
        deal_board_cards(session, "river", 1)


def deal_board_cards(session: BattleSession, street: BattleStreet, count: int) -> list[str]:
    burned = burn_card(session)
    if burned:
        record_table_event(
            session,
            "burn",
            f"Burn · {street_label(street)}",
            street=street,
            burn_card=burned,
        )

    cards = draw(session, count)
    session.board.extend(cards)
    event: BattleTableEventType
    if street == "flop":
        event = "deal_flop"
    elif street == "turn":
        event = "deal_turn"
    else:
        event = "deal_river"
    record_table_event(
        session,
        event,
        f"发 {street_label(street)}",
        street=street,
        cards=cards,
    )
    return cards


def record_table_event(
    session: BattleSession,
    event: BattleTableEventType,
    label: str,
    street: BattleStreet | None = None,
    seat_index: int | None = None,
    cards: list[str] | None = None,
    burn_card: str | None = None,
) -> None:
    session.table_events.append(
        TableEvent(
            id=f"evt_{session.hand_number:03d}_{len(session.table_events) + 1:04d}",
            event=event,
            street=street or session.street,
            label=label,
            seat_index=seat_index,
            cards=list(cards or []),
            burn_card=burn_card,
            pot_bb=session.pot_bb,
            created_at=now_iso(),
        )
    )


def burn_card(session: BattleSession) -> str | None:
    burned = draw(session, 1)
    session.burned_cards.extend(burned)
    return burned[0] if burned else None


def draw(session: BattleSession, count: int) -> list[str]:
    cards = session.deck[:count]
    del session.deck[:count]
    return cards


def fresh_deck(rng: Random) -> list[str]:
    cards = full_deck_codes()
    rng.shuffle(cards)
    return cards


def full_deck_codes() -> list[str]:
    return [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]


def preflop_first_actor(session: BattleSession) -> int:
    if session.table_size == 2:
        return next(player.index for player in session.players if player.position == "SB")
    first_relative_index = 0 if session.table_size == 3 else 3
    return (session.dealer_index + first_relative_index) % session.table_size


def postflop_first_actor(session: BattleSession) -> int | None:
    return next_actor_after(session, session.dealer_index)


def next_actor_after(session: BattleSession, start_index: int) -> int | None:
    total = len(session.players)
    for offset in range(1, total + 1):
        index = (start_index + offset) % total
        player = session.players[index]
        if player.folded or player.all_in:
            continue
        if player.index not in session.acted_this_street or player.street_bet_bb < session.current_bet_bb:
            return index
    return None


def count_contenders(session: BattleSession) -> int:
    return sum(1 for player in session.players if not player.folded)


def count_players_who_can_act(session: BattleSession) -> int:
    return sum(1 for player in session.players if not player.folded and not player.all_in)


def betting_closed_for_hand(session: BattleSession) -> bool:
    return count_contenders(session) > 1 and count_players_who_can_act(session) <= 1


def positions_for(table_size: int) -> list[str]:
    layouts = {
        2: ["SB", "BB"],
        3: ["BTN", "SB", "BB"],
        4: ["BTN", "SB", "BB", "CO"],
        5: ["BTN", "SB", "BB", "UTG", "CO"],
        6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
        7: ["BTN", "SB", "BB", "UTG", "LJ", "HJ", "CO"],
        8: ["BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO"],
        9: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"],
    }
    if table_size not in layouts:
        raise ValueError(f"Unsupported table size: {table_size}")
    return layouts[table_size]


def assign_positions(session: BattleSession) -> None:
    positions = positions_for(session.table_size)
    for player in session.players:
        relative_index = (player.index - session.dealer_index + session.table_size) % session.table_size
        player.position = positions[relative_index]


def normalize_seat(index: int, table_size: int) -> int:
    if table_size <= 0:
        return 0
    return min(max(index, 0), table_size - 1)


def street_label(street: BattleStreet) -> str:
    labels = {
        "preflop": "Preflop",
        "flop": "Flop",
        "turn": "Turn",
        "river": "River",
        "showdown": "Showdown",
        "complete": "Complete",
    }
    return labels[street]


def build_action_timeline(action_log: list[ActionEvent]) -> list[BattleActionStreetSnapshot]:
    street_order: list[BattleStreet] = ["preflop", "flop", "turn", "river", "showdown", "complete"]
    grouped: dict[BattleStreet, list[BattleActionSnapshot]] = {street: [] for street in street_order}
    for event in action_log:
        grouped.setdefault(event.street, []).append(event.snapshot())

    return [
        BattleActionStreetSnapshot(
            street=street,
            label=street_label(street),
            actions=actions,
        )
        for street in street_order
        if (actions := grouped.get(street))
    ]


def build_replay_events(
    action_log: list[ActionEvent],
    table_events: list[TableEvent],
) -> list[BattleReplayEventSnapshot]:
    merged: list[tuple[str, int, BattleReplayEventType, ActionEvent | TableEvent]] = []
    for index, event in enumerate(table_events):
        merged.append((event.created_at, index * 2, "table", event))
    for index, event in enumerate(action_log):
        merged.append((event.created_at, index * 2 + 1, "action", event))
    merged.sort(key=lambda item: (item[0], item[1]))

    replay_events: list[BattleReplayEventSnapshot] = []
    for sequence, (_, _, kind, event) in enumerate(merged, start=1):
        if kind == "action":
            assert isinstance(event, ActionEvent)
            replay_events.append(
                BattleReplayEventSnapshot(
                    id=f"rep_{sequence:04d}",
                    sequence=sequence,
                    kind="action",
                    street=event.street,
                    label=f"{event.position} {event.agent_name} {event.label}",
                    created_at=event.created_at,
                    seat_index=event.seat_index,
                    position=event.position,
                    agent_id=event.agent_id,
                    agent_name=event.agent_name,
                    action_id=event.id,
                    action=event.action,
                    amount_bb=round_bb(event.amount_bb),
                    total_bet_bb=round_bb(event.total_bet_bb),
                    pot_bb=round_bb(event.pot_bb),
                    equity=None if event.equity is None else round(event.equity, 3),
                    pot_odds=None if event.pot_odds is None else round(event.pot_odds, 3),
                    note=event.note,
                    decision=event.decision.snapshot() if event.decision else None,
                )
            )
            continue

        assert isinstance(event, TableEvent)
        replay_events.append(
            BattleReplayEventSnapshot(
                id=f"rep_{sequence:04d}",
                sequence=sequence,
                kind="table",
                street=event.street,
                label=event.label,
                created_at=event.created_at,
                seat_index=event.seat_index,
                table_event_id=event.id,
                table_event=event.event,
                cards=list(event.cards),
                burn_card=event.burn_card,
                pot_bb=round_bb(event.pot_bb),
            )
        )

    return replay_events


def build_review_insights(session: BattleSession, decisions: list[ActionEvent]) -> list[BattleReviewInsightSnapshot]:
    insights: list[BattleReviewInsightSnapshot] = []

    if session.result:
        insights.append(
            BattleReviewInsightSnapshot(
                id="result",
                title="牌局结算",
                detail=session.result.summary,
                icon="sparkles",
                accent="#8B5CF6",
                seat_index=session.result.winners[0] if session.result.winners else None,
            )
        )

    chosen_edges: list[tuple[float, ActionEvent, BattleDecisionSnapshot]] = []
    close_spots: list[tuple[float, ActionEvent, BattleDecisionSnapshot]] = []
    for event in decisions:
        if not event.decision:
            continue
        decision = event.decision.snapshot()
        if decision.ev_delta_bb is not None:
            chosen_edges.append((decision.ev_delta_bb, event, decision))
        close_spots.append((decision.confidence, event, decision))

    if chosen_edges:
        edge, event, decision = max(chosen_edges, key=lambda item: item[0])
        alternative = f" vs {decision.best_alternative_label}" if decision.best_alternative_label else ""
        insights.append(
            BattleReviewInsightSnapshot(
                id="best_edge",
                title="最高 EV 选择",
                detail=f"{event.position} {event.agent_name} 选择 {event.label}，EV {edge:+.1f}BB{alternative}。",
                icon="arrow.triangle.branch",
                accent="#13C8A6",
                seat_index=event.seat_index,
                action_id=event.id,
            )
        )

    if close_spots:
        confidence, event, decision = min(close_spots, key=lambda item: item[0])
        insights.append(
            BattleReviewInsightSnapshot(
                id="close_spot",
                title="边缘决策点",
                detail=f"{event.position} {event.agent_name} 的 {event.label} 信心 {confidence:.0%}，{decision.summary}",
                icon="target",
                accent="#F59E0B",
                seat_index=event.seat_index,
                action_id=event.id,
            )
        )

    if not insights:
        insights.append(
            BattleReviewInsightSnapshot(
                id="waiting",
                title="复盘准备中",
                detail="推进牌局后会生成 Agent 决策依据、EV 对比和关键街道摘要。",
                icon="hourglass",
                accent="#64748B",
            )
        )

    return insights[:3]


def action_label(action: BattleActionType) -> str:
    labels = {
        "blind": "盲注",
        "fold": "弃牌",
        "check": "过牌",
        "call": "跟注",
        "bet": "下注",
        "raise": "加注",
        "all_in": "全下",
    }
    return labels[action]


def made_hand_label(score: int) -> str:
    labels = {
        "Royal Flush": "皇家同花顺",
        "Straight Flush": "同花顺",
        "Four of a Kind": "四条",
        "Full House": "葫芦",
        "Flush": "同花",
        "Straight": "顺子",
        "Three of a Kind": "三条",
        "Two Pair": "两对",
        "Pair": "一对",
        "High Card": "高牌",
    }
    class_name = EVALUATOR.class_to_string(EVALUATOR.get_rank_class(score))
    return labels.get(class_name, class_name)


def build_tasks(session: BattleSession, observer_seat: int) -> list[BattleTaskSnapshot]:
    observer = session.players[observer_seat]
    observer_trace = observer.last_action.decision if observer.last_action and observer.last_action.decision else None
    equity = estimate_equity(session, observer) if session.street not in ("complete", "showdown") else None
    equity_text = "摊牌完成" if equity is None else f"{equity:.0%} equity"
    if observer_trace and observer_trace.equity is not None:
        equity_text = f"{observer_trace.equity:.0%} equity"
    pressure_state: Literal["done", "running", "queued"] = "running" if observer.index == session.current_actor else "done"
    return [
        BattleTaskSnapshot(
            id="range",
            title="范围推演",
            subtitle=(
                f"{observer.position} · {observer_trace.range_bucket}"
                if observer_trace
                else f"{observer.position} {observer.agent.name} · {street_label(session.street)}"
            ),
            icon="scope",
            accent="#8B5CF6",
            state="done",
        ),
        BattleTaskSnapshot(
            id="equity",
            title="胜率刷新",
            subtitle=equity_text,
            icon="percent",
            accent="#13C8A6",
            state="done" if equity is not None else "queued",
        ),
        BattleTaskSnapshot(
            id="line",
            title="下注线判断",
            subtitle=(
                f"建议总额 {round_bb(observer_trace.recommended_total_bb)}BB"
                if observer_trace
                else f"当前池 {round_bb(session.pot_bb)}BB"
            ),
            icon="chart.line.uptrend.xyaxis",
            accent="#F59E0B",
            state=pressure_state,
        ),
        BattleTaskSnapshot(
            id="exploit",
            title="对手倾向",
            subtitle=observer_trace.summary if observer_trace else f"{count_contenders(session)} 名牌手仍在池内",
            icon="eye.fill",
            accent="#EF4444",
            state="running" if session.current_actor is not None else "done",
        ),
    ]


def round_bb(value: float) -> float:
    return round(value * 2) / 2


def round_money(value: float) -> float:
    return round(value + 1e-9, 2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
