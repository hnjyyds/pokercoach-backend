from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PokerAction = Literal["fold", "call", "raise", "check", "bet"]


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)


class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2)


class User(BaseModel):
    id: str
    name: str
    email: str
    level: str
    streak_days: int
    skill_score: int


class AuthResponse(BaseModel):
    token: str
    user: User


class ModuleCard(BaseModel):
    id: str
    title: str
    subtitle: str
    progress: float = Field(ge=0, le=1)
    icon: str
    accent: str


class DailyPlan(BaseModel):
    title: str
    target_minutes: int
    completed_minutes: int
    focus: str
    modules: list[ModuleCard]


class DashboardResponse(BaseModel):
    user: User
    daily_plan: DailyPlan
    recent_mistakes: list[str]
    next_drill_id: str


class Choice(BaseModel):
    action: PokerAction
    label: str
    sizing: str | None = None


class PreflopScenario(BaseModel):
    id: str
    position: str
    hand: str
    table_state: str
    villain_action: str
    stack_depth_bb: int
    pot_bb: float
    choices: list[Choice]
    recommended_action: PokerAction
    recommended_sizing: str
    concept_tags: list[str]
    explanation: str


class AnswerRequest(BaseModel):
    action: PokerAction


class DecisionResult(BaseModel):
    is_correct: bool
    recommendation: str
    explanation: str
    concept_tags: list[str]
    next_prompt: str


class QuizCoachMessageSnapshot(BaseModel):
    id: str
    role: Literal["agent", "user"]
    content: str
    created_at: str


class HandQuiz(BaseModel):
    id: str
    hero_hand: str
    villain_hand: str
    board: str
    question: str
    options: list[str]
    answer: str
    explanation: str
    source_agent: str = "Ivy"
    agent_icon: str = "sparkles"
    agent_accent: str = "#8B5CF6"
    thesis: str = "先识别牌型，再判断摊牌结果。"
    street: str = "river"
    position: str = "BTN"
    stack_depth_bb: int = 100
    pot_bb: float = 8.5
    difficulty: str = "新手"
    concept_tags: list[str] = Field(default_factory=list)
    coach_messages: list[QuizCoachMessageSnapshot] = Field(default_factory=list)


class HandQuizGenerateRequest(BaseModel):
    focus: str = Field(default="牌力识别", min_length=2, max_length=40)
    difficulty: str = Field(default="新手", min_length=2, max_length=20)
    street: str = Field(default="river", min_length=3, max_length=20)


class OddsRequest(BaseModel):
    hero_hand: str
    board: str = ""
    outs: int = Field(ge=0, le=20)


class OddsResponse(BaseModel):
    hero_hand: str
    board: str
    outs: int
    turn_or_river_probability: float
    by_river_probability: float
    coaching_note: str
    is_mock: bool = True
