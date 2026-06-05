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


class HandQuiz(BaseModel):
    id: str
    hero_hand: str
    villain_hand: str
    board: str
    question: str
    options: list[str]
    answer: str
    explanation: str


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
