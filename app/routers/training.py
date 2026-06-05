from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.data import HAND_QUIZZES, SCENARIOS
from app.dependencies import require_current_user
from app.schemas import AnswerRequest, DecisionResult, HandQuiz, PreflopScenario, User


router = APIRouter(prefix="/training", tags=["training"])


@router.get("/preflop", response_model=list[PreflopScenario])
def preflop_scenarios(_: User = Depends(require_current_user)) -> list[PreflopScenario]:
    return SCENARIOS


@router.post("/preflop/{scenario_id}/answer", response_model=DecisionResult)
def answer_preflop(
    scenario_id: str,
    payload: AnswerRequest,
    _: User = Depends(require_current_user),
) -> DecisionResult:
    scenario = next((item for item in SCENARIOS if item.id == scenario_id), None)
    if scenario is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found")

    is_correct = payload.action == scenario.recommended_action
    return DecisionResult(
        is_correct=is_correct,
        recommendation=f"{scenario.recommended_action.upper()} {scenario.recommended_sizing}",
        explanation=scenario.explanation,
        concept_tags=scenario.concept_tags,
        next_prompt="下一题会继续强化同一个概念。" if not is_correct else "保持这个节奏，下一题会提高一点难度。",
    )


@router.get("/hand-quiz", response_model=list[HandQuiz])
def hand_quiz(_: User = Depends(require_current_user)) -> list[HandQuiz]:
    return HAND_QUIZZES


@router.post("/hand-quiz/{quiz_id}/answer", response_model=DecisionResult)
def answer_hand_quiz(
    quiz_id: str,
    payload: dict[str, str],
    _: User = Depends(require_current_user),
) -> DecisionResult:
    quiz = next((item for item in HAND_QUIZZES if item.id == quiz_id), None)
    if quiz is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")

    selected = payload.get("answer", "")
    is_correct = selected == quiz.answer
    return DecisionResult(
        is_correct=is_correct,
        recommendation=quiz.answer,
        explanation=quiz.explanation,
        concept_tags=["牌型识别", "摊牌判断"],
        next_prompt="把牌型顺序练成反射，会显著减少线下局低级失误。",
    )
