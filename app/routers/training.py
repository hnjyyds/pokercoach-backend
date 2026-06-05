from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.data import SCENARIOS
from app.dependencies import require_current_user
from app.mistakes import (
    BattleMistakeDetail,
    BattleMistakeSummary,
    CoachMessageRequest,
    add_coach_message,
    get_mistake,
    list_mistakes,
)
from app.quiz_agent import (
    answer_hand_quiz as answer_generated_hand_quiz,
    coach_hand_quiz,
    generate_hand_quiz,
    list_hand_quizzes,
)
from app.schemas import AnswerRequest, DecisionResult, HandQuiz, HandQuizGenerateRequest, PreflopScenario, User


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
def hand_quiz(user: User = Depends(require_current_user)) -> list[HandQuiz]:
    return list_hand_quizzes(user.id)


@router.post("/hand-quiz/generate", response_model=HandQuiz, status_code=status.HTTP_201_CREATED)
def generate_quiz(
    payload: HandQuizGenerateRequest,
    user: User = Depends(require_current_user),
) -> HandQuiz:
    return generate_hand_quiz(user.id, payload)


@router.get("/mistakes", response_model=list[BattleMistakeSummary])
def mistakes(user: User = Depends(require_current_user)) -> list[BattleMistakeSummary]:
    return list_mistakes(user.id)


@router.get("/mistakes/{mistake_id}", response_model=BattleMistakeDetail)
def mistake_detail(
    mistake_id: str,
    user: User = Depends(require_current_user),
) -> BattleMistakeDetail:
    return get_mistake(mistake_id, user.id)


@router.post("/mistakes/{mistake_id}/coach", response_model=BattleMistakeDetail)
def coach_mistake(
    mistake_id: str,
    payload: CoachMessageRequest,
    user: User = Depends(require_current_user),
) -> BattleMistakeDetail:
    return add_coach_message(mistake_id, user.id, payload)


@router.post("/hand-quiz/{quiz_id}/answer", response_model=DecisionResult)
def answer_hand_quiz(
    quiz_id: str,
    payload: dict[str, str],
    user: User = Depends(require_current_user),
) -> DecisionResult:
    return answer_generated_hand_quiz(user.id, quiz_id, payload.get("answer", ""))


@router.post("/hand-quiz/{quiz_id}/coach", response_model=HandQuiz)
def coach_quiz(
    quiz_id: str,
    payload: CoachMessageRequest,
    user: User = Depends(require_current_user),
) -> HandQuiz:
    return coach_hand_quiz(user.id, quiz_id, payload.message)
