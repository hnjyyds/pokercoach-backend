from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException, status

from app.data import HAND_QUIZZES
from app.schemas import (
    DecisionResult,
    HandQuiz,
    HandQuizGenerateRequest,
    QuizCoachMessageSnapshot,
)


GENERATED_QUIZZES: dict[str, list[HandQuiz]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_sentence(value: str) -> str:
    return value.strip().rstrip("。.!！?")


def list_hand_quizzes(owner_id: str) -> list[HandQuiz]:
    base_quizzes = [quiz_with_agent_context(quiz) for quiz in HAND_QUIZZES]
    generated = GENERATED_QUIZZES.get(owner_id, [])
    generated_ids = {quiz.id for quiz in generated}
    return generated + [quiz for quiz in base_quizzes if quiz.id not in generated_ids]


def generate_hand_quiz(owner_id: str, request: HandQuizGenerateRequest) -> HandQuiz:
    template = choose_template(request)
    quiz = HandQuiz(
        id=f"hq_agent_{uuid4().hex[:10]}",
        hero_hand=template["hero_hand"],
        villain_hand=template["villain_hand"],
        board=template["board"],
        question=template["question"],
        options=template["options"],
        answer=template["answer"],
        explanation=template["explanation"],
        source_agent=template["agent"],
        agent_icon=template["agent_icon"],
        agent_accent=template["agent_accent"],
        thesis=template["thesis"],
        street=request.street,
        position=template["position"],
        stack_depth_bb=template["stack_depth_bb"],
        pot_bb=template["pot_bb"],
        difficulty=request.difficulty,
        concept_tags=template["concept_tags"],
        coach_messages=[
            QuizCoachMessageSnapshot(
                id=f"msg_{uuid4().hex[:10]}",
                role="agent",
                content=(
                    f"这题的论点是：{clean_sentence(template['thesis'])}。先只看牌面图形、位置和底池，"
                    "不要急着背答案。"
                ),
                created_at=now_iso(),
            )
        ],
    )
    GENERATED_QUIZZES.setdefault(owner_id, []).insert(0, quiz)
    return quiz


def answer_hand_quiz(owner_id: str, quiz_id: str, selected: str) -> DecisionResult:
    quiz = get_hand_quiz(owner_id, quiz_id)
    is_correct = selected == quiz.answer
    return DecisionResult(
        is_correct=is_correct,
        recommendation=quiz.answer,
        explanation=quiz.explanation,
        concept_tags=quiz.concept_tags or ["牌型识别", "摊牌判断"],
        next_prompt=(
            "可以继续追问导师：如果有效后手更深，答案会不会改变？"
            if is_correct
            else "建议围绕当前题目的论点追问导师，把错因拆成位置、牌面和底池三层。"
        ),
    )


def coach_hand_quiz(owner_id: str, quiz_id: str, message: str) -> HandQuiz:
    quiz = get_hand_quiz(owner_id, quiz_id)
    updated = quiz.model_copy(deep=True)
    updated.coach_messages.append(
        QuizCoachMessageSnapshot(
            id=f"msg_{uuid4().hex[:10]}",
            role="user",
            content=message.strip(),
            created_at=now_iso(),
        )
    )
    updated.coach_messages.append(
        QuizCoachMessageSnapshot(
            id=f"msg_{uuid4().hex[:10]}",
            role="agent",
            content=coach_reply(updated, message),
            created_at=now_iso(),
        )
    )
    replace_generated_quiz(owner_id, updated)
    return updated


def get_hand_quiz(owner_id: str, quiz_id: str) -> HandQuiz:
    quiz = next((item for item in list_hand_quizzes(owner_id) if item.id == quiz_id), None)
    if quiz is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")
    return quiz


def replace_generated_quiz(owner_id: str, quiz: HandQuiz) -> None:
    generated = GENERATED_QUIZZES.setdefault(owner_id, [])
    for index, item in enumerate(generated):
        if item.id == quiz.id:
            generated[index] = quiz
            return
    generated.insert(0, quiz)


def quiz_with_agent_context(quiz: HandQuiz) -> HandQuiz:
    if quiz.concept_tags:
        return quiz
    return quiz.model_copy(
        update={
            "source_agent": "Ivy",
            "agent_icon": "brain.head.profile",
            "agent_accent": "#10B981",
            "thesis": "从摊牌结构倒推：先找最佳五张牌，再比较牌型层级。",
            "street": "river",
            "position": "BTN",
            "stack_depth_bb": 100,
            "pot_bb": 12.5,
            "difficulty": "新手",
            "concept_tags": ["牌型识别", "摊牌判断"],
            "coach_messages": [
                QuizCoachMessageSnapshot(
                    id=f"msg_seed_{quiz.id}",
                    role="agent",
                    content="这题可以当成牌型比较训练：先找 Hero 最佳五张，再找对手最佳五张。",
                    created_at=now_iso(),
                )
            ],
        }
    )


def choose_template(request: HandQuizGenerateRequest) -> dict:
    normalized_focus = request.focus.lower()
    if "赔率" in request.focus or "outs" in normalized_focus:
        return {
            "agent": "Nash",
            "agent_icon": "percent",
            "agent_accent": "#14B8A6",
            "hero_hand": "As Qs",
            "villain_hand": "Kh Kd",
            "board": "Ts 7s 2c",
            "question": "面对下注，Hero 的继续理由主要是什么？",
            "options": ["同花听牌 + 高张权益", "已经成牌领先", "只能弃牌"],
            "answer": "同花听牌 + 高张权益",
            "explanation": "Hero 还没有成牌领先，但有同花听牌和高张改进空间。关键不是牌面强度，而是权益实现和下注价格是否允许继续。",
            "thesis": "听牌题先算可改进牌，再看底池赔率是否支持继续。",
            "position": "BTN",
            "stack_depth_bb": 100,
            "pot_bb": 9.5,
            "concept_tags": ["底池赔率", "听牌", "权益实现"],
        }
    if "范围" in request.focus:
        return {
            "agent": "River",
            "agent_icon": "scope",
            "agent_accent": "#8B5CF6",
            "hero_hand": "Ac Qd",
            "villain_hand": "9h 9s",
            "board": "Qh 7c 4d",
            "question": "这手牌的核心论点是什么？",
            "options": ["顶对要保护但不盲目打大", "必须全下", "只能慢打"],
            "answer": "顶对要保护但不盲目打大",
            "explanation": "顶对好踢脚有价值下注动机，但在深筹码下不能把所有较差牌赶走，也要保留对手弱 Q 和中对继续。",
            "thesis": "价值下注不是越大越好，目标是让更差范围继续付费。",
            "position": "CO",
            "stack_depth_bb": 150,
            "pot_bb": 7.0,
            "concept_tags": ["范围优势", "价值下注", "深筹码"],
        }
    return {
        "agent": "Ivy",
        "agent_icon": "sparkles",
        "agent_accent": "#F59E0B",
        "hero_hand": "8c 8d",
        "villain_hand": "As Kd",
        "board": "8h Ac Ks 2d 2s",
        "question": "Hero 最终牌型是什么？",
        "options": ["三条", "葫芦", "两对"],
        "answer": "葫芦",
        "explanation": "Hero 的口袋对子命中三条，再配公共牌对子，最终组成葫芦。",
        "thesis": "摊牌判断要从最佳五张牌出发，而不是只看手牌名字。",
        "position": "BB",
        "stack_depth_bb": 80,
        "pot_bb": 18.0,
        "concept_tags": ["牌型识别", "摊牌判断"],
    }


def coach_reply(quiz: HandQuiz, message: str) -> str:
    lowered = message.lower()
    thesis = clean_sentence(quiz.thesis)
    if "为什么" in message or "why" in lowered:
        return (
            f"因为这题的核心论点是“{thesis}”。你要先用牌面图形确认最佳五张，"
            f"再把它放回 {quiz.position} 位置、{quiz.stack_depth_bb}BB 有效后手和 {quiz.pot_bb:g}BB 底池里判断。"
        )
    if "如果" in message or "深" in message or "短" in message or "short" in lowered or "deep" in lowered:
        return (
            "如果后手更深，边缘成牌和听牌的处理会更谨慎，因为反向隐含赔率变大；"
            "如果后手更短，强听牌和顶对更容易进入承诺底池。"
        )
    return (
        f"围绕当前题目，我会先抓三个点：{', '.join(quiz.concept_tags[:3])}。"
        f"本题推荐答案是“{quiz.answer}”，但真正要记住的是：{thesis}。"
    )
