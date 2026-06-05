from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from app.poker_eval import MadeHandExplanation, describe_showdown


class AgentToolResult(BaseModel):
    tool: str = Field(min_length=1)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    result: dict[str, Any]


class AgentToolSpec(BaseModel):
    tool: str = Field(min_length=1)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)


class AgentToolSelection(BaseModel):
    selected_tools: list[str] = Field(default_factory=list, max_length=6)
    reason: str = Field(min_length=4, max_length=160)


TOOL_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selected_tools", "reason"],
    "properties": {
        "selected_tools": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "reason": {"type": "string", "minLength": 4, "maxLength": 160},
    },
}


QUIZ_GENERATION_TOOL_CATALOG = [
    AgentToolSpec(
        tool="position_and_stack_classifier",
        title="位置与后手分类",
        purpose="判断题目中的座位、有效后手深度和底池规模。",
        inputs=["position", "effective_stack_bb", "pot_bb"],
    ),
    AgentToolSpec(
        tool="pot_context",
        title="底池上下文",
        purpose="读取底池大小和新手训练重点。",
        inputs=["pot_bb", "beginner_focus"],
    ),
    AgentToolSpec(
        tool="decision_validator",
        title="答案校验",
        purpose="锁定合法选项、推荐答案和解释，防止改写题目事实。",
        inputs=["allowed_options", "recommended_answer", "reason"],
    ),
]


QUIZ_COACH_TOOL_CATALOG = [
    AgentToolSpec(
        tool="showdown_evaluator",
        title="摊牌牌型评估",
        purpose="找出最佳五张牌、牌型层级和摊牌胜负。",
        inputs=["hero_hand", "villain_hand", "board"],
    ),
    AgentToolSpec(
        tool="answer_validator",
        title="答题结论校验",
        purpose="读取当前题目的正确答案和锁定解释。",
        inputs=["correct_answer", "locked_explanation"],
    ),
    AgentToolSpec(
        tool="stack_context",
        title="后手深度上下文",
        purpose="判断有效后手深度，用于解释短后手、标准后手、深后手差异。",
        inputs=["position", "effective_stack_bb", "pot_bb"],
    ),
]


MISTAKE_COACH_TOOL_CATALOG = [
    AgentToolSpec(
        tool="spot_snapshot",
        title="错题场景快照",
        purpose="读取错题发生时的位置、街次、底池、后手、SPR 和标签。",
        inputs=["position", "street", "pot_bb", "stack_bb", "spr", "tags"],
    ),
    AgentToolSpec(
        tool="ev_action_compare",
        title="动作 EV 对比",
        purpose="比较用户动作、推荐动作和候选动作 EV。",
        inputs=["candidates", "user_action_label", "recommended_action_label", "ev_delta_bb"],
    ),
]


def serialize_tools(tools: Iterable[AgentToolResult]) -> list[dict[str, Any]]:
    return [tool.model_dump(mode="json") for tool in tools]


def serialize_tool_catalog(tools: Iterable[AgentToolSpec]) -> list[dict[str, Any]]:
    return [tool.model_dump(mode="json") for tool in tools]


def find_tool(tool_context: Iterable[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((tool for tool in tool_context if tool.get("tool") == name), None)


def normalize_selected_tool_names(
    selected_tools: Iterable[str],
    available_tools: Iterable[AgentToolSpec],
    fallback: Iterable[str],
) -> list[str]:
    selected_tool_list = list(selected_tools)
    allowed = {tool.tool for tool in available_tools}
    normalized = []
    seen = set()
    for selected_tool in selected_tool_list:
        if selected_tool in allowed and selected_tool not in seen:
            normalized.append(selected_tool)
            seen.add(selected_tool)
    if normalized:
        return normalized
    if not selected_tool_list:
        return []
    return [tool for tool in fallback if tool in allowed]


def stack_bucket(stack_depth_bb: float) -> str:
    if stack_depth_bb <= 40:
        return "短后手"
    if stack_depth_bb >= 140:
        return "深后手"
    return "标准后手"


def position_and_stack_classifier_tool(
    position: str,
    effective_stack_bb: float,
    pot_bb: float | None = None,
) -> AgentToolResult:
    result: dict[str, Any] = {
        "position": position,
        "effective_stack_bb": effective_stack_bb,
        "stack_bucket": stack_bucket(effective_stack_bb),
    }
    if pot_bb is not None:
        result["pot_bb"] = pot_bb
    return AgentToolResult(
        tool="position_and_stack_classifier",
        title="位置与后手分类",
        purpose="判断当前座位、有效后手深度和底池规模，为策略解释提供边界。",
        result=result,
    )


def stack_context_tool(position: str, effective_stack_bb: float, pot_bb: float) -> AgentToolResult:
    return AgentToolResult(
        tool="stack_context",
        title="后手深度上下文",
        purpose="把有效后手归入短后手、标准后手或深后手，提醒 Agent 解释不同筹码深度的策略差异。",
        result={
            "bucket": stack_bucket(effective_stack_bb),
            "effective_stack_bb": effective_stack_bb,
            "position": position,
            "pot_bb": pot_bb,
        },
    )


def pot_context_tool(pot_bb: float, beginner_focus: str) -> AgentToolResult:
    return AgentToolResult(
        tool="pot_context",
        title="底池上下文",
        purpose="给 Agent 提供底池大小和本题新手训练重点。",
        result={
            "pot_bb": pot_bb,
            "beginner_focus": beginner_focus,
        },
    )


def decision_validator_tool(
    allowed_options: list[str],
    recommended_answer: str,
    reason: str,
) -> AgentToolResult:
    return AgentToolResult(
        tool="decision_validator",
        title="答案校验",
        purpose="锁定题目的合法选项、推荐答案和算法/题库解释，避免 LLM 改答案。",
        result={
            "allowed_options": allowed_options,
            "recommended_answer": recommended_answer,
            "reason": reason,
        },
    )


def answer_validator_tool(correct_answer: str, locked_explanation: str) -> AgentToolResult:
    return AgentToolResult(
        tool="answer_validator",
        title="答题结论校验",
        purpose="锁定当前题目的正确答案和解释，供追问时引用。",
        result={
            "correct_answer": correct_answer,
            "locked_explanation": locked_explanation,
        },
    )


def showdown_evaluator_tool(
    hero_hand: str,
    villain_hand: str,
    board: str,
) -> AgentToolResult | None:
    showdown = describe_showdown(hero_hand=hero_hand, villain_hand=villain_hand, board=board)
    if showdown is None:
        return None
    return AgentToolResult(
        tool="showdown_evaluator",
        title="摊牌牌型评估",
        purpose="用确定性牌型算法找出最佳五张牌、牌型层级和摊牌胜负。",
        result={
            "hero": made_hand_payload(showdown.hero),
            "villain": made_hand_payload(showdown.villain) if showdown.villain else None,
            "winner": showdown.winner,
            "teaching_summary": showdown.content,
        },
    )


def spot_snapshot_tool(scenario: Any) -> AgentToolResult:
    return AgentToolResult(
        tool="spot_snapshot",
        title="错题场景快照",
        purpose="锁定错题发生时的位置、街次、底池、后手和 SPR，供复盘解释。",
        result={
            "position": read_value(scenario, "position"),
            "street": read_value(scenario, "street"),
            "pot_bb": read_value(scenario, "pot_bb"),
            "stack_bb": read_value(scenario, "stack_bb"),
            "spr": read_value(scenario, "spr"),
            "tags": read_value(scenario, "tags", []),
        },
    )


def ev_action_compare_tool(
    candidates: Iterable[Any],
    user_action_label: str,
    recommended_action_label: str,
    ev_delta_bb: float,
) -> AgentToolResult:
    return AgentToolResult(
        tool="ev_action_compare",
        title="动作 EV 对比",
        purpose="比较用户动作与推荐动作的 EV 差距，并保留候选动作的原因。",
        result={
            "user_action": user_action_label,
            "recommended_action": recommended_action_label,
            "ev_delta_bb": round(ev_delta_bb, 2),
            "candidates": [dump_candidate(candidate) for candidate in candidates],
        },
    )


def made_hand_payload(made_hand: MadeHandExplanation) -> dict[str, Any]:
    return {
        "made_hand": made_hand.label,
        "detail": made_hand.detail,
        "best_cards": list(made_hand.best_cards),
    }


def dump_candidate(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump(mode="json")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {
        "action": read_value(candidate, "action"),
        "label": read_value(candidate, "label"),
        "target_total_bb": read_value(candidate, "target_total_bb"),
        "ev_bb": read_value(candidate, "ev_bb"),
        "weight": read_value(candidate, "weight"),
        "is_recommended": read_value(candidate, "is_recommended"),
        "reason": read_value(candidate, "reason"),
    }


def read_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)
