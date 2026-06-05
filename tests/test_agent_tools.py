from __future__ import annotations

from app.agent_tools import (
    QUIZ_COACH_TOOL_CATALOG,
    decision_validator_tool,
    ev_action_compare_tool,
    normalize_selected_tool_names,
    position_and_stack_classifier_tool,
    serialize_tools,
    showdown_evaluator_tool,
    spot_snapshot_tool,
)
from app.mistakes import BattleMistakeCandidate


def test_showdown_evaluator_tool_serializes_generic_result() -> None:
    tool = showdown_evaluator_tool(
        hero_hand="Qh Qd",
        villain_hand="As Kh",
        board="Qs 2d 2c 7h 9c",
    )

    assert tool is not None
    payload = tool.model_dump(mode="json")
    assert payload["tool"] == "showdown_evaluator"
    assert payload["title"] == "摊牌牌型评估"
    assert payload["purpose"]
    assert payload["result"]["hero"]["made_hand"] == "葫芦"
    assert payload["result"]["winner"] == "hero"
    assert "Hero 的最佳五张牌" in payload["result"]["teaching_summary"]


def test_strategy_tools_share_agent_contract() -> None:
    serialized = serialize_tools(
        [
            position_and_stack_classifier_tool("BTN", 150, 7.0),
            decision_validator_tool(["弃牌", "跟注"], "弃牌", "位置太差"),
        ]
    )

    assert len(serialized) == 2
    for tool in serialized:
        assert {"tool", "title", "purpose", "result"} == set(tool)
        assert isinstance(tool["result"], dict)
    assert serialized[0]["result"]["stack_bucket"] == "深后手"
    assert serialized[1]["result"]["recommended_answer"] == "弃牌"


def test_ev_action_compare_tool_serializes_candidates() -> None:
    tool = ev_action_compare_tool(
        candidates=[
            BattleMistakeCandidate(
                action="fold",
                label="弃牌",
                target_total_bb=0,
                ev_bb=0.2,
                weight=0.6,
                is_recommended=True,
                reason="避免被主导。",
            )
        ],
        user_action_label="跟注",
        recommended_action_label="弃牌",
        ev_delta_bb=1.124,
    )

    payload = tool.model_dump(mode="json")
    assert payload["tool"] == "ev_action_compare"
    assert payload["result"]["ev_delta_bb"] == 1.12
    assert payload["result"]["candidates"][0]["is_recommended"] is True


def test_spot_snapshot_tool_accepts_plain_mapping() -> None:
    tool = spot_snapshot_tool(
        {
            "position": "CO",
            "street": "preflop",
            "pot_bb": 4.0,
            "stack_bb": 100,
            "spr": 25,
            "tags": ["CO", "被主导"],
        }
    )

    assert tool.result["position"] == "CO"
    assert tool.result["tags"] == ["CO", "被主导"]


def test_agent_tool_selection_can_choose_no_tools() -> None:
    assert normalize_selected_tool_names([], QUIZ_COACH_TOOL_CATALOG, ["answer_validator"]) == []
    assert normalize_selected_tool_names(["missing"], QUIZ_COACH_TOOL_CATALOG, ["answer_validator"]) == [
        "answer_validator"
    ]
