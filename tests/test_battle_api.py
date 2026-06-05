from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.battle import SESSIONS
from app.main import app
from app.mistakes import MISTAKES


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("POKERCOACH_BATTLE_STORE_DIR", str(tmp_path / "battle-store"))
    monkeypatch.setenv("POKERCOACH_MISTAKE_STORE_DIR", str(tmp_path / "mistake-store"))
    SESSIONS.clear()
    MISTAKES.clear()
    with TestClient(app) as test_client:
        yield test_client
    SESSIONS.clear()
    MISTAKES.clear()


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        json={"email": "alex@example.com", "password": "password"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def registered_auth_headers(client: TestClient, email: str) -> dict[str, str]:
    response = client.post(
        "/auth/register",
        json={"name": "Guest User", "email": email, "password": "password"},
    )
    assert response.status_code == 201
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def assert_observer_privacy(snapshot: dict, observer_seat: int) -> None:
    assert snapshot["observer_seat"] == observer_seat
    for seat in snapshot["seats"]:
        hole_cards = seat["hole_cards"]
        if seat["index"] == observer_seat or snapshot["is_complete"]:
            assert isinstance(hole_cards, list)
            assert len(hole_cards) == 2
        else:
            assert hole_cards is None


def latest_decision_action(snapshot: dict) -> dict | None:
    for action in reversed(snapshot["recent_actions"]):
        if action["decision"] is not None:
            return action
    return None


def test_battle_routes_require_authentication(client: TestClient) -> None:
    assert client.get("/battle/agents").status_code == 401
    assert client.post(
        "/battle/sessions",
        json={"table_size": 6, "observer_seat": 2, "starting_stack_bb": 100},
    ).status_code == 401
    assert client.get("/battle/sessions/bat_missing/history").status_code == 401
    assert client.get("/battle/sessions/bat_missing/hands").status_code == 401


def test_battle_contract_create_observe_advance_and_next_hand(client: TestClient) -> None:
    headers = auth_headers(client)

    agents_response = client.get("/battle/agents", headers=headers)
    assert agents_response.status_code == 200
    agents = agents_response.json()
    assert len(agents) >= 6
    assert {
        "id",
        "name",
        "style",
        "avatar_seed",
        "accent",
        "bio",
        "archetype",
        "mastery_label",
        "gto_score",
        "exploit_score",
        "postflop_score",
        "risk_profile",
        "strategy_tags",
    }.issubset(agents[0])
    assert agents[0]["mastery_label"] == "大师级"
    assert agents[0]["gto_score"] >= 86
    assert agents[0]["postflop_score"] >= 86
    assert agents[0]["strategy_tags"]

    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 2,
            "starting_stack_bb": 100,
            "seed": "api-contract-seed-0",
        },
    )
    assert create_response.status_code == 201
    snapshot = create_response.json()
    session_id = snapshot["id"]

    assert snapshot["table_size"] == 6
    assert snapshot["hand_number"] == 1
    assert snapshot["street"] == "preflop"
    assert snapshot["pot_bb"] == 1.5
    assert snapshot["current_bet_bb"] == 1
    assert snapshot["min_raise_bb"] == 1
    assert snapshot["burned_cards"] == []
    assert snapshot["active_seat"] is not None
    assert len(snapshot["seats"]) == 6
    assert_observer_privacy(snapshot, 2)
    assert [event["event"] for event in snapshot["table_events"]] == [
        "hand_start",
        "blind_posted",
        "blind_posted",
    ]
    assert [event["sequence"] for event in snapshot["replay_events"]] == [1, 2, 3, 4, 5]
    assert [event["kind"] for event in snapshot["replay_events"]] == ["table", "action", "table", "action", "table"]
    assert snapshot["replay_events"][0]["table_event"] == "hand_start"
    assert snapshot["replay_events"][1]["action"] == "blind"
    assert snapshot["replay_events"][1]["amount_bb"] == 0.5
    assert snapshot["replay_events"][1]["pot_bb"] == 0.5
    assert snapshot["replay_events"][2]["table_event"] == "blind_posted"
    assert snapshot["replay_events"][-1]["table_event"] == "blind_posted"
    assert snapshot["table_events"][0]["street"] == "preflop"
    assert snapshot["table_events"][0]["cards"] == []
    assert snapshot["table_events"][1]["seat_index"] == 1
    assert snapshot["table_events"][2]["seat_index"] == 2

    assert len(snapshot["action_timeline"]) == 1
    preflop_group = snapshot["action_timeline"][0]
    assert preflop_group["street"] == "preflop"
    assert [action["action"] for action in preflop_group["actions"]] == ["blind", "blind"]
    assert all(action["decision"] is None for action in preflop_group["actions"])

    switched_response = client.get(
        f"/battle/sessions/{session_id}?observer_seat=4",
        headers=headers,
    )
    assert switched_response.status_code == 200
    switched_snapshot = switched_response.json()
    assert_observer_privacy(switched_snapshot, 4)
    assert switched_snapshot["seats"][2]["hole_cards"] is None

    advanced = switched_snapshot
    decision_action = None
    for _ in range(12):
        advance_response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 4, "steps": 1},
        )
        assert advance_response.status_code == 200
        advanced = advance_response.json()
        decision_action = latest_decision_action(advanced)
        if decision_action is not None:
            break

    assert decision_action is not None
    decision = decision_action["decision"]
    assert decision["source"]
    assert decision["engine"] in {"range_chart", "treys_monte_carlo"}
    assert decision["equity_samples"] >= 0
    assert "大师级" in decision["policy_profile"]
    assert decision["chosen_action"] is not None
    assert decision["chosen_label"]
    assert decision["chosen_ev_bb"] is not None
    assert decision["best_alternative_action"] is not None
    assert decision["best_alternative_ev_bb"] is not None
    assert decision["ev_delta_bb"] is not None
    assert decision["hand_class"]
    assert decision["range_bucket"]
    assert decision["range_role"]
    assert 0 < decision["range_frequency"] <= 1
    assert 0 <= decision["pot_odds"] <= 1
    assert decision["spr"] > 0
    assert 0.45 <= decision["confidence"] <= 0.98
    assert decision["recommended_total_bb"] >= 0
    assert decision["tags"]
    assert len(decision["candidates"]) >= 2
    assert sum(1 for candidate in decision["candidates"] if candidate["is_chosen"]) == 1
    assert any(group["actions"] for group in advanced["action_timeline"])
    latest_replay_action = next(
        event for event in reversed(advanced["replay_events"]) if event["kind"] == "action"
    )
    assert latest_replay_action["action_id"] == decision_action["id"]
    assert latest_replay_action["decision"] == decision
    assert latest_replay_action["agent_name"] == decision_action["agent_name"]
    assert_observer_privacy(advanced, 4)

    complete = advanced
    for _ in range(220):
        if complete["is_complete"]:
            break
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 4, "steps": 8},
        )
        assert response.status_code == 200
        complete = response.json()
    else:
        pytest.fail("battle session did not complete through the API")

    assert complete["street"] == "complete"
    assert complete["result"] is not None
    assert len(complete["board"]) == 5
    assert len(complete["burned_cards"]) == 3
    table_event_names = [event["event"] for event in complete["table_events"]]
    assert "deal_flop" in table_event_names
    assert "deal_turn" in table_event_names
    assert "deal_river" in table_event_names
    assert "showdown" in table_event_names
    assert table_event_names[-1] == "hand_complete"
    assert complete["replay_events"][-1]["table_event"] == "hand_complete"
    assert [event["sequence"] for event in complete["replay_events"]] == list(
        range(1, len(complete["replay_events"]) + 1)
    )
    assert len(complete["replay_events"]) >= sum(len(group["actions"]) for group in complete["action_timeline"])
    assert all(len(seat["hole_cards"] or []) == 2 for seat in complete["seats"])
    assert complete["result"]["side_pots"]
    assert complete["result"]["showdown_details"]
    first_showdown = complete["result"]["showdown_details"][0]
    assert {
        "seat_index",
        "position",
        "agent_id",
        "agent_name",
        "hole_cards",
        "made_hand",
        "hand_rank",
        "is_winner",
        "won_bb",
    }.issubset(first_showdown)
    assert len(first_showdown["hole_cards"]) == 2

    history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=4",
        headers=headers,
    )
    assert history_response.status_code == 200
    history = history_response.json()
    assert history["id"] == f"{session_id}_hand_1"
    assert history["session_id"] == session_id
    assert history["hand_number"] == 1
    assert history["table_size"] == 6
    assert history["seed"] == "api-contract-seed-0"
    assert history["is_complete"] is True
    assert history["observer_seat"] == 4
    assert history["board"] == complete["board"]
    assert history["burned_cards"] == complete["burned_cards"]
    assert history["result"] == complete["result"]
    assert history["action_count"] >= len(complete["recent_actions"])
    assert history["decision_count"] > 0
    assert history["action_timeline"]
    assert [event["event"] for event in history["table_events"]] == table_event_names
    assert history["replay_events"] == complete["replay_events"]
    assert history["replay_events"][-1]["table_event"] == "hand_complete"
    assert history["review_insights"]
    assert {"id", "title", "detail", "icon", "accent", "seat_index", "action_id"}.issubset(history["review_insights"][0])
    assert all(len(seat["hole_cards"] or []) == 2 for seat in history["seats"])

    next_response = client.post(
        f"/battle/sessions/{session_id}/next-hand",
        headers=headers,
        json={"observer_seat": 4},
    )
    assert next_response.status_code == 200
    next_snapshot = next_response.json()
    assert next_snapshot["id"] == session_id
    assert next_snapshot["hand_number"] == 2
    assert next_snapshot["street"] == "preflop"
    assert next_snapshot["board"] == []
    assert next_snapshot["burned_cards"] == []
    assert next_snapshot["pot_bb"] == 1.5
    assert next_snapshot["result"] is None
    assert [event["event"] for event in next_snapshot["table_events"]] == [
        "hand_start",
        "blind_posted",
        "blind_posted",
    ]
    assert [event["kind"] for event in next_snapshot["replay_events"]] == ["table", "action", "table", "action", "table"]
    assert next_snapshot["table_events"][0]["id"].startswith("evt_002")
    assert [action["action"] for action in next_snapshot["action_timeline"][0]["actions"]] == ["blind", "blind"]
    assert_observer_privacy(next_snapshot, 4)

    hands_response = client.get(
        f"/battle/sessions/{session_id}/hands",
        headers=headers,
    )
    assert hands_response.status_code == 200
    hand_summaries = hands_response.json()
    assert len(hand_summaries) == 1
    assert hand_summaries[0]["id"] == f"{session_id}_hand_1"
    assert hand_summaries[0]["hand_number"] == 1
    assert hand_summaries[0]["board"] == complete["board"]
    assert hand_summaries[0]["winners"] == complete["result"]["winners"]
    assert hand_summaries[0]["action_count"] == history["action_count"]
    assert hand_summaries[0]["decision_count"] == history["decision_count"]
    assert hand_summaries[0]["replay_count"] == len(history["replay_events"])
    assert hand_summaries[0]["completed_at"]

    archived_history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=1&hand_number=1",
        headers=headers,
    )
    assert archived_history_response.status_code == 200
    archived_history = archived_history_response.json()
    assert archived_history["hand_number"] == 1
    assert archived_history["observer_seat"] == 1
    assert archived_history["result"] == complete["result"]
    assert archived_history["action_count"] == history["action_count"]
    assert archived_history["replay_events"] == history["replay_events"]
    assert all(len(seat["hole_cards"] or []) == 2 for seat in archived_history["seats"])
    assert next(seat for seat in archived_history["seats"] if seat["index"] == 1)["is_observer"] is True

    missing_hand_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=1&hand_number=99",
        headers=headers,
    )
    assert missing_hand_response.status_code == 404


def test_battle_route_errors_are_contractual(client: TestClient) -> None:
    headers = auth_headers(client)

    invalid_create = client.post(
        "/battle/sessions",
        headers=headers,
        json={"table_size": 6, "observer_seat": 9, "starting_stack_bb": 100},
    )
    assert invalid_create.status_code == 422

    missing = client.get("/battle/sessions/not-found", headers=headers)
    assert missing.status_code == 404

    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={"table_size": 6, "observer_seat": 2, "starting_stack_bb": 100},
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    next_too_early = client.post(
        f"/battle/sessions/{session_id}/next-hand",
        headers=headers,
        json={"observer_seat": 2},
    )
    assert next_too_early.status_code == 409


def test_player_wrong_action_is_recorded_as_reviewable_mistake(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 2,
            "observer_seat": 0,
            "player_seat": 0,
            "mode": "play",
            "starting_stack_bb": 100,
            "seed": "mistake-seed-3",
        },
    )
    assert create_response.status_code == 201
    snapshot = create_response.json()
    session_id = snapshot["id"]
    assert snapshot["active_seat"] == 0
    assert snapshot["seats"][0]["is_human"] is True

    action_response = client.post(
        f"/battle/sessions/{session_id}/player-action",
        headers=headers,
        json={"observer_seat": 0, "action": "fold"},
    )
    assert action_response.status_code == 200

    list_response = client.get("/training/mistakes", headers=headers)
    assert list_response.status_code == 200
    mistakes = list_response.json()
    assert mistakes
    mistake = mistakes[0]
    assert mistake["id"].startswith("mis_")
    assert mistake["position"] == "SB"
    assert mistake["hero_cards"] == ["Jc", "Ts"]
    assert mistake["user_action_label"] == "弃牌"
    assert mistake["recommended_action_label"] in {"加注", "下注"}
    assert mistake["ev_delta_bb"] > 0

    detail_response = client.get(f"/training/mistakes/{mistake['id']}", headers=headers)
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["session_id"] == session_id
    assert detail["scenario"]["session_id"] == session_id
    assert detail["scenario"]["hand_number"] == 1
    assert detail["scenario"]["hero_cards"] == ["Jc", "Ts"]
    assert detail["scenario"]["table_seats"]
    assert detail["candidates"]
    assert any(candidate["is_recommended"] for candidate in detail["candidates"])
    assert "EV" in detail["why_wrong"]
    assert detail["correct_play"]
    assert detail["coach_messages"][0]["role"] == "agent"

    coach_response = client.post(
        f"/training/mistakes/{mistake['id']}/coach",
        headers=headers,
        json={"message": "为什么不能弃牌？"},
    )
    assert coach_response.status_code == 200
    coached = coach_response.json()
    assert coached["coach_messages"][-2]["role"] == "user"
    assert coached["coach_messages"][-1]["role"] == "agent"
    assert "SPR" in coached["coach_messages"][-1]["content"]


def test_battle_sessions_are_private_to_the_creating_user(client: TestClient) -> None:
    owner_headers = auth_headers(client)
    other_headers = registered_auth_headers(client, "other-battle-user@example.com")

    create_response = client.post(
        "/battle/sessions",
        headers=owner_headers,
        json={
            "table_size": 6,
            "observer_seat": 2,
            "starting_stack_bb": 100,
            "seed": "private-session-seed",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    assert client.get(
        f"/battle/sessions/{session_id}?observer_seat=4",
        headers=other_headers,
    ).status_code == 404
    assert client.post(
        f"/battle/sessions/{session_id}/advance",
        headers=other_headers,
        json={"observer_seat": 4, "steps": 1},
    ).status_code == 404
    assert client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=4",
        headers=other_headers,
    ).status_code == 404
    assert client.get(
        f"/battle/sessions/{session_id}/hands",
        headers=other_headers,
    ).status_code == 404
    assert client.post(
        f"/battle/sessions/{session_id}/next-hand",
        headers=other_headers,
        json={"observer_seat": 4},
    ).status_code == 404

    owner_snapshot = client.get(
        f"/battle/sessions/{session_id}?observer_seat=2",
        headers=owner_headers,
    )
    assert owner_snapshot.status_code == 200
    assert owner_snapshot.json()["id"] == session_id

    SESSIONS.clear()

    assert client.get(
        f"/battle/sessions/{session_id}?observer_seat=4",
        headers=other_headers,
    ).status_code == 404
    restored_owner_snapshot = client.get(
        f"/battle/sessions/{session_id}?observer_seat=2",
        headers=owner_headers,
    )
    assert restored_owner_snapshot.status_code == 200
    assert restored_owner_snapshot.json()["id"] == session_id


def test_battle_session_list_is_private_and_restores_from_disk(client: TestClient) -> None:
    owner_headers = auth_headers(client)
    other_headers = registered_auth_headers(client, "other-list-user@example.com")

    first_response = client.post(
        "/battle/sessions",
        headers=owner_headers,
        json={
            "table_size": 6,
            "observer_seat": 2,
            "starting_stack_bb": 100,
            "seed": "owner-list-first",
        },
    )
    assert first_response.status_code == 201
    first_id = first_response.json()["id"]

    second_response = client.post(
        "/battle/sessions",
        headers=owner_headers,
        json={
            "table_size": 9,
            "observer_seat": 4,
            "starting_stack_bb": 100,
            "seed": "owner-list-second",
        },
    )
    assert second_response.status_code == 201
    second_id = second_response.json()["id"]

    other_response = client.post(
        "/battle/sessions",
        headers=other_headers,
        json={
            "table_size": 2,
            "observer_seat": 1,
            "starting_stack_bb": 100,
            "seed": "other-list-session",
        },
    )
    assert other_response.status_code == 201
    other_id = other_response.json()["id"]

    advance_response = client.post(
        f"/battle/sessions/{first_id}/advance",
        headers=owner_headers,
        json={"observer_seat": 2, "steps": 2},
    )
    assert advance_response.status_code == 200

    SESSIONS.clear()

    owner_list_response = client.get("/battle/sessions", headers=owner_headers)
    assert owner_list_response.status_code == 200
    owner_sessions = owner_list_response.json()
    owner_ids = [session["id"] for session in owner_sessions]
    assert owner_ids[:2] == [first_id, second_id]
    assert other_id not in owner_ids
    assert {
        "id",
        "table_size",
        "hand_number",
        "street",
        "stage_label",
        "pot_bb",
        "active_seat",
        "completed_hand_count",
        "is_complete",
        "last_event_label",
        "created_at",
        "updated_at",
    }.issubset(owner_sessions[0])
    assert owner_sessions[0]["last_event_label"]
    assert owner_sessions[0]["updated_at"] >= owner_sessions[0]["created_at"]

    other_list_response = client.get("/battle/sessions", headers=other_headers)
    assert other_list_response.status_code == 200
    other_sessions = other_list_response.json()
    assert [session["id"] for session in other_sessions] == [other_id]


def test_battle_snapshot_resumes_live_session_by_id(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 0,
            "starting_stack_bb": 100,
            "seed": "resume-session-seed",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    advanced = create_response.json()
    for _ in range(6):
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 0, "steps": 1},
        )
        assert response.status_code == 200
        advanced = response.json()
        if len(advanced["recent_actions"]) > 3:
            break

    resume_response = client.get(
        f"/battle/sessions/{session_id}?observer_seat=5",
        headers=headers,
    )
    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["id"] == session_id
    assert resumed["hand_number"] == advanced["hand_number"]
    assert resumed["street"] == advanced["street"]
    assert resumed["board"] == advanced["board"]
    assert resumed["burned_cards"] == advanced["burned_cards"]
    assert resumed["pot_bb"] == advanced["pot_bb"]
    assert resumed["current_bet_bb"] == advanced["current_bet_bb"]
    assert resumed["active_seat"] == advanced["active_seat"]
    assert resumed["action_timeline"] == advanced["action_timeline"]
    assert resumed["replay_events"] == advanced["replay_events"]
    assert_observer_privacy(resumed, 5)
    if not resumed["is_complete"]:
        assert next(seat for seat in resumed["seats"] if seat["index"] == 0)["hole_cards"] is None


def test_battle_session_persists_after_memory_cache_clear(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 1,
            "starting_stack_bb": 100,
            "seed": "disk-resume-seed",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    advanced = create_response.json()
    for _ in range(8):
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 1, "steps": 1},
        )
        assert response.status_code == 200
        advanced = response.json()
        if len(advanced["recent_actions"]) > 4:
            break

    SESSIONS.clear()

    restored_response = client.get(
        f"/battle/sessions/{session_id}?observer_seat=4",
        headers=headers,
    )
    assert restored_response.status_code == 200
    restored = restored_response.json()
    assert restored["id"] == session_id
    assert restored["observer_seat"] == 4
    assert restored["hand_number"] == advanced["hand_number"]
    assert restored["street"] == advanced["street"]
    assert restored["board"] == advanced["board"]
    assert restored["burned_cards"] == advanced["burned_cards"]
    assert restored["pot_bb"] == advanced["pot_bb"]
    assert restored["current_bet_bb"] == advanced["current_bet_bb"]
    assert restored["action_timeline"] == advanced["action_timeline"]
    assert restored["replay_events"] == advanced["replay_events"]
    assert_observer_privacy(restored, 4)

    continued_response = client.post(
        f"/battle/sessions/{session_id}/advance",
        headers=headers,
        json={"observer_seat": 4, "steps": 1},
    )
    assert continued_response.status_code == 200
    continued = continued_response.json()
    assert continued["id"] == session_id
    assert len(continued["replay_events"]) >= len(restored["replay_events"])
    assert continued["hand_number"] == restored["hand_number"]


@pytest.mark.parametrize(
    ("table_size", "expected_positions", "initial_active_position"),
    [
        (2, ["SB", "BB"], "SB"),
        (9, ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"], "UTG"),
    ],
)
def test_battle_api_supports_table_sizes_with_real_agent_decisions(
    client: TestClient,
    table_size: int,
    expected_positions: list[str],
    initial_active_position: str,
) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": table_size,
            "observer_seat": min(1, table_size - 1),
            "starting_stack_bb": 100,
            "seed": f"table-size-contract-{table_size}",
        },
    )
    assert create_response.status_code == 201
    snapshot = create_response.json()
    session_id = snapshot["id"]

    assert [seat["position"] for seat in snapshot["seats"]] == expected_positions
    active_seat = next(seat for seat in snapshot["seats"] if seat["is_active"])
    assert active_seat["position"] == initial_active_position
    assert_observer_privacy(snapshot, min(1, table_size - 1))

    advanced = snapshot
    decision_action = None
    for _ in range(16):
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 0, "steps": 1},
        )
        assert response.status_code == 200
        advanced = response.json()
        decision_action = latest_decision_action(advanced)
        if decision_action is not None:
            break

    assert decision_action is not None
    assert decision_action["position"] in expected_positions
    assert decision_action["decision"]["policy_profile"].endswith("大师级")
    assert decision_action["decision"]["candidates"]
    assert_observer_privacy(advanced, 0)


def test_completed_hand_history_and_next_hand_survive_memory_cache_clear(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 2,
            "starting_stack_bb": 100,
            "seed": "completed-hand-persist-seed",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    complete = create_response.json()
    for _ in range(220):
        if complete["is_complete"]:
            break
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 2, "steps": 8},
        )
        assert response.status_code == 200
        complete = response.json()
    else:
        pytest.fail("battle session did not complete before persistence restore")

    SESSIONS.clear()

    history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=4",
        headers=headers,
    )
    assert history_response.status_code == 200
    history = history_response.json()
    assert history["hand_number"] == 1
    assert history["is_complete"] is True
    assert history["result"] == complete["result"]
    assert history["replay_events"] == complete["replay_events"]
    assert next(seat for seat in history["seats"] if seat["index"] == 4)["is_observer"] is True

    next_response = client.post(
        f"/battle/sessions/{session_id}/next-hand",
        headers=headers,
        json={"observer_seat": 4},
    )
    assert next_response.status_code == 200
    next_snapshot = next_response.json()
    assert next_snapshot["hand_number"] == 2
    assert next_snapshot["street"] == "preflop"
    assert next_snapshot["board"] == []
    assert next_snapshot["burned_cards"] == []
    assert next_snapshot["result"] is None
    assert_observer_privacy(next_snapshot, 4)


def test_replay_events_reconstruct_visible_table_state(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 0,
            "starting_stack_bb": 100,
            "seed": "api-contract-seed-0",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    snapshot = create_response.json()
    for _ in range(220):
        if snapshot["is_complete"]:
            break
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 0, "steps": 8},
        )
        assert response.status_code == 200
        snapshot = response.json()
    else:
        pytest.fail("battle session did not complete through replay reconstruction path")

    replay_events = snapshot["replay_events"]
    assert replay_events
    assert [event["sequence"] for event in replay_events] == list(range(1, len(replay_events) + 1))
    assert replay_events[-1]["table_event"] == "hand_complete"

    board: list[str] = []
    latest_action_by_seat: dict[int, dict] = {}
    for event in replay_events:
        if event["kind"] == "table" and event["table_event"] in {"deal_flop", "deal_turn", "deal_river"}:
            board.extend(event["cards"])
        if event["kind"] == "action":
            latest_action_by_seat[event["seat_index"]] = event

    assert board == snapshot["board"]
    assert replay_events[-1]["pot_bb"] == snapshot["pot_bb"]

    timeline_actions = [
        action
        for group in snapshot["action_timeline"]
        for action in group["actions"]
    ]
    expected_latest_by_seat = {action["seat_index"]: action for action in timeline_actions}
    assert expected_latest_by_seat
    assert set(latest_action_by_seat) == set(expected_latest_by_seat)
    for seat_index, replay_action in latest_action_by_seat.items():
        expected = expected_latest_by_seat[seat_index]
        assert replay_action["action_id"] == expected["id"]
        assert replay_action["action"] == expected["action"]
        assert replay_action["amount_bb"] == expected["amount_bb"]
        assert replay_action["total_bet_bb"] == expected["total_bet_bb"]
        assert replay_action["pot_bb"] == expected["pot_bb"]

    first_deal_index = next(
        index
        for index, event in enumerate(replay_events)
        if event["kind"] == "table" and event["table_event"] in {"deal_flop", "deal_turn", "deal_river"}
    )
    assert all(
        event["table_event"] not in {"deal_flop", "deal_turn", "deal_river"}
        for event in replay_events[:first_deal_index]
        if event["kind"] == "table"
    )

    switched_response = client.get(
        f"/battle/sessions/{session_id}?observer_seat=3",
        headers=headers,
    )
    assert switched_response.status_code == 200
    switched_snapshot = switched_response.json()
    assert switched_snapshot["observer_seat"] == 3
    assert switched_snapshot["replay_events"] == replay_events
    assert all(len(seat["hole_cards"] or []) == 2 for seat in switched_snapshot["seats"])


def test_latest_history_remains_last_completed_hand_after_next_hand_starts(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post(
        "/battle/sessions",
        headers=headers,
        json={
            "table_size": 6,
            "observer_seat": 2,
            "starting_stack_bb": 100,
            "seed": "history-default-seed",
        },
    )
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    snapshot = create_response.json()
    for _ in range(220):
        if snapshot["is_complete"]:
            break
        response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 2, "steps": 8},
        )
        assert response.status_code == 200
        snapshot = response.json()
    else:
        pytest.fail("first hand did not complete")

    first_history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=2",
        headers=headers,
    )
    assert first_history_response.status_code == 200
    first_history = first_history_response.json()
    assert first_history["hand_number"] == 1
    assert first_history["is_complete"] is True
    assert first_history["replay_events"][-1]["table_event"] == "hand_complete"

    next_response = client.post(
        f"/battle/sessions/{session_id}/next-hand",
        headers=headers,
        json={"observer_seat": 4},
    )
    assert next_response.status_code == 200
    next_snapshot = next_response.json()
    assert next_snapshot["hand_number"] == 2
    assert next_snapshot["is_complete"] is False

    default_history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=4",
        headers=headers,
    )
    assert default_history_response.status_code == 200
    default_history = default_history_response.json()
    assert default_history["hand_number"] == 1
    assert default_history["observer_seat"] == 4
    assert default_history["replay_events"] == first_history["replay_events"]
    assert next(seat for seat in default_history["seats"] if seat["index"] == 4)["is_observer"] is True

    for _ in range(4):
        advance_response = client.post(
            f"/battle/sessions/{session_id}/advance",
            headers=headers,
            json={"observer_seat": 4, "steps": 1},
        )
        assert advance_response.status_code == 200

    archived_history_response = client.get(
        f"/battle/sessions/{session_id}/history?observer_seat=1&hand_number=1",
        headers=headers,
    )
    assert archived_history_response.status_code == 200
    archived_history = archived_history_response.json()
    assert archived_history["hand_number"] == 1
    assert archived_history["observer_seat"] == 1
    assert archived_history["replay_events"] == first_history["replay_events"]
    assert archived_history["result"] == first_history["result"]
