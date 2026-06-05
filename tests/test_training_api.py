from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.quiz_agent import GENERATED_QUIZZES


@pytest.fixture()
def client() -> Iterator[TestClient]:
    GENERATED_QUIZZES.clear()
    with TestClient(app) as test_client:
        yield test_client
    GENERATED_QUIZZES.clear()


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/auth/login",
        json={"email": "alex@example.com", "password": "password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_agent_generated_hand_quiz_contract(client: TestClient) -> None:
    headers = auth_headers(client)

    listing_response = client.get("/training/hand-quiz", headers=headers)
    assert listing_response.status_code == 200
    seed_quiz = listing_response.json()[0]
    assert {"source_agent", "thesis", "concept_tags", "coach_messages"}.issubset(seed_quiz)
    assert seed_quiz["coach_messages"][0]["role"] == "agent"

    generated_response = client.post(
        "/training/hand-quiz/generate",
        headers=headers,
        json={"focus": "底池赔率", "difficulty": "进阶", "street": "flop"},
    )
    assert generated_response.status_code == 201
    generated = generated_response.json()
    assert generated["id"].startswith("hq_agent_")
    assert generated["source_agent"] == "Nash"
    assert generated["street"] == "flop"
    assert generated["hero_hand"]
    assert generated["board"]
    assert generated["thesis"]
    assert generated["coach_messages"][0]["role"] == "agent"

    answer_response = client.post(
        f"/training/hand-quiz/{generated['id']}/answer",
        headers=headers,
        json={"answer": generated["answer"]},
    )
    assert answer_response.status_code == 200
    assert answer_response.json()["is_correct"] is True
    assert answer_response.json()["concept_tags"]

    coach_response = client.post(
        f"/training/hand-quiz/{generated['id']}/coach",
        headers=headers,
        json={"message": "为什么这里不是已经领先？"},
    )
    assert coach_response.status_code == 200
    coached = coach_response.json()
    assert [message["role"] for message in coached["coach_messages"][-2:]] == ["user", "agent"]
    assert generated["thesis"].rstrip("。.!！?") in coached["coach_messages"][-1]["content"]


def test_showdown_coach_answers_card_shape_before_strategy(client: TestClient) -> None:
    headers = auth_headers(client)
    generated_response = client.post(
        "/training/hand-quiz/generate",
        headers=headers,
        json={"focus": "牌力识别", "difficulty": "新手", "street": "river"},
    )
    assert generated_response.status_code == 201
    quiz = generated_response.json()

    coach_response = client.post(
        f"/training/hand-quiz/{quiz['id']}/coach",
        headers=headers,
        json={"message": "Why full house?"},
    )

    assert coach_response.status_code == 200
    reply = coach_response.json()["coach_messages"][-1]["content"]
    assert "最佳五张牌" in reply
    assert "三张" in reply
    assert "一对" in reply
    assert "葫芦" in reply
    assert "Hero 赢" in reply
    assert "BB 位置" not in reply
