from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.battle import (
    BattleAdvanceRequest,
    BattleAgentProfile,
    BattleHandHistorySnapshot,
    BattleHandSummarySnapshot,
    BattleNextHandRequest,
    BattlePlayerActionRequest,
    BattleSessionCreate,
    BattleSessionSnapshot,
    BattleSessionSummarySnapshot,
    advance_battle_session,
    apply_player_battle_action,
    battle_hand_history,
    create_battle_session,
    get_battle_session,
    list_agents,
    list_battle_hand_summaries,
    list_battle_sessions,
    start_next_hand,
)
from app.dependencies import require_current_user
from app.schemas import User


router = APIRouter(prefix="/battle", tags=["battle"])


@router.get("/agents", response_model=list[BattleAgentProfile])
def battle_agents(_: User = Depends(require_current_user)) -> list[BattleAgentProfile]:
    return list_agents()


@router.post("/sessions", response_model=BattleSessionSnapshot, status_code=status.HTTP_201_CREATED)
def create_battle(
    payload: BattleSessionCreate,
    user: User = Depends(require_current_user),
) -> BattleSessionSnapshot:
    session = create_battle_session(payload, owner_id=user.id, owner_name=user.name)
    return session.snapshot(payload.player_seat if payload.mode == "play" and payload.player_seat is not None else payload.observer_seat)


@router.get("/sessions", response_model=list[BattleSessionSummarySnapshot])
def battle_sessions(user: User = Depends(require_current_user)) -> list[BattleSessionSummarySnapshot]:
    return list_battle_sessions(owner_id=user.id)


@router.get("/sessions/{session_id}", response_model=BattleSessionSnapshot)
def battle_snapshot(
    session_id: str,
    observer_seat: int = 0,
    user: User = Depends(require_current_user),
) -> BattleSessionSnapshot:
    session = get_battle_session(session_id, owner_id=user.id)
    return session.snapshot(observer_seat)


@router.get("/sessions/{session_id}/history", response_model=BattleHandHistorySnapshot)
def battle_history(
    session_id: str,
    observer_seat: int = 0,
    hand_number: int | None = None,
    user: User = Depends(require_current_user),
) -> BattleHandHistorySnapshot:
    return battle_hand_history(session_id, observer_seat, hand_number, owner_id=user.id)


@router.get("/sessions/{session_id}/hands", response_model=list[BattleHandSummarySnapshot])
def battle_hands(
    session_id: str,
    user: User = Depends(require_current_user),
) -> list[BattleHandSummarySnapshot]:
    return list_battle_hand_summaries(session_id, owner_id=user.id)


@router.post("/sessions/{session_id}/advance", response_model=BattleSessionSnapshot)
def advance_battle(
    session_id: str,
    payload: BattleAdvanceRequest,
    user: User = Depends(require_current_user),
) -> BattleSessionSnapshot:
    session = advance_battle_session(session_id, payload.steps, owner_id=user.id)
    return session.snapshot(payload.observer_seat)


@router.post("/sessions/{session_id}/player-action", response_model=BattleSessionSnapshot)
def player_battle_action(
    session_id: str,
    payload: BattlePlayerActionRequest,
    user: User = Depends(require_current_user),
) -> BattleSessionSnapshot:
    session = apply_player_battle_action(session_id, payload, owner_id=user.id)
    return session.snapshot(payload.observer_seat)


@router.post("/sessions/{session_id}/next-hand", response_model=BattleSessionSnapshot)
def next_battle_hand(
    session_id: str,
    payload: BattleNextHandRequest,
    user: User = Depends(require_current_user),
) -> BattleSessionSnapshot:
    session = start_next_hand(session_id, owner_id=user.id)
    return session.snapshot(payload.observer_seat)
