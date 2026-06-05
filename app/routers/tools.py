from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import require_current_user
from app.schemas import OddsRequest, OddsResponse, User


router = APIRouter(prefix="/tools", tags=["tools"])


@router.post("/odds", response_model=OddsResponse)
def odds(payload: OddsRequest, _: User = Depends(require_current_user)) -> OddsResponse:
    turn_or_river = min(payload.outs * 0.0217, 0.95)
    by_river = min(payload.outs * 0.04, 0.95)
    return OddsResponse(
        hero_hand=payload.hero_hand,
        board=payload.board,
        outs=payload.outs,
        turn_or_river_probability=round(turn_or_river, 3),
        by_river_probability=round(by_river, 3),
        coaching_note="这是基于 2/4 法则的教学近似值，适合新手快速估算；正式版本可接入完整胜率计算器。",
    )
