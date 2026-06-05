from __future__ import annotations

from fastapi import APIRouter, Depends

from app.data import SCENARIOS
from app.dependencies import require_current_user
from app.mistakes import dashboard_mistake_texts
from app.schemas import DailyPlan, DashboardResponse, ModuleCard, User


router = APIRouter(tags=["users"])


@router.get("/me", response_model=User)
def me(user: User = Depends(require_current_user)) -> User:
    return user


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(user: User = Depends(require_current_user)) -> DashboardResponse:
    return DashboardResponse(
        user=user,
        daily_plan=DailyPlan(
            title="今日 10 分钟训练",
            target_minutes=10,
            completed_minutes=4,
            focus="按钮位开放范围 + 大盲防守",
            modules=[
                ModuleCard(
                    id="preflop",
                    title="翻前范围",
                    subtitle="12 题情景决策",
                    progress=0.36,
                    icon="scope",
                    accent="#0F766E",
                ),
                ModuleCard(
                    id="hand_quiz",
                    title="牌力识别",
                    subtitle="快速判断摊牌结果",
                    progress=0.62,
                    icon="suit.club.fill",
                    accent="#D97706",
                ),
                ModuleCard(
                    id="odds",
                    title="赔率工具",
                    subtitle="outs 与补牌概率",
                    progress=0.18,
                    icon="percent",
                    accent="#DC2626",
                ),
            ],
        ),
        recent_mistakes=dashboard_mistake_texts(user.id),
        next_drill_id=SCENARIOS[0].id,
    )
