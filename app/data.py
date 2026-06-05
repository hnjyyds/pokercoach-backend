from __future__ import annotations

from app.schemas import Choice, HandQuiz, PreflopScenario, User


DEMO_USER = User(
    id="usr_demo",
    name="Alex",
    email="alex@example.com",
    level="新手进阶",
    streak_days=6,
    skill_score=684,
)

USERS: dict[str, User] = {DEMO_USER.email: DEMO_USER}
TOKENS: dict[str, str] = {"mock-token-alex": DEMO_USER.email}

SCENARIOS: list[PreflopScenario] = [
    PreflopScenario(
        id="pf_001",
        position="BTN",
        hand="A9s",
        table_state="前面玩家全部弃牌",
        villain_action="无人入池",
        stack_depth_bb=100,
        pot_bb=1.5,
        choices=[
            Choice(action="fold", label="弃牌"),
            Choice(action="call", label="跟注", sizing="1BB"),
            Choice(action="raise", label="加注", sizing="2.5BB"),
        ],
        recommended_action="raise",
        recommended_sizing="2.5BB",
        concept_tags=["位置优势", "偷盲", "同花高牌"],
        explanation="按钮位面对无人入池时范围可以明显打开。这类同花高牌有阻断顶张范围、同花潜力和位置优势，标准策略是开放加注而不是 limp。",
    ),
    PreflopScenario(
        id="pf_002",
        position="CO",
        hand="KQo",
        table_state="UTG 加注到 2.5BB，其余弃牌",
        villain_action="UTG open raise",
        stack_depth_bb=100,
        pot_bb=4.0,
        choices=[
            Choice(action="fold", label="弃牌"),
            Choice(action="call", label="跟注", sizing="2.5BB"),
            Choice(action="raise", label="3-bet", sizing="8BB"),
        ],
        recommended_action="fold",
        recommended_sizing="-",
        concept_tags=["反向隐含赔率", "位置", "范围压制"],
        explanation="这类高张非同花组合面对 UTG 强范围容易被顶张强范围和高对子压制。CO 位置还没有绝对优势，新手阶段建议先弃牌，减少被主导牌拖入大底池。",
    ),
    PreflopScenario(
        id="pf_003",
        position="BB",
        hand="77",
        table_state="BTN 加注到 2.5BB，SB 弃牌",
        villain_action="BTN open raise",
        stack_depth_bb=80,
        pot_bb=4.0,
        choices=[
            Choice(action="fold", label="弃牌"),
            Choice(action="call", label="防守跟注", sizing="1.5BB"),
            Choice(action="raise", label="3-bet", sizing="9BB"),
        ],
        recommended_action="call",
        recommended_sizing="补 1.5BB",
        concept_tags=["盲注防守", "口袋对子", "成套价值"],
        explanation="中小口袋对子在大盲面对按钮位宽范围时有足够权益防守。跟注能保留对手宽范围，击中暗三条时有很好的隐含赔率。",
    ),
]

HAND_QUIZZES: list[HandQuiz] = [
    HandQuiz(
        id="hq_001",
        hero_hand="Ah Kh",
        villain_hand="Qs Qd",
        board="Th Jh 2c 3h 9s",
        question="河牌摊牌谁赢？",
        options=["Hero", "Villain", "平局"],
        answer="Hero",
        explanation="Hero 用手牌和公共牌组成高张同花，击败 Villain 的一对高张。",
    ),
    HandQuiz(
        id="hq_002",
        hero_hand="8c 8d",
        villain_hand="As Kd",
        board="8h Ac Ks 2d 2s",
        question="Hero 最终牌型是什么？",
        options=["三条", "葫芦", "两对"],
        answer="葫芦",
        explanation="Hero 的口袋对子命中三条，再配公共牌对子，最终是葫芦。",
    ),
]
