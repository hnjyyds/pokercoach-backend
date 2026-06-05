from __future__ import annotations

import unittest

from app.battle import (
    BrainDecision,
    BattleSessionCreate,
    BattlePlayerActionRequest,
    DecisionTrace,
    apply_decision,
    apply_player_battle_action,
    advance_street,
    advance_battle_session,
    create_battle_session,
    decide_action,
    hand_notation,
    list_agents,
    next_actor_after,
    positions_for,
    settle_showdown,
    start_next_hand,
)
from fastapi import HTTPException


class BattleEngineTests(unittest.TestCase):
    def decision_stub(self, action: str, target_total_bb: float) -> BrainDecision:
        return BrainDecision(
            action=action,
            target_total_bb=target_total_bb,
            equity=0.8,
            pot_odds=0.2,
            note="test decision",
            trace=DecisionTrace(
                source="test",
                engine="test_engine",
                equity_samples=0,
                policy_profile="test profile",
                hand_class="AA",
                range_bucket="价值进攻",
                range_role="价值3bet",
                range_frequency=1.0,
                board_texture="翻前",
                equity=0.8,
                pot_odds=0.2,
                spr=3,
                pressure=0.5,
                confidence=0.9,
                recommended_total_bb=target_total_bb,
                tags=["test"],
                summary="test",
            ),
        )

    def test_positions_for_supported_table_sizes(self) -> None:
        self.assertEqual(positions_for(2), ["SB", "BB"])
        self.assertEqual(positions_for(3), ["BTN", "SB", "BB"])
        self.assertEqual(positions_for(4), ["BTN", "SB", "BB", "CO"])
        self.assertEqual(positions_for(5), ["BTN", "SB", "BB", "UTG", "CO"])
        self.assertEqual(positions_for(6), ["BTN", "SB", "BB", "UTG", "HJ", "CO"])
        self.assertEqual(positions_for(7), ["BTN", "SB", "BB", "UTG", "LJ", "HJ", "CO"])
        self.assertEqual(positions_for(8), ["BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO"])
        self.assertEqual(positions_for(9), ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"])

    def test_hand_notation_normalizes_pairs_suited_and_offsuit(self) -> None:
        self.assertEqual(hand_notation(["As", "Ah"]), "AA")
        self.assertEqual(hand_notation(["Ts", "As"]), "ATs")
        self.assertEqual(hand_notation(["Kd", "Ac"]), "AKo")

    def test_seed_replays_the_same_deal(self) -> None:
        payload = BattleSessionCreate(table_size=6, observer_seat=2, seed="fixed-seed")
        first = create_battle_session(payload)
        second = create_battle_session(payload)

        first_hands = [player.hole_cards for player in first.players]
        second_hands = [player.hole_cards for player in second.players]
        self.assertEqual(first_hands, second_hands)

    def test_initial_stack_uses_payload_depth(self) -> None:
        session = create_battle_session(
            BattleSessionCreate(
                table_size=6,
                observer_seat=2,
                starting_stack_bb=40,
                seed="starting-stack-depth",
            )
        )

        self.assertEqual([player.stack_bb for player in session.players], [40, 39.5, 39, 40, 40, 40])
        self.assertFalse(session.snapshot(observer_seat=2).is_session_complete)

    def test_play_mode_marks_human_seat_and_waits_for_player_action(self) -> None:
        session = create_battle_session(
            BattleSessionCreate(
                table_size=2,
                observer_seat=0,
                player_seat=0,
                mode="play",
                seed="human-player-heads-up",
            ),
            owner_id="usr_demo",
            owner_name="Alex",
        )

        self.assertEqual(session.mode, "play")
        self.assertEqual(session.player_seat, 0)
        self.assertTrue(session.players[0].is_human)
        self.assertEqual(session.current_actor, 0)

        unchanged_actions = len(session.action_log)
        advance_battle_session(session.id, 4, owner_id="usr_demo")

        self.assertEqual(session.current_actor, 0)
        self.assertEqual(len(session.action_log), unchanged_actions)

        next_session = apply_player_battle_action(
            session.id,
            BattlePlayerActionRequest(observer_seat=0, action="call"),
            owner_id="usr_demo",
        )
        self.assertGreater(len(next_session.action_log), unchanged_actions)
        self.assertFalse(next_session.players[0].last_action is None)
        self.assertIn(next_session.players[0].last_action.action, {"call", "check", "all_in"})

    def test_initial_deal_uses_one_unique_deck_for_every_table_size(self) -> None:
        for table_size in (2, 6, 9):
            with self.subTest(table_size=table_size):
                session = create_battle_session(
                    BattleSessionCreate(
                        table_size=table_size,
                        observer_seat=0,
                        seed=f"deck-integrity-{table_size}",
                    )
                )
                dealt_cards = [card for player in session.players for card in player.hole_cards]

                self.assertEqual(len(dealt_cards), table_size * 2)
                self.assertEqual(len(set(dealt_cards)), table_size * 2)
                self.assertEqual(len(session.deck), 52 - table_size * 2)
                self.assertTrue(set(dealt_cards).isdisjoint(session.deck))

    def test_board_streets_burn_before_flop_turn_and_river(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="burn-flow-seed"))
        session.deck = ["Ac", "2c", "3c", "4c", "Ad", "5d", "Ah", "6h"]
        session.board = []
        session.burned_cards = []
        session.street = "preflop"

        advance_street(session)

        self.assertEqual(session.street, "flop")
        self.assertEqual(session.burned_cards, ["Ac"])
        self.assertEqual(session.board, ["2c", "3c", "4c"])
        self.assertEqual([event.event for event in session.table_events[-2:]], ["burn", "deal_flop"])
        self.assertEqual(session.table_events[-2].burn_card, "Ac")
        self.assertEqual(session.table_events[-1].cards, ["2c", "3c", "4c"])

        advance_street(session)

        self.assertEqual(session.street, "turn")
        self.assertEqual(session.burned_cards, ["Ac", "Ad"])
        self.assertEqual(session.board, ["2c", "3c", "4c", "5d"])
        self.assertEqual([event.event for event in session.table_events[-2:]], ["burn", "deal_turn"])
        self.assertEqual(session.table_events[-2].burn_card, "Ad")
        self.assertEqual(session.table_events[-1].cards, ["5d"])

        advance_street(session)

        self.assertEqual(session.street, "river")
        self.assertEqual(session.burned_cards, ["Ac", "Ad", "Ah"])
        self.assertEqual(session.board, ["2c", "3c", "4c", "5d", "6h"])
        self.assertEqual([event.event for event in session.table_events[-2:]], ["burn", "deal_river"])
        self.assertEqual(session.table_events[-2].burn_card, "Ah")
        self.assertEqual(session.table_events[-1].cards, ["6h"])

    def test_initial_snapshot_posts_blinds_and_keeps_other_hole_cards_private(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=2, seed="privacy-seed"))
        snapshot = session.snapshot(observer_seat=2)

        self.assertEqual(snapshot.pot_bb, 1.5)
        self.assertEqual(snapshot.street, "preflop")
        self.assertEqual(snapshot.active_seat, 3)
        self.assertEqual(len(snapshot.seats), 6)
        self.assertEqual(len(snapshot.seats[2].hole_cards or []), 2)
        self.assertEqual(len(snapshot.action_timeline), 1)
        self.assertEqual(snapshot.action_timeline[0].street, "preflop")
        self.assertEqual([action.action for action in snapshot.action_timeline[0].actions], ["blind", "blind"])
        self.assertEqual([event.event for event in snapshot.table_events], ["hand_start", "blind_posted", "blind_posted"])
        self.assertEqual([event.sequence for event in snapshot.replay_events], [1, 2, 3, 4, 5])
        self.assertEqual([event.kind for event in snapshot.replay_events], ["table", "action", "table", "action", "table"])
        self.assertEqual(snapshot.replay_events[0].table_event, "hand_start")
        self.assertEqual(snapshot.replay_events[1].action, "blind")
        self.assertEqual(snapshot.replay_events[1].seat_index, 1)
        self.assertEqual(snapshot.replay_events[1].amount_bb, 0.5)
        self.assertEqual(snapshot.replay_events[2].table_event, "blind_posted")
        self.assertEqual(snapshot.replay_events[-1].table_event, "blind_posted")
        self.assertEqual(snapshot.table_events[0].street, "preflop")
        self.assertEqual(snapshot.table_events[1].seat_index, 1)
        self.assertEqual(snapshot.table_events[2].seat_index, 2)
        self.assertEqual(snapshot.seats[2].agent.mastery_label, "大师级")
        self.assertGreaterEqual(snapshot.seats[2].agent.gto_score, 86)
        self.assertGreaterEqual(snapshot.seats[2].agent.postflop_score, 86)
        self.assertGreaterEqual(snapshot.seats[2].agent.exploit_score, 86)
        self.assertTrue(snapshot.seats[2].agent.strategy_tags)
        for seat in snapshot.seats:
            if seat.index != 2:
                self.assertIsNone(seat.hole_cards)
        self.assertTrue(all(action.decision is None for action in snapshot.recent_actions))

    def test_agent_profiles_expose_master_strategy_scores(self) -> None:
        agents = list_agents()

        self.assertGreaterEqual(len(agents), 6)
        for agent in agents:
            self.assertEqual(agent.mastery_label, "大师级")
            self.assertGreaterEqual(agent.gto_score, 86)
            self.assertLessEqual(agent.gto_score, 99)
            self.assertGreaterEqual(agent.exploit_score, 86)
            self.assertLessEqual(agent.exploit_score, 99)
            self.assertGreaterEqual(agent.postflop_score, 86)
            self.assertLessEqual(agent.postflop_score, 99)
            self.assertTrue(agent.archetype)
            self.assertTrue(agent.risk_profile)
            self.assertIn("GTO", agent.strategy_tags)

    def test_agent_action_includes_structured_strategy_trace(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=2, seed="trace-seed"))
        active_before = session.current_actor
        self.assertIsNotNone(active_before)
        assert active_before is not None

        advance_battle_session(session.id, 1)
        snapshot = session.snapshot(observer_seat=active_before)
        agent_action = snapshot.recent_actions[-1]

        self.assertEqual(snapshot.observer_seat, active_before)
        self.assertEqual(agent_action.seat_index, active_before)
        self.assertIsNotNone(agent_action.decision)
        assert agent_action.decision is not None
        self.assertIn("GTO", agent_action.decision.source)
        self.assertEqual(agent_action.decision.engine, "range_chart")
        self.assertEqual(agent_action.decision.equity_samples, 0)
        self.assertIn("大师级", agent_action.decision.policy_profile)
        self.assertEqual(agent_action.decision.board_texture, "翻前")
        self.assertEqual(agent_action.decision.hand_class, hand_notation(session.players[active_before].hole_cards))
        self.assertGreater(agent_action.decision.range_frequency, 0)
        self.assertLessEqual(agent_action.decision.range_frequency, 1)
        self.assertTrue(agent_action.decision.range_role)
        self.assertGreaterEqual(agent_action.decision.confidence, 0.45)
        self.assertLessEqual(agent_action.decision.confidence, 0.98)
        self.assertGreater(agent_action.decision.spr, 0)
        self.assertTrue(agent_action.decision.tags)
        self.assertGreaterEqual(len(agent_action.decision.candidates), 2)
        self.assertAlmostEqual(sum(candidate.weight for candidate in agent_action.decision.candidates), 1, delta=0.01)
        self.assertEqual(sum(1 for candidate in agent_action.decision.candidates if candidate.is_chosen), 1)
        chosen_candidate = next(candidate for candidate in agent_action.decision.candidates if candidate.is_chosen)
        self.assertEqual(chosen_candidate.action, agent_action.action)
        self.assertEqual(agent_action.decision.chosen_action, chosen_candidate.action)
        self.assertEqual(agent_action.decision.chosen_label, chosen_candidate.label)
        self.assertEqual(agent_action.decision.chosen_ev_bb, chosen_candidate.ev_bb)
        self.assertIsNotNone(agent_action.decision.best_alternative_action)
        self.assertIsNotNone(agent_action.decision.best_alternative_ev_bb)
        self.assertIsNotNone(agent_action.decision.ev_delta_bb)
        self.assertIsInstance(chosen_candidate.reason, str)
        self.assertGreater(len(chosen_candidate.reason), 0)
        self.assertEqual(snapshot.action_timeline[0].street, "preflop")
        self.assertEqual(len(snapshot.action_timeline[0].actions), 3)
        self.assertEqual(snapshot.action_timeline[0].actions[-1].id, agent_action.id)
        self.assertEqual(snapshot.replay_events[-1].kind, "action")
        self.assertEqual(snapshot.replay_events[-1].action_id, agent_action.id)
        self.assertIsNotNone(snapshot.replay_events[-1].decision)
        self.assertIn(session.players[active_before].position, snapshot.tasks[0].subtitle)
        self.assertIn(agent_action.decision.range_bucket, snapshot.tasks[0].subtitle)
        self.assertNotIn(agent_action.decision.hand_class, snapshot.tasks[0].subtitle)

    def test_preflop_range_profile_drives_position_and_3bet_roles(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="range-profile-seed"))
        opener = session.players[3]
        opener.position = "UTG"
        opener.hole_cards = ["As", "Ah"]
        session.current_actor = opener.index
        session.current_bet_bb = 1
        session.acted_this_street = set()

        open_decision = decide_action(session, opener)
        self.assertEqual(open_decision.action, "raise")
        self.assertEqual(open_decision.trace.range_role, "价值开池")
        self.assertEqual(open_decision.trace.range_frequency, 1.0)

        defender = session.players[4]
        defender.position = "HJ"
        defender.hole_cards = ["As", "Ks"]
        defender.street_bet_bb = 0
        session.current_actor = defender.index
        session.current_bet_bb = 3
        session.acted_this_street = {opener.index}

        defend_decision = decide_action(session, defender)
        self.assertEqual(defend_decision.trace.range_role, "价值3bet")
        self.assertGreaterEqual(defend_decision.trace.range_frequency, 0.85)

    def test_postflop_strategy_uses_texture_position_and_spr(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="dry-aa"))
        session.street = "flop"
        session.board = ["Qs", "7d", "2c"]
        session.pot_bb = 12
        session.current_bet_bb = 0
        session.min_raise_bb = 1
        session.current_actor = 0
        session.dealer_index = 0
        session.acted_this_street = set()
        session.action_log = []

        for player in session.players:
            player.folded = True
            player.all_in = False
            player.stack_bb = 94
            player.street_bet_bb = 0
            player.total_committed_bb = 0
            player.last_action = None

        hero = session.players[0]
        villain = session.players[1]
        hero.folded = False
        villain.folded = False
        hero.position = "BTN"
        villain.position = "BB"
        hero.hole_cards = ["As", "Ah"]
        villain.hole_cards = ["Kd", "Kh"]

        decision = decide_action(session, hero)

        self.assertEqual(decision.action, "bet")
        self.assertEqual(decision.target_total_bb, 7)
        self.assertIn("牌面纹理", decision.trace.source)
        self.assertEqual(decision.trace.engine, "treys_monte_carlo")
        self.assertEqual(decision.trace.equity_samples, 96)
        self.assertIn("大师级", decision.trace.policy_profile)
        self.assertEqual(decision.trace.board_texture, "干燥高张面")
        self.assertEqual(decision.trace.range_bucket, "价值进攻")
        self.assertEqual(decision.trace.range_role, "极化价值下注")
        self.assertGreater(decision.trace.range_frequency, 0.85)
        chosen = next(candidate for candidate in decision.trace.candidates if candidate.is_chosen)
        self.assertEqual(chosen.action, "bet")

    def test_postflop_draw_can_mix_semibluff_raise(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="draw-ak"))
        session.street = "flop"
        session.board = ["Qc", "Jc", "2d"]
        session.pot_bb = 12
        session.current_bet_bb = 5
        session.min_raise_bb = 4
        session.current_actor = 0
        session.dealer_index = 0
        session.acted_this_street = set()
        session.action_log = []

        for player in session.players:
            player.folded = True
            player.all_in = False
            player.stack_bb = 94
            player.street_bet_bb = 0
            player.total_committed_bb = 0
            player.last_action = None

        hero = session.players[0]
        villain = session.players[1]
        hero.folded = False
        villain.folded = False
        hero.position = "BTN"
        villain.position = "BB"
        hero.hole_cards = ["Ac", "Kc"]
        villain.hole_cards = ["9h", "9d"]

        decision = decide_action(session, hero)

        self.assertEqual(decision.action, "raise")
        self.assertEqual(decision.trace.board_texture, "湿润听牌面")
        self.assertEqual(decision.trace.range_role, "半诈唬加注")
        self.assertGreater(decision.trace.equity or 0, decision.trace.pot_odds)
        self.assertTrue(any(candidate.action == "raise" for candidate in decision.trace.candidates))

    def test_battle_completes_with_chip_conservation_and_showdown_privacy_release(self) -> None:
        table_size = 9
        starting_stack = 100
        session = create_battle_session(
            BattleSessionCreate(
                table_size=table_size,
                observer_seat=4,
                starting_stack_bb=starting_stack,
                seed="complete-flow-seed",
            )
        )

        for _ in range(180):
            advance_battle_session(session.id, 1)
            snapshot = session.snapshot(observer_seat=4)
            self.assertGreaterEqual(snapshot.pot_bb, 0)
            self.assertLessEqual(len(snapshot.board), 5)
            self.assertTrue(all(action.amount_bb >= 0 for action in snapshot.recent_actions))
            if snapshot.is_complete:
                break
        else:
            self.fail("battle did not complete within 180 actions")

        self.assertEqual(session.street, "complete")
        self.assertIsNotNone(session.result)
        self.assertIsNone(session.current_actor)
        self.assertAlmostEqual(sum(player.stack_bb for player in session.players), table_size * starting_stack)

        complete_snapshot = session.snapshot(observer_seat=4)
        self.assertTrue(all(len(seat.hole_cards or []) == 2 for seat in complete_snapshot.seats))
        self.assertTrue(complete_snapshot.replay_events)
        self.assertEqual(complete_snapshot.replay_events[-1].table_event, "hand_complete")
        self.assertGreaterEqual(
            len(complete_snapshot.replay_events),
            len(session.action_log) + len(session.table_events),
        )
        if len(session.board) == 5:
            self.assertEqual(len(session.burned_cards), 3)
            visible_cards = set(session.board + session.burned_cards)
            dealt_cards = {card for player in session.players for card in player.hole_cards}
            self.assertTrue(visible_cards.isdisjoint(session.deck))
            self.assertTrue(set(session.burned_cards).isdisjoint(session.board))
            self.assertTrue(set(session.burned_cards).isdisjoint(dealt_cards))

    def test_showdown_splits_main_and_side_pots_by_commitment(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="side-pot-seed"))
        session.board = ["2c", "3d", "4h", "8s", "9c"]
        session.pot_bb = 75
        session.street = "river"
        session.current_actor = None
        session.acted_this_street = set()

        for player in session.players:
            player.hole_cards = ["5c", "7d"]
            player.folded = True
            player.all_in = False
            player.stack_bb = 100
            player.street_bet_bb = 0
            player.total_committed_bb = 0

        short_stack = session.players[0]
        middle_stack = session.players[1]
        deep_stack = session.players[2]
        short_stack.hole_cards = ["As", "Ah"]
        middle_stack.hole_cards = ["Kd", "Kh"]
        deep_stack.hole_cards = ["Qs", "Qh"]

        for player, committed, remaining in [
            (short_stack, 10, 0),
            (middle_stack, 30, 0),
            (deep_stack, 30, 70),
        ]:
            player.folded = False
            player.all_in = remaining == 0
            player.stack_bb = remaining
            player.total_committed_bb = committed
        session.players[3].total_committed_bb = 5
        session.players[3].stack_bb = 95

        settle_showdown(session)

        self.assertEqual(session.street, "complete")
        self.assertIsNotNone(session.result)
        assert session.result is not None
        self.assertEqual([pot.amount_bb for pot in session.result.side_pots], [35, 40])
        self.assertEqual(session.result.side_pots[0].eligible_seats, [0, 1, 2])
        self.assertEqual(session.result.side_pots[0].winners, [0])
        self.assertEqual(session.result.side_pots[1].eligible_seats, [1, 2])
        self.assertEqual(session.result.side_pots[1].winners, [1])
        self.assertEqual(short_stack.stack_bb, 35)
        self.assertEqual(middle_stack.stack_bb, 40)
        self.assertEqual(deep_stack.stack_bb, 70)
        self.assertEqual(len(session.result.showdown_details), 3)
        self.assertEqual(
            [(detail.seat_index, detail.made_hand, detail.is_winner, detail.won_bb) for detail in session.result.showdown_details],
            [(0, "一对", True, 35), (1, "一对", True, 40), (2, "一对", False, 0)],
        )
        self.assertIn("主池", session.result.summary)
        self.assertIn("边池 1", session.result.summary)

    def test_all_in_runout_skips_dead_betting_rounds(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=2, seed="runout-seed"))
        session.board = ["2c", "3d", "4h"]
        session.burned_cards = []
        session.deck = ["6s", "8s", "Td", "9c"]
        session.pot_bb = 30
        session.street = "flop"
        session.current_bet_bb = 0
        session.current_actor = None
        session.acted_this_street = set()
        session.action_log = []

        for player in session.players:
            player.hole_cards = ["5c", "7d"]
            player.folded = True
            player.all_in = False
            player.stack_bb = 100
            player.street_bet_bb = 0
            player.total_committed_bb = 0
            player.last_action = None

        for player, cards, all_in, stack in [
            (session.players[0], ["As", "Ah"], True, 0),
            (session.players[1], ["Kd", "Kh"], True, 0),
            (session.players[2], ["Qs", "Qh"], False, 90),
        ]:
            player.hole_cards = cards
            player.folded = False
            player.all_in = all_in
            player.stack_bb = stack
            player.total_committed_bb = 10

        advance_battle_session(session.id, 1)

        self.assertEqual(session.street, "complete")
        self.assertEqual(session.board, ["2c", "3d", "4h", "8s", "9c"])
        self.assertEqual(session.burned_cards, ["6s", "Td"])
        self.assertIsNone(session.current_actor)
        self.assertEqual(session.action_log, [])
        self.assertIsNotNone(session.result)
        assert session.result is not None
        self.assertEqual(session.result.side_pots[0].winners, [0])
        self.assertTrue(session.result.showdown_details[0].is_winner)
        self.assertEqual(session.result.showdown_details[0].made_hand, "一对")

    def test_short_all_in_does_not_reopen_action_or_reduce_min_raise(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=0, seed="short-all-in-seed"))
        session.street = "flop"
        session.board = ["2c", "7d", "Th"]
        session.pot_bb = 20
        session.current_bet_bb = 10
        session.min_raise_bb = 10
        session.current_actor = 2
        session.acted_this_street = {0, 1}
        session.action_log = []

        for player in session.players:
            player.folded = True
            player.all_in = False
            player.stack_bb = 100
            player.street_bet_bb = 0
            player.total_committed_bb = 0
            player.last_action = None

        prior_raiser = session.players[0]
        caller = session.players[1]
        short_stack = session.players[2]

        prior_raiser.hole_cards = ["As", "Ah"]
        caller.hole_cards = ["Qd", "Qh"]
        short_stack.hole_cards = ["Kc", "Kh"]

        for player in [prior_raiser, caller, short_stack]:
            player.folded = False
        for player in [prior_raiser, caller]:
            player.stack_bb = 90
            player.street_bet_bb = 10
            player.total_committed_bb = 10
        short_stack.stack_bb = 15

        apply_decision(session, short_stack, self.decision_stub("raise", 15))

        self.assertTrue(short_stack.all_in)
        self.assertEqual(session.current_bet_bb, 15)
        self.assertEqual(session.min_raise_bb, 10)
        self.assertEqual(session.acted_this_street, {0, 1, 2})
        self.assertEqual(next_actor_after(session, 2), 0)

        follow_up = decide_action(session, prior_raiser)
        self.assertEqual(follow_up.action, "call")

    def test_next_hand_requires_completed_hand(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=2, seed="not-yet-complete"))

        with self.assertRaises(HTTPException) as context:
            start_next_hand(session.id)

        self.assertEqual(context.exception.status_code, 409)

    def test_next_hand_rotates_button_and_redeals(self) -> None:
        session = create_battle_session(BattleSessionCreate(table_size=6, observer_seat=2, seed="multi-hand-seed"))
        first_hand_observer_cards = list(session.players[2].hole_cards)
        first_dealer = session.dealer_index

        for _ in range(180):
            advance_battle_session(session.id, 1)
            if session.street == "complete":
                break
        else:
            self.fail("first hand did not complete")

        next_session = start_next_hand(session.id)
        next_snapshot = next_session.snapshot(observer_seat=2)

        self.assertEqual(next_session.id, session.id)
        self.assertEqual(next_session.hand_number, 2)
        self.assertEqual(next_session.dealer_index, (first_dealer + 1) % session.table_size)
        self.assertEqual(next_session.street, "preflop")
        self.assertEqual(next_session.board, [])
        self.assertEqual(next_session.burned_cards, [])
        self.assertEqual(next_session.pot_bb, 1.5)
        self.assertEqual(len(next_session.action_log), 2)
        self.assertEqual(next_snapshot.seats[next_session.dealer_index].position, "BTN")
        self.assertNotEqual(first_hand_observer_cards, next_session.players[2].hole_cards)
        self.assertEqual(len(next_snapshot.seats[2].hole_cards or []), 2)
        for seat in next_snapshot.seats:
            if seat.index != 2:
                self.assertIsNone(seat.hole_cards)

    def test_next_hand_eliminates_busted_agent_without_rebuy(self) -> None:
        session = create_battle_session(
            BattleSessionCreate(
                table_size=6,
                observer_seat=2,
                starting_stack_bb=40,
                seed="busted-agent-leaves",
            )
        )
        busted_agent_id = session.players[4].agent.id
        short_agent_id = session.players[0].agent.id
        session.players[0].stack_bb = 7
        session.players[4].stack_bb = 0
        session.street = "complete"

        next_session = start_next_hand(session.id)
        short_stack_survivor = next(player for player in next_session.players if player.agent.id == short_agent_id)

        self.assertEqual(next_session.table_size, 5)
        self.assertEqual(len(next_session.players), 5)
        self.assertNotIn(busted_agent_id, [player.agent.id for player in next_session.players])
        self.assertLessEqual(short_stack_survivor.stack_bb, 7)
        self.assertEqual(next_session.street, "preflop")
        self.assertFalse(next_session.is_session_complete)

    def test_next_hand_marks_terminal_session_when_only_one_agent_has_chips(self) -> None:
        session = create_battle_session(
            BattleSessionCreate(
                table_size=2,
                observer_seat=0,
                starting_stack_bb=20,
                seed="terminal-table",
            )
        )
        session.players[0].stack_bb = 0
        session.players[1].stack_bb = 40
        session.street = "complete"

        same_session = start_next_hand(session.id)
        snapshot = same_session.snapshot(observer_seat=0)

        self.assertTrue(same_session.is_session_complete)
        self.assertEqual(same_session.street, "complete")
        self.assertTrue(snapshot.is_complete)
        self.assertTrue(snapshot.is_session_complete)


if __name__ == "__main__":
    unittest.main()
