"""Unit tests for WebGameUi — no Flask, no threads, pure logic."""
import queue
import threading
import unittest
from pathlib import Path

from game.campaign.chat import ChatRoom
from game.datatypes.command import Command
from game.datatypes.game_map import Region
from game.datatypes.game_obs import Observation, RegionObservation
from game.datatypes.state import GameState
from web.web_game_ui import WebGameUi

from tests.helpers import map_with_regions


def _make_obs(viewer_id, turn, regions):
    """Shorthand: build Observation from a list of (region_id, owner, troops, is_capital, base_growth)."""
    regs = [None]
    for rid, owner, troops, cap, growth in regions:
        regs.append(RegionObservation(rid, owner, troops, cap, growth))
    return Observation(viewer_id, turn, tuple(regs))


class TestWebGameUiLaunch(unittest.TestCase):
    """Event-based session launch."""

    def test_ask_launch_blocks_until_prepare(self):
        ui = WebGameUi()
        ui.prepare_launch("/tmp/foo", True)
        session_dir, is_new = ui.ask_launch()
        self.assertEqual(str(session_dir), "/tmp/foo")
        self.assertTrue(is_new)

    def test_prepare_launch_resets_state(self):
        ui = WebGameUi()
        ui._phase = "playing"
        ui._observation = {"x": 1}
        ui._battle_changes = ["a"]
        ui._winner = 1
        ui._pending_commands = [Command(1, 2, 5, 1)]
        ui.prepare_launch("/tmp/bar", False)
        self.assertEqual(ui._phase, "lobby")
        self.assertIsNone(ui._observation)
        self.assertEqual(ui._battle_changes, [])
        self.assertIsNone(ui._winner)
        self.assertEqual(ui._pending_commands, [])


class TestWebGameUiPhaseTransitions(unittest.TestCase):
    """show_* methods set correct phase."""

    def setUp(self):
        self.ui = WebGameUi()
        self.a = Region("a", [2], 4)
        self.a.owner = 1
        self.a.troops = 80
        self.b = Region("b", [1], 4)
        self.b.owner = 2
        self.b.troops = 80
        self.m = map_with_regions([None, self.a, self.b])
        self.state = GameState(self.m, num_players=2)

    def test_show_game_start_sets_playing(self):
        self.ui.prepare_launch("/tmp/s", True)
        self.ui.ask_launch()
        self.ui.show_game_start(self.state)
        self.assertEqual(self.ui._phase, "playing")
        self.assertEqual(self.ui._num_players, 2)
        self.assertEqual(self.ui._turn, 1)

    def test_show_turn_results_sets_result(self):
        self.ui._game_map = self.m
        self.ui.show_turn_results(self.state, [])
        self.assertEqual(self.ui._phase, "result")

    def test_show_game_result_sets_over(self):
        self.ui._game_map = self.m
        self.ui.show_game_result(self.state)
        self.assertEqual(self.ui._phase, "over")
        self.assertIsNone(self.ui._winner)  # 0 is falsy, but None means no winner found


class TestWebGameUiObservation(unittest.TestCase):
    """Observation serialization and fog-of-war."""

    def setUp(self):
        self.ui = WebGameUi()
        self.a = Region("a", [2], 4)
        self.a.owner = 1
        self.a.is_capital = True
        self.b = Region("b", [1], 5)
        self.b.owner = 2
        self.m = map_with_regions([None, self.a, self.b])
        self.state = GameState(self.m, num_players=2)
        self.ui.prepare_launch("/tmp/s", True)
        self.ui.ask_launch()
        self.ui.show_game_start(self.state)

    def test_own_regions_have_full_info(self):
        """己方完整信息 + 敌方仅归属，中立跳过。"""
        obs = _make_obs(1, 1, [(1, 1, 80, True, 4),
                                (2, 2, None, None, None),
                                (3, 0, None, None, None)])
        self.ui._game_map = map_with_regions([
            None,
            Region("a", [2], 1),
            Region("b", [1, 3], 1),
            Region("c", [2], 1),
        ])
        self.ui.show_observation(obs)
        d = self.ui._observation
        self.assertEqual(len(d["regions"]), 2)  # 敌我都有，中立跳过

    def test_enemy_regions_show_owner_only(self):
        obs = _make_obs(1, 1, [(1, 1, 80, True, 4), (2, 2, None, None, None)])
        self.ui._game_map = self.m
        self.ui.show_observation(obs)
        d = self.ui._observation
        # Both regions should appear (owner != 0)
        self.assertEqual(len(d["regions"]), 2)
        own = [r for r in d["regions"] if r["owner"] == 1][0]
        self.assertEqual(own["troops"], 80)
        self.assertTrue(own["is_capital"])
        enemy = [r for r in d["regions"] if r["owner"] == 2][0]
        self.assertIsNone(enemy["troops"])
        self.assertIsNone(enemy["is_capital"])

    def test_neutral_regions_skipped(self):
        obs = _make_obs(1, 1, [(1, 1, 80, True, 4), (2, 0, None, None, None)])
        self.ui._game_map = self.m
        self.ui.show_observation(obs)
        d = self.ui._observation
        self.assertEqual(len(d["regions"]), 1)

    def test_obs_seq_increments(self):
        obs = _make_obs(1, 1, [(1, 1, 80, True, 4)])
        self.ui._game_map = self.m
        s0 = self.ui._obs_seq
        self.ui.show_observation(obs)
        self.assertEqual(self.ui._obs_seq, s0 + 1)


class TestWebGameUiCommands(unittest.TestCase):
    """Command accumulation, submission, and Queue-based blocking."""

    def setUp(self):
        self.ui = WebGameUi()
        self.ui.prepare_launch("/tmp/s", True)
        self.ui.ask_launch()

    def test_add_pending_appears_in_snapshot(self):
        cmd = Command(1, 2, 10, 1)
        self.ui._game_map = map_with_regions([
            None,
            Region("a", [2], 4),
            Region("b", [1], 4),
        ])
        regions = [{"id": 1, "name": "a", "owner": 1, "troops": 50, "adjacent": [2], "is_capital": False}]
        self.ui._current_player = 1
        err = self.ui.add_pending_command(cmd, regions)
        self.assertIsNone(err)
        snap = self.ui.snapshot()
        self.assertEqual(len(snap["pending_commands"]), 1)
        self.assertIn("a", snap["pending_commands"][0]["source_name"])

    def test_add_pending_respects_command_limit(self):
        """max_commands(1 owned) = 1, so second command should be rejected."""
        self.ui._game_map = map_with_regions([
            None,
            Region("a", [2], 4),
            Region("b", [1], 4),
        ])
        regions = [{"id": 1, "name": "a", "owner": 1, "troops": 50, "adjacent": [2], "is_capital": False}]
        self.ui._current_player = 1
        err1 = self.ui.add_pending_command(Command(1, 2, 10, 1), regions)
        self.assertIsNone(err1)
        err2 = self.ui.add_pending_command(Command(1, 2, 5, 1), regions)
        self.assertIn("已达指令上限", err2)
        self.assertEqual(len(self.ui._pending_commands), 1)

    def test_add_pending_limit_scales_with_owned_regions(self):
        """4 owned regions → max_commands(4) = ceil(4/3) = 2."""
        self.ui._game_map = map_with_regions([
            None,
            Region("a", [2, 3], 4),
            Region("b", [1], 4),
            Region("c", [1], 4),
            Region("d", [1], 4),
        ])
        regions = [
            {"id": 1, "name": "a", "owner": 1, "troops": 50, "adjacent": [2, 3], "is_capital": False},
            {"id": 2, "name": "b", "owner": 1, "troops": 10, "adjacent": [1], "is_capital": False},
            {"id": 3, "name": "c", "owner": 1, "troops": 10, "adjacent": [1], "is_capital": False},
            {"id": 4, "name": "d", "owner": 1, "troops": 10, "adjacent": [1], "is_capital": False},
        ]
        self.ui._current_player = 1
        err1 = self.ui.add_pending_command(Command(1, 2, 5, 1), regions)
        self.assertIsNone(err1)
        err2 = self.ui.add_pending_command(Command(1, 3, 5, 1), regions)
        self.assertIsNone(err2)
        err3 = self.ui.add_pending_command(Command(2, 1, 5, 1), regions)
        self.assertIn("已达指令上限", err3)
        self.assertEqual(len(self.ui._pending_commands), 2)

    def test_submit_puts_into_queue(self):
        cmds = [Command(1, 2, 5, 1)]
        self.ui._pending_commands = cmds
        self.ui.submit_commands()
        result = self.ui._cmd_queue.get(timeout=0.5)
        self.assertEqual(result, cmds)
        self.assertEqual(self.ui._pending_commands, [])

    def test_collect_commands_blocks_then_returns(self):
        expected = [Command(1, 2, 3, 1)]

        def _put():
            import time
            time.sleep(0.05)
            self.ui._cmd_queue.put(expected)

        t = threading.Thread(target=_put)
        t.start()
        result = self.ui.collect_commands(None, 1)
        t.join()
        self.assertEqual([(c.source, c.target, c.troops) for c in result],
                         [(1, 2, 3)])


class TestWebGameUiAI(unittest.TestCase):
    """AI player detection and delegation."""

    def setUp(self):
        self.ui = WebGameUi()
        self.a = Region("a", [2], 4)
        self.a.owner = 1
        self.a.troops = 80
        self.b = Region("b", [1], 4)
        self.b.owner = 2
        self.b.troops = 80
        self.m = map_with_regions([None, self.a, self.b])
        self.state = GameState(self.m, num_players=2)

    def test_show_observation_skipped_for_ai_player(self):
        self.ui._ai_cfg = {2: {"type": "random"}}
        self.ui._game_map = self.m
        obs = _make_obs(2, 1, [(2, 2, 80, True, 4)])
        self.ui.show_observation(obs)
        self.assertIsNone(self.ui._observation)

    def test_collect_commands_delegates_to_ai(self):
        class MockOpponent:
            def act(self, state):
                return [Command(2, 1, 10, 2)]
        self.ui._opponents = {2: MockOpponent()}
        cmds = self.ui.collect_commands(self.state, 2)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].player, 2)

    def test_human_collect_blocks_even_when_ai_configured(self):
        """AI cfg set for player 2, but player 1 is human — should block on queue."""
        self.ui._ai_cfg = {2: {"type": "random"}}

        def _put():
            import time
            time.sleep(0.05)
            self.ui._cmd_queue.put([Command(1, 2, 3, 1)])

        t = threading.Thread(target=_put)
        t.start()
        result = self.ui.collect_commands(self.state, 1)
        t.join()
        self.assertEqual(result[0].player, 1)


class TestWebGameUiSnapshot(unittest.TestCase):
    """Snapshot thread-safety and completeness."""

    def setUp(self):
        self.ui = WebGameUi()
        self.ui.prepare_launch("/tmp/s", True)
        self.ui.ask_launch()
        self.a = Region("a", [2], 4)
        self.a.owner = 1
        self.a.troops = 80
        self.b = Region("b", [1], 4)
        self.b.owner = 2
        self.b.troops = 80
        self.m = map_with_regions([None, self.a, self.b])
        self.state = GameState(self.m, num_players=2)
        self.ui.show_game_start(self.state)

    def test_snapshot_concurrent_reads_do_not_crash(self):
        errors = []

        def read():
            try:
                for _ in range(100):
                    self.ui.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_snapshot_includes_all_keys(self):
        snap = self.ui.snapshot()
        for k in ("phase", "turn", "num_players", "current_player",
                   "observation", "battle_changes", "winner",
                   "pending_commands", "obs_seq", "error", "cmd_error",
                   "map_path", "session_name", "chat_messages"):
            self.assertIn(k, snap)


class TestWebGameUiDiplomacy(unittest.TestCase):
    """run_diplomacy + add_chat_message + snapshot chat_messages。"""

    def setUp(self):
        self.a = Region("a", [2], 4)
        self.a.owner = 1
        self.a.troops = 10
        self.b = Region("b", [1], 4)
        self.b.owner = 2
        self.b.troops = 10

    def test_run_diplomacy_stores_chat_room(self):
        ui = WebGameUi()
        ui.prepare_launch("/tmp/diptest", True)
        m = map_with_regions([None, self.a, self.b])
        state = GameState(m, num_players=2)
        chat = ChatRoom()
        ui.run_diplomacy(state, chat)
        snap = ui.snapshot()
        self.assertEqual(snap["chat_messages"], [])
        self.assertEqual(snap["session_name"], "diptest")

    def test_add_chat_message(self):
        ui = WebGameUi()
        ui.prepare_launch("/tmp/diptest2", True)
        m = map_with_regions([None, self.a, self.b])
        state = GameState(m, num_players=2)
        chat = ChatRoom()
        ui.run_diplomacy(state, chat)
        ui.add_chat_message(1, "Hello diplomacy")
        snap = ui.snapshot()
        self.assertEqual(len(snap["chat_messages"]), 1)
        self.assertEqual(snap["chat_messages"][0]["text"], "Hello diplomacy")
        self.assertEqual(snap["chat_messages"][0]["sender_id"], 1)

    def test_add_chat_message_no_chat_room_is_noop(self):
        ui = WebGameUi()
        ui.prepare_launch("/tmp/diptest3", True)
        ui.add_chat_message(1, "should be ignored")
        snap = ui.snapshot()
        self.assertEqual(snap["chat_messages"], [])

    def test_snapshot_includes_map_path(self):
        ui = WebGameUi()
        ui.prepare_launch("/tmp/diptest4", True)
        m = map_with_regions([None, self.a, self.b])
        state = GameState(m, num_players=2)
        ui.show_turn_start(state, Path("/tmp/maps/map_turn_001.png"))
        snap = ui.snapshot()
        self.assertEqual(snap["map_path"], "/tmp/maps/map_turn_001.png")

    def test_submit_commands_prevents_double_commit(self):
        """Bug 1: submit 后 phase->waiting + obs 清空，/wait 不会再跳回 /play。"""
        ui = WebGameUi()
        ui.prepare_launch("/tmp/bug1", True)
        m = map_with_regions([None, self.a, self.b])
        state = GameState(m, num_players=2)
        ui.show_game_start(state)
        obs = state.get_observation(1)
        ui.show_observation(obs)
        ui.submit_commands()
        snap = ui.snapshot()
        self.assertEqual(snap["phase"], "waiting")
        self.assertIsNone(snap["observation"])

    def test_add_chat_message_saves_to_disk(self):
        """Bug 2: 人类发言后应持久化到 disk，防止下回合被 load 覆盖。"""
        import tempfile
        ui = WebGameUi()
        ui.prepare_launch("/tmp/bug2", True)
        m = map_with_regions([None, self.a, self.b])
        state = GameState(m, num_players=2)
        chat = ChatRoom()
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "chat.json"
            # 模拟 Runner 传 Path 对象（非 str）
            ui.run_diplomacy(state, chat, save_path=save_path)
            ui.add_chat_message(1, "人类发言")
            self.assertTrue(save_path.exists())
            # 从磁盘重新加载，验证人类发言已持久化
            chat2 = ChatRoom()
            chat2.load(str(save_path))
            msgs = chat2.get_history()
            self.assertEqual(len(msgs), 1)
            self.assertEqual(msgs[0].text, "人类发言")
