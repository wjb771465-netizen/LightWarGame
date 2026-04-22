import io
import os
import tempfile
import unittest

from game.datatypes.game_map import GameMap, Region
from game.datatypes.state import GameState
from game.ui import display
from game.ui import input_handler
from game.ui.terminal_ui import TerminalGameUi

from tests.helpers import map_with_regions as _map_with_regions


class TestDisplay(unittest.TestCase):
    def test_show_turn_start_writes_turn(self) -> None:
        buf = io.StringIO()
        a = Region("a", [], 4)
        a.owner = 1
        m = _map_with_regions([None, a])
        s = GameState(m, num_players=2, turn=7)
        display.show_turn_start(s, out=buf)
        self.assertIn("7", buf.getvalue())

    def test_show_game_result_winner(self) -> None:
        buf = io.StringIO()
        a = Region("a", [], 4)
        a.owner = 2
        m = _map_with_regions([None, a])
        s = GameState(m, num_players=2)
        s.settle()
        display.show_game_result(s, out=buf)
        self.assertIn("2", buf.getvalue())
        self.assertIn("获胜", buf.getvalue())


class TestWaitPressToStart(unittest.TestCase):
    def test_prints_prompt_and_reads_once(self) -> None:
        buf = io.StringIO()
        calls: list[str] = []

        def fake_input(prompt: str) -> str:
            calls.append(prompt)
            return ""

        input_handler.wait_press_to_start(input_fn=fake_input, out=buf)
        self.assertIn("回车", buf.getvalue())
        self.assertEqual(calls, [""])


class TestInputHandler(unittest.TestCase):
    def test_collect_one_valid_command(self) -> None:
        a = Region("a", [2], 10)
        a.owner = 1
        a.troops = 5
        b = Region("b", [1], 4)
        b.owner = 2
        b.troops = 1
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        lines = iter(["1,2,3", ""])
        cmds = input_handler.collect_commands_for_player(
            s, 1, input_fn=lambda _: next(lines)
        )
        self.assertEqual(len(cmds), 1)
        c0 = cmds[0]
        self.assertEqual((c0.source, c0.target, c0.troops, c0.player), (1, 2, 3, 1))

    def test_no_land_returns_empty(self) -> None:
        a = Region("a", [], 4)
        a.owner = 2
        m = _map_with_regions([None, a])
        s = GameState(m, num_players=2)
        cmds = input_handler.collect_commands_for_player(s, 1, input_fn=lambda _: "")
        self.assertEqual(cmds, [])


class TestTerminalGameUi(unittest.TestCase):
    def test_implements_collect_via_runner_shape(self) -> None:
        ui = TerminalGameUi(out=io.StringIO(), input_fn=lambda _: "")
        a = Region("a", [2], 4)
        a.owner = 1
        a.troops = 5
        b = Region("b", [1], 4)
        b.owner = 2
        b.troops = 5
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        ui.show_game_start(s)
        self.assertEqual(ui.collect_commands(s, 1), [])


class TestMapRenderer(unittest.TestCase):
    def test_name_normalize(self) -> None:
        from game.ui.map_renderer import _normalize_name
        cases = [
            ("内蒙古自治区", "内蒙古"),
            ("广西壮族自治区", "广西"),
            ("新疆维吾尔自治区", "新疆"),
            ("宁夏回族自治区", "宁夏"),
            ("北京市", "北京"),
            ("四川省", "四川"),
            ("西藏自治区", "西藏"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_name(raw), expected)

    def test_render_creates_file(self) -> None:
        from game.ui.map_renderer import render_map
        from init_game import fixed_capitals
        state = fixed_capitals([1, 2])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_map.png")
            render_map(state, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
