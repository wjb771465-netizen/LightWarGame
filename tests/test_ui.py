import io
import unittest

from game.datatypes.command import Command
from game.datatypes.game_map import GameMap, Region
from game.datatypes.state import GameState
from game.ui import display
from game.ui import input_handler
from game.ui.terminal_ui import TerminalGameUi


def _map_with_regions(regions):
    m = GameMap.__new__(GameMap)
    m.regions = regions
    return m


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
        display.show_game_result(s, out=buf)
        self.assertIn("2", buf.getvalue())
        self.assertIn("获胜", buf.getvalue())


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
        self.assertEqual(cmds[0], Command(1, 2, 3, 1))

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


if __name__ == "__main__":
    unittest.main()
