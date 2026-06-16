import unittest
from unittest.mock import MagicMock, patch

from game.datatypes.command import Command
from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from game.ui.terminal_ui import TerminalGameUi
from tests.helpers import map_with_regions


def _make_state():
    a = Region("A", [2], 4); a.owner = 1; a.troops = 10; a.is_capital = True
    b = Region("B", [1], 4); b.owner = 2; b.troops = 10; b.is_capital = True
    return GameState(map_with_regions([None, a, b]), num_players=2)


def _make_ui(opponents):
    ui = TerminalGameUi()
    ui._opponents = opponents
    ui._ai_cfg = {pid: {} for pid in opponents}
    return ui


class TestAIGameUi(unittest.TestCase):

    def test_ai_returns_command(self):
        """AI 玩家：opponent.act 返回 Command 列表 → collect_commands 直接返回。"""
        state = _make_state()
        expected_cmd = Command(source=2, target=1, troops=5, player=2)

        opponent = MagicMock()
        opponent.act.return_value = [expected_cmd]
        ui = _make_ui({2: opponent})

        result = ui.collect_commands(state, player_id=2)

        self.assertEqual(result, [expected_cmd])
        opponent.act.assert_called_once_with(state)

    def test_human_player_delegates_terminal(self):
        """非 AI 玩家：collect_commands 委托给 input_handler，不触发 opponent。"""
        state = _make_state()
        opponent = MagicMock()
        ui = _make_ui({2: opponent})

        with patch("game.ui.input_handler.collect_commands_for_player", return_value=[]) as mock:
            result = ui.collect_commands(state, player_id=1)
            mock.assert_called_once_with(state, 1, None)

        opponent.act.assert_not_called()
        self.assertEqual(result, [])

    def test_opponent_returns_empty(self):
        """opponent.act 返回 [] → collect_commands 返回 []。"""
        state = _make_state()
        opponent = MagicMock()
        opponent.act.return_value = []
        ui = _make_ui({2: opponent})

        result = ui.collect_commands(state, player_id=2)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
