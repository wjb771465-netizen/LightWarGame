"""
game/ui/ai_game_ui.py 的单元测试。

用 mock 替代真实模型，验证 collect_commands 的路由逻辑。
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from game.datatypes.command import Command
from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from game.ui.ai_game_ui import AIGameUi
from game.ui.terminal_ui import TerminalGameUi
from tests.helpers import map_with_regions


def _make_state():
    # 1(p1,10) ↔ 2(p2,10)，最小 2-player 场景
    a = Region("A", [2], 4); a.owner = 1; a.troops = 10; a.is_capital = True
    b = Region("B", [1], 4); b.owner = 2; b.troops = 10; b.is_capital = True
    return GameState(map_with_regions([None, a, b]), num_players=2)


def _make_ui(policies):
    obs_enc = MagicMock()
    obs_enc.encode.return_value = np.zeros(10, dtype=np.float32)
    act_enc = MagicMock()
    act_enc.mask.return_value = np.ones(5, dtype=bool)
    return AIGameUi(policies, obs_enc, act_enc), obs_enc, act_enc


class TestAIGameUi(unittest.TestCase):

    def test_ai_returns_command(self):
        """AI 玩家：policy 返回有效 action → collect_commands 返回非空 Command 列表。"""
        state = _make_state()
        expected_cmd = Command(source=2, target=1, troops=5, player=2)

        policy = MagicMock()
        policy.predict.return_value = 3
        ui, _, act_enc = _make_ui({2: policy})
        act_enc.decode.return_value = expected_cmd

        result = ui.collect_commands(state, player_id=2)

        self.assertEqual(result, [expected_cmd])
        policy.predict.assert_called_once()

    def test_human_player_delegates_terminal(self):
        """非 AI 玩家：collect_commands 委托给 TerminalGameUi。"""
        state = _make_state()
        ui, _, _ = _make_ui({2: MagicMock()})  # 只有 player 2 是 AI

        with patch.object(TerminalGameUi, "collect_commands", return_value=[]) as mock_terminal:
            result = ui.collect_commands(state, player_id=1)
            mock_terminal.assert_called_once_with(state, 1)

        self.assertEqual(result, [])

    def test_noop_action_returns_empty(self):
        """policy 返回 0（no-op）→ decode 返回 None → collect_commands 返回 []。"""
        state = _make_state()
        policy = MagicMock()
        policy.predict.return_value = 0
        ui, _, act_enc = _make_ui({2: policy})
        act_enc.decode.return_value = None

        result = ui.collect_commands(state, player_id=2)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
