"""测试公共工具。

使用方式：
    from tests.helpers import map_with_regions, PlaceholderGameUi

    def test_something(self) -> None:
        a = Region("name", [adjacent_ids], base_growth)
        ...
"""

from pathlib import Path
from typing import List

from game.datatypes.command import Command
from game.datatypes.game_map import GameMap
from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState
from game.ui_ports import GameUiPort


def map_with_regions(regions):
    """构造 GameMap，绕过配置加载直接注入 regions 列表（index 0 为 None，1-indexed）。"""
    m = GameMap.__new__(GameMap)
    m.regions = regions
    m.capitals = []
    return m


class PlaceholderGameUi(GameUiPort):
    """占位 UI：对局期 show_* 全 no-op，collect_commands 返回空列表。测试专用。"""

    def ask_launch(self) -> tuple[Path, bool]:
        raise NotImplementedError

    def show_game_start(self, state: GameState) -> None:
        pass

    def wait_after_welcome(self) -> None:
        pass

    def show_turn_start(self, state: GameState, map_path: Path) -> None:
        pass

    def show_state(self, state: GameState) -> None:
        pass

    def show_observation(self, obs: Observation) -> None:
        pass

    def show_game_result(self, state: GameState) -> None:
        pass

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        return []
