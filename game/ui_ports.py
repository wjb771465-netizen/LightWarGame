from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from game.datatypes.command import Command
from game.datatypes.state import GameState


@runtime_checkable
class GameUiPort(Protocol):
    """
    游戏 UI 端口：输出 `show_*`，输入 `collect_commands`。
    Runner 按约定顺序调用；终端 / Web 等各自实现本协议。
    """

    def show_game_start(self, state: GameState) -> None:
        """欢迎与简要规则。"""
        ...

    def show_turn_start(self, state: GameState) -> None:
        """本回合开始（如第几回合）。"""
        ...

    def show_state(self, state: GameState, player_id: int) -> None:
        """版图信息（当前可全量，后续可改为迷雾）。"""
        ...

    def show_turn_results(self, state: GameState) -> None:
        """本回合结算战报（可先空实现）。"""
        ...

    def show_game_result(self, state: GameState) -> None:
        """终局与胜利者等。"""
        ...

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        """提示并收集该玩家本回合指令。"""
        ...


class PlaceholderGameUi:
    """占位：`show_*` 为空，`collect_commands` 返回空列表。"""

    def show_game_start(self, state: GameState) -> None:
        pass

    def show_turn_start(self, state: GameState) -> None:
        pass

    def show_state(self, state: GameState, player_id: int) -> None:
        pass

    def show_turn_results(self, state: GameState) -> None:
        pass

    def show_game_result(self, state: GameState) -> None:
        pass

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        return []
