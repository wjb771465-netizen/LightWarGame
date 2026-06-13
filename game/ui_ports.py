from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from game.datatypes.command import Command
from game.datatypes.game_obs import Observation
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

    def wait_after_welcome(self) -> None:
        """欢迎信息展示完毕后，等待玩家确认再进入对局。"""
        ...

    def show_turn_start(self, state: GameState) -> None:
        """本回合开始（如第几回合）。"""
        ...

    def show_state(self, state: GameState) -> None:
        """全图版图信息（上帝视角，与当前操作玩家无关）。"""
        ...

    def show_observation(self, obs: Observation) -> None:
        """玩家观测摘要（短小）；中立格不展示。"""
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

    def wait_after_welcome(self) -> None:
        pass

    def show_turn_start(self, state: GameState) -> None:
        pass

    def show_state(self, state: GameState) -> None:
        pass

    def show_observation(self, obs: Observation) -> None:
        pass

    def show_turn_results(self, state: GameState) -> None:
        pass

    def show_game_result(self, state: GameState) -> None:
        pass

    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        return []
