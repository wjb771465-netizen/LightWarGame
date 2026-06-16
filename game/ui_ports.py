from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from game.campaign.chat import ChatRoom
from game.datatypes.command import Command
from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState


class GameUiPort(ABC):
    """
    游戏 UI 端口：ask_launch 处理启动期，show_* 输出对局信息，collect_commands 收集指令。
    Runner 按约定顺序调用；终端 / Web 等各自继承实现。
    """

    @abstractmethod
    def ask_launch(self) -> tuple[Path, bool]:
        """处理全部启动交互，返回 (session_dir, is_new)。"""

    @abstractmethod
    def show_game_start(self, state: GameState) -> None:
        """欢迎与简要规则。"""

    @abstractmethod
    def wait_after_welcome(self) -> None:
        """欢迎信息展示完毕后，等待玩家确认再进入对局。"""

    @abstractmethod
    def show_turn_start(self, state: GameState, map_path: Path) -> None:
        """本回合开始（如第几回合）。"""

    @abstractmethod
    def show_state(self, state: GameState) -> None:
        """全图版图信息（上帝视角，与当前操作玩家无关）。"""

    @abstractmethod
    def show_observation(self, obs: Observation) -> None:
        """玩家观测摘要（短小）；中立格不展示。"""

    @abstractmethod
    def show_game_result(self, state: GameState) -> None:
        """终局与胜利者等。"""

    @abstractmethod
    def collect_commands(self, state: GameState, player_id: int) -> List[Command]:
        """提示并收集该玩家本回合指令。"""

    def show_turn_results(self, state: GameState,
                          battle_report: list[tuple[int, int, int]]) -> None:
        """本回合结算战报；默认 no-op，子类按需覆盖。"""

    def run_diplomacy(self, state: GameState, chat_room: ChatRoom,
                      save_path: Optional[Path] = None,
                      battle_report: list[tuple[int, int, int]] | None = None) -> None:
        """外交阶段；默认 no-op，AI UI 覆盖。"""
