from __future__ import annotations

from typing import List, Optional

from game.datatypes.command import Command, CommandResult
from game.datatypes.game_map import GameMap
from game.datatypes.game_obs import Observation, build_observation


class GameState:
    """游戏状态"""

    def __init__(
        self,
        game_map: GameMap,
        num_players: int,
        turn: int = 1,
    ) -> None:
        self.game_map = game_map
        self.num_players = num_players
        self.turn = turn
        self.active_players = list(range(1, num_players + 1))

    def settle(self) -> bool:
        """
        扫图更新 active_players。
        若已终局（至多一名仍有地）返回 True 且不改 turn；
        否则 turn += 1 并返回 False。
        """
        seen: set[int] = set()
        regions = self.game_map.regions
        for i in range(1, len(regions)):
            r = regions[i]
            if r is None:
                continue
            o = r.owner
            if 1 <= o <= self.num_players:
                seen.add(o)
        self.active_players = sorted(seen)
        if len(self.active_players) <= 1:
            return True
        self.turn += 1
        return False

    def get_observation(self, viewer_id: int) -> Observation:
        return build_observation(self.game_map, self.turn, viewer_id)

    def winner(self) -> Optional[int]:
        """地图上仅剩一名配置内玩家有地时返回其 id，否则 None。"""
        ap = self.active_players
        if len(ap) == 1:
            return ap[0]
        return None

    def is_command_valid(self, cmd: Command) -> bool:
        m = self.game_map
        if not m.valid_id(cmd.source) or not m.valid_id(cmd.target):
            return False
        src = m.get(cmd.source)
        assert src is not None
        if src.owner != cmd.player:
            return False
        if cmd.troops >= src.troops:
            return False
        if not m.are_adjacent(cmd.source, cmd.target):
            return False
        return True

    def resolve_turn(self, commands: List[Command]) -> List[CommandResult]:
        """按顺序执行指令并执行版图兵力增长（不推进回合，回合由 settle 处理）。"""
        results: List[CommandResult] = []
        for cmd in commands:
            if not self.is_command_valid(cmd):
                res = CommandResult(cmd, False)
                res.reason = "兵力不足或地区已变"
                results.append(res)
                continue
            self.game_map.move_troops(cmd.source, cmd.target, cmd.troops, cmd.player)
            results.append(CommandResult(cmd, True))
        self.game_map.troop_growth()
        return results
