from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from game.datatypes.command import Command
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
        若已终局（至多一名仍有地）返回 True ；
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
        self.game_map.troop_growth()
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

    def check_cmds(self, commands: List[Command]) -> List[Command]:
        """在不动地图的前提下筛出可执行指令（单条合法 + 同源派出总和 ≤ 该源兵力−1），顺序与入参一致。"""
        m = self.game_map
        valid_cmds = [cmd for cmd in commands if self.is_command_valid(cmd)]
        sum_by_src: Dict[int, int] = defaultdict(int)
        for cmd in valid_cmds:
            sum_by_src[cmd.source] += cmd.troops

        result: List[Command] = []
        for cmd in valid_cmds:
            src = m.get(cmd.source)
            assert src is not None
            if sum_by_src[cmd.source] <= src.troops - 1:
                result.append(cmd)
        return result

    def apply_cmds(self, commands: List[Command]) -> None:
        """
        对已通过 check_cmds 的指令同时结算：每条指令聚到达（围困折半）并扣源 → Region.battle。
        """
        m = self.game_map
        incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for cmd in commands:
            t = cmd.troops
            if m.is_surrounded(cmd.source):
                t = math.floor(t * 0.5)
            incoming[cmd.target][cmd.player] += t
            m.regions[cmd.source].troops -= cmd.troops

        for dst, by_player in incoming.items():
            total_in = sum(by_player.values())
            if total_in <= 0:
                continue
            m.regions[dst].battle({p: c for p, c in by_player.items() if c > 0})
