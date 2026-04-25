from __future__ import annotations

from typing import List, Optional, Set

from game.datatypes.command import Command
from game.datatypes.state import GameState
from .base_opponent import BaseOpponent


class RuleOpponent(BaseOpponent):
    """规则对手。

    每回合行动逻辑：
      1. 攻击指令（ceil(quota/2) 条）
         - 兵源：邻敌方地区优先，其次邻中立；并列取兵最多者
         - 目标：优先敌方，其次中立；并列取兵最少者
      2. 调兵指令（floor(quota/2) 条）
         - 兵源：己方邻居数最少（最孤立）的地区，并列取兵最多者
         - 目标：兵源邻居中己方兵最多的地区
         - 兵力：floor(troops / 2)
      同一地区不同时作为攻击源和调兵源；攻击优先。
    """

    def act(self, state: GameState) -> List[Command]:
        regions = state.game_map.regions
        player = self.player_id
        owned = sum(1 for r in regions[1:] if r is not None and r.owner == player)
        max_cmds = max(1, owned // 3)
        attack_quota = (max_cmds + 1) // 2
        move_quota = max_cmds // 2

        attacks = self._attacks(state, attack_quota)
        used: Set[int] = {cmd.source for cmd in attacks}
        moves = self._moves(state, move_quota, used)
        return attacks + moves

    # ------------------------------------------------------------------
    def _attacks(self, state: GameState, quota: int) -> List[Command]:
        regions = state.game_map.regions
        player = self.player_id

        adj_enemy: List[tuple] = []
        adj_neutral: List[tuple] = []

        for rid, r in enumerate(regions[1:], 1):
            if r is None or r.owner != player or r.troops <= 1:
                continue
            neighbors = [regions[n] for n in r.adjacent if regions[n] is not None]
            if any(n.owner not in (0, player) for n in neighbors):
                adj_enemy.append((rid, r))
            elif any(n.owner == 0 for n in neighbors):
                adj_neutral.append((rid, r))

        # 邻敌优先，并列取兵多者
        candidates = (
            sorted(adj_enemy, key=lambda x: -x[1].troops)
            + sorted(adj_neutral, key=lambda x: -x[1].troops)
        )

        cmds: List[Command] = []
        for rid, r in candidates:
            if len(cmds) >= quota:
                break
            target = self._pick_attack_target(state, rid, player)
            if target is None:
                continue
            cmds.append(Command(rid, target, r.troops - 1, player))

        return cmds

    def _pick_attack_target(
        self, state: GameState, src: int, player: int
    ) -> Optional[int]:
        regions = state.game_map.regions
        r = regions[src]
        neighbors = [(n, regions[n]) for n in r.adjacent if regions[n] is not None]
        enemy = [(n, reg) for n, reg in neighbors if reg.owner not in (0, player)]
        neutral = [(n, reg) for n, reg in neighbors if reg.owner == 0]
        targets = enemy or neutral
        if not targets:
            return None
        return min(targets, key=lambda x: x[1].troops)[0]

    # ------------------------------------------------------------------
    def _moves(
        self, state: GameState, quota: int, excluded: Set[int]
    ) -> List[Command]:
        if quota == 0:
            return []

        regions = state.game_map.regions
        player = self.player_id

        # (rid, region, own_neighbor_count)
        candidates: List[tuple] = []
        for rid, r in enumerate(regions[1:], 1):
            if r is None or r.owner != player or r.troops <= 1 or rid in excluded:
                continue
            own_nbrs = [n for n in r.adjacent if regions[n] is not None and regions[n].owner == player]
            if not own_nbrs:
                continue
            candidates.append((rid, r, len(own_nbrs), own_nbrs))

        # 邻己方数最少优先，并列取兵最多者
        candidates.sort(key=lambda x: (x[2], -x[1].troops))

        cmds: List[Command] = []
        for rid, r, _, own_nbrs in candidates:
            if len(cmds) >= quota:
                break
            target = max(own_nbrs, key=lambda n: regions[n].troops)
            amount = min(r.troops // 2, r.troops - 1)
            cmds.append(Command(rid, target, amount, player))

        return cmds
