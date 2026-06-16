from __future__ import annotations

from typing import Dict, List, Optional, Set

from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from .base_opponent import BaseOpponent


class RuleOpponent(BaseOpponent):
    """规则对手。

    信息边界：
      Observation：己方全量（兵力/首都/增长），敌方/中立仅 owner
      GameMap：仅用于邻接关系和 capitals 列表（地图公开知识）
      不访问敌方/中立地区的 troops / is_capital / base_growth

    每回合行动逻辑：
      1. 攻击指令（ceil(quota/2) 条）
         - 兵源：邻敌方地区优先，其次邻中立；并列取兵最多者
         - 目标：优先敌方首都，其次敌方，其次中立
      2. 调兵指令（floor(quota/2) 条）
         - 兵源：己方邻居数最少（最孤立）的地区，并列取兵最多者
         - 目标：兵源邻居中己方兵最多的地区
         - 兵力：floor(troops / 2)
      同一地区不同时作为攻击源和调兵源；攻击优先。
    """

    def act(self, state: GameState) -> List[Command]:
        obs = state.get_observation(self.player_id)
        gm = state.game_map

        my: Dict[int, dict] = {}
        enemy_ids: Set[int] = set()
        neutral_ids: Set[int] = set()

        for r_obs in obs.regions[1:]:
            if r_obs is None:
                continue
            if r_obs.owner == self.player_id:
                my[r_obs.region_id] = {
                    "troops": r_obs.troops,
                    "is_capital": r_obs.is_capital,
                }
            elif r_obs.owner == 0:
                neutral_ids.add(r_obs.region_id)
            else:
                enemy_ids.add(r_obs.region_id)

        max_cmds = max_commands(len(my))
        attack_quota = (max_cmds + 1) // 2
        move_quota = max_cmds // 2

        attacks = self._attacks(gm, my, enemy_ids, neutral_ids, attack_quota)
        used: Set[int] = {cmd.source for cmd in attacks}
        moves = self._moves(gm, my, used, move_quota)
        return attacks + moves

    # ------------------------------------------------------------------
    def _attacks(
        self,
        gm: GameMap,
        my: Dict[int, dict],
        enemy_ids: Set[int],
        neutral_ids: Set[int],
        quota: int,
    ) -> List[Command]:
        enemy_capitals = set(gm.capitals) & enemy_ids

        adj_enemy: List[tuple] = []   # (rid, troops, has_enemy_capital)
        adj_neutral: List[tuple] = []  # (rid, troops)

        for rid, info in my.items():
            if info["troops"] <= 1:
                continue
            region = gm.regions[rid]
            nbrs = [n for n in region.adjacent if gm.regions[n] is not None]
            if any(n in enemy_ids for n in nbrs):
                has_cap = any(n in enemy_capitals for n in nbrs)
                adj_enemy.append((rid, info["troops"], has_cap))
            elif any(n in neutral_ids for n in nbrs):
                adj_neutral.append((rid, info["troops"]))

        # 邻敌优先（敌首都 > 普通敌），其次邻中立；并列取兵多者
        adj_enemy.sort(key=lambda x: (-x[2], -x[1]))
        adj_neutral.sort(key=lambda x: -x[1])
        candidates = adj_enemy + adj_neutral

        cmds: List[Command] = []
        for rid, troops, *_ in candidates:
            if len(cmds) >= quota:
                break
            target = self._pick_attack_target(
                gm, rid, my, enemy_ids, neutral_ids, enemy_capitals,
            )
            if target is None:
                continue
            cmds.append(Command(rid, target, troops - 1, self.player_id))

        return cmds

    def _pick_attack_target(
        self,
        gm: GameMap,
        src: int,
        my: Dict[int, dict],
        enemy_ids: Set[int],
        neutral_ids: Set[int],
        enemy_capitals: Set[int],
    ) -> Optional[int]:
        region = gm.regions[src]
        nbrs = [n for n in region.adjacent if gm.regions[n] is not None]

        # 优先级：敌方首都 > 普通敌方 > 中立；同优先级取 rid 最小
        cap_targets = [n for n in nbrs if n in enemy_capitals]
        enemy_targets = [n for n in nbrs if n in enemy_ids and n not in enemy_capitals]
        neutral_targets = [n for n in nbrs if n in neutral_ids]

        for targets in (cap_targets, enemy_targets, neutral_targets):
            if targets:
                return min(targets)
        return None

    # ------------------------------------------------------------------
    def _moves(
        self,
        gm: GameMap,
        my: Dict[int, dict],
        excluded: Set[int],
        quota: int,
    ) -> List[Command]:
        if quota == 0:
            return []

        # (rid, troops, own_neighbor_count, own_nbrs)
        candidates: List[tuple] = []
        for rid, info in my.items():
            if info["troops"] <= 1 or rid in excluded:
                continue
            region = gm.regions[rid]
            own_nbrs = [
                n for n in region.adjacent
                if gm.regions[n] is not None and n in my
            ]
            if not own_nbrs:
                continue
            candidates.append((rid, info["troops"], len(own_nbrs), own_nbrs))

        # 邻己方数最少优先，并列取兵最多者
        candidates.sort(key=lambda x: (x[2], -x[1]))

        cmds: List[Command] = []
        for rid, troops, _, own_nbrs in candidates:
            if len(cmds) >= quota:
                break
            target = max(own_nbrs, key=lambda n: my[n]["troops"])
            amount = min(troops // 2, troops - 1)
            cmds.append(Command(rid, target, amount, self.player_id))

        return cmds
