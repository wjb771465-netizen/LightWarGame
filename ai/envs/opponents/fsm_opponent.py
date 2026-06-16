"""状态机 AI 对手。

BaseFsmOpponent：抽象骨架 — 状态流转、成员声明
FsmOpponent：具体实现 — 感知、转移、三个状态 handler

扩展方式：
  1. 继承 BaseFsmOpponent
  2. __init__ 中 self._states 注册 name -> handler
  3. 实现 observe() / transition() / do_<state>()
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

from game.constants import max_commands
from game.datatypes.command import Command
from game.datatypes.game_obs import Observation
from game.datatypes.state import GameState
from .base_opponent import BaseOpponent


class BaseFsmOpponent(BaseOpponent, ABC):
    """状态机 AI 抽象基类。

    子类需定义 _initial_state 类属性，并实现 observe() / transition()
    和状态 handler（通过 self._states 注册）。

    生命周期：
      reset()          -> self.state = self._initial_state
      act(game_state)  -> observe() -> transition() -> execute
    """

    _initial_state: str

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.state: str = self._initial_state
        self._states: Dict[str, callable] = {}

        self.my: Dict[int, dict] = {}
        self.enemy_ids: Set[int] = set()
        self.neutral_ids: Set[int] = set()
        self.total: int = 0
        self.avg_troops: float = 0.0
        self.threatened: Dict[int, dict] = {}
        self.safe_front: Dict[int, dict] = {}
        self.rear: Dict[int, dict] = {}
        self.game_map = None

    def reset(self) -> None:
        self.state = self._initial_state

    def act(self, game_state: GameState) -> List[Command]:
        self.observe(game_state)
        self.state = self.transition()
        return self._states[self.state]()

    @abstractmethod
    def observe(self, game_state: GameState) -> None:
        ...

    @abstractmethod
    def transition(self) -> str:
        ...


class FsmOpponent(BaseFsmOpponent):
    """具体状态机对手。

    三种状态：
      EXPAND  — 无敌方相邻，全速占领中立
      ATTACK  — 有敌相邻但无受威胁前线，偏进攻
      DEFEND  — 有受威胁前线，偏防守

    信息边界：
      Observation：己方全量（兵力/首都/增长），敌方/中立仅 owner
      GameMap：仅用于邻接关系和 capitals 列表（地图公开知识）
      不访问敌方/中立地区的 troops / is_capital / base_growth
    """

    STATE_EXPAND = "expand"
    STATE_ATTACK = "attack"
    STATE_DEFEND = "defend"
    _initial_state = STATE_EXPAND

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self._states = {
            self.STATE_EXPAND: self.do_expand,
            self.STATE_ATTACK: self.do_attack,
            self.STATE_DEFEND: self.do_defend,
        }

    def observe(self, game_state: GameState) -> None:
        obs = game_state.get_observation(self.player_id)
        self.game_map = game_state.game_map

        self.my = {}
        self.enemy_ids = set()
        self.neutral_ids = set()

        for r_obs in obs.regions[1:]:
            if r_obs is None:
                continue
            if r_obs.owner == self.player_id:
                self.my[r_obs.region_id] = {
                    "troops": r_obs.troops,
                    "is_capital": r_obs.is_capital,
                    "base_growth": r_obs.base_growth,
                }
            elif r_obs.owner == 0:
                self.neutral_ids.add(r_obs.region_id)
            else:
                self.enemy_ids.add(r_obs.region_id)

        self.total = max_commands(len(self.my))
        self.avg_troops = (
            sum(r["troops"] for r in self.my.values()) / max(len(self.my), 1)
        )

        self.threatened = {}
        self.safe_front = {}
        self.rear = {}

        for rid, info in self.my.items():
            region = self.game_map.regions[rid]
            adj_enemy = any(n in self.enemy_ids for n in region.adjacent)
            if adj_enemy:
                if info["troops"] < self.avg_troops * 0.5:
                    self.threatened[rid] = info
                else:
                    self.safe_front[rid] = info
            else:
                self.rear[rid] = info

    def transition(self) -> str:
        if not self._has_adjacent_enemy():
            return self.STATE_EXPAND
        if self.threatened:
            return self.STATE_DEFEND
        return self.STATE_ATTACK

    def do_expand(self) -> List[Command]:
        """EXPAND：无敌方相邻，全速占领邻接中立。"""
        candidates: List[Tuple[int, int]] = []

        for rid, info in self.my.items():
            if info["troops"] <= 1:
                continue
            region = self.game_map.regions[rid]
            if region is not None and any(n in self.neutral_ids for n in region.adjacent):
                candidates.append((info["troops"], rid))

        if not candidates:
            return []

        candidates.sort(key=lambda x: -x[0])

        cmds: List[Command] = []
        used: Set[int] = set()

        for _, src_id in candidates:
            if len(cmds) >= self.total:
                break
            if src_id in used:
                continue
            region = self.game_map.regions[src_id]
            for nid in region.adjacent:
                if nid in self.neutral_ids and len(cmds) < self.total:
                    amount = self.my[src_id]["troops"] - 1
                    cmds.append(Command(src_id, nid, amount, self.player_id))
                    used.add(src_id)
                    break

        return cmds

    def do_attack(self) -> List[Command]:
        """ATTACK：有敌相邻但无受威胁前线，偏进攻。"""
        attack_q = min(self.total, (self.total + 1) // 2 + 1)
        move_q = self.total - attack_q

        cmds = self._issue_attacks(attack_q, include_neutral=True)
        used = {c.source for c in cmds}
        cmds.extend(self._issue_moves(move_q, used, targets=self.safe_front))
        return cmds

    def do_defend(self) -> List[Command]:
        """DEFEND：有受威胁前线，偏防守。受威胁前线不进攻。"""
        attack_q = max(1, self.total // 3)
        move_q = self.total - attack_q

        cmds = self._issue_attacks(attack_q, include_neutral=False)
        used = {c.source for c in cmds}
        cmds.extend(self._issue_moves(move_q, used, targets=self.threatened))
        return cmds

    def _has_adjacent_enemy(self) -> bool:
        for rid in self.my:
            region = self.game_map.regions[rid]
            if region is not None and any(n in self.enemy_ids for n in region.adjacent):
                return True
        return False

    def _issue_attacks(self, quota: int, include_neutral: bool) -> List[Command]:
        if quota == 0 or not self.safe_front:
            return []

        enemy_capitals = {
            rid for rid in getattr(self.game_map, 'capitals', [])
            if rid in self.enemy_ids
        }

        AttackKey = Tuple[int, int, int, int]
        candidates: List[AttackKey] = []

        for rid in self.safe_front:
            info = self.my[rid]
            if info["troops"] <= 1:
                continue
            region = self.game_map.regions[rid]
            for nid in region.adjacent:
                if nid in self.enemy_ids:
                    prio = 20 if nid in enemy_capitals else 10
                elif nid in self.neutral_ids and include_neutral:
                    prio = 0
                else:
                    continue
                candidates.append((-prio, -info["troops"], rid, nid))

        candidates.sort()

        cmds: List[Command] = []
        used: Set[int] = set()

        for _, _, src_id, tgt_id in candidates:
            if len(cmds) >= quota:
                break
            if src_id in used:
                continue

            troops = self.my[src_id]["troops"]
            lo = max(1, int(troops * 0.5))
            hi = max(lo, int(troops * 0.8))
            amount = random.randint(lo, hi) if hi > lo else lo
            amount = min(amount, troops - 1)

            cmds.append(Command(src_id, tgt_id, amount, self.player_id))
            used.add(src_id)

        return cmds

    def _issue_moves(
        self, quota: int, used: Set[int], targets: Dict[int, dict]
    ) -> List[Command]:
        if quota == 0 or not self.rear or not targets:
            return []

        MoveKey = Tuple[int, int, int, int]
        candidates: List[MoveKey] = []

        for src_id, src_info in self.rear.items():
            if src_info["troops"] <= 1:
                continue
            if src_id in used:
                continue
            if src_info["is_capital"] and src_info["troops"] <= 10:
                continue
            region = self.game_map.regions[src_id]
            for nid in region.adjacent:
                if nid in targets:
                    candidates.append((
                        self.my[nid]["troops"],
                        -src_info["troops"],
                        src_id,
                        nid,
                    ))

        candidates.sort()

        cmds: List[Command] = []
        used_now: Set[int] = set()

        for _, _, src_id, tgt_id in candidates:
            if len(cmds) >= quota:
                break
            if src_id in used or src_id in used_now:
                continue

            troops = self.my[src_id]["troops"]
            lo = max(1, int(troops * 0.3))
            hi = max(lo, int(troops * 0.6))
            amount = random.randint(lo, hi) if hi > lo else lo
            amount = min(amount, troops - 1)

            cmds.append(Command(src_id, tgt_id, amount, self.player_id))
            used_now.add(src_id)

        return cmds
