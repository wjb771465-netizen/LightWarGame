from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from game.utils import parse_map_config


class Region:
    def __init__(self, name: str, adjacent: List[int], base_growth: int):
        self.name = name
        self.adjacent = adjacent
        self.base_growth = base_growth
        self.owner = 0
        self.troops = 0
        self.is_capital = False
        self.is_special = False
        self.growth_multiplier = 1.0

    def is_adjacent_to(self, other_id: int) -> bool:
        return other_id in self.adjacent

    def battle(self, strengths: Dict[int, int]) -> None:
        """
        按各方有效兵力结算本地区：就地更新 owner、troops。

        strengths: 各势力到达本地区兵力。
        规则：最大者胜；剩余兵力 = 全局最大 − 全局第二（严格小于最大者的最大兵力，无则为 0）。
        并列最大时若守方在并列中则守方胜；否则多方进攻并列且守方不在并列中 → 中立、兵力 0。
        """
        defender = self.owner
        d = dict(strengths)
        d[defender] = d.get(defender, 0) + self.troops
        pos = [v for v in d.values() if v > 0]
        if not pos:
            self.owner = 0
            self.troops = 0
            return
        mx = max(pos)
        winners = {k for k, v in d.items() if v == mx and v > 0}
        second = max((v for v in d.values() if 0 < v < mx), default=0)
        remain = mx - second
        if len(winners) == 1:
            w = next(iter(winners))
            self.owner = w
            self.troops = remain
            return
        if defender in winners:
            self.owner = defender
            self.troops = remain
            return
        self.owner = 0
        self.troops = 0


class GameMap:
    """版图：合法 id / 邻接校验、兵力增长、按指令更新地区。"""

    __slots__ = ("regions", "_config_name")

    def __init__(self, config_name: str = "cn") -> None:
        self._config_name = config_name
        self.regions = self._load_regions(parse_map_config(config_name))

    def assign_capitals(self, capitals: Sequence[int]) -> None:
        """按玩家顺序分配首都：capitals[p-1] 为玩家 p 的首都地区 id（1-based 地区编号）。"""
        seen: set[int] = set()
        for player_num, capital_idx in enumerate(capitals, 1):
            assert self.valid_id(capital_idx), f"invalid capital region id: {capital_idx}"
            assert capital_idx not in seen, f"duplicate capital region id: {capital_idx}"
            seen.add(capital_idx)
            r = self.regions[capital_idx]
            assert r is not None
            r.owner = player_num
            r.troops = 80
            r.is_capital = True
            r.base_growth = 8

    def _load_regions(self, data: Dict[str, Any]) -> List[Optional[Region]]:
        spec = data["regions"]
        tr = data.get("initial_troops_range", [5, 10])
        gr = data.get("base_growth_range", [4, 6])
        tr_lo, tr_hi = int(tr[0]), int(tr[1])
        gr_lo, gr_hi = int(gr[0]), int(gr[1])
        by_id = {int(r["id"]): r for r in spec}
        out: List[Optional[Region]] = [None]
        for i in range(1, 32):
            assert i in by_id, f"map config missing region id {i}"
            rec = by_id[i]
            adjacent = [int(x) for x in rec["adjacent"]]
            bg = rec.get("base_growth")
            if bg is None:
                bg = random.randint(gr_lo, gr_hi)
            reg = Region(str(rec["name"]), adjacent, int(bg))
            reg.troops = random.randint(tr_lo, tr_hi)
            reg.growth_multiplier = float(rec.get("growth_multiplier", 1.0))
            reg.is_special = bool(rec.get("is_special", False))
            out.append(reg)
        return out

    def valid_id(self, idx: int) -> bool:
        if idx < 1 or idx >= len(self.regions):
            return False
        return self.regions[idx] is not None

    def get(self, idx: int) -> Optional[Region]:
        if not self.valid_id(idx):
            return None
        return self.regions[idx]

    def are_adjacent(self, a: int, b: int) -> bool:
        ra = self.get(a)
        if ra is None:
            return False
        return ra.is_adjacent_to(b) and self.valid_id(b)

    def is_surrounded(self, idx: int) -> bool:
        r = self.regions[idx]
        if r is None or r.owner == 0 or r.is_capital:
            return False
        return not any(
            self.regions[n] is not None and self.regions[n].owner == r.owner for n in r.adjacent
        )

    def troop_growth(self) -> None:
        for i in range(1, len(self.regions)):
            r = self.regions[i]
            if r is None:
                continue
            if r.owner >= 1:
                r.troops += r.base_growth
            elif r.owner == 0:
                r.troops += 1
