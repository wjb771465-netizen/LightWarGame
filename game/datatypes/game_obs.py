from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from game.datatypes.game_map import GameMap


@dataclass(frozen=True)
class RegionObservation:
    """
    玩家视角下单地区信息。
    仅己方：troops / is_capital / base_growth 有值；中立与敌方只暴露 owner。
    """

    region_id: int
    owner: int
    troops: Optional[int]
    is_capital: Optional[bool]
    base_growth: Optional[int]


@dataclass(frozen=True)
class Observation:
    viewer_id: int
    turn: int
    regions: Tuple[Optional[RegionObservation], ...]


def build_observation(game_map: GameMap, turn: int, viewer_id: int) -> Observation:
    """构建观测：己方全量，非己方仅归属。"""
    regs: List[Optional[RegionObservation]] = [None] * len(game_map.regions)
    for i in range(1, len(game_map.regions)):
        r = game_map.regions[i]
        if r is None:
            continue
        if r.owner == viewer_id:
            regs[i] = RegionObservation(
                region_id=i,
                owner=r.owner,
                troops=r.troops,
                is_capital=r.is_capital,
                base_growth=r.base_growth,
            )
        else:
            regs[i] = RegionObservation(
                region_id=i,
                owner=r.owner,
                troops=None,
                is_capital=None,
                base_growth=None,
            )
    return Observation(viewer_id=viewer_id, turn=turn, regions=tuple(regs))
