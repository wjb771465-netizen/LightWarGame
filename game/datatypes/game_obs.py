from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from game.datatypes.game_map import GameMap


@dataclass(frozen=True)
class RegionObservation:
    """玩家视角下某一地区的可见信息；兵力为 None 表示未知。"""

    region_id: int
    name: str
    adjacent: Tuple[int, ...]
    owner: int
    troops: Optional[int]
    is_capital: bool
    base_growth: int


@dataclass(frozen=True)
class Observation:
    viewer_id: int
    turn: int
    regions: Tuple[Optional[RegionObservation], ...]


def build_observation(game_map: GameMap, turn: int, viewer_id: int) -> Observation:
    """中立地兵力一律未知"""
    regs: List[Optional[RegionObservation]] = [None] * len(game_map.regions)
    for i in range(1, len(game_map.regions)):
        r = game_map.regions[i]
        if r is None:
            continue
        adj = tuple(r.adjacent)
        if r.owner == viewer_id:
            tile = RegionObservation(
                region_id=i,
                name=r.name,
                adjacent=adj,
                owner=r.owner,
                troops=r.troops,
                is_capital=r.is_capital,
                base_growth=r.base_growth,
            )
        elif r.owner != 0:
            tile = RegionObservation(
                region_id=i,
                name=r.name,
                adjacent=adj,
                owner=r.owner,
                troops=None,
                is_capital=r.is_capital,
                base_growth=r.base_growth,
            )
        else:
            tile = RegionObservation(
                region_id=i,
                name=r.name,
                adjacent=adj,
                owner=0,
                troops=None,
                is_capital=r.is_capital,
                base_growth=r.base_growth,
            )
        regs[i] = tile
    return Observation(viewer_id=viewer_id, turn=turn, regions=tuple(regs))
