"""游戏初始化：返回可直接传给 GameRunner 的 GameState。

扩展示例：
    # 固定首都开局
    from init_game import fixed_capitals
    state = fixed_capitals([5, 20])

    # 从存档恢复
    from init_game import from_save
    state = from_save("save.json")
"""

from __future__ import annotations

import random

from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.save_load import load_game

NUM_PLAYERS = 2
MAP_CONFIG = "cn"


def random_capitals(num_players: int = NUM_PLAYERS, map_config: str = MAP_CONFIG) -> GameState:
    """随机选取首都，各玩家起点80兵。"""
    m = GameMap(map_config)
    region_ids = [i for i in range(1, len(m.regions)) if m.valid_id(i)]
    capitals = random.sample(region_ids, num_players)
    m.assign_capitals(capitals)
    return GameState(m, num_players=num_players)


def fixed_capitals(capitals: list[int], map_config: str = MAP_CONFIG) -> GameState:
    """指定首都 id 列表开局，capitals[i] 对应玩家 i+1。"""
    m = GameMap(map_config)
    m.assign_capitals(capitals)
    return GameState(m, num_players=len(capitals))


def from_save(path: str = "save.json") -> GameState:
    """从存档文件恢复游戏状态。文件不存在时抛出 FileNotFoundError。"""
    return load_game(path)
