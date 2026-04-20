"""测试公共工具。

使用方式：
    from tests.helpers import map_with_regions

    def test_something(self) -> None:
        a = Region("a", [2], 4); a.owner = 1; a.troops = 10
        b = Region("b", [1], 4); b.owner = 2; b.troops = 5
        m = map_with_regions([None, a, b])
"""

from game.datatypes.game_map import GameMap


def map_with_regions(regions):
    """构造 GameMap，绕过配置加载直接注入 regions 列表（index 0 为 None，1-indexed）。"""
    m = GameMap.__new__(GameMap)
    m.regions = regions
    return m
