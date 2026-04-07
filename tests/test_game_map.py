import unittest

from game.datatypes.command import Command
from game.datatypes.game_map import GameMap, Region
from game.datatypes.state import GameState

from tests.constants import MAP_CONFIG


def _map_with_regions(regions):
    m = GameMap.__new__(GameMap)
    m.regions = regions
    return m


class TestGameMap(unittest.TestCase):
    def test_valid_id_and_adjacent(self) -> None:
        r1 = Region("a", [2], 4)
        r2 = Region("b", [1], 4)
        m = _map_with_regions([None, r1, r2])
        self.assertFalse(m.valid_id(0))
        self.assertTrue(m.valid_id(1))
        self.assertTrue(m.valid_id(2))
        self.assertFalse(m.valid_id(3))
        self.assertTrue(m.are_adjacent(1, 2))
        self.assertTrue(m.are_adjacent(2, 1))
        self.assertFalse(m.are_adjacent(1, 99))

    def test_region_is_adjacent_to(self) -> None:
        r = Region("x", [3, 5], 1)
        self.assertTrue(r.is_adjacent_to(3))
        self.assertFalse(r.is_adjacent_to(2))

    def test_troop_growth_player_and_neutral(self) -> None:
        a = Region("a", [], 10)
        a.owner = 1
        a.troops = 5
        n = Region("n", [], 3)
        n.owner = 0
        n.troops = 2
        m = _map_with_regions([None, a, n])
        m.troop_growth()
        self.assertEqual(a.troops, 15)
        self.assertEqual(n.troops, 3)

    def test_apply_cmd_friendly_merge(self) -> None:
        a = Region("a", [2], 1)
        a.owner = 1
        a.troops = 10
        b = Region("b", [1], 1)
        b.owner = 1
        b.troops = 3
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        raw = [Command(1, 2, 4, 1)]
        self.assertEqual(s.check_cmds(raw), raw)
        s.apply_cmds(raw)
        self.assertEqual(a.troops, 6)
        self.assertEqual(b.troops, 7)

    def test_game_map_loads_cn_config(self) -> None:
        m = GameMap(MAP_CONFIG)
        self.assertEqual(len(m.regions), 32)
        self.assertTrue(m.valid_id(1))
        self.assertTrue(m.valid_id(31))
        self.assertTrue(m.are_adjacent(1, 2))

    def test_assign_capitals(self) -> None:
        m = GameMap(MAP_CONFIG)
        m.assign_capitals([1, 2])
        r1 = m.get(1)
        r2 = m.get(2)
        assert r1 is not None and r2 is not None
        self.assertEqual(r1.owner, 1)
        self.assertEqual(r1.troops, 80)
        self.assertTrue(r1.is_capital)
        self.assertEqual(r1.base_growth, 8)
        self.assertEqual(r2.owner, 2)
        self.assertEqual(r2.troops, 80)

    def test_assign_capitals_rejects_duplicate(self) -> None:
        m = GameMap(MAP_CONFIG)
        with self.assertRaises(AssertionError):
            m.assign_capitals([3, 3])


class TestRegionBattle(unittest.TestCase):
    def test_unique_winner_remain_is_max_minus_second(self) -> None:
        # 本地 7 + 己方到达 8 = 15，敌方到达 8 → 胜方 1，剩 15-8=7
        r = Region("x", [], 1)
        r.owner = 1
        r.troops = 7
        r.battle({1: 8, 2: 8})
        self.assertEqual(r.owner, 1)
        self.assertEqual(r.troops, 7)

    def test_attacker_unique_winner(self) -> None:
        r = Region("x", [], 1)
        r.owner = 1
        r.troops = 5
        r.battle({2: 10})
        self.assertEqual(r.owner, 2)
        self.assertEqual(r.troops, 5)

    def test_tie_max_defender_priority(self) -> None:
        r = Region("x", [], 1)
        r.owner = 1
        r.troops = 10
        r.battle({2: 10})
        self.assertEqual(r.owner, 1)
        self.assertEqual(r.troops, 10)

    def test_tie_attackers_only_becomes_neutral(self) -> None:
        r = Region("x", [], 1)
        r.owner = 1
        r.troops = 5
        r.battle({2: 10, 3: 10})
        self.assertEqual(r.owner, 0)
        self.assertEqual(r.troops, 0)

    def test_defender_key_defaulted_zero(self) -> None:
        r = Region("x", [], 1)
        r.owner = 1
        r.troops = 2
        r.battle({2: 10, 3: 10})
        self.assertEqual(r.owner, 0)
        self.assertEqual(r.troops, 0)

    def test_all_zero_strengths_neutral(self) -> None:
        r = Region("x", [], 1)
        r.owner = 2
        r.troops = 0
        r.battle({1: 0, 2: 0})
        self.assertEqual(r.owner, 0)
        self.assertEqual(r.troops, 0)

    def test_neutral_defender_tie_with_attacker_keeps_neutral(self) -> None:
        r = Region("x", [], 1)
        r.owner = 0
        r.troops = 10
        r.battle({1: 10})
        self.assertEqual(r.owner, 0)
        self.assertEqual(r.troops, 10)


if __name__ == "__main__":
    unittest.main()
