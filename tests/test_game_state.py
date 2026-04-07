import unittest

from game.datatypes.command import Command
from game.datatypes.game_map import GameMap, Region
from game.datatypes.state import GameState

from tests.constants import MAP_CONFIG


def _map_with_regions(regions):
    m = GameMap.__new__(GameMap)
    m.regions = regions
    return m


class TestGameState(unittest.TestCase):
    def test_init_with_map(self) -> None:
        m = GameMap(MAP_CONFIG)
        m.assign_capitals([5, 10])
        s = GameState(m, num_players=2)
        self.assertEqual(s.turn, 1)
        self.assertEqual(s.num_players, 2)
        self.assertEqual(s.active_players, [1, 2])
        self.assertIs(s.game_map, m)
        self.assertIsNotNone(s.game_map.get(5))
        self.assertIsNotNone(s.game_map.get(10))
        self.assertEqual(s.game_map.get(5).owner, 1)
        self.assertEqual(s.game_map.get(10).owner, 2)

    def test_apply_cmd_friendly_move_then_growth(self) -> None:
        a = Region("a", [2], 10)
        a.owner = 1
        a.troops = 20
        b = Region("b", [1], 10)
        b.owner = 1
        b.troops = 5
        c = Region("c", [], 4)
        c.owner = 2
        c.troops = 10
        m = _map_with_regions([None, a, b, c])
        s = GameState(m, num_players=2, turn=1)
        raw = [Command(1, 2, 5, 1)]
        valid = s.check_cmds(raw)
        self.assertEqual(valid, raw)
        s.apply_cmds(valid)
        self.assertEqual(a.troops, 15)
        self.assertEqual(b.troops, 10)
        self.assertEqual(s.turn, 1)
        self.assertFalse(s.settle())
        self.assertEqual(a.troops, 25)
        self.assertEqual(b.troops, 20)
        self.assertEqual(c.troops, 14)
        self.assertEqual(s.turn, 2)

    def test_apply_cmd_invalid_skips_move_growth_still(self) -> None:
        a = Region("a", [2], 4)
        a.owner = 1
        a.troops = 10
        b = Region("b", [1], 4)
        b.owner = 2
        b.troops = 3
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        raw = [Command(1, 2, 5, 2)]
        self.assertEqual(s.check_cmds(raw), [])
        s.apply_cmds([])
        self.assertEqual(a.troops, 10)
        self.assertEqual(b.troops, 3)
        self.assertEqual(s.turn, 1)
        self.assertFalse(s.settle())
        self.assertEqual(a.troops, 14)
        self.assertEqual(b.troops, 7)
        self.assertEqual(s.turn, 2)

    def test_apply_cmd_rejects_same_source_over_budget(self) -> None:
        a = Region("a", [2, 3], 4)
        a.owner = 1
        a.troops = 10
        b = Region("b", [1], 4)
        b.owner = 1
        b.troops = 1
        c = Region("c", [1], 4)
        c.owner = 2
        c.troops = 5
        m = _map_with_regions([None, a, b, c])
        s = GameState(m, num_players=2)
        raw = [
            Command(1, 2, 5, 1),
            Command(1, 3, 5, 1),
        ]
        self.assertEqual(s.check_cmds(raw), [])
        s.apply_cmds([])
        self.assertEqual(a.troops, 10)
        self.assertEqual(b.troops, 1)
        self.assertEqual(c.troops, 5)

    def test_two_survivors_on_map(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        a.troops = 5
        b = Region("b", [], 4)
        b.owner = 2
        b.troops = 5
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        self.assertEqual(len(s.active_players), 2)

    def test_one_survivor_on_map(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        a.troops = 5
        b = Region("b", [], 4)
        b.owner = 1
        b.troops = 3
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        s.settle()
        self.assertEqual(s.active_players, [1])

    def test_active_players_sorted(self) -> None:
        a = Region("a", [], 4)
        a.owner = 2
        b = Region("b", [], 4)
        b.owner = 1
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        self.assertEqual(s.active_players, [1, 2])

    def test_settle_continues_increments_turn(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        b = Region("b", [], 4)
        b.owner = 2
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2, turn=1)
        self.assertFalse(s.settle())
        self.assertEqual(s.turn, 2)
        self.assertEqual(s.active_players, [1, 2])

    def test_settle_finishes_without_turn_increment(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        b = Region("b", [], 4)
        b.owner = 1
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2, turn=3)
        self.assertTrue(s.settle())
        self.assertEqual(s.turn, 3)
        self.assertEqual(s.active_players, [1])

    def test_winner_not_single_returns_none(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        b = Region("b", [], 4)
        b.owner = 2
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        self.assertIsNone(s.winner())

    def test_winner_single_survivor(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        b = Region("b", [], 4)
        b.owner = 1
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        s.settle()
        self.assertEqual(s.winner(), 1)

    def test_winner_no_one_with_land(self) -> None:
        a = Region("a", [], 4)
        a.owner = 0
        m = _map_with_regions([None, a])
        s = GameState(m, num_players=2)
        s.settle()
        self.assertEqual(s.active_players, [])
        self.assertIsNone(s.winner())


if __name__ == "__main__":
    unittest.main()
