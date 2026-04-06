import unittest

from game.datatypes.game_map import GameMap, Region
from game.datatypes.state import GameState
from game.runner import GameRunner
from game.ui_ports import PlaceholderGameUi


def _map_with_regions(regions):
    m = GameMap.__new__(GameMap)
    m.regions = regions
    return m


class TestGameRunner(unittest.TestCase):
    def test_run_until_no_resolve_when_already_finished(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        a.troops = 5
        b = Region("b", [], 4)
        b.owner = 1
        b.troops = 3
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        self.assertEqual(len(s.active_players), 1)
        r = GameRunner(s, PlaceholderGameUi())
        self.assertIsNone(r.run())
        self.assertEqual(s.turn, 1)

    def test_run_single_turn_placeholder_advances_turn(self) -> None:
        a = Region("a", [2], 4)
        a.owner = 1
        a.troops = 10
        b = Region("b", [1], 4)
        b.owner = 2
        b.troops = 10
        m = _map_with_regions([None, a, b])
        s = GameState(m, num_players=2)
        self.assertEqual(len(s.active_players), 2)
        r = GameRunner(s, PlaceholderGameUi())
        cont = r.run_single_turn()
        self.assertTrue(cont)
        self.assertEqual(s.turn, 2)
        self.assertEqual(r.last_turn_results, [])

    def test_run_single_turn_when_finished_returns_false(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        a.troops = 5
        m = _map_with_regions([None, a])
        s = GameState(m, num_players=2)
        self.assertEqual(len(s.active_players), 1)
        r = GameRunner(s, PlaceholderGameUi())
        self.assertFalse(r.run_single_turn())


if __name__ == "__main__":
    unittest.main()
