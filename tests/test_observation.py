import unittest

from game.datatypes.game_map import GameMap, Region
from game.datatypes.game_obs import build_observation
from game.datatypes.state import GameState

from tests.helpers import map_with_regions as _map_with_regions


class TestObservation(unittest.TestCase):
    def test_own_full_enemy_neutral_unknown_troops(self) -> None:
        a = Region("a", [2], 4)
        a.owner = 1
        a.troops = 10
        b = Region("b", [1], 4)
        b.owner = 2
        b.troops = 99
        n = Region("n", [], 3)
        n.owner = 0
        n.troops = 7
        m = _map_with_regions([None, a, b, n])
        s = GameState(m, num_players=3, turn=3)
        obs = s.get_observation(1)
        self.assertEqual(obs.viewer_id, 1)
        self.assertEqual(obs.turn, 3)
        oa = obs.regions[1]
        ob = obs.regions[2]
        on = obs.regions[3]
        assert oa is not None and ob is not None and on is not None
        self.assertEqual(oa.troops, 10)
        self.assertEqual(oa.is_capital, False)
        self.assertEqual(oa.base_growth, 4)
        self.assertIsNone(ob.troops)
        self.assertIsNone(ob.is_capital)
        self.assertIsNone(ob.base_growth)
        self.assertEqual(ob.owner, 2)
        self.assertIsNone(on.troops)
        self.assertIsNone(on.is_capital)
        self.assertIsNone(on.base_growth)
        self.assertEqual(on.owner, 0)


if __name__ == "__main__":
    unittest.main()
