import unittest

from game.datatypes.game_map import GameMap, Region
from game.datatypes.game_obs import build_observation, observation_to_dict
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


class TestObservationToDict(unittest.TestCase):
    def _make_state(self):
        a = Region("北京", [2, 3], 5)
        a.owner = 1
        a.troops = 12
        a.is_capital = True
        b = Region("天津", [1, 3], 4)
        b.owner = 2
        b.troops = 8
        c = Region("河北", [1, 2], 4)
        c.owner = 0
        c.troops = 5
        m = _map_with_regions([None, a, b, c])
        return GameState(m, num_players=2, turn=3), m

    def test_contract_shape(self):
        s, m = self._make_state()
        obs = s.get_observation(1)
        d = observation_to_dict(obs, m)
        self.assertEqual(d["turn"], 3)
        self.assertEqual(d["player"], 1)
        self.assertIsInstance(d["regions"], list)
        self.assertEqual(len(d["regions"]), 3)

    def test_own_region_full_data(self):
        s, m = self._make_state()
        obs = s.get_observation(1)
        d = observation_to_dict(obs, m)
        r1 = next(r for r in d["regions"] if r["id"] == 1)
        self.assertEqual(r1["owner"], 1)
        self.assertEqual(r1["troops"], 12)
        self.assertTrue(r1["is_capital"])
        self.assertEqual(r1["base_growth"], 5)
        self.assertEqual(r1["adjacent"], [2, 3])
        self.assertEqual(r1["name"], "北京")

    def test_enemy_region_adjacent_included_troops_null(self):
        s, m = self._make_state()
        obs = s.get_observation(1)
        d = observation_to_dict(obs, m)
        r2 = next(r for r in d["regions"] if r["id"] == 2)
        self.assertEqual(r2["owner"], 2)
        self.assertIsNone(r2["troops"])
        self.assertIsNone(r2["is_capital"])
        self.assertIsNone(r2["base_growth"])
        self.assertEqual(r2["adjacent"], [1, 3])

    def test_neutral_region_adjacent_included(self):
        s, m = self._make_state()
        obs = s.get_observation(1)
        d = observation_to_dict(obs, m)
        r3 = next(r for r in d["regions"] if r["id"] == 3)
        self.assertEqual(r3["owner"], 0)
        self.assertIsNone(r3["troops"])
        self.assertEqual(r3["adjacent"], [1, 2])


if __name__ == "__main__":
    unittest.main()
