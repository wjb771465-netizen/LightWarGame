import os
import tempfile
import unittest

from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from game.save_load import load_game, save_game

from tests.helpers import map_with_regions as _map_with_regions


def _two_player_state(turn: int = 3) -> GameState:
    a = Region("a", [2], 8)
    a.owner = 1
    a.troops = 45
    a.is_capital = True
    b = Region("b", [1, 3], 4)
    b.owner = 2
    b.troops = 12
    c = Region("c", [2], 3)
    c.owner = 0
    c.troops = 7
    m = _map_with_regions([None, a, b, c])
    m._config_name = "cn"
    state = GameState(m, num_players=2, turn=turn)
    state.active_players = [1, 2]
    return state


class TestSaveLoad(unittest.TestCase):
    def test_save_creates_valid_json(self) -> None:
        import json

        state = _two_player_state()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "save.json")
            save_game(state, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertIn("map_config", data)
        self.assertIn("num_players", data)
        self.assertIn("turn", data)
        self.assertIn("regions", data)

    def test_round_trip_preserves_turn_and_num_players(self) -> None:
        state = _two_player_state(turn=7)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "save.json")
            save_game(state, path)
            loaded = load_game(path)
        self.assertEqual(loaded.turn, 7)
        self.assertEqual(loaded.num_players, 2)

    def test_round_trip_preserves_region_state(self) -> None:
        state = _two_player_state()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "save.json")
            save_game(state, path)
            loaded = load_game(path)
        r1 = loaded.game_map.regions[1]
        assert r1 is not None
        self.assertEqual(r1.owner, 1)
        self.assertEqual(r1.troops, 45)
        self.assertTrue(r1.is_capital)
        self.assertEqual(r1.base_growth, 8)
        r2 = loaded.game_map.regions[2]
        assert r2 is not None
        self.assertEqual(r2.owner, 2)
        self.assertEqual(r2.troops, 12)
        r3 = loaded.game_map.regions[3]
        assert r3 is not None
        self.assertEqual(r3.owner, 0)
        self.assertEqual(r3.troops, 7)

    def test_round_trip_rebuilds_active_players(self) -> None:
        state = _two_player_state()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "save.json")
            save_game(state, path)
            loaded = load_game(path)
        self.assertEqual(loaded.active_players, [1, 2])

    def test_round_trip_single_active_player(self) -> None:
        a = Region("a", [], 4)
        a.owner = 1
        a.troops = 10
        m = _map_with_regions([None, a])
        m._config_name = "cn"
        state = GameState(m, num_players=2, turn=5)
        state.active_players = [1]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "save.json")
            save_game(state, path)
            loaded = load_game(path)
        self.assertEqual(loaded.active_players, [1])

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_game("/tmp/nonexistent_lightwar_save.json")


if __name__ == "__main__":
    unittest.main()
