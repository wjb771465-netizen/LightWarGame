import io
import unittest

from game.datatypes.game_map import GameMap, Region
from game.datatypes.game_obs import Observation, RegionObservation
from game.ui import display


class TestShowObservation(unittest.TestCase):
    def test_skips_neutral_shows_own_and_enemy(self) -> None:
        buf = io.StringIO()
        obs = Observation(
            viewer_id=1,
            turn=2,
            regions=(
                None,
                RegionObservation(1, 1, 10, False, 4),
                RegionObservation(2, 2, None, None, None),
                RegionObservation(3, 0, None, None, None),
            ),
        )
        display.show_observation(obs, buf)
        text = buf.getvalue()
        self.assertIn("己方", text)
        self.assertIn("   1", text)
        self.assertIn("10", text)
        self.assertIn("敌·P2", text)
        self.assertIn("   2", text)
        self.assertNotIn("   3", text)


if __name__ == "__main__":
    unittest.main()
