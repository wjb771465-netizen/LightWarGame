"""
ai/envs/rewards/ 的单元测试。

使用 map_with_regions + GameState 构造 curr，StateSnapshot 构造 prev，
不依赖随机初始化或完整游戏流程。
"""

import copy
import unittest
from types import SimpleNamespace

from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from tests.helpers import map_with_regions

from ai.envs.utils import StateSnapshot
from ai.envs.rewards import build_reward_functions
from ai.envs.rewards.win_lose_reward import WinLoseReward
from ai.envs.rewards.territory_reward import TerritoryReward
from ai.envs.rewards.capital_capture_reward import CapitalCaptureReward
from ai.envs.rewards.step_penalty_reward import StepPenaltyReward


def _make_state(regions, active_players=None, num_players=2):
    s = GameState(map_with_regions(regions), num_players=num_players)
    if active_players is not None:
        s.active_players = active_players
    return s


def _two_region_map(p1_troops=10, p2_troops=10):
    a = Region("A", [2], 4); a.owner = 1; a.troops = p1_troops
    b = Region("B", [1], 4); b.owner = 2; b.troops = p2_troops
    return [None, a, b]


def _snap(state: GameState) -> StateSnapshot:
    return StateSnapshot.from_state(state)


class TestWinLoseReward(unittest.TestCase):

    def setUp(self):
        self.rf = WinLoseReward(win=1.0, lose=-1.0)

    def _state(self, active):
        return _make_state(_two_region_map(), active_players=active)

    def test_win(self):
        prev = _snap(self._state([1, 2]))
        curr = self._state([1])
        self.assertAlmostEqual(self.rf.get_reward(prev, curr, player_id=1, terminated=True), 1.0)

    def test_lose(self):
        prev = _snap(self._state([1, 2]))
        curr = self._state([2])
        self.assertAlmostEqual(self.rf.get_reward(prev, curr, player_id=1, terminated=True), -1.0)

    def test_draw_timeout(self):
        prev = _snap(self._state([1, 2]))
        curr = self._state([1, 2])
        self.assertAlmostEqual(self.rf.get_reward(prev, curr, player_id=1, terminated=True), 0.0)

    def test_not_terminated_returns_zero(self):
        prev = _snap(self._state([1, 2]))
        curr = self._state([1, 2])
        self.assertAlmostEqual(self.rf.get_reward(prev, curr, player_id=1, terminated=False), 0.0)


class TestTerritoryReward(unittest.TestCase):
    """势函数奖励：F(s,s') = φ(s') − φ(s)，测试前须 reset() 初始化 _prev_phi。"""

    def setUp(self):
        self.rf = TerritoryReward(territory_gain=0.05, territory_loss=-0.05)

    def _state_with_owners(self, owners):
        regions = [None]
        for i, owner in enumerate(owners, 1):
            r = Region(str(i), [], 4)
            r.owner = owner
            r.troops = 5
            regions.append(r)
        return _make_state(regions)

    def test_no_change_returns_zero(self):
        # φ([1,2]) = 0.05*1 + (-0.05)*1 = 0.0；prev = curr → reward = 0
        state = self._state_with_owners([1, 2])
        self.rf.reset(state, player_id=1)
        self.assertAlmostEqual(
            self.rf.get_reward(_snap(state), state, player_id=1, terminated=False), 0.0
        )

    def test_gain_one_territory(self):
        # prev φ = 0.05*1 + 0 = 0.05；curr φ = 0.05*2 = 0.10 → +0.05
        prev_state = self._state_with_owners([1, 0])
        self.rf.reset(prev_state, player_id=1)
        curr = self._state_with_owners([1, 1])
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), 0.05)

    def test_lose_one_territory(self):
        # prev φ = 0.05*2 = 0.10；curr φ = 0.05*1 + (-0.05)*1 = 0.0 → -0.10
        prev_state = self._state_with_owners([1, 1])
        self.rf.reset(prev_state, player_id=1)
        curr = self._state_with_owners([2, 1])
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), -0.10)

    def test_enemy_loses_territory(self):
        # prev φ = 0.05*1 + (-0.05)*1 = 0.0；curr φ = 0.05*1 = 0.05 → +0.05
        prev_state = self._state_with_owners([1, 2])
        self.rf.reset(prev_state, player_id=1)
        curr = self._state_with_owners([1, 0])
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), 0.05)


class TestCapitalCaptureReward(unittest.TestCase):
    """势函数奖励：F = φ(curr) − φ(prev)，测试前须 reset() 初始化 _prev_phi。"""

    def setUp(self):
        self.rf = CapitalCaptureReward(capital_capture=0.2)

    def _two_capital_state(self, r1_owner, r2_owner):
        a = Region("A", [2], 4); a.owner = r1_owner; a.troops = 10; a.is_capital = True
        b = Region("B", [1], 4); b.owner = r2_owner; b.troops = 10; b.is_capital = True
        return _make_state([None, a, b])

    def test_no_capture_returns_zero(self):
        # φ(p1 owns 1 cap) = 0.2；prev = curr → reward = 0
        state = self._two_capital_state(1, 2)
        self.rf.reset(state, player_id=1)
        self.assertAlmostEqual(
            self.rf.get_reward(_snap(state), state, player_id=1, terminated=False), 0.0
        )

    def test_capture_enemy_capital(self):
        # prev φ = 0.2；curr φ = 0.4 → +0.2
        prev_state = self._two_capital_state(1, 2)
        self.rf.reset(prev_state, player_id=1)
        curr = self._two_capital_state(1, 1)
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), 0.2)

    def test_own_capital_recaptured(self):
        # prev φ = 0（p1 无首都）；curr φ = 0.2 → +0.2
        prev_state = self._two_capital_state(2, 2)
        self.rf.reset(prev_state, player_id=1)
        curr = self._two_capital_state(1, 2)
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), 0.2)

    def test_capture_two_capitals(self):
        # prev φ = 0（p1 无首都）；curr φ = 0.4（p1 占 2 首都）→ +0.4
        a = Region("A", [2, 3], 4); a.owner = 2; a.troops = 10; a.is_capital = True
        b = Region("B", [1, 3], 4); b.owner = 3; b.troops = 10; b.is_capital = True
        c = Region("C", [1, 2], 4); c.owner = 1; c.troops = 10
        prev_state = _make_state([None, a, b, c], num_players=3)
        self.rf.reset(prev_state, player_id=1)
        a2 = copy.copy(a); a2.owner = 1
        b2 = copy.copy(b); b2.owner = 1
        curr = _make_state([None, a2, b2, copy.copy(c)], num_players=3)
        self.assertAlmostEqual(self.rf.get_reward(_snap(prev_state), curr, player_id=1, terminated=False), 0.4)


class TestBuildRewardFunctions(unittest.TestCase):

    def _cfg(self, win=1.0, lose=-1.0, territory_gain=0.05, territory_loss=-0.05,
             capital_capture=0.2, step_penalty=-0.1):
        return SimpleNamespace(win=win, lose=lose, shaped=SimpleNamespace(
            territory_gain=territory_gain, territory_loss=territory_loss,
            capital_capture=capital_capture, step_penalty=step_penalty,
        ))

    def test_returns_four_components(self):
        rfs = build_reward_functions(self._cfg())
        self.assertEqual(len(rfs), 4)
        self.assertIsInstance(rfs[0], WinLoseReward)
        self.assertIsInstance(rfs[1], TerritoryReward)
        self.assertIsInstance(rfs[2], CapitalCaptureReward)
        self.assertIsInstance(rfs[3], StepPenaltyReward)

    def test_custom_config_applied(self):
        rfs = build_reward_functions(self._cfg(win=5.0, lose=-3.0, capital_capture=1.0, step_penalty=-0.5))
        self.assertAlmostEqual(rfs[0]._win, 5.0)
        self.assertAlmostEqual(rfs[0]._lose, -3.0)
        self.assertAlmostEqual(rfs[2]._capital_capture, 1.0)
        self.assertAlmostEqual(rfs[3]._penalty, -0.5)


if __name__ == "__main__":
    unittest.main()
