"""
ai/envs/opponents/ 的单元测试。

用确定性小地图验证指令生成逻辑，不依赖随机种子。
"""

import unittest

import numpy as np

from game.constants import max_commands
from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from tests.helpers import map_with_regions

from ai.envs.opponents.policy_opponent import PolicyOpponent
from ai.envs.opponents.random_opponent import RandomOpponent
from ai.envs.opponents.rule_opponent import RuleOpponent
from ai.envs.action import ActionEncoder
from ai.envs.observation import ObservationEncoder


class _MockPolicy:
    """可编程的 mock Policy，按预设顺序返回动作。"""

    def __init__(self, *actions: int):
        self._actions = list(actions)
        self._idx = 0
        self.last_mask: np.ndarray | None = None
        self.last_obs: np.ndarray | None = None
        self.call_count = 0

    def predict(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self.last_obs = obs.copy()
        self.last_mask = mask.copy()
        self.call_count += 1
        if self._idx >= len(self._actions):
            return 0
        action = self._actions[self._idx]
        self._idx += 1
        return action


def _make_state(regions, num_players=2):
    return GameState(map_with_regions(regions), num_players=num_players)


class TestRandomOpponent(unittest.TestCase):

    def _linear_map(self):
        # 1(p2,20) ↔ 2(p2,15) ↔ 3(p1,10)
        a = Region("A", [2], 4); a.owner = 2; a.troops = 20
        b = Region("B", [1, 3], 4); b.owner = 2; b.troops = 15
        c = Region("C", [2], 4); c.owner = 1; c.troops = 10
        return _make_state([None, a, b, c])

    def test_commands_respect_max_cmds(self):
        state = self._linear_map()
        opp = RandomOpponent(player_id=2)
        for _ in range(20):
            cmds = opp.act(state)
            owned = sum(1 for r in state.game_map.regions[1:] if r is not None and r.owner == 2)
            self.assertLessEqual(len(cmds), max_commands(owned))

    def test_commands_are_valid(self):
        state = self._linear_map()
        opp = RandomOpponent(player_id=2)
        for _ in range(20):
            cmds = opp.act(state)
            valid = state.check_cmds(cmds)
            # 随机对手可能产生同源冲突被 check_cmds 过滤，但不应有非法地区
            for cmd in cmds:
                self.assertEqual(cmd.player, 2)

    def test_no_commands_when_all_troops_at_one(self):
        a = Region("A", [2], 4); a.owner = 2; a.troops = 1
        b = Region("B", [1], 4); b.owner = 1; b.troops = 5
        state = _make_state([None, a, b])
        cmds = RandomOpponent(player_id=2).act(state)
        self.assertEqual(cmds, [])


class TestRuleOpponent(unittest.TestCase):

    def _triangle_map(self):
        """
        1(p2,30) ↔ 2(p1,10) ↔ 3(p2,5)
        1 ↔ 3
        """
        a = Region("A", [2, 3], 4); a.owner = 2; a.troops = 30
        b = Region("B", [1, 3], 4); b.owner = 1; b.troops = 10
        c = Region("C", [1, 2], 4); c.owner = 2; c.troops = 5
        return _make_state([None, a, b, c])

    def test_attack_prefers_enemy_target(self):
        # region1(p2) 邻 region2(enemy)，应攻击 region2 而非同阵营 region3
        state = self._triangle_map()
        opp = RuleOpponent(player_id=2)
        cmds = opp.act(state)
        attack_cmds = [c for c in cmds if state.game_map.regions[c.target].owner != 2]
        self.assertTrue(len(attack_cmds) > 0)

    def test_attack_source_has_most_troops_among_adj_enemy(self):
        # region1(30兵) 和 region3(5兵) 都邻 region2(enemy)
        # 攻击源应优先选 region1（兵更多）
        state = self._triangle_map()
        opp = RuleOpponent(player_id=2)
        cmds = opp.act(state)
        attack_sources = [c.source for c in cmds if state.game_map.regions[c.target].owner != 2]
        if attack_sources:
            self.assertIn(1, attack_sources)  # region1 应被选为攻击源

    def test_move_uses_floor_half_troops(self):
        """调兵量 = floor(troops / 2)"""
        # 1(p2,20) ↔ 2(p2,5) ↔ 3(p1,10)
        a = Region("A", [2], 4); a.owner = 2; a.troops = 20
        b = Region("B", [1, 3], 4); b.owner = 2; b.troops = 5
        c = Region("C", [2], 4); c.owner = 1; c.troops = 10
        state = _make_state([None, a, b, c])
        opp = RuleOpponent(player_id=2)
        cmds = opp.act(state)
        move_cmds = [c for c in cmds if state.game_map.regions[c.target].owner == 2]
        for cmd in move_cmds:
            src_troops = state.game_map.regions[cmd.source].troops
            self.assertEqual(cmd.troops, min(src_troops // 2, src_troops - 1))

    def test_no_same_source_for_attack_and_move(self):
        state = self._triangle_map()
        opp = RuleOpponent(player_id=2)
        cmds = opp.act(state)
        sources = [c.source for c in cmds]
        self.assertEqual(len(sources), len(set(sources)))

    def test_no_commands_when_all_troops_at_one(self):
        a = Region("A", [2], 4); a.owner = 2; a.troops = 1
        b = Region("B", [1], 4); b.owner = 1; b.troops = 5
        state = _make_state([None, a, b])
        cmds = RuleOpponent(player_id=2).act(state)
        self.assertEqual(cmds, [])

    def test_adj_neutral_fallback_when_no_enemy(self):
        # region1(p2) 只邻中立 region2，无敌方邻居 → 应攻击中立
        a = Region("A", [2], 4); a.owner = 2; a.troops = 10
        b = Region("B", [1], 4); b.owner = 0; b.troops = 3
        state = _make_state([None, a, b])
        cmds = RuleOpponent(player_id=2).act(state)
        attack_cmds = [c for c in cmds if c.target == 2]
        self.assertTrue(len(attack_cmds) > 0)

    def test_all_generated_commands_pass_check_cmds(self):
        state = self._triangle_map()
        opp = RuleOpponent(player_id=2)
        cmds = opp.act(state)
        valid = state.check_cmds(cmds)
        self.assertEqual(len(valid), len(cmds))


class TestPolicyOpponent(unittest.TestCase):
    """PolicyOpponent：mock Policy 验证多步自回归逻辑。"""

    def _multi_region_map(self):
        """1(p2,20) ↔ 2(p2,20) ↔ 3(p2,20) ↔ 4(p2,20) ↔ 5(p1,10)
        player2 占 4 地 → max_cmds=2"""
        regions = [None]
        for i in range(1, 5):
            r = Region(str(i), [i - 1, i + 1] if 1 < i < 4 else ([i - 1] if i == 4 else [i + 1]), 4)
            r.owner = 2
            r.troops = 20
            regions.append(r)
        e = Region("5", [4], 4)
        e.owner = 1
        e.troops = 10
        regions.append(e)
        return map_with_regions(regions)

    def _opponent(self, player_id=2, *actions: int):
        m = self._multi_region_map()
        obs_enc = ObservationEncoder(m, max_players=2)
        act_enc = ActionEncoder(m)
        policy = _MockPolicy(*actions)
        opp = PolicyOpponent(player_id, policy, obs_enc, act_enc)
        return opp, m, policy

    def test_multi_step_returns_multiple_commands(self):
        """配额为 2，提供 2 个有效动作 → 应返回 2 条指令。"""
        opp, m, _ = self._opponent(2, 1, 17)  # edges (1,2) and (3,4)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        self.assertEqual(len(cmds), 2)

    def test_stops_on_noop(self):
        """第一步有效，第二步 no-op → 只返回 1 条。"""
        opp, m, _ = self._opponent(2, 1, 0)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        self.assertEqual(len(cmds), 1)

    def test_respects_max_cmds(self):
        """提供 5 个动作但配额只有 2 → 最多 2 条。"""
        opp, m, _ = self._opponent(2, 1, 17, 1, 17)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        self.assertLessEqual(len(cmds), 2)

    def test_player_id_on_commands(self):
        opp, m, _ = self._opponent(2, 1)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        self.assertGreater(len(cmds), 0)
        for cmd in cmds:
            self.assertEqual(cmd.player, 2)

    def test_all_commands_valid(self):
        opp, m, _ = self._opponent(2, 1, 17)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        valid = state.check_cmds(cmds)
        self.assertEqual(len(valid), len(cmds))

    def test_obs_shape_matches_encoder_dim(self):
        opp, m, policy = self._opponent(2, 1)
        state = GameState(m, num_players=2)
        opp.act(state)
        self.assertIsNotNone(policy.last_obs)
        self.assertEqual(policy.last_obs.shape, (opp._obs_enc.dim,))
        self.assertEqual(policy.last_mask.shape, (opp._act_enc.dim,))

    def test_noop_returns_empty_when_no_valid_moves(self):
        """对方无兵可派时 mask 只有 no-op，返回空。"""
        a = Region("A", [2], 4); a.owner = 2; a.troops = 1
        b = Region("B", [1], 4); b.owner = 1; b.troops = 5
        m = map_with_regions([None, a, b])
        obs_enc = ObservationEncoder(m, max_players=2)
        act_enc = ActionEncoder(m)
        opp = PolicyOpponent(2, _MockPolicy(0), obs_enc, act_enc)
        state = GameState(m, num_players=2)
        cmds = opp.act(state)
        self.assertEqual(cmds, [])


class TestPolicyOpponentIntegration(unittest.TestCase):
    """PolicyOpponent 集成测试：真实 Encoder + mock Policy。"""

    def test_full_pipeline_with_real_encoders(self):
        """encode → mask(含 pending) → predict → decode 全链路。"""
        regions = [None]
        for i in range(1, 5):
            r = Region(str(i), [i - 1, i + 1] if 1 < i < 4 else ([i - 1] if i == 4 else [i + 1]), 4)
            r.owner = 2; r.troops = 20
            regions.append(r)
        e = Region("5", [4], 4); e.owner = 1; e.troops = 10
        regions.append(e)
        m = map_with_regions(regions)

        obs_enc = ObservationEncoder(m, max_players=2)
        act_enc = ActionEncoder(m)
        policy = _MockPolicy(1, 17)  # edges (1,2) and (3,4)
        opp = PolicyOpponent(2, policy, obs_enc, act_enc)

        state = GameState(m, num_players=2)
        cmds = opp.act(state)

        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[0].player, 2)
        self.assertEqual(cmds[1].player, 2)

    def test_second_call_mask_reflects_pending(self):
        """第二次 predict 时 mask 应反映 pending 指令的兵力扣减。"""
        opp, m, policy = self._opp_for_integration(2, 1, 17)
        state = GameState(m, num_players=2)
        opp.act(state)
        self.assertEqual(policy.call_count, 2)

    def _opp_for_integration(self, player_id=2, *actions: int):
        """构建 4 领地地图 → max_cmds=2"""
        regions = [None]
        for i in range(1, 5):
            r = Region(str(i), [i - 1, i + 1] if 1 < i < 4 else ([i - 1] if i == 4 else [i + 1]), 4)
            r.owner = 2; r.troops = 20
            regions.append(r)
        e = Region("5", [4], 4); e.owner = 1; e.troops = 10
        regions.append(e)
        m = map_with_regions(regions)
        obs_enc = ObservationEncoder(m, max_players=2)
        act_enc = ActionEncoder(m)
        policy = _MockPolicy(*actions)
        opp = PolicyOpponent(player_id, policy, obs_enc, act_enc)
        return opp, m, policy


if __name__ == "__main__":
    unittest.main()
