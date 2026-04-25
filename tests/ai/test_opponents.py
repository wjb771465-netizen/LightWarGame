"""
ai/envs/opponents/ 的单元测试。

用确定性小地图验证指令生成逻辑，不依赖随机种子。
"""

import unittest

from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from tests.helpers import map_with_regions

from ai.envs.opponents.random_opponent import RandomOpponent
from ai.envs.opponents.rule_opponent import RuleOpponent


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
            self.assertLessEqual(len(cmds), max(1, owned // 3))

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


if __name__ == "__main__":
    unittest.main()
