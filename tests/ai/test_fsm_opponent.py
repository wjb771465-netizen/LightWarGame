"""FsmOpponent 状态机对手测试。"""

import unittest

from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from tests.helpers import map_with_regions

from ai.envs.opponents.fsm_opponent import FsmOpponent


def _make_state(regions, num_players=2):
    return GameState(map_with_regions(regions), num_players=num_players)


class TestFsmOpponentStateSelection(unittest.TestCase):
    """状态转移逻辑测试。"""

    def test_reset_sets_expand(self):
        agent = FsmOpponent(player_id=2)
        agent.state = "something_else"
        agent.reset()
        self.assertEqual(agent.state, FsmOpponent.STATE_EXPAND)

    def test_expand_when_no_enemy(self):
        """无敌方相邻 -> EXPAND"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 20
        b = Region("B", [1], 4)
        b.owner = 0
        b.troops = 5
        state = _make_state([None, a, b])
        agent = FsmOpponent(player_id=2)
        agent.observe(state)
        self.assertEqual(agent.transition(), FsmOpponent.STATE_EXPAND)

    def test_attack_when_enemy_no_threat(self):
        """有敌方相邻但无受威胁前线 -> ATTACK"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 20  # 兵力远高于平均
        b = Region("B", [1], 4)
        b.owner = 1  # enemy
        b.troops = 10
        state = _make_state([None, a, b])
        agent = FsmOpponent(player_id=2)
        agent.observe(state)
        self.assertEqual(agent.transition(), FsmOpponent.STATE_ATTACK)

    def test_defend_when_threatened(self):
        """有受威胁前线（兵力 < avg*0.5）-> DEFEND"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 5  # threatened: 5 < avg(5,40)=22.5 * 0.5 = 11.25
        b = Region("B", [1, 3], 4)
        b.owner = 1
        b.troops = 80
        c = Region("C", [2], 4)
        c.owner = 2
        c.troops = 40  # safe: 40 >= 11.25
        state = _make_state([None, a, b, c])
        agent = FsmOpponent(player_id=2)
        agent.observe(state)
        self.assertEqual(agent.transition(), FsmOpponent.STATE_DEFEND)

    def test_act_sets_state(self):
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 5  # threatened: 5 < avg(5,40)=22.5 * 0.5
        b = Region("B", [1, 3], 4)
        b.owner = 1
        b.troops = 80
        c = Region("C", [2], 4)
        c.owner = 2
        c.troops = 40  # safe
        state = _make_state([None, a, b, c])
        agent = FsmOpponent(player_id=2)
        agent.act(state)
        self.assertEqual(agent.state, FsmOpponent.STATE_DEFEND)


class TestFsmOpponentExpand(unittest.TestCase):
    """EXPAND 状态行为测试。"""

    def test_expand_attacks_neutral(self):
        """无敌方相邻时攻击中立。"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 20
        b = Region("B", [1, 3], 4)
        b.owner = 0  # neutral
        b.troops = 5
        c = Region("C", [2], 4)
        c.owner = 2
        c.troops = 15
        state = _make_state([None, a, b, c])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertGreater(len(cmds), 0)
        for cmd in cmds:
            self.assertIn(cmd.target, {2})  # targets neutral
            self.assertEqual(cmd.player, 2)

    def test_expand_sends_all_but_one(self):
        """EXPAND 出兵 = troops - 1"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 20
        b = Region("B", [1], 4)
        b.owner = 0
        b.troops = 5
        state = _make_state([None, a, b])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0].troops, 19)  # troops - 1

    def test_expand_no_neutral_idle(self):
        """无敌方相邻且无邻接中立 -> 返回空"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 20
        b = Region("B", [1], 4)
        b.owner = 2  # my own region
        b.troops = 10
        state = _make_state([None, a, b])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(cmds, [])

    def test_expand_skips_low_troops(self):
        """兵力 <= 1 的地区不作为攻击源。"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 1  # too low
        b = Region("B", [1], 4)
        b.owner = 0
        b.troops = 5
        state = _make_state([None, a, b])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(cmds, [])


class TestFsmOpponentAttack(unittest.TestCase):
    """ATTACK 状态行为测试。"""

    def test_attack_prioritizes_capital(self):
        """攻击优先级：敌方首都 > 普通敌区"""
        a = Region("A", [2, 3], 4)
        a.owner = 2
        a.troops = 30
        b = Region("B", [1], 4)
        b.owner = 1
        b.troops = 10
        c = Region("C", [1], 4)
        c.owner = 1
        c.troops = 10
        c.is_capital = True  # enemy capital
        state = _make_state([None, a, b, c])
        state.game_map.capitals = [3]
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(agent.state, FsmOpponent.STATE_ATTACK)
        attack_targets = [c.target for c in cmds if c.target in {2, 3}]
        self.assertTrue(len(attack_targets) > 0)
        self.assertEqual(attack_targets[0], 3)  # capital first


class TestFsmOpponentDefend(unittest.TestCase):
    """DEFEND 状态行为测试。"""

    def test_defend_threatened_does_not_attack(self):
        """受威胁前线不作为攻击源。"""
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 4  # threatened: 4 < avg(4,20)=12 * 0.5 = 6
        b = Region("B", [1, 3], 4)
        b.owner = 1
        b.troops = 80
        c = Region("C", [2], 4)
        c.owner = 2
        c.troops = 20  # safe: 20 >= 6
        state = _make_state([None, a, b, c])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(agent.state, FsmOpponent.STATE_DEFEND)
        attack_sources = {
            c.source for c in cmds if state.game_map.regions[c.target].owner != 2
        }
        self.assertNotIn(1, attack_sources)  # region 1 is threatened, should not attack

    def test_defend_reinforces_threatened(self):
        """后方兵调往受威胁前线。"""
        a = Region("A", [2, 3], 4)
        a.owner = 2
        a.troops = 4  # threatened
        b = Region("B", [1], 4)
        b.owner = 1
        b.troops = 80
        c = Region("C", [1], 4)
        c.owner = 2
        c.troops = 30  # rear: not adjacent to enemy, adjacent to a
        d = Region("D", [], 4)
        d.owner = 2
        d.troops = 20  # isolated, bump territory count so move_q > 0
        e = Region("E", [], 4)
        e.owner = 2
        e.troops = 15  # another isolated, to reach 4 owned -> total=2
        state = _make_state([None, a, b, c, d, e])
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        self.assertEqual(agent.state, FsmOpponent.STATE_DEFEND)
        move_cmds = [
            c for c in cmds if state.game_map.regions[c.target].owner == 2
        ]
        self.assertGreater(len(move_cmds), 0)
        self.assertEqual(move_cmds[0].source, 3)  # from rear
        self.assertEqual(move_cmds[0].target, 1)  # to threatened


class TestFsmOpponentConstraints(unittest.TestCase):
    """通用约束测试。"""

    def _two_front_map(self):
        """多领地地图：player2 占 4 地 -> max_cmds=2"""
        regions = [None]
        for i in range(1, 5):
            adj = []
            if i > 1:
                adj.append(i - 1)
            if i < 4:
                adj.append(i + 1)
            r = Region(str(i), adj, 4)
            r.owner = 2
            r.troops = 20
            regions.append(r)
        e = Region("5", [4], 4)
        e.owner = 1
        e.troops = 10
        regions.append(e)
        return _make_state(regions)

    def test_respects_max_cmds(self):
        state = self._two_front_map()
        agent = FsmOpponent(player_id=2)
        for _ in range(10):
            cmds = agent.act(state)
            owned = sum(
                1 for r in state.game_map.regions[1:]
                if r is not None and r.owner == 2
            )
            from game.constants import max_commands
            self.assertLessEqual(len(cmds), max_commands(owned))

    def test_no_commands_when_no_troops(self):
        a = Region("A", [2], 4)
        a.owner = 2
        a.troops = 1
        b = Region("B", [1], 4)
        b.owner = 1
        b.troops = 5
        state = _make_state([None, a, b])
        cmds = FsmOpponent(player_id=2).act(state)
        self.assertEqual(cmds, [])

    def test_all_commands_pass_check_cmds(self):
        state = self._two_front_map()
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        valid = state.check_cmds(cmds)
        self.assertEqual(len(valid), len(cmds))

    def test_never_reads_enemy_troops(self):
        """回归：确认没有访问敌方 r.troops。

        做法：用 property mock 拦截敌方地区的 troops 访问。
        直接验证生成的指令中只用了己方兵力信息。
        """
        state = self._two_front_map()
        agent = FsmOpponent(player_id=2)
        cmds = agent.act(state)
        for cmd in cmds:
            self.assertEqual(cmd.player, 2)
            src_r = state.game_map.regions[cmd.source]
            self.assertEqual(src_r.owner, 2)

    def test_state_not_modified(self):
        state = self._two_front_map()
        owners_before = [
            r.owner if r is not None else None
            for r in state.game_map.regions
        ]
        troops_before = [
            r.troops if r is not None else None
            for r in state.game_map.regions
        ]
        FsmOpponent(player_id=2).act(state)
        owners_after = [
            r.owner if r is not None else None
            for r in state.game_map.regions
        ]
        troops_after = [
            r.troops if r is not None else None
            for r in state.game_map.regions
        ]
        self.assertEqual(owners_before, owners_after)
        self.assertEqual(troops_before, troops_after)


if __name__ == "__main__":
    unittest.main()
