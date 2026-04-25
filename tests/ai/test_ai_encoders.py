"""
ai/envs/observation.py 和 ai/envs/action.py 的单元测试。

使用 map_with_regions 构造最小确定性地图，避免依赖随机初始化。
"""

import math
import unittest

from game.datatypes.game_map import Region
from game.datatypes.game_obs import build_observation
from tests.helpers import map_with_regions
from ai.envs.observation import ObservationEncoder
from ai.envs.action import ActionEncoder


def _make_map():
    """
    最小 3 地区地图：
      1(北) ↔ 2(南)，2 ↔ 3(东)，1 不与 3 相邻。
      player1 拥有 region 1（20 兵），player2 拥有 region 2（10 兵），region 3 中立（5 兵）。
    """
    a = Region("北", [2], 4); a.owner = 1; a.troops = 20
    b = Region("南", [1, 3], 4); b.owner = 2; b.troops = 10
    c = Region("东", [2], 4); c.owner = 0; c.troops = 5
    return map_with_regions([None, a, b, c])


class TestObservationEncoder(unittest.TestCase):

    def setUp(self):
        self.gm = _make_map()
        self.enc = ObservationEncoder(self.gm, max_players=2)
        self.obs = build_observation(self.gm, turn=1, viewer_id=1)
        self.vec = self.enc.encode(self.obs)

    def test_dim_matches_formula(self):
        # obs_dim = num_regions × (max_players + 6)
        expected = 3 * (2 + 6)
        self.assertEqual(self.enc.dim, expected)

    def test_encoded_shape_matches_dim(self):
        self.assertEqual(self.vec.shape, (self.enc.dim,))

    def test_encoded_dtype_float32(self):
        self.assertEqual(self.vec.dtype.name, "float32")

    def test_space_shape_matches_dim(self):
        self.assertEqual(self.enc.space.shape, (self.enc.dim,))

    def test_own_region_onehot_at_index1(self):
        # viewer=1 拥有 region1；owner_onehot index 1 应为 1.0
        F = self.enc._F
        base = 0 * F  # region1 在 idx=0
        self.assertAlmostEqual(self.vec[base + 1], 1.0)  # index 1 = viewer self
        self.assertAlmostEqual(self.vec[base + 0], 0.0)  # index 0 = neutral

    def test_neutral_region_onehot_at_index0(self):
        # region3 中立；owner_onehot index 0 应为 1.0
        F = self.enc._F
        base = 2 * F  # region3 在 idx=2
        self.assertAlmostEqual(self.vec[base + 0], 1.0)

    def test_enemy_region_onehot_at_index2(self):
        # player2 拥有 region2；viewer-relative index = 2
        F = self.enc._F
        base = 1 * F  # region2 在 idx=1
        self.assertAlmostEqual(self.vec[base + 2], 1.0)

    def test_own_region_is_visible(self):
        F = self.enc._F
        base = 0 * F
        is_visible_offset = (2 + 1) + 3  # max_players+1 + 3 scalars before is_visible
        self.assertAlmostEqual(self.vec[base + is_visible_offset], 1.0)

    def test_foggy_region_is_not_visible(self):
        # region2 归 player2，viewer=1 看不到兵力 → is_visible=0
        F = self.enc._F
        base = 1 * F
        is_visible_offset = (2 + 1) + 3
        self.assertAlmostEqual(self.vec[base + is_visible_offset], 0.0)

    def test_foggy_region_troops_is_zero(self):
        # foggy 时 troops_norm 应为 0
        F = self.enc._F
        base = 1 * F
        troops_offset = 2 + 1  # max_players+1
        self.assertAlmostEqual(self.vec[base + troops_offset], 0.0)

    def test_adj_to_my_territory_set_for_enemy_neighbor(self):
        # region2 与 region1（己方）相邻 → is_adj_to_my_territory=1
        F = self.enc._F
        base = 1 * F
        adj_offset = (2 + 1) + 4
        self.assertAlmostEqual(self.vec[base + adj_offset], 1.0)

    def test_adj_to_my_territory_zero_for_nonadjacent(self):
        # region3 不与 region1 相邻 → is_adj_to_my_territory=0
        F = self.enc._F
        base = 2 * F
        adj_offset = (2 + 1) + 4
        self.assertAlmostEqual(self.vec[base + adj_offset], 0.0)


class TestActionEncoder(unittest.TestCase):

    def setUp(self):
        self.gm = _make_map()
        self.enc = ActionEncoder(self.gm)
        self.obs = build_observation(self.gm, turn=1, viewer_id=1)

    def test_edge_list_sorted(self):
        edges = self.enc._edges
        self.assertEqual(edges, sorted(edges))

    def test_edge_list_contains_all_directed_edges(self):
        # 1↔2, 2↔3 → 4 条有向边
        edges = set(self.enc._edges)
        self.assertIn((1, 2), edges)
        self.assertIn((2, 1), edges)
        self.assertIn((2, 3), edges)
        self.assertIn((3, 2), edges)
        self.assertEqual(len(edges), 4)

    def test_dim_equals_edges_times_buckets_plus_one(self):
        self.assertEqual(self.enc.dim, 4 * 4 + 1)

    def test_space_n_equals_dim(self):
        self.assertEqual(self.enc.space.n, self.enc.dim)

    def test_noop_always_valid(self):
        mask = self.enc.mask(self.obs, commands_issued=0, max_commands=1)
        self.assertTrue(mask[0])

    def test_only_noop_when_quota_exhausted(self):
        mask = self.enc.mask(self.obs, commands_issued=3, max_commands=3)
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[0])

    def test_enemy_src_not_in_mask(self):
        # region2 归 player2，viewer=1 不应能从 region2 出兵
        mask = self.enc.mask(self.obs, commands_issued=0, max_commands=3)
        B = 4
        for edge_idx, (src, _) in enumerate(self.enc._edges):
            if src == 2:  # player2 的地区
                base = 1 + edge_idx * B
                self.assertFalse(mask[base:base + B].any(),
                                 f"edge ({src}→_) should not be valid for viewer=1")

    def test_own_src_in_mask(self):
        # region1 归 player1（20 兵 > 1）→ edge (1,2) 应合法
        mask = self.enc.mask(self.obs, commands_issued=0, max_commands=3)
        edges = self.enc._edges
        edge_idx = edges.index((1, 2))
        base = 1 + edge_idx * 4
        self.assertTrue(mask[base:base + 4].any())

    def test_decode_noop_returns_none(self):
        self.assertIsNone(self.enc.decode(0, player_id=1, game_map=self.gm))

    def test_decode_troops_25_percent(self):
        # region1 有 20 兵，available=19，bucket 0.25 → floor(19*0.25)=4
        edge_idx = self.enc._edges.index((1, 2))
        action = 1 + edge_idx * 4 + 0  # bucket_idx=0 → 25%
        cmd = self.enc.decode(action, player_id=1, game_map=self.gm)
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.source, 1)
        self.assertEqual(cmd.target, 2)
        self.assertEqual(cmd.troops, math.floor(19 * 0.25))

    def test_decode_troops_full(self):
        # bucket 1.0 → floor(19*1.0)=19
        edge_idx = self.enc._edges.index((1, 2))
        action = 1 + edge_idx * 4 + 3  # bucket_idx=3 → 100%
        cmd = self.enc.decode(action, player_id=1, game_map=self.gm)
        self.assertEqual(cmd.troops, 19)

    def test_decode_player_id_set(self):
        edge_idx = self.enc._edges.index((1, 2))
        action = 1 + edge_idx * 4 + 0
        cmd = self.enc.decode(action, player_id=1, game_map=self.gm)
        self.assertEqual(cmd.player, 1)


if __name__ == "__main__":
    unittest.main()
