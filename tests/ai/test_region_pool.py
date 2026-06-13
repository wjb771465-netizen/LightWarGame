"""ai/algos/region_pool.py 与 LwgEnv.set_capitals() 的单元测试。"""

import json
import os
import tempfile
import unittest

from ai.algos.region_pool import RegionPool
from ai.envs.env import LwgEnv

CONFIG = "duel/region_selfplay"


class TestRegionPoolAdd(unittest.TestCase):

    def test_add_and_available_regions(self):
        pool = RegionPool(history=3)
        pool.add(1, "/tmp/r1.zip", 100)
        pool.add(2, "/tmp/r2.zip", 200)
        self.assertEqual(pool.available_regions(), [1, 2])

    def test_available_regions_empty(self):
        self.assertEqual(RegionPool(history=3).available_regions(), [])

    def test_latest_returns_max_step(self):
        pool = RegionPool(history=3)
        pool.add(1, "/tmp/s100.zip", 100)
        pool.add(1, "/tmp/s200.zip", 200)
        self.assertEqual(pool.latest(1).step, 200)

    def test_latest_none_unknown_region(self):
        self.assertIsNone(RegionPool(history=3).latest(99))

    def test_eviction_respects_history(self):
        """history=2 时第 3 条触发淘汰，最旧的 step=100 被移除。"""
        pool = RegionPool(history=2)
        pool.add(1, "/tmp/s100.zip", 100)
        pool.add(1, "/tmp/s200.zip", 200)
        pool.add(1, "/tmp/s300.zip", 300)
        self.assertEqual(pool.latest(1).step, 300)
        # 只剩 2 条，step=100 已淘汰
        self.assertIsNone(pool.latest(99))  # 其他地区无记录，间接确认结构正常


class TestRegionPoolSampleOpponent(unittest.TestCase):

    def test_sample_excludes_given_region(self):
        pool = RegionPool(history=3)
        pool.add(1, "/tmp/r1.zip", 100)
        pool.add(2, "/tmp/r2.zip", 200)
        rid, _ = pool.sample_opponent(exclude_region=1)
        self.assertEqual(rid, 2)

    def test_sample_none_only_excluded_region(self):
        pool = RegionPool(history=3)
        pool.add(1, "/tmp/r1.zip", 100)
        self.assertIsNone(pool.sample_opponent(exclude_region=1))

    def test_sample_none_empty_pool(self):
        self.assertIsNone(RegionPool(history=3).sample_opponent(exclude_region=1))

    def test_sample_returns_latest_entry(self):
        """抽到的是该地区最新的 checkpoint（latest）。"""
        pool = RegionPool(history=3)
        pool.add(2, "/tmp/r2_old.zip", 100)
        pool.add(2, "/tmp/r2_new.zip", 500)
        rid, entry = pool.sample_opponent(exclude_region=1)
        self.assertEqual(rid, 2)
        self.assertEqual(entry.step, 500)

    def test_sample_covers_multiple_regions(self):
        """多地区时随机抽样能覆盖到不同地区。"""
        pool = RegionPool(history=3)
        for r in range(2, 8):
            pool.add(r, f"/tmp/r{r}.zip", r * 100)
        seen = set()
        for _ in range(200):
            rid, _ = pool.sample_opponent(exclude_region=1)
            seen.add(rid)
        self.assertGreater(len(seen), 1)

    def test_sample_strategy_uniform(self):
        """strategy=uniform 时可能返回非最新 entry。"""
        pool = RegionPool(history=3)
        pool.add(2, "/tmp/r2_old.zip", 100)
        pool.add(2, "/tmp/r2_new.zip", 500)
        seen_steps = set()
        for _ in range(100):
            _, entry = pool.sample_opponent(exclude_region=1, strategy="uniform")
            seen_steps.add(entry.step)
        self.assertIn(100, seen_steps)

    def test_sample_strategy_progress(self):
        """strategy=progress 时不崩溃。"""
        pool = RegionPool(history=3)
        pool.add(2, "/tmp/r2.zip", 200)
        rid, entry = pool.sample_opponent(exclude_region=1, strategy="progress")
        self.assertEqual(rid, 2)
        self.assertIsNotNone(entry)

    def test_sample_strategy_elo_falls_back(self):
        """strategy=elo 且 ELO=None 时不崩溃（退化 uniform）。"""
        pool = RegionPool(history=3)
        pool.add(2, "/tmp/r2.zip", 200)
        rid, entry = pool.sample_opponent(exclude_region=1, strategy="elo")
        self.assertEqual(rid, 2)
        self.assertIsNotNone(entry)


class TestRegionPoolSaveLoad(unittest.TestCase):

    def test_roundtrip(self):
        pool = RegionPool(history=3)
        pool.add(4, "/tmp/r4_100.zip", 100)
        pool.add(4, "/tmp/r4_200.zip", 200)
        pool.add(20, "/tmp/r20_150.zip", 150)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pool.json")
            pool.save(path)
            loaded = RegionPool.load(path)

        self.assertEqual(loaded.available_regions(), [4, 20])
        self.assertEqual(loaded.latest(4).step, 200)
        self.assertEqual(loaded.latest(20).step, 150)

    def test_save_is_valid_json(self):
        pool = RegionPool(history=2)
        pool.add(1, "/tmp/a.zip", 100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pool.json")
            pool.save(path)
            with open(path) as f:
                data = json.load(f)
        self.assertIn("history", data)
        self.assertIn("regions", data)
        self.assertIn("1", data["regions"])

    def test_atomic_write_leaves_no_tmp(self):
        pool = RegionPool(history=2)
        pool.add(1, "/tmp/a.zip", 100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pool.json")
            pool.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))


class TestLwgEnvSetCapitals(unittest.TestCase):

    def setUp(self):
        self.env = LwgEnv(CONFIG)

    def test_set_capitals_updates_config(self):
        self.env.set_capitals(5, 20)
        self.assertEqual(self.env.config.game.capitals, [5, 20])
        self.assertEqual(self.env.config.game.capital_mode, "fixed")

    def test_set_capitals_agent_at_correct_index(self):
        """agent_id=1 时首都列表 index 0 是 agent，index 1 是对手。"""
        self.env.set_capitals(7, 15)
        self.assertEqual(self.env.config.game.capitals[0], 7)
        self.assertEqual(self.env.config.game.capitals[1], 15)

    def test_set_capitals_reflected_after_reset(self):
        self.env.set_capitals(4, 20)
        self.env.reset()
        regions = self.env._state.game_map.regions
        capital_owners = [r.owner for r in regions[1:] if r is not None and r.is_capital]
        self.assertIn(1, capital_owners)  # agent(player 1) 有首都
        self.assertIn(2, capital_owners)  # 对手(player 2) 有首都

    def test_set_capitals_can_change_between_resets(self):
        self.env.set_capitals(4, 20)
        self.env.reset()
        self.env.set_capitals(10, 25)
        self.env.reset()
        self.assertEqual(self.env.config.game.capitals, [10, 25])


if __name__ == "__main__":
    unittest.main()
