"""ai/algos/opponent_pool.py 的单元测试。"""

import unittest

from ai.algos.opponent_pool import OpponentPool, PoolEntry


class TestPoolEntry(unittest.TestCase):

    def test_fields_default(self):
        e = PoolEntry(path="/tmp/ckpt.zip", step=1000)
        self.assertEqual(e.path, "/tmp/ckpt.zip")
        self.assertEqual(e.step, 1000)
        self.assertIsNone(e.elo)

    def test_fields_with_elo(self):
        e = PoolEntry(path="/tmp/ckpt.zip", step=2000, elo=1500.0)
        self.assertEqual(e.elo, 1500.0)


class TestOpponentPoolAdd(unittest.TestCase):

    def test_add_within_limit_no_eviction(self):
        pool = OpponentPool(max_size=3)
        evicted = pool.add("/tmp/a.zip", 100)
        self.assertIsNone(evicted)
        self.assertEqual(len(pool), 1)

    def test_add_triggers_time_eviction(self):
        pool = OpponentPool(max_size=2, eviction_mode="time")
        pool.add("/tmp/old.zip", 100)
        pool.add("/tmp/mid.zip", 200)
        evicted = pool.add("/tmp/new.zip", 300)
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.step, 100)  # step 最小 = 最旧
        self.assertEqual(evicted.path, "/tmp/old.zip")
        self.assertEqual(len(pool), 2)

    def test_add_triggers_elo_eviction(self):
        pool = OpponentPool(max_size=2, eviction_mode="elo")
        pool.add("/tmp/strong.zip", 100, elo=1600.0)
        pool.add("/tmp/weak.zip", 200, elo=1200.0)
        evicted = pool.add("/tmp/mid.zip", 300, elo=1400.0)
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.elo, 1200.0)  # elo 最小
        self.assertEqual(evicted.path, "/tmp/weak.zip")

    def test_elo_none_treated_as_zero(self):
        pool = OpponentPool(max_size=2, eviction_mode="elo")
        pool.add("/tmp/a.zip", 100, elo=500.0)
        pool.add("/tmp/b.zip", 200)  # elo=None → 0
        evicted = pool.add("/tmp/c.zip", 300, elo=300.0)
        self.assertEqual(evicted.path, "/tmp/b.zip")  # None→0 < 500

    def test_unknown_eviction_mode_raises(self):
        with self.assertRaises(ValueError):
            OpponentPool(max_size=3, eviction_mode="fifo")

    def test_evicted_is_returned_entry_not_in_pool(self):
        pool = OpponentPool(max_size=1, eviction_mode="time")
        pool.add("/tmp/first.zip", 100)
        evicted = pool.add("/tmp/second.zip", 200)
        self.assertEqual(evicted.path, "/tmp/first.zip")
        self.assertNotIn("/tmp/first.zip", pool)
        self.assertIn("/tmp/second.zip", pool)


class TestOpponentPoolQuery(unittest.TestCase):

    def setUp(self):
        self.pool = OpponentPool(max_size=5)
        self.pool.add("/tmp/s100.zip", 100)
        self.pool.add("/tmp/s300.zip", 300)
        self.pool.add("/tmp/s200.zip", 200)

    def test_latest_returns_max_step(self):
        e = self.pool.latest()
        self.assertIsNotNone(e)
        self.assertEqual(e.step, 300)
        self.assertEqual(e.path, "/tmp/s300.zip")

    def test_latest_returns_none_when_empty(self):
        pool = OpponentPool(max_size=3)
        self.assertIsNone(pool.latest())

    def test_sample_valid_index(self):
        e = self.pool.sample(0)
        self.assertIsNotNone(e)
        self.assertEqual(e.step, 100)

    def test_sample_out_of_range(self):
        self.assertIsNone(self.pool.sample(-1))
        self.assertIsNone(self.pool.sample(99))

    def test_contains(self):
        self.assertIn("/tmp/s100.zip", self.pool)
        self.assertNotIn("/tmp/ghost.zip", self.pool)

    def test_len(self):
        self.assertEqual(len(self.pool), 3)
        self.pool.add("/tmp/s400.zip", 400)
        self.assertEqual(len(self.pool), 4)


class TestOpponentPoolEdgeCases(unittest.TestCase):

    def test_max_size_one(self):
        pool = OpponentPool(max_size=1)
        self.assertIsNone(pool.add("/tmp/a.zip", 100))
        evicted = pool.add("/tmp/b.zip", 200)
        self.assertEqual(evicted.path, "/tmp/a.zip")
        self.assertEqual(len(pool), 1)

    def test_all_same_step_time_mode_evicts_first(self):
        """step 相同时 min() 返回第一个匹配项（稳定行为）。"""
        pool = OpponentPool(max_size=2, eviction_mode="time")
        pool.add("/tmp/a.zip", 100)
        pool.add("/tmp/b.zip", 100)
        evicted = pool.add("/tmp/c.zip", 100)
        self.assertEqual(evicted.path, "/tmp/a.zip")


if __name__ == "__main__":
    unittest.main()
