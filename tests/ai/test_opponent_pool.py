"""ai/algos/opponent_pool.py 的单元测试。"""

import unittest

from ai.algos.opponent_pool import OpponentPool, PoolEntry


class TestPoolEntry(unittest.TestCase):

    def test_fields_default(self):
        e = PoolEntry(step=1000)
        self.assertEqual(e.step, 1000)
        self.assertIsNone(e.elo)

    def test_fields_with_elo(self):
        e = PoolEntry(step=2000, elo=1500.0)
        self.assertEqual(e.elo, 1500.0)


class TestOpponentPoolAdd(unittest.TestCase):

    def test_add_within_limit_no_eviction(self):
        pool = OpponentPool(max_size=3)
        evicted, _, accepted = pool.add(100)
        self.assertIsNone(evicted)
        self.assertTrue(accepted)
        self.assertEqual(len(pool), 1)

    def test_add_triggers_time_eviction(self):
        pool = OpponentPool(max_size=2, eviction_mode="time")
        pool.add(100)
        pool.add(200)
        evicted, _, _ = pool.add(300)
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.step, 100)
        self.assertEqual(len(pool), 2)

    def test_add_triggers_elo_eviction(self):
        pool = OpponentPool(max_size=2, eviction_mode="elo")
        pool.add(100, elo=1600.0)
        pool.add(200, elo=1200.0)
        evicted, _, _ = pool.add(300, elo=1400.0)
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.elo, 1200.0)

    def test_elo_none_treated_as_zero(self):
        pool = OpponentPool(max_size=2, eviction_mode="elo")
        pool.add(100, elo=500.0)
        pool.add(200)  # elo=None → 0
        evicted, _, _ = pool.add(300, elo=300.0)
        self.assertIsNone(evicted.elo)

    def test_unknown_eviction_mode_raises(self):
        with self.assertRaises(ValueError):
            OpponentPool(max_size=3, eviction_mode="fifo")

    def test_evicted_is_returned_entry_not_in_pool(self):
        pool = OpponentPool(max_size=1, eviction_mode="time")
        pool.add(100)
        evicted, _, _ = pool.add(200)
        self.assertEqual(evicted.step, 100)
        self.assertNotIn(100, pool)
        self.assertIn(200, pool)


class TestOpponentPoolQuery(unittest.TestCase):

    def setUp(self):
        self.pool = OpponentPool(max_size=5)
        self.pool.add(100)
        self.pool.add(300)
        self.pool.add(200)

    def test_latest_returns_max_step(self):
        e = self.pool.latest()
        self.assertIsNotNone(e)
        self.assertEqual(e.step, 300)

    def test_latest_returns_none_when_empty(self):
        pool = OpponentPool(max_size=3)
        self.assertIsNone(pool.latest())

    def test_get_by_step(self):
        e = self.pool.get(100)
        self.assertIsNotNone(e)
        self.assertEqual(e.step, 100)

    def test_get_missing_step(self):
        self.assertIsNone(self.pool.get(999))

    def test_contains(self):
        self.assertIn(100, self.pool)
        self.assertNotIn(999, self.pool)

    def test_len(self):
        self.assertEqual(len(self.pool), 3)
        self.pool.add(400)
        self.assertEqual(len(self.pool), 4)


class TestOpponentPoolSampling(unittest.TestCase):

    def setUp(self):
        self.pool = OpponentPool(max_size=10)
        self.pool.add(100)
        self.pool.add(300)
        self.pool.add(200)

    def test_uniform_returns_entry_from_pool(self):
        e = self.pool.sample_uniform()
        self.assertIsNotNone(e)
        self.assertIn(e.step, {100, 200, 300})

    def test_uniform_empty_returns_none(self):
        pool = OpponentPool(max_size=3)
        self.assertIsNone(pool.sample_uniform())

    def test_progress_returns_entry_from_pool(self):
        e = self.pool.sample_progress()
        self.assertIsNotNone(e)
        self.assertIn(e.step, {100, 200, 300})

    def test_progress_empty_returns_none(self):
        pool = OpponentPool(max_size=3)
        self.assertIsNone(pool.sample_progress())

    def test_progress_higher_step_more_likely(self):
        pool = OpponentPool(max_size=10)
        pool.add(100)
        pool.add(10000)
        counts = {100: 0, 10000: 0}
        for _ in range(200):
            e = pool.sample_progress(lam=1.0, s=100.0, D=1000.0)
            counts[e.step] += 1
        self.assertGreater(counts[10000], counts[100])

    def test_elo_returns_entry_from_pool(self):
        e = self.pool.sample_elo()
        self.assertIsNotNone(e)
        self.assertIn(e.step, {100, 200, 300})

    def test_elo_empty_returns_none(self):
        pool = OpponentPool(max_size=3)
        self.assertIsNone(pool.sample_elo())

    def test_elo_all_none_falls_back_to_uniform(self):
        e = self.pool.sample_elo()
        self.assertIsNotNone(e)

    def test_elo_higher_elo_more_likely(self):
        pool = OpponentPool(max_size=10)
        pool.add(100, elo=800.0)
        pool.add(200, elo=2000.0)
        counts = {100: 0, 200: 0}
        for _ in range(200):
            e = pool.sample_elo(lam=1.0, s=100.0)
            counts[e.step] += 1
        self.assertGreater(counts[200], counts[100])


class TestOpponentPoolEdgeCases(unittest.TestCase):

    def test_max_size_one(self):
        pool = OpponentPool(max_size=1)
        evicted, _, accepted = pool.add(100)
        self.assertIsNone(evicted)
        self.assertTrue(accepted)
        evicted, _, _ = pool.add(200)
        self.assertEqual(evicted.step, 100)
        self.assertEqual(len(pool), 1)

    def test_all_same_step_time_mode_evicts_first(self):
        pool = OpponentPool(max_size=2, eviction_mode="time")
        pool.add(100)
        pool.add(200)
        evicted, _, _ = pool.add(300)
        self.assertEqual(evicted.step, 100)

    def test_iter_yields_all_entries(self):
        pool = OpponentPool(max_size=5)
        pool.add(100)
        pool.add(200)
        self.assertEqual({e.step for e in pool}, {100, 200})


class TestOpponentPoolElo(unittest.TestCase):

    def test__update_elo_win(self):
        pool = OpponentPool(max_size=3)
        pool.add(100, elo=1200.0)

        new_agent, new_opp = pool._update_elo(100, agent_elo=1200.0, score=1.0)
        self.assertAlmostEqual(new_agent, 1200.0 + 32.0 * 0.5, places=1)
        self.assertAlmostEqual(new_opp, 1200.0 - 32.0 * 0.5, places=1)
        self.assertAlmostEqual(pool.get(100).elo, new_opp, places=1)

    def test__update_elo_opponent_not_in_pool(self):
        pool = OpponentPool(max_size=3)
        new_agent, new_opp = pool._update_elo(999, agent_elo=1400.0, score=1.0)
        expected = 1.0 / (1.0 + 10.0 ** ((1200.0 - 1400.0) / 400.0))
        self.assertAlmostEqual(new_agent, 1400.0 + 32.0 * (1.0 - expected), places=1)

    def test__update_elo_default_elo_when_none(self):
        pool = OpponentPool(max_size=3)
        pool.add(100)  # elo=None

        new_agent, new_opp = pool._update_elo(100, agent_elo=1200.0, score=0.0)
        self.assertAlmostEqual(new_agent, 1200.0 - 16.0, places=1)
        self.assertAlmostEqual(new_opp, 1200.0 + 16.0, places=1)


if __name__ == "__main__":
    unittest.main()
