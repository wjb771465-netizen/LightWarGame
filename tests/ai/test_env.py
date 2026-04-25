"""
ai/envs/env.py 的冒烟测试。

验证：obs 维度、action mask、随机 rollout 不崩溃、截断逻辑。
"""

import tempfile
import os
import unittest

import numpy as np

from ai.envs.env import LwgEnv

CONFIG = "1v1/vsbaseline"


class TestLwgEnvInit(unittest.TestCase):

    def setUp(self):
        self.env = LwgEnv(CONFIG)

    def test_observation_space_shape(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_obs_dtype_float32(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_in_space(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_action_space_size(self):
        # action_dim = E*B + 1，cn 地图约 145 条有向边，B=4
        self.assertGreater(self.env.action_space.n, 1)

    def test_reset_returns_info_dict(self):
        _, info = self.env.reset()
        self.assertIsInstance(info, dict)


class TestLwgEnvActionMasks(unittest.TestCase):

    def setUp(self):
        self.env = LwgEnv(CONFIG)
        self.env.reset()

    def test_mask_length_equals_action_space(self):
        mask = self.env.action_masks()
        self.assertEqual(len(mask), self.env.action_space.n)

    def test_noop_always_valid(self):
        self.assertTrue(self.env.action_masks()[0])

    def test_mask_is_bool(self):
        self.assertEqual(self.env.action_masks().dtype, bool)


class TestLwgEnvStep(unittest.TestCase):

    def setUp(self):
        self.env = LwgEnv(CONFIG)
        self.env.reset()

    def test_noop_step_returns_correct_shapes(self):
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_truncated_at_max_turns(self):
        env = LwgEnv(CONFIG)
        env.reset()
        env.config.game.max_turns = 3
        truncated = False
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        self.assertTrue(truncated or terminated)


class TestLwgEnvRender(unittest.TestCase):

    def test_render_saves_png(self):
        env = LwgEnv(CONFIG)
        env.reset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "turn.png")
            env.render(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


class TestLwgEnvRollout(unittest.TestCase):

    def test_random_rollout_100_episodes(self):
        """随机合法动作跑 100 局不崩溃，每局必须结束。"""
        env = LwgEnv(CONFIG)
        for _ in range(100):
            env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action = int(np.random.choice(np.where(mask)[0]))
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        # 能到这里说明没有异常
        self.assertTrue(True)

    def test_obs_always_in_space_during_rollout(self):
        """每步 obs 都在合法空间内。"""
        env = LwgEnv(CONFIG)
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        for _ in range(30):
            mask = env.action_masks()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, _, terminated, truncated, _ = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()
