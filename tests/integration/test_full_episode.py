"""
集成测试：完整对局 + 地图渲染。

默认跳过，需手动触发：
    RUN_INTEGRATION=1 conda run -n chinese_war_game python -m unittest tests.integration.test_full_episode -v

渲染产物保存到 saves/integration/，供肉眼检查。
"""

import os
import random
import unittest

import numpy as np

from ai.envs.env import LwgEnv

_RUN = os.getenv("RUN_INTEGRATION")
_OUT_DIR = "saves/integration"
CONFIG = "two_players/vsbaseline"


@unittest.skipUnless(_RUN, "set RUN_INTEGRATION=1 to run")
class TestFullEpisode(unittest.TestCase):

    def test_full_episode_with_renders(self):
        """完整跑一局，每 10 回合 + 终局各存一张地图 PNG，打印对局摘要。"""
        random.seed(42)
        np.random.seed(42)
        os.makedirs(_OUT_DIR, exist_ok=True)

        env = LwgEnv(CONFIG)
        obs, _ = env.reset()

        total_reward = 0.0
        step = 0

        while True:
            # 每 10 步渲染一次
            if step % 10 == 0:
                env.render(os.path.join(_OUT_DIR, f"turn_{step:04d}.png"))

            mask = env.action_masks()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                break

        # 终局渲染
        env.render(os.path.join(_OUT_DIR, f"turn_{step:04d}_final.png"))

        winner = env._state.winner()
        outcome = (
            "agent wins" if winner == env.agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )

        print(f"\n--- 对局摘要 ---")
        print(f"  回合数:    {step}")
        print(f"  结果:      {outcome}")
        print(f"  总奖励:    {total_reward:.1f}")
        print(f"  渲染目录:  {os.path.abspath(_OUT_DIR)}")

        # 结构性断言（不断言具体值，仅验证流程完整）
        self.assertGreater(step, 0)
        self.assertTrue(terminated or truncated)
        self.assertTrue(os.path.exists(os.path.join(_OUT_DIR, f"turn_{step:04d}_final.png")))
