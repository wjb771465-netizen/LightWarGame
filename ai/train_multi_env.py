# train_multi_env.py (GNN 多环境并行训练 - 完全修复版)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ai.envs.env import ChineseWarGameEnv
from ai.algorithm.gnn_policy import CustomGNNPolicy

import os
from typing import Callable


# === 1. 全局 mask 函数（必须是顶层函数，可 pickle）===
def get_action_mask(env):
    """从 env 的 info 中提取 action_masks"""
    return env._get_info()["action_masks"]


# === 2. 环境构造器（每个子进程独立创建）===
def make_env(player_id: int, rank: int) -> Callable[[], ChineseWarGameEnv]:
    def _init():
        env = ChineseWarGameEnv(player_id=player_id)
        # 正确传入 mask 函数（位置参数，无 mask_fn=）
        env = ActionMasker(env, get_action_mask)
        return env
    return _init


# === 3. 主训练逻辑（必须放在 if __name__ == '__main__':）===
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Windows 打包 exe 时需要，普通运行可保留

    # === 配置并行环境数 ===
    num_cpu = os.cpu_count() or 4
    n_envs = min(8, num_cpu)
    print(f"使用 {n_envs} 个并行环境训练")

    # === 创建向量化环境 ===
    vec_env = SubprocVecEnv([
        make_env(player_id=1, rank=i) for i in range(n_envs)
    ])
    # 调试时可切换为 DummyVecEnv（单进程）
    #vec_env = DummyVecEnv([make_env(player_id=1, rank=i) for i in range(n_envs)])

    print("多环境创建完成。")

    # === 初始化 MaskablePPO + GNN 策略 ===
    model = MaskablePPO(
        policy=CustomGNNPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,   # 保持总 rollout ≈ 2048
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs_multi/",
        device="auto"
    )

    print("开始多环境并行训练...")

    # === 训练 ===
    model.learn(
        total_timesteps=1_000_000,
        log_interval=4
    )

    # === 保存模型 ===
    model.save("./models/war_1v1_ai_gnn_multi")
    vec_env.close()

    print("训练完成！模型已保存至 ./models/war_1v1_ai_gnn_multi")