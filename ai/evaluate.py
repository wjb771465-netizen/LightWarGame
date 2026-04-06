#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估训练好的 PPO 模型 vs 随机策略
- 100 局对战
- 统计胜率（基于总奖励 > 0 判定为胜）
- 打印每局总奖励 + 最终统计
- 单文件，零依赖，复制即用
"""

import os
import numpy as np
from sb3_contrib import MaskablePPO
from ai.envs.env import ChineseWarGameEnv

# --------------------------- 配置 ---------------------------
MODEL_PATH = "./models/war_1v1_ai_gnn.zip"  # ← 修改为你的模型路径
N_GAMES = 100  # 对战局数
SEED = 42


# -----------------------------------------------------------

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件未找到: {path}")
    return MaskablePPO.load(path, device="cpu")


def play_one_game(model, player_id=1):
    """返回 (AI 总奖励, 是否获胜)"""
    env = ChineseWarGameEnv(player_id=player_id)
    obs, info = env.reset(seed=SEED + hash(player_id))

    total_reward = 0.0
    done = False

    while not done:
        # AI 动作
        if env.player_id == player_id:
            # 使用 env 公开的 info（reset/step 都返回）
            action_masks = info.get("action_mask")
            if action_masks is None:
                # 兼容旧版 env（若没有 mask 则全为 1）
                action_masks = np.ones(env.action_space.n, dtype=np.float32)
            action, _ = model.predict(obs, action_masks=action_masks)
        else:
            # 随机策略：从合法动作中随机选
            mask = env._get_info()["action_mask"]
            valid = np.where(mask > 0)[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0

        obs, reward, done, _, info = env.step(int(action))
        total_reward += reward

    # 胜负判定：总奖励 > 0 视为胜利（因为胜利奖励 +100，失败 -100）
    win = total_reward > 0
    return total_reward, win


def main():
    print(f"加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"开始 {N_GAMES} 局对战（AI vs 随机）...")
    wins = 0
    total_rewards = []

    for i in range(1, N_GAMES + 1):
        reward, win = play_one_game(model, player_id=1)
        wins += int(win)
        total_rewards.append(reward)
        print(f"第 {i:3d} 局: {'胜' if win else '负'} | 总奖励: {reward:+6.1f} | 当前胜率: {wins / i:.1%}")

    win_rate = wins / N_GAMES
    avg_reward = np.mean(total_rewards)

    print("\n" + "=" * 60)
    print(f"评估完成！")
    print(f"总对战: {N_GAMES} 局")
    print(f"AI 获胜: {wins} 局")
    print(f"胜率: {win_rate:6.1%}")
    print(f"平均奖励: {avg_reward:+6.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


def main():
    print(f"加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"开始 {N_GAMES} 局对战（AI vs 随机）...")
    wins = 0
    for i in range(1, N_GAMES + 1):
        win = play_one_game(model, player_id=1)
        wins += int(win)
        print(f"第 {i:3d} 局: {'胜' if win else '负'} | 当前胜率: {wins / i:.1%}")

    win_rate = wins / N_GAMES
    print("\n" + "=" * 50)
    print(f"评估完成！")
    print(f"总对战: {N_GAMES} 局")
    print(f"AI 获胜: {wins} 局")
    print(f"胜率: {win_rate:.1%}")
    print("=" * 50)


if __name__ == "__main__":
    main()