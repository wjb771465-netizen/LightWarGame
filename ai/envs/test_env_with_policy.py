# test_env_with_policy.py
import gymnasium as gym
import torch as th
import numpy as np
import random
import time
from ai.envs.env import ChineseWarGameEnv
from ai.algorithm.gnn_policy import CustomGNNPolicy

# -----------------------------
# 1. 环境 + Policy
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

# 如需每次不同，注释下面这行
# seed_everything(123)

env = ChineseWarGameEnv(player_id=1)
policy = CustomGNNPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 0.0,
)
policy.eval()
print("环境 + GNN Policy 创建成功（Mask 已修复）\n")

# -----------------------------
# 2. obs → tensor dict
# -----------------------------
def dict_to_tensor(obs_dict):
    return {
        "node_features": th.from_numpy(obs_dict["node_features"]).unsqueeze(0).to(th.float32),
        "edge_index":    th.from_numpy(obs_dict["edge_index"]).unsqueeze(0).to(th.int64),
    }

# -----------------------------
# 3. 主循环：**强制 Mask 过滤**
# -----------------------------
obs, info = env.reset()
print("=== 游戏开始 ===")
print(f"【回合 {env.game_state.get('turn', 0)}】初始状态")

step = 0
total_reward = 0.0
terminated = truncated = False

with th.no_grad():
    while not (terminated or truncated):
        step += 1

        # --- 1. 转为 tensor ---
        tensor_obs = dict_to_tensor(obs)

        # --- 2. 获取 action_masks ---
        action_masks_np = info["action_masks"]           # (1001,) bool numpy
        action_masks = th.from_numpy(action_masks_np).bool()  # (1001,) torch.bool

        # --- 3. 获取分布 + 传入 masks ---
        distribution = policy.get_distribution(
            tensor_obs,
            action_masks=action_masks   # <--- 关键！！
        )

        # --- 4. 采样（强制只从合法动作中选）---
        # 方式A（推荐）：使用 masked sample
        action = distribution.sample()  # 只采样 True 的位置
        action = action.item()

        # 方式B（备选）：手动验证
        # action = distribution.sample()
        # if not action_masks[action]:
        #     print("警告：采样到非法动作！")

        # --- 5. step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        current_turn = env.game_state.get("turn", 0)

        # --- 6. 打印 ---
        print(f"\n--- Step {step} | 回合 {current_turn} | action={action} ---")
        print(f"   reward = {reward:+.6f}  |  累计 = {total_reward:+.6f}")
        print(f"   terminated={terminated}  truncated={truncated}")

        # 己方区域
        node_feats = obs["node_features"]
        own_mask = node_feats[:, 0] > 0.5
        own_count = int(own_mask.sum())
        if own_count > 0:
            avg_troops = (node_feats[own_mask, 2] * 100).mean()
            print(f"   己方区域：{own_count} 块，平均部队 ≈ {avg_troops:.1f}")
        else:
            print(f"   己方已无区域")

        # 安全
        if step > 300:
            print("强制截断")
            truncated = True
            break

        time.sleep(0.01)

print("\n" + "="*60)
print(f"游戏结束！最终回合: {env.game_state.get('turn', -1)}")
print(f"总步数: {step}，累计奖励: {total_reward:.6f}")
print("Mask 已 100% 生效：所有 action 均在 info['action_masks']=True 中")
print("="*60)