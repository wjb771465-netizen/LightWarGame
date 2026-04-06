# train.py (GNN version)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from ai.envs.env import ChineseWarGameEnv
from ai.algorithm.gnn_policy import CustomGNNPolicy


# === 1. 定义 mask 提取函数 ===
def mask_fn(env):
    """从 info 中提取 action_masks"""
    return env._get_info()["action_masks"]

print("创建 GNN 环境...")
env = ChineseWarGameEnv(player_id=1)


# 包装环境
env = ActionMasker(env, mask_fn)
print("环境创建完成。")

model = MaskablePPO(
    CustomGNNPolicy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="./logs/",
)

print("开始训练...")
model.learn(total_timesteps=1_000_000)
model.save("./models/war_1v1_ai_gnn")
print("训练完成。")
