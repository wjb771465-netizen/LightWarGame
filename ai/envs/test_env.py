# test_env.py
import gymnasium as gym
import numpy as np
import random
import time
from ai.envs.env import ChineseWarGameEnv          # <-- 你的环境
from game.fog_of_war import create_fog_view_for_player
from game.display import display_fog_game_state


def seed_everything(seed: int = 42):
    """固定随机种子（可注释掉让每次不同）"""
    random.seed(seed)
    np.random.seed(seed)


def play_one_episode(env: gym.Env, episode_idx: int) -> dict:
    """
    玩一局完整游戏，返回统计信息
    """
    obs, info = env.reset()
    print(f"\n{'='*20} 第 {episode_idx+1} 局 开始 {'='*20}")

    step = 0
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        step += 1

        # ---------- 随机合法动作 ----------
        mask = info["action_masks"]

        valid_actions = np.flatnonzero(mask)
        action = int(np.random.choice(valid_actions))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        current_turn = env.game_state.get("turn", 0)

        # 打印关键信息（可自行调低频率）
        print(f"Step {step:3d} | 回合 {current_turn:3d} | action={action:3d} | "
              f"reward={reward:+.4f} | 累计={total_reward:+.4f}")

        # 己方区域统计（可视化）
        troops = obs["node_features"][:, 2] * 100
        own = obs["node_features"][:, 0]
        own_count = int(own.sum())
        if own_count > 0:
            avg_troops = troops[own > 0.5].mean()
            print(f"  -> 己方区域：{own_count} 块，平均部队 ≈ {avg_troops:.1f}")
        else:
            print("  -> 己方已无区域")

#        # 防死循环
#        if step > 300:
#            print(">>> 强制截断（step > 300）")
#            truncated = True
#           break

        #time.sleep(0.01)   # 控制台不刷太快

    # ---------- 局结束统计 ----------
    final_turn = env.game_state.get('turn', -1)
    own_regions = int(obs["node_features"][:, 0].sum())
    stats = {
        "episode": episode_idx + 1,
        "steps": step,
        "total_reward": total_reward,
        "final_turn": final_turn,
        "own_regions": own_regions,
    }

    print("\n" + "="*60)
    print(f"第 {episode_idx+1} 局结束！")
    print(f"  总步数: {step}，累计奖励: {total_reward:+.4f}")
    print(f"  最终回合: {final_turn}，己方剩余区域: {own_regions}")
    print("="*60)

    # 可视化最终战局（雾视图）
    fog_view = create_fog_view_for_player(env.game_state, 1)
    #display_fog_game_state(fog_view)

    return stats


def main():
    # 如需每次运行结果不同，请注释下面这行
    # seed_everything(123)

    env: gym.Env = ChineseWarGameEnv(player_id=1)
    print("环境创建成功")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)

    all_stats = []
    N_GAMES = 120

    for i in range(N_GAMES):
        stats = play_one_episode(env, i)
        all_stats.append(stats)
        # 每局之间稍作停顿，防止画面叠加
        time.sleep(1)

    # ---------- 5 局汇总 ----------
    print("\n\n" + "#" * 70)
    print("5 局测试汇总")
    print("#" * 70)
    for s in all_stats:
        print(f"第 {s['episode']} 局 | 步数 {s['steps']:3d} | 奖励 {s['total_reward']:+.2f} | "
              f"回合 {s['final_turn']:3d} | 己方区域 {s['own_regions']:2d}")
    # 计算平均
    avg_reward = np.mean([s["total_reward"] for s in all_stats])
    avg_steps  = np.mean([s["steps"] for s in all_stats])
    print(f"\n平均奖励: {avg_reward:+.2f}  | 平均步数: {avg_steps:.1f}")
    print("#" * 70)


if __name__ == "__main__":
    main()