from __future__ import annotations

import argparse
import os

from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.renders.utils import latest_model_dir, model_path, render_out_dir


def run_render(policy: SB3Policy, scenario: str, out_dir: str, num_episodes: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(num_episodes):
        env = LwgEnv(scenario)
        obs, _ = env.reset()
        step = 0
        while True:
            env.render(os.path.join(out_dir, f"ep{ep:02d}_turn_{step:04d}.png"))
            action = policy.predict(obs, env.action_masks())
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            if terminated or truncated:
                break

        env.render(os.path.join(out_dir, f"ep{ep:02d}_turn_{step:04d}_final.png"))
        winner = env._state.winner()
        outcome = (
            "agent wins" if winner == env.agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )
        print(f"ep {ep:02d} | {step} 回合 | {outcome}")

    print(f"渲染图像已保存至 {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染已训练的 LightWarGame 智能体对局")
    parser.add_argument("--scenario", type=str, default='1v1/vsbaseline',
                        help="env 配置名，如 two_players/vsbaseline")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="训练结果目录；不传则自动使用该 scenario 最新的 run")
    parser.add_argument("--episodes", type=int, default=1,
                        help="渲染局数（default: 1）")
    args = parser.parse_args()

    model_dir = args.model_dir or latest_model_dir(args.scenario)
    mp = model_path(model_dir)
    assert os.path.exists(mp + ".zip"), f"找不到模型：{mp}.zip，请先训练"

    out_dir = render_out_dir(args.scenario)
    run_render(SB3Policy(mp), args.scenario, out_dir, args.episodes)


if __name__ == "__main__":
    main()
