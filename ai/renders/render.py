from __future__ import annotations

import argparse
import os
from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.renders.utils import latest_model_dir, make_video, model_path, render_out_dir


def _render_episode(policy: SB3Policy, scenario: str, ep: int, out_dir: str) -> tuple[int, int | None]:
    png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
    os.makedirs(png_dir, exist_ok=True)

    env = LwgEnv(scenario)
    obs, _ = env.reset()
    step = 0
    while True:
        env.render(os.path.join(png_dir, f"turn_{step:04d}.png"))
        action = policy.predict(obs, env.action_masks())
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break

    env.render(os.path.join(png_dir, f"turn_{step:04d}_final.png"))
    return step, env._state.winner(), env.agent_id



def run_render(policy: SB3Policy, scenario: str, out_dir: str, num_episodes: int, fps: int = 2) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(num_episodes):
        step, winner, agent_id = _render_episode(policy, scenario, ep, out_dir)

        png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
        video_path = os.path.join(out_dir, f"ep{ep:02d}", f"ep{ep:02d}.mp4")
        if fps > 0: 
            make_video(png_dir, video_path, fps) 

        outcome = (
            "agent wins" if winner == agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )
        print(f"ep {ep:02d} | {step} 回合 | {outcome} → {video_path}")

    print(f"渲染结果已保存至 {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染已训练的 LightWarGame 智能体对局")
    parser.add_argument("--scenario", type=str, default="1v1/vsbaseline",
                        help="env 配置名，如 1v1/vsbaseline")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="训练结果目录；不传则自动使用该 scenario 最新的 run")
    parser.add_argument("--episodes", type=int, default=1,
                        help="渲染局数（default: 1）")
    parser.add_argument("--fps", type=int, default=2,
                        help="视频帧率（default: 2）")
    args = parser.parse_args()

    model_dir = args.model_dir or latest_model_dir(args.scenario)
    mp = model_path(model_dir)
    assert os.path.exists(mp + ".zip"), f"找不到模型：{mp}.zip，请先训练"

    out_dir = render_out_dir(args.scenario)
    run_render(SB3Policy(mp), args.scenario, out_dir, args.episodes, args.fps)


if __name__ == "__main__":
    main()
