from __future__ import annotations

import argparse
import logging
import os
from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.renders.utils import latest_model_dir, make_video, resolve_model_path, render_out_dir


def _render_episode(policy: SB3Policy, env: LwgEnv, ep: int,
                    out_dir: str) -> tuple[int, int | None, int]:
    """渲染一局对战为 PNG 帧序列。env 须已由调用方配置好 capital/opponent/max_turns。"""
    png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
    os.makedirs(png_dir, exist_ok=True)

    obs, _ = env.reset()

    turn = 0
    env.render(os.path.join(png_dir, f"turn_{turn:04d}.png"))
    prev_episode_steps = env._episode_steps

    while True:
        action = policy.predict(obs, env.action_masks())
        obs, _, terminated, truncated, _ = env.step(action)

        if env._episode_steps > prev_episode_steps or terminated or truncated:
            turn += 1
            env.render(os.path.join(png_dir, f"turn_{turn:04d}.png"))
            prev_episode_steps = env._episode_steps

        if terminated or truncated:
            break

    return turn, env._state.winner(), env.agent_id



def render(policy: SB3Policy, env: LwgEnv, out_dir: str,
           num_episodes: int = 1, fps: int = 2) -> list[str]:
    """渲染多局对战为视频。env 须已由调用方配置好 capital/opponent/max_turns。"""
    os.makedirs(out_dir, exist_ok=True)
    videos: list[str] = []
    for ep in range(num_episodes):
        step, winner, agent_id = _render_episode(policy, env, ep, out_dir)

        png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
        video_path = os.path.join(out_dir, f"ep{ep:02d}", f"ep{ep:02d}.mp4")
        if fps > 0:
            make_video(png_dir, video_path, fps)
        videos.append(video_path)

        outcome = (
            "agent wins" if winner == agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )
        logging.info("ep %02d | %d 回合 | %s → %s", ep, step, outcome, video_path)

    logging.info("渲染结果已保存至 %s", out_dir)
    return videos


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="渲染已训练的 LightWarGame 智能体对局")
    parser.add_argument("--scenario", type=str, default="duel/vsbaseline",
                        help="env 配置名，如 duel/vsbaseline")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="训练结果目录；不传则自动使用该 scenario 最新的 run")
    parser.add_argument("--episodes", type=int, default=1,
                        help="渲染局数（default: 1）")
    parser.add_argument("--fps", type=int, default=2,
                        help="视频帧率（default: 2）")
    parser.add_argument("--max-turns", type=int, default=60,
                        help="最大回合数（default: 60）")
    parser.add_argument("--agent-capital", type=int, default=None,
                        help="AI 首都省份 ID（覆盖场景配置）")
    parser.add_argument("--opponent-capital", type=int, default=None,
                        help="对手首都省份 ID（覆盖场景配置）")
    args = parser.parse_args()

    raw = args.model_dir or latest_model_dir(args.scenario)
    mp = resolve_model_path(raw)

    if isinstance(mp, tuple):
        agent_path, opp_path = mp
    else:
        agent_path, opp_path = mp, None

    agent_policy = SB3Policy(path=agent_path)

    env = LwgEnv(args.scenario)
    if args.max_turns is not None:
        env.config.game.max_turns = args.max_turns
    if args.agent_capital is not None and args.opponent_capital is not None:
        env.set_capitals(args.agent_capital, args.opponent_capital)

    opp_policy = SB3Policy(path=opp_path) if opp_path else None
    if opp_policy is not None:
        from ai.envs.opponents import PolicyOpponent
        env.opponent = PolicyOpponent(
            player_id=2, policy=opp_policy,
            obs_encoder=env.obs_encoder, act_encoder=env.act_encoder,
        )

    out_dir = render_out_dir(args.scenario)
    render(agent_policy, env, out_dir, args.episodes, args.fps)


if __name__ == "__main__":
    main()
