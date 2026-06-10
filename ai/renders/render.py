from __future__ import annotations

import argparse
import logging
import os
from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.renders.utils import latest_model_dir, make_video, resolve_model_path, render_out_dir


def _render_episode(policy: SB3Policy, scenario: str, ep: int, out_dir: str,
                    max_turns: int | None = None,
                    agent_capital: int | None = None,
                    opponent_capital: int | None = None,
                    opponent_policy: SB3Policy | None = None) -> tuple[int, int | None]:
    png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
    os.makedirs(png_dir, exist_ok=True)

    env = LwgEnv(scenario)
    if max_turns is not None:
        env.config.game.max_turns = max_turns
    if agent_capital is not None and opponent_capital is not None:
        env.set_capitals(agent_capital, opponent_capital)
    if opponent_policy is not None:
        from ai.envs.opponents import PolicyOpponent
        env.opponent = PolicyOpponent(
            player_id=2, policy=opponent_policy,
            obs_encoder=env.obs_encoder, act_encoder=env.act_encoder,
        )
    obs, _ = env.reset()

    # 渲染初始状态（第 0 回合）
    turn = 0
    env.render(os.path.join(png_dir, f"turn_{turn:04d}.png"))
    prev_episode_steps = env._episode_steps

    while True:
        action = policy.predict(obs, env.action_masks())
        obs, _, terminated, truncated, _ = env.step(action)

        # 回合结算后或游戏结束时才渲染帧
        if env._episode_steps > prev_episode_steps or terminated or truncated:
            turn += 1
            env.render(os.path.join(png_dir, f"turn_{turn:04d}.png"))
            prev_episode_steps = env._episode_steps

        if terminated or truncated:
            break

    return turn, env._state.winner(), env.agent_id



def run_render(policy: SB3Policy, scenario: str, out_dir: str, num_episodes: int, fps: int = 2,
               max_turns: int | None = None,
               agent_capital: int | None = None,
               opponent_capital: int | None = None,
               opponent_policy: SB3Policy | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(num_episodes):
        step, winner, agent_id = _render_episode(policy, scenario, ep, out_dir,
                                                 max_turns, agent_capital, opponent_capital,
                                                 opponent_policy)

        png_dir = os.path.join(out_dir, f"ep{ep:02d}", "png")
        video_path = os.path.join(out_dir, f"ep{ep:02d}", f"ep{ep:02d}.mp4")
        if fps > 0: 
            make_video(png_dir, video_path, fps) 

        outcome = (
            "agent wins" if winner == agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )
        logging.info("ep %02d | %d 回合 | %s → %s", ep, step, outcome, video_path)

    logging.info("渲染结果已保存至 %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="渲染已训练的 LightWarGame 智能体对局")
    parser.add_argument("--scenario", type=str, default="1v1/vsbaseline",
                        help="env 配置名，如 1v1/vsbaseline")
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
    opp_policy = SB3Policy(path=opp_path) if opp_path else None

    out_dir = render_out_dir(args.scenario)
    run_render(agent_policy, args.scenario, out_dir, args.episodes, args.fps,
               max_turns=args.max_turns,
               agent_capital=args.agent_capital,
               opponent_capital=args.opponent_capital,
               opponent_policy=opp_policy)


if __name__ == "__main__":
    main()
