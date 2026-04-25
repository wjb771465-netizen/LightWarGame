"""
PPO 训练入口。

训练：
    conda run -n chinese_war_game python -m ai.trainer --scenario two_players/vsbaseline

渲染（加载已训练模型，运行对局并存图）：
    conda run -n chinese_war_game python -m ai.trainer --scenario two_players/vsbaseline --render

或使用预设脚本：
    bash scripts/train_vsrandom.sh
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from ai.args import get_config
from ai.envs.env import LwgEnv


def _resolve_save_dir(args) -> str:
    if args.save_dir is not None:
        return args.save_dir
    return os.path.join("ai", "checkpoints", args.scenario)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args) -> None:
    save_dir = _resolve_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    scenario = args.scenario
    env = make_vec_env(lambda: LwgEnv(scenario), n_envs=1)
    eval_env = make_vec_env(lambda: LwgEnv(scenario), n_envs=1)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": args.net_arch},
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        seed=args.seed,
    )

    project = args.wandb_project or args.scenario.split("/")[0]
    exp_name = args.exp_name or args.scenario.split("/")[-1]

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=save_dir,
            name_prefix="lwg_ppo",
        ),
    ]

    if args.use_eval:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            verbose=1,
        ))

    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(project=project, name=exp_name, config=vars(args))
        callbacks.append(WandbCallback())

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(os.path.join(save_dir, "final"))
    print(f"模型已保存至 {save_dir}/final.zip")


def render(args) -> None:
    """加载 final.zip，运行完整对局并存图（训练后用）。"""
    save_dir = _resolve_save_dir(args)
    model_path = os.path.join(save_dir, "final")
    assert os.path.exists(model_path + ".zip"), f"找不到模型：{model_path}.zip，请先训练"

    render_dir = os.path.join(save_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    model = MaskablePPO.load(model_path)
    scenario = args.scenario

    for ep in range(args.render_episodes):
        env = LwgEnv(scenario)
        obs, _ = env.reset()
        step = 0
        while True:
            env.render(os.path.join(render_dir, f"ep{ep:02d}_turn_{step:04d}.png"))
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            step += 1
            if terminated or truncated:
                break

        env.render(os.path.join(render_dir, f"ep{ep:02d}_turn_{step:04d}_final.png"))
        winner = env._state.winner()
        outcome = (
            "agent wins" if winner == env.agent_id
            else "opponent wins" if winner is not None
            else "draw (timeout)"
        )
        print(f"ep {ep:02d} | {step} 回合 | {outcome}")

    print(f"渲染图像已保存至 {render_dir}")


def main() -> None:
    parser = get_config()
    args = parser.parse_args()
    _set_seeds(args.seed)

    if args.render:
        render(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
