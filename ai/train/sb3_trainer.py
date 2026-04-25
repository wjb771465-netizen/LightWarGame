from __future__ import annotations

import os
import random
from datetime import datetime

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from ai.train.args import get_config
from ai.envs.env import LwgEnv


def _resolve_save_dir(args) -> str:
    if args.save_dir is not None:
        return args.save_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("ai", "train", "results", args.scenario, f"run_{ts}")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args) -> None:
    save_dir = _resolve_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    tb_log_dir = os.path.join(save_dir, "tb")

    scenario = args.scenario
    env = VecMonitor(make_vec_env(lambda: LwgEnv(scenario), n_envs=1))
    eval_env = VecMonitor(make_vec_env(lambda: LwgEnv(scenario), n_envs=1))

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
        tensorboard_log=tb_log_dir,
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

        wandb.init(
            project=project,
            name=exp_name,
            config=vars(args),
            dir=save_dir,
            sync_tensorboard=True,
            monitor_gym=True,
        )
        callbacks.append(WandbCallback(verbose=2))
    else:
        model.set_logger(configure(folder=None, format_strings=["stdout"]))

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(os.path.join(save_dir, "final"))
    print(f"模型已保存至 {save_dir}/final.zip")


def main() -> None:
    parser = get_config()
    args = parser.parse_args()
    _set_seeds(args.seed)
    train(args)


if __name__ == "__main__":
    main()
