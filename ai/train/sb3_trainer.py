from __future__ import annotations

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.train.metrics import WinRateCallback


def _resolve_save_dir(args) -> str:
    if args.save_dir is not None:
        return args.save_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("ai", "train", "results", args.scenario, f"run_{ts}")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Sb3Trainer:

    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = _resolve_save_dir(args)
        os.makedirs(self.save_dir, exist_ok=True)
        _set_seeds(args.seed)

    def train(self) -> None:
        env   = self.create_env()
        agent = self.create_agent(env)
        self._win_cb = WinRateCallback(window=self.args.win_rate_window)
        self._init_logging()
        self.run(agent, env)

    def create_env(self) -> VecEnv:
        scenario = self.args.scenario
        return VecMonitor(
            make_vec_env(lambda: LwgEnv(scenario), n_envs=self.args.n_envs, vec_env_cls=SubprocVecEnv, monitor_kwargs=None),
            info_keywords=("win", "turn"),
        )

    def create_agent(self, env: VecEnv) -> SB3Policy:
        resume = getattr(self.args, "resume_from", None)
        if resume:
            return SB3Policy(path=resume, env=env)
        return SB3Policy(env=env, args=self.args, tb_log_dir=os.path.join(self.save_dir, "tb"))

    def run(self, agent: SB3Policy, env: VecEnv) -> None:
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while agent.num_timesteps < total:
            agent.learn(min(chunk, total - agent.num_timesteps), callback=[self._win_cb])
            step = agent.num_timesteps
            agent.save(os.path.join(self.save_dir, f"ckpt_{step}"))
            self.log_metrics(self._collect_metrics(), step)
        agent.save(os.path.join(self.save_dir, "final"))
        print(f"模型已保存至 {self.save_dir}/final.zip")

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self.args.wandb:
            import wandb
            wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        else:
            logging.info("step=%d %s", step, metrics)

    def _collect_metrics(self) -> dict:
        t = self._win_cb._tracker
        return {
            "win_rate_global": t.win_rate_global,
            "win_rate_window": t.win_rate_window,
        }

    def _init_logging(self) -> None:
        if self.args.wandb:
            import wandb
            wandb.init(
                project=self.args.wandb_project or self.args.scenario.split("/")[0],
                name=self.args.exp_name or self.args.scenario.split("/")[-1],
                config=vars(self.args),
                dir=self.save_dir,
                sync_tensorboard=True,
                monitor_gym=True,
            )
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
