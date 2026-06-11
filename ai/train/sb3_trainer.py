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
            ckpt = os.path.join(self.save_dir, f"ckpt_{step}")
            agent.save(ckpt)
            if self.args.use_eval:
                self.run_eval(ckpt, env, step)
            self.log_metrics(self._collect_metrics(), step)
        agent.save(os.path.join(self.save_dir, "final"))
        logging.info("模型已保存至 %s/final.zip", self.save_dir)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def run_eval(self, ckpt: str, env, step: int) -> list:
        """评估 agent vs 对手并记录指标。子类覆写 eval_opponent_specs 以接入 pool。"""
        specs = self.eval_opponent_specs(env)
        if not specs:
            return []

        from ai.train.eval import evaluate, aggregate_win_rate, aggregate_avg_turns

        logging.info("[Eval] step=%d n_envs=%d episodes_per=%d",
                     step, len(specs), self.args.eval_episodes)
        results = evaluate(ckpt, specs, self.args.scenario, self.args.eval_episodes)

        self.log_metrics({
            "eval/win_rate": aggregate_win_rate(results),
            "eval/avg_turns": aggregate_avg_turns(results),
            "eval/episodes": sum(r.episodes for r in results),
        }, step)
        return results

    def eval_opponent_specs(self, env) -> list[dict]:
        """eval_n_envs 个固定对手 spec。SelfPlay/RegionSelfPlay 子类覆写以接入 pool。"""
        eval_n_envs = self.args.eval_n_envs or self.args.n_envs
        opponent_id = self._opponent_id(env)

        if self.args.eval_opponent_path:
            return eval_n_envs * [
                {"type": "policy", "player_id": opponent_id,
                 "path": self.args.eval_opponent_path},
            ]
        return eval_n_envs * [
            {"type": self.args.eval_opponent, "player_id": opponent_id},
        ]

    def _opponent_id(self, env) -> int:
        agent_id: int = env.get_attr("agent_id")[0]
        num_players: int = env.get_attr("config")[0].game.num_players
        return next(p for p in range(1, num_players + 1) if p != agent_id)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

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
        logging.basicConfig(level=logging.INFO, format="%(message)s")
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
