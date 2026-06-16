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


def _format_eval_specs(specs: list[dict]) -> str:
    """将 eval spec 列表格式化为可读的对手摘要字符串。"""
    def _label(s: dict) -> str:
        if s["type"] == "policy":
            step = s.get("path", "").rsplit("ckpt_", 1)[-1]
            return f"s{step}"
        return s["type"]
    return ", ".join(_label(s) for s in specs[:8]) + (", ..." if len(specs) > 8 else "")


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
        return SB3Policy(env=env, args=self.args, tb_log_dir=self.save_dir)

    def run(self, agent: SB3Policy, env: VecEnv) -> None:
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while agent.num_timesteps < total:
            agent.learn(min(chunk, total - agent.num_timesteps), callback=[self._win_cb])
            agent._model._custom_logger = True  # 复用 SummaryWriter，避免多 tfevents 文件
            step = agent.num_timesteps
            ckpt = os.path.join(self.save_dir, f"ckpt_{step}")
            agent.save(ckpt)
            if self.args.use_eval:
                self.eval(ckpt, env, step)
        agent.save(os.path.join(self.save_dir, "final"))
        logging.info("模型已保存至 %s/final.zip", self.save_dir)
        self.render(os.path.join(self.save_dir, "final"))

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval(self, ckpt: str, env, step: int, region: int | None = None) -> list:
        """评估 agent vs 对手并记录指标。子类覆写 choose_eval_opponents 以接入 pool。"""
        freq = max(1, self.args.eval_opponent_freq)
        ckpt_idx = step // max(1, self.args.checkpoint_freq)
        include_fixed = self.args.eval_opponent and (ckpt_idx % freq == 0)

        specs = self.choose_eval_opponents(env, include_fixed=include_fixed, region=region)
        if not specs:
            return []

        from ai.train.eval import evaluate, aggregate_win_rate, aggregate_avg_turns

        summary = _format_eval_specs(specs)
        logging.info("[Eval] step=%d n=%d eps=%d fixed=%s [%s]",
                     step, len(specs), self.args.eval_episodes, include_fixed, summary)
        results = evaluate(ckpt, specs, self.args.scenario, self.args.eval_episodes,
                           agent_capital=region)

        # 每种固定对手单独出指标
        by_type: dict[str, list] = {}
        for r in results:
            t = r.opponent_spec["type"]
            if t == "policy":
                continue
            by_type.setdefault(t, []).append(r)
        for opp_type, group in by_type.items():
            self.log_eval_metrics({
                f"vs_{opp_type}/win_rate": aggregate_win_rate(group),
                f"vs_{opp_type}/avg_turns": aggregate_avg_turns(group),
            }, step, region=region)

        return results

    def choose_eval_opponents(self, env, include_fixed: bool = True, region: int | None = None) -> list[dict]:
        """eval_n_envs 个固定对手 spec。SelfPlay/RegionSelfPlay 子类覆写以接入 pool。"""
        eval_n_envs = self.args.eval_n_envs or self.args.n_envs
        opponent_id = self._opponent_id(env)

        if self.args.eval_opponent_path:
            return eval_n_envs * [
                {"type": "policy", "player_id": opponent_id,
                 "path": self.args.eval_opponent_path},
            ]
        if include_fixed and self.args.eval_opponent:
            return self._fixed_opponent_specs(eval_n_envs, opponent_id)
        return []

    def _fixed_opponent_specs(self, eval_n_envs: int, opponent_id: int) -> list[dict]:
        opp_types = [s.strip() for s in self.args.eval_opponent.split(",")]
        n_types = len(opp_types)

        if n_types > eval_n_envs:
            picked = random.sample(opp_types, eval_n_envs)
            logging.warning(
                "[Eval] 固定对手类型数(%d) > eval 进程数(%d)，随机选 %d 种: %s",
                n_types, eval_n_envs, eval_n_envs, picked,
            )
            return [{"type": t, "player_id": opponent_id} for t in picked]

        if n_types < eval_n_envs:
            logging.warning(
                "[Eval] 固定对手类型数(%d) < eval 进程数(%d)，循环填充至 %d",
                n_types, eval_n_envs, eval_n_envs,
            )
            specs = []
            i = 0
            while len(specs) < eval_n_envs:
                specs.append({"type": opp_types[i % n_types], "player_id": opponent_id})
                i += 1
            return specs

        # n_types == eval_n_envs
        return [{"type": t, "player_id": opponent_id} for t in opp_types]

    def _opponent_id(self, env) -> int:
        agent_id: int = env.get_attr("agent_id")[0]
        num_players: int = env.get_attr("config")[0].game.num_players
        return next(p for p in range(1, num_players + 1) if p != agent_id)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_eval_metrics(self, metrics: dict, step: int, region: int | None = None) -> None:
        """记录 eval 指标到 W&B（训练指标走 TensorBoard sync，不在此重复）。"""
        if self.args.wandb:
            import wandb
            data = {"global_step": step}
            data.update({"eval/" + k: v for k, v in metrics.items() if v is not None})
            wandb.log(data)
        else:
            logging.info("step=%d %s", step, metrics)

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
            )
            # sync_tensorboard 下 wandb.log(step=...) 被忽略，改用 global_step 对齐 x 轴
            wandb.define_metric("eval/*", step_metric="global_step")
            wandb.define_metric("*", step_metric="global_step", hidden=True)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, ckpt: str, agent_capital: int | None = None) -> None:
        """训练结束后渲染一局对战视频（本地 PNG + MP4，若启用 wandb 则上传）。"""
        if not self.args.eval_opponent:
            return

        from ai.algos.policy import SB3Policy
        from ai.renders.render import _render_episode
        from ai.renders.utils import make_video

        agent_policy = SB3Policy(path=ckpt)
        opp_types = [s.strip() for s in self.args.eval_opponent.split(",")]
        videos = []
        wandb_enabled = self.args.wandb

        for opp_type in opp_types:
            out_dir = os.path.join(self.save_dir, "eval_videos", opp_type)
            _render_episode(agent_policy, self.args.scenario, 0, out_dir,
                            opponent_spec={"type": opp_type, "player_id": 2},
                            agent_capital=agent_capital)
            png_dir = os.path.join(out_dir, "ep00", "png")
            video_path = os.path.join(out_dir, "ep00", "replay.mp4")
            make_video(png_dir, video_path, fps=4)
            if wandb_enabled:
                import wandb
                videos.append(wandb.Video(video_path, caption=f"vs_{opp_type}"))
            logging.info("[Render] vs %s → %s", opp_type, video_path)

        if videos:
            wandb.log({"eval/videos": videos})
