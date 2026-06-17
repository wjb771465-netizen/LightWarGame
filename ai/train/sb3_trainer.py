from __future__ import annotations

import logging
import os
import random

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.train.metrics import WinRateCallback
from ai.train.utils import (
    checkpoint_path,
    final_model_path,
    format_eval_specs,
    render_paths,
    resolve_save_dir,
    set_seeds,
)


class Sb3Trainer:

    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = resolve_save_dir(args.scenario, args.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        set_seeds(args.seed)
        self.env = self.create_env()
        self.agent = self.create_agent(self.env, tb_log_dir=self.save_dir)
        self._win_cb = WinRateCallback(window=self.args.win_rate_window)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._init_logging()

        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while self.agent.num_timesteps < total:
            steps = min(chunk, total - self.agent.num_timesteps)
            self.agent.learn(steps, callback=[self._win_cb])
            self.agent._model._custom_logger = True
            step = self.agent.num_timesteps
            ckpt = checkpoint_path(self.save_dir, step)
            self.agent.save(ckpt)
            if self.args.use_eval:
                self.eval(ckpt, step)
        final = final_model_path(self.save_dir)
        self.agent.save(final)
        logging.info("模型已保存至 %s/final.zip", self.save_dir)
        self.render(final, save_dir=self.save_dir)

    # ------------------------------------------------------------------
    # Env / Agent factories
    # ------------------------------------------------------------------

    def create_env(self) -> VecEnv:
        scenario = self.args.scenario

        def _wrap_env():
            import sys, traceback as _tb
            env = LwgEnv(scenario)
            # 给所有 Gym API 方法包一层异常捕获，确保 traceback 进 stderr
            for _attr in ("step", "reset", "action_masks", "render", "close",
                          "set_opponent", "set_capitals"):
                _orig = getattr(env, _attr, None)
                if _orig is None:
                    continue
                def _safe(method=_orig, name=_attr):
                    def _wrapper(*a, **kw):
                        try:
                            return method(*a, **kw)
                        except Exception:
                            print(f"[SubprocEnv] {name} crashed:", file=sys.stderr, flush=True)
                            _tb.print_exc(file=sys.stderr)
                            sys.stderr.flush()
                            raise
                    return _wrapper
                setattr(env, _attr, _safe())
            return env

        return VecMonitor(
            make_vec_env(_wrap_env, n_envs=self.args.n_envs,
                         vec_env_cls=SubprocVecEnv, monitor_kwargs=None),
            info_keywords=("win", "turn"),
        )

    def create_agent(self, env: VecEnv, tb_log_dir: str) -> SB3Policy:
        resume = getattr(self.args, "resume_from", None)
        if resume:
            return SB3Policy(path=resume, env=env)
        return SB3Policy(env=env, args=self.args, tb_log_dir=tb_log_dir)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval(self, ckpt: str, step: int, region: int | None = None) -> list:
        """评估 agent vs 对手并记录指标。子类覆写 choose_eval_opponents 以接入 pool。"""
        freq = max(1, self.args.eval_opponent_freq)
        ckpt_idx = step // max(1, self.args.checkpoint_freq)
        include_fixed = self.args.eval_opponent and (ckpt_idx % freq == 0)

        specs = self.choose_eval_opponents(include_fixed=include_fixed, region=region)
        if not specs:
            return []

        from ai.train.eval import evaluate, aggregate_win_rate, aggregate_avg_turns

        summary = format_eval_specs(specs)
        logging.info("[Eval] step=%d n=%d eps=%d fixed=%s [%s]",
                     step, len(specs), self.args.eval_episodes, include_fixed, summary)
        results = evaluate(ckpt, specs, self.args.scenario, self.args.eval_episodes,
                           agent_capital=region)

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

    def choose_eval_opponents(self, include_fixed: bool = True, region: int | None = None) -> list[dict]:
        """eval_n_envs 个固定对手 spec。SelfPlay/RegionSelfPlay 子类覆写以接入 pool。"""
        eval_n_envs = self.args.eval_n_envs or self.args.n_envs
        opponent_id = self._opponent_id()

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

        return [{"type": t, "player_id": opponent_id} for t in opp_types]

    def _opponent_id(self) -> int:
        agent_id: int = self.env.get_attr("agent_id")[0]
        num_players: int = self.env.get_attr("config")[0].game.num_players
        return next(p for p in range(1, num_players + 1) if p != agent_id)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_eval_metrics(self, metrics: dict, step: int, region: int | None = None) -> None:
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
            wandb.define_metric("eval/*", step_metric="global_step")
            wandb.define_metric("*", step_metric="global_step", hidden=True)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, ckpt: str, *, save_dir: str, agent_capital: int | None = None,
               opponent_capital: int | None = None, max_turns: int = 60) -> None:
        if not self.args.eval_opponent:
            return

        from ai.algos.policy import SB3Policy
        from ai.renders.render import render as render_episodes

        if self.args.wandb:
            import wandb

        agent_policy = SB3Policy(path=ckpt)
        opp_types = [s.strip() for s in self.args.eval_opponent.split(",")]
        wandb_videos = []

        base, wandb_key = render_paths(save_dir)
        for opp_type in opp_types:
            env = LwgEnv(self.args.scenario)
            env.config.game.max_turns = max_turns
            if agent_capital is not None and opponent_capital is not None:
                env.set_capitals(agent_capital, opponent_capital)
            env.set_opponent({"type": opp_type, "player_id": 2})

            out_dir = os.path.join(base, "eval_videos", opp_type)
            video_paths = render_episodes(agent_policy, env, out_dir, 1, fps=4)
            if self.args.wandb and video_paths:
                wandb_videos.append(wandb.Video(video_paths[0], caption=f"vs_{opp_type}", format="mp4"))

        if wandb_videos:
            wandb.log({wandb_key: wandb_videos})
