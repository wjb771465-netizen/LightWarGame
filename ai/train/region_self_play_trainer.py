from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.algos.region_pool import RegionPool
from ai.envs.env import LwgEnv
from ai.envs.opponents import PolicyOpponent
from ai.train.metrics import WinRateCallback
from ai.train.self_play_trainer import SelfPlayTrainer


class RegionSelfPlayTrainer(SelfPlayTrainer):
    """每个地区维护独立模型，支持多线程并行训练，对手从地区策略池中随机抽取。"""

    def __init__(self, args) -> None:
        super().__init__(args)
        raw = getattr(args, "region_self_play_regions", None)
        self.regions: list[int] = (
            list(range(1, 32)) if raw is None
            else [int(x.strip()) for x in raw.split(",")]
        )
        self.pool = RegionPool(history=args.region_pool_history)
        self._log_lock = threading.Lock()
        torch.set_num_threads(args.n_training_threads)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._init_logging()

        scenario = self.args.scenario
        envs: dict[int, VecMonitor] = {}
        agents: dict[int, SB3Policy] = {}
        win_cbs: dict[int, WinRateCallback] = {}

        for R in self.regions:
            os.makedirs(self._region_dir(R), exist_ok=True)
            env = VecMonitor(
                make_vec_env(lambda: LwgEnv(scenario), n_envs=self.args.n_envs, vec_env_cls=SubprocVecEnv, monitor_kwargs=None),
                info_keywords=("win", "turn"),
            )
            envs[R] = env
            agents[R] = SB3Policy(
                env=env, args=self.args,
                tb_log_dir=os.path.join(self._region_dir(R), "tb"),
            )
            win_cbs[R] = WinRateCallback(window=self.args.win_rate_window)

        max_workers = min(self.args.parallel_regions, len(self.regions))
        pending: set[int] = set(self.regions)
        pending_lock = threading.Lock()

        def worker() -> None:
            while True:
                with pending_lock:
                    if not pending:
                        return
                    R = pending.pop()
                self._train_region(R, envs[R], agents[R], win_cbs[R])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(max_workers)]
            for f in as_completed(futures):
                f.result()  # propagate any exception from worker

        for R in self.regions:
            agents[R].save(os.path.join(self._region_dir(R), "final"))
        print(f"[RegionSelfPlay] 训练完成，模型保存至 {self.save_dir}")

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def _train_region(self, R: int, env, agent, win_cb) -> None:
        """Train a single region to completion."""
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq

        while agent.num_timesteps < total:
            # (1) Sample opponent — RegionPool handles its own locking
            result = self.pool.sample_opponent(exclude_region=R)

            if result is None:
                # 冷启动：池子还没有任何 checkpoint，用规则对手占位
                opp_region = next(r for r in self.regions if r != R)
                opp = self._make_warmup_opponent("rule", player_id=2)
            else:
                opp_region, entry = result
                # Model load happens OUTSIDE the pool lock (expensive I/O)
                opp = PolicyOpponent(
                    player_id=2,
                    policy=SB3Policy(path=entry.path),
                    obs_encoder=env.get_attr("obs_encoder")[0],
                    act_encoder=env.get_attr("act_encoder")[0],
                )

            # (2) Configure environment with opponent and capitals
            env.env_method("set_opponent", opp)
            env.env_method("set_capitals", R, opp_region)

            # (3) Train one chunk
            steps = min(chunk, total - agent.num_timesteps)
            agent.learn(steps, callback=[win_cb])
            step = agent.num_timesteps

            # (4) Save checkpoint (unique path per region, no contention)
            ckpt = os.path.join(self._region_dir(R), f"ckpt_{step}")
            agent.save(ckpt)

            # (5) Update shared pool (thread-safe internally)
            self.pool.add(R, ckpt + ".zip", step)
            self.pool.save(os.path.join(self.save_dir, "pool_state.json"))

            # (6) Log metrics
            t = win_cb._tracker
            self.log_metrics({
                f"region_{R}/win_rate_global": t.win_rate_global,
                f"region_{R}/win_rate_window": t.win_rate_window,
            }, step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Thread-safe metrics logging."""
        if self.args.wandb:
            import wandb
            with self._log_lock:
                wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        else:
            import logging
            logging.info("step=%d %s", step, metrics)

    def _region_dir(self, R: int) -> str:
        return os.path.join(self.save_dir, f"region_{R}")
