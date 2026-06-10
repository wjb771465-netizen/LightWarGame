from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.algos.region_pool import RegionPool
from ai.envs.env import LwgEnv
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
        self.pool = RegionPool(history=args.self_play_pool_size)
        self._log_lock = threading.Lock()
        torch.set_num_threads(args.n_training_threads)

    # ------------------------------------------------------------------
    # Opponent sampling (region-aware override)
    # ------------------------------------------------------------------

    def _sample_opponent_specs(self, pool, n_total: int, opponent_id: int,
                               exclude_region: int) -> list[dict]:
        """从 RegionPool 中有放回采样，spec 额外携带 opp_region。"""
        strategy = self.args.pool_sampling_strategy
        lam = self.args.sampling_lam
        scale = self.args.sampling_scale

        specs = []
        for _ in range(n_total):
            result = pool.sample_opponent(
                exclude_region=exclude_region, strategy=strategy,
                lam=lam, s=scale, progress_D=self.args.progress_D,
            )
            if result is not None:
                rid, entry = result
                specs.append({
                    "type": "policy", "player_id": opponent_id,
                    "path": entry.path.replace(".zip", ""),
                    "opp_region": rid,
                })
            else:
                specs.append({"type": "rule", "player_id": opponent_id})
        return specs

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
                make_vec_env(lambda: LwgEnv(scenario), n_envs=self.args.n_envs,
                             vec_env_cls=SubprocVecEnv, monitor_kwargs=None),
                info_keywords=("win", "turn"),
            )
            envs[R] = env
            agents[R] = SB3Policy(
                env=env, args=self.args,
                tb_log_dir=os.path.join(self._region_dir(R), "tb"),
            )
            win_cbs[R] = WinRateCallback(window=self.args.win_rate_window)

        n_opponents = self.args.n_opponents or self.args.n_envs
        n_envs = self.args.n_envs
        if n_envs % n_opponents != 0:
            raise ValueError(
                f"--n-envs ({n_envs}) 须为 --n-opponents ({n_opponents}) 的整数倍"
            )
        per_opponent = n_envs // n_opponents

        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq

        while True:
            active = [R for R in self.regions if agents[R].num_timesteps < total]
            if not active:
                break

            # Phase 1: snapshot opponents from SAME pool state
            round_specs: dict[int, list[dict]] = {}
            for R in active:
                round_specs[R] = self._sample_opponent_specs(
                    self.pool, n_opponents, opponent_id=2, exclude_region=R)

            # Phase 2: train all regions, per-env opponent + capital
            def _train_chunk(R: int) -> None:
                agent = agents[R]
                env = envs[R]
                win_cb = win_cbs[R]
                specs = round_specs[R]
                steps = min(chunk, total - agent.num_timesteps)

                opp_info = []
                for i, spec in enumerate(specs):
                    indices = list(range(i * per_opponent, (i + 1) * per_opponent))
                    if not indices:
                        continue
                    env.env_method("set_opponent", spec, indices=indices)
                    opp_region = spec.get("opp_region")
                    if opp_region is None:
                        opp_region = next(r for r in self.regions if r != R)
                    env.env_method("set_capitals", R, opp_region, indices=indices)
                    if spec["type"] == "policy":
                        opp_info.append((opp_region, int(spec["path"].rsplit("ckpt_", 1)[-1])))
                    else:
                        opp_info.append((opp_region, spec["type"]))
                info_str = ", ".join(f"R{r}->s{step}" if isinstance(step, int) else f"R{r}->{step}"
                                     for r, step in opp_info[:8])
                if len(opp_info) > 8:
                    info_str += ", ..."
                logging.info("[RegionSP R=%d] agent=%d, opps=[%s], envs=%d, per=%d",
                             R, R, info_str, n_envs, per_opponent)

                agent.learn(steps, callback=[win_cb])
                step = agent.num_timesteps

                ckpt = os.path.join(self._region_dir(R), f"ckpt_{step}")
                agent.save(ckpt)
                self.pool.add(R, ckpt + ".zip", step)

                t = win_cb._tracker
                self.log_metrics({
                    f"region_{R}/win_rate_global": t.win_rate_global,
                    f"region_{R}/win_rate_window": t.win_rate_window,
                }, step)

            max_workers = max(1, min(self.args.parallel_regions, len(active)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_to_R = {executor.submit(_train_chunk, R): R for R in active}
                for f in as_completed(fut_to_R):
                    f.result()

            self.pool.save(os.path.join(self.save_dir, "pool_state.json"))

        for R in self.regions:
            agents[R].save(os.path.join(self._region_dir(R), "final"))
        logging.info("[RegionSelfPlay] 训练完成，模型保存至 %s", self.save_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self.args.wandb:
            import wandb
            with self._log_lock:
                wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        else:
            logging.info("step=%d %s", step, metrics)

    def _region_dir(self, R: int) -> str:
        return os.path.join(self.save_dir, f"region_{R}")
